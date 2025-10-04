import os
import json
import librosa
import numpy as np
import torch
import torchaudio
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, WhisperProcessor, WhisperForConditionalGeneration, ClapModel, ClapProcessor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize models
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def initialize_wav2vec_model(model_name):
    """Initialize Wav2Vec2 model with configurable model name"""
    print(f"Loading Wav2Vec2 model: {model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Get hidden dimension from model config
    hidden_dim = model.config.hidden_size
    print(f"Wav2Vec2 hidden dimension: {hidden_dim}")
    
    return feature_extractor, model, hidden_dim

def is_clap_model(model_name):
    """Check if the model is a CLAP model"""
    return "clap" in model_name.lower()

def get_model_mark(model_name):
    """Extract model mark from model name for file suffixes"""
    return model_name.replace("/", "-").replace("_", "-")

def initialize_clap_model(model_name):
    """Initialize CLAP model with configurable model name"""
    print(f"Loading CLAP model: {model_name}")
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # CLAP models typically output 512-dimensional features
    hidden_dim = 512
    print(f"CLAP hidden dimension: {hidden_dim}")
    
    return processor, model, hidden_dim

print(f"Using device: {device}")

def initialize_whisper_model():
    """Initialize Whisper model for transcript generation"""
    print("Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return processor, model

def load_frame_timestamps(timestamp_file):
    """Load frame timestamps from file"""
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                frame_name = parts[0]
                start_time = float(parts[1])
                end_time = float(parts[2])
                timestamps.append((frame_name, start_time, end_time))
    return timestamps

def extract_frame_features(video_id, audio_path, frame_timestamps, model_processor, audio_model, hidden_dim, model_name):
    """Extract audio features from frame-aligned segments using CLAP or Wav2Vec2"""
    try:
        # Load full audio with appropriate sampling rate
        is_clap = is_clap_model(model_name)
        target_sr = 48000 if is_clap else 16000
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        frame_features = []
        
        for i, (frame_name, start_time, end_time) in enumerate(frame_timestamps):
            # Extract segment
            start_sample = int(start_time * target_sr)
            end_sample = int(end_time * target_sr)
            
            # Ensure valid segment bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample <= start_sample:
                # Invalid segment, use zeros
                segment_length = int(0.5 * target_sr)  # 0.5 second default
                if is_clap:
                    segment = torch.zeros(segment_length, dtype=torch.float32).numpy()
                else:
                    segment = torch.zeros(segment_length, dtype=torch.float32)
            else:
                segment = audio[start_sample:end_sample]
                if not is_clap:
                    # Convert to torch tensor for wav2vec
                    segment = torch.tensor(segment, dtype=torch.float32)
            
            try:
                if is_clap:
                    # Process with CLAP
                    # Ensure minimum length for CLAP (typically needs at least some audio)
                    min_samples = int(0.1 * target_sr)  # 0.1 seconds at 48kHz = 4800 samples
                    if len(segment) < min_samples:
                        padding = min_samples - len(segment)
                        segment = np.pad(segment, (0, padding), mode='constant', constant_values=0)
                    
                    inputs = model_processor(audios=segment, return_tensors="pt", sampling_rate=target_sr)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        features = audio_model.get_audio_features(**inputs)  # [1, hidden_dim]
                        frame_features.append(features.cpu())
                else:
                    # Process through wav2vec (ensure minimum length)
                    if len(segment) < 400:  # Wav2Vec2 minimum length
                        segment = torch.cat([segment, torch.zeros(400 - len(segment))])
                    
                    segment = segment.unsqueeze(0).to(device)
                    with torch.no_grad():
                        wav2vec_outputs = audio_model(segment)
                        # Use mean pooling over sequence length to get fixed-size features
                        features = wav2vec_outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
                        frame_features.append(features.cpu())
                        
            except Exception as e:
                model_type = "CLAP" if is_clap else "Wav2Vec2"
                print(f"{model_type} error for {video_id}, {frame_name}: {e}")
                # Fallback: zero features
                frame_features.append(torch.zeros(1, hidden_dim))
        
        # Stack all frame features
        if frame_features:
            stacked_features = torch.cat(frame_features, dim=0)  # [16, hidden_dim]
        else:
            print(f"ERROR: {video_id} - No valid features extracted, using zero tensor")
            stacked_features = torch.zeros(16, hidden_dim)
            
        return stacked_features
        
    except Exception as e:
        print(f"Error processing features for {video_id}: {e}")
        # Return zero features
        return torch.zeros(16, hidden_dim)

def generate_frame_transcripts(video_id, audio_path, frame_timestamps, whisper_processor, whisper_model):
    """Generate transcripts from frame-aligned audio segments using whisper"""
    try:
        # Load full audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        frame_transcripts = []
        
        for i, (frame_name, start_time, end_time) in enumerate(frame_timestamps):
            # Extract segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Ensure valid segment bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample <= start_sample:
                # Invalid segment
                transcript = ""
            else:
                segment = audio[start_sample:end_sample]
                
                # Generate transcript if segment is long enough
                if len(segment) > 0.1 * sr:  # At least 0.1 seconds
                    try:
                        # Process with Whisper
                        input_features = whisper_processor(segment, sampling_rate=16000, return_tensors="pt").input_features
                        input_features = input_features.to(device, dtype=torch_dtype)  # Match model dtype
                        
                        with torch.no_grad():
                            predicted_ids = whisper_model.generate(input_features)
                            transcript = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                            
                    except Exception as e:
                        print(f"ERROR: Whisper failed for {video_id}, {frame_name}: {e}")
                        transcript = ""
                else:
                    transcript = ""
            
            frame_transcripts.append(transcript.strip())
        
        return frame_transcripts
        
    except Exception as e:
        print(f"Error processing transcripts for {video_id}: {e}")
        # Return empty transcripts
        return [""] * 16

def process_frame_aligned_audio(video_id, audio_path, frame_timestamps, model_processor, audio_model, hidden_dim, skip_transcripts=False, whisper_processor=None, whisper_model=None, skip_features=False, model_name=None):
    """Wrapper function to process audio features and/or transcripts"""
    
    # Initialize defaults
    frame_features = None
    frame_transcripts = []
    
    # Extract features if not skipping
    if not skip_features:
        frame_features = extract_frame_features(video_id, audio_path, frame_timestamps, model_processor, audio_model, hidden_dim, model_name)
    else:
        # Use zero features when skipping
        frame_features = torch.zeros(16, hidden_dim)
    
    # Generate transcripts if not skipping
    if not skip_transcripts:
        frame_transcripts = generate_frame_transcripts(video_id, audio_path, frame_timestamps, whisper_processor, whisper_model)
    else:
        # Use empty transcripts when skipping
        frame_transcripts = [""] * 16
    
    return frame_features, frame_transcripts

def process_global_audio(audio_path, audio_model, hidden_dim, model_processor=None, model_name=None):
    """Process full audio file for global features using CLAP or Wav2Vec2"""
    try:
        # Load full audio with appropriate sampling rate
        is_clap = is_clap_model(model_name) if model_name else False
        target_sr = 48000 if is_clap else 16000
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        if is_clap and model_processor:
            # Process with CLAP
            inputs = model_processor(audios=audio, return_tensors="pt", sampling_rate=target_sr)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                global_features = audio_model.get_audio_features(**inputs)  # [1, hidden_dim]
                
            return global_features.cpu().squeeze(0)  # [hidden_dim]
        else:
            # Process through wav2vec
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                wav2vec_outputs = audio_model(audio_tensor)
                # Use mean pooling over sequence length to get fixed-size features
                global_features = wav2vec_outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
                
            return global_features.cpu().squeeze(0)  # [hidden_dim]
        
    except Exception as e:
        model_type = "CLAP" if (is_clap_model(model_name) if model_name else False) else "Wav2Vec2"
        print(f"Error processing global audio {audio_path} with {model_type}: {e}")
        return torch.zeros(hidden_dim)

def create_dummy_audio_features(video_id, hidden_dim):
    """为没有音频的视频创建dummy features，维度与模型输出一致"""
    print(f"Generating dummy audio features for {video_id} (dim: {hidden_dim})")
    frame_features = torch.zeros(16, hidden_dim)  # 16帧特征
    global_features = torch.zeros(hidden_dim)     # 全局特征
    frame_transcripts = [""] * 16                 # 空frame转录
    return frame_features, global_features, frame_transcripts

def load_existing_results(dataset_name, model_mark, skip_transcripts=False):
    """Load existing processing results to enable resume functionality"""
    existing_frame_features = []
    existing_global_features = []
    existing_frame_transcripts = []
    existing_video_ids = set()
    
    # Try to load existing features with model mark
    frame_features_path = f"data/{dataset_name}/fea/audio_features_frames_{model_mark}.pt"
    global_features_path = f"data/{dataset_name}/fea/audio_features_global_{model_mark}.pt"
    video_ids_path = f"data/{dataset_name}/fea/audio_video_ids_{model_mark}.txt"
    transcripts_path = f"data/{dataset_name}/transcript_frames.jsonl"
    
    if os.path.exists(frame_features_path) and os.path.exists(global_features_path) and os.path.exists(video_ids_path):
        try:
            # Load existing features
            frame_features_data = torch.load(frame_features_path)
            global_features_data = torch.load(global_features_path)
            
            # Check if both are dictionaries (new format)
            if isinstance(frame_features_data, dict) and isinstance(global_features_data, dict):
                # New format: both are dictionaries with video IDs as keys
                existing_video_list = list(global_features_data.keys())
                
                # Load existing transcripts (only if not skipping)
                existing_transcripts_dict = {}
                if not skip_transcripts and os.path.exists(transcripts_path):
                    with open(transcripts_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            existing_transcripts_dict[data['vid']] = data['frame_transcripts']
                
                # Convert to lists for merging
                for video_id in existing_video_list:
                    existing_frame_features.append(frame_features_data[video_id])
                    existing_global_features.append(global_features_data[video_id])
                    if skip_transcripts:
                        existing_frame_transcripts.append([""] * 16)  # Empty transcripts when skipping
                    else:
                        existing_frame_transcripts.append(existing_transcripts_dict.get(video_id, [""] * 16))
                    existing_video_ids.add(video_id)
            elif isinstance(global_features_data, dict):
                # Mixed format: frame features is tensor, global is dict
                existing_video_list = list(global_features_data.keys())
                
                # Load existing transcripts (only if not skipping)
                existing_transcripts_dict = {}
                if not skip_transcripts and os.path.exists(transcripts_path):
                    with open(transcripts_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            existing_transcripts_dict[data['vid']] = data['frame_transcripts']
                
                # Convert to lists for merging
                for i, video_id in enumerate(existing_video_list):
                    existing_frame_features.append(frame_features_data[i])
                    existing_global_features.append(global_features_data[video_id])
                    if skip_transcripts:
                        existing_frame_transcripts.append([""] * 16)  # Empty transcripts when skipping
                    else:
                        existing_frame_transcripts.append(existing_transcripts_dict.get(video_id, [""] * 16))
                    existing_video_ids.add(video_id)
            else:
                # Old format: tensor with separate video ID file
                # Load existing video IDs
                with open(video_ids_path, 'r') as f:
                    existing_video_list = [line.strip() for line in f.readlines()]
                
                # Load existing transcripts (only if not skipping)
                existing_transcripts_dict = {}
                if not skip_transcripts and os.path.exists(transcripts_path):
                    with open(transcripts_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line.strip())
                            existing_transcripts_dict[data['vid']] = data['frame_transcripts']
                
                # Convert to lists for merging
                for i, video_id in enumerate(existing_video_list):
                    existing_frame_features.append(frame_features_tensor[i])
                    existing_global_features.append(global_features_data[i])
                    if skip_transcripts:
                        existing_frame_transcripts.append([""] * 16)  # Empty transcripts when skipping
                    else:
                        existing_frame_transcripts.append(existing_transcripts_dict.get(video_id, [""] * 16))
                    existing_video_ids.add(video_id)
            
            print(f"Loaded {len(existing_video_list)} existing processed videos")
            
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")
            print("Starting from scratch...")
            existing_frame_features = []
            existing_global_features = []
            existing_frame_transcripts = []
            existing_video_ids = set()
    else:
        print("No existing results found, starting from scratch...")
    
    return existing_frame_features, existing_global_features, existing_frame_transcripts, existing_video_ids

def process_dataset(dataset_name, model_name, skip_transcripts=False, skip_features=False):
    """Process dataset for frame-level and global audio features with resume capability"""
    
    # Initialize audio model conditionally
    model_processor, audio_model, hidden_dim = None, None, None
    model_mark = get_model_mark(model_name)
    
    if not skip_features:
        if is_clap_model(model_name):
            model_processor, audio_model, hidden_dim = initialize_clap_model(model_name)
        else:
            model_processor, audio_model, hidden_dim = initialize_wav2vec_model(model_name)
    else:
        # Still need hidden_dim for tensor shapes when skipping features
        if is_clap_model(model_name):
            hidden_dim = 512  # CLAP models
        elif "CAiRE" in model_name:
            hidden_dim = 1024
        else:
            hidden_dim = 768  # Default for facebook models
    
    # Initialize whisper model conditionally
    whisper_processor, whisper_model = None, None
    if not skip_transcripts:
        whisper_processor, whisper_model = initialize_whisper_model()
    
    frames_dir = f"data/{dataset_name}/frames_16"
    audios_dir = f"data/{dataset_name}/audios"
    
    # Create output directories
    os.makedirs(f"data/{dataset_name}/fea", exist_ok=True)
    
    # Get list of all video IDs from frames directory
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    
    video_dirs = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
    print(f"Found {len(video_dirs)} total videos in frames directory")
    
    # Load existing results with model mark
    all_frame_features, all_global_features, all_frame_transcripts, existing_video_ids = load_existing_results(dataset_name, model_mark, skip_transcripts)
    all_video_ids = list(existing_video_ids)
    
    # Filter out already processed videos
    remaining_videos = [vid for vid in video_dirs if vid not in existing_video_ids]
    print(f"Already processed: {len(existing_video_ids)} videos")
    print(f"Remaining to process: {len(remaining_videos)} videos")
    
    # Process remaining videos only
    if remaining_videos:
        print(f"Processing {len(remaining_videos)} remaining videos...")
        for video_id in tqdm(remaining_videos, desc="Processing remaining videos"):
            video_frames_dir = os.path.join(frames_dir, video_id)
            timestamp_file = os.path.join(video_frames_dir, "frame_timestamps.txt")
            audio_path = os.path.join(audios_dir, f"{video_id}_full.wav")
            
            # Check if timestamp file exists
            if not os.path.exists(timestamp_file):
                print(f"ERROR: Timestamp file not found for {video_id}: {timestamp_file}")
                continue
                
            # Check if audio file exists
            if not os.path.exists(audio_path):
                if dataset_name in ['TwitterVideo', 'FVC']:
                    print(f"INFO: No audio found for {video_id} in {dataset_name}, generating dummy features")
                    frame_features, global_features, frame_transcripts = create_dummy_audio_features(video_id, hidden_dim)
                    # Store dummy results
                    all_frame_features.append(frame_features)
                    all_global_features.append(global_features)
                    all_frame_transcripts.append(frame_transcripts)
                    all_video_ids.append(video_id)
                    continue
                else:
                    print(f"ERROR: Audio file not found for {video_id}: {audio_path}")
                    continue
            
            # Load frame timestamps
            frame_timestamps = load_frame_timestamps(timestamp_file)
            if len(frame_timestamps) != 16:
                print(f"ERROR: Expected 16 frame timestamps for {video_id}, got {len(frame_timestamps)} - skipping video")
                continue
            elif len(frame_timestamps) == 0:
                print(f"ERROR: No timestamps found in file for {video_id}: {timestamp_file}")
                continue
            
            # Process frame-level audio
            frame_features, frame_transcripts = process_frame_aligned_audio(video_id, audio_path, frame_timestamps, model_processor, audio_model, hidden_dim, skip_transcripts, whisper_processor, whisper_model, skip_features, model_name)
            
            # Process global audio
            if not skip_features:
                global_features = process_global_audio(audio_path, audio_model, hidden_dim, model_processor, model_name)
            else:
                global_features = torch.zeros(hidden_dim)
            
            # Store results
            all_frame_features.append(frame_features)
            all_global_features.append(global_features)
            all_frame_transcripts.append(frame_transcripts)
            all_video_ids.append(video_id)
    else:
        print("All videos have already been processed!")
    
    # Final processing statistics
    total_available = len(video_dirs)
    successfully_processed = len(all_video_ids)
    newly_processed = successfully_processed - len(existing_video_ids)
    skipped = total_available - successfully_processed
    
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total videos available: {total_available}")
    print(f"Previously processed: {len(existing_video_ids)}")
    print(f"Newly processed: {newly_processed}")
    print(f"Total successfully processed: {successfully_processed}")
    print(f"Skipped due to errors: {skipped}")
    
    if skipped > 0:
        print(f"WARNING: {skipped} videos were skipped due to missing files or invalid timestamps")
    
    if all_video_ids:
        # Save features (only if not skipping)
        if not skip_features:
            # Create dictionaries for both frame and global features with video IDs as keys
            frame_features_dict = {}
            global_features_dict = {}
            for video_id, frame_feat, global_feat in zip(all_video_ids, all_frame_features, all_global_features):
                frame_features_dict[video_id] = frame_feat
                global_features_dict[video_id] = global_feat
            
            # Save features as dictionaries with model mark
            torch.save(frame_features_dict, f"data/{dataset_name}/fea/audio_features_frames_{model_mark}.pt")
            torch.save(global_features_dict, f"data/{dataset_name}/fea/audio_features_global_{model_mark}.pt")
            
            # Save video ID mapping for reference with model mark
            with open(f"data/{dataset_name}/fea/audio_video_ids_{model_mark}.txt", 'w') as f:
                for video_id in all_video_ids:
                    f.write(f"{video_id}\n")
        
        # Save frame transcripts (only if not skipping)
        if not skip_transcripts:
            frame_transcripts_data = []
            for video_id, transcripts in zip(all_video_ids, all_frame_transcripts):
                frame_transcripts_data.append({
                    'vid': video_id,
                    'frame_transcripts': transcripts
                })
            
            with open(f"data/{dataset_name}/transcript_frames.jsonl", 'w', encoding='utf-8') as f:
                for item in frame_transcripts_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n=== OUTPUT FILES ===")
        if not skip_features:
            print(f"Frame features: dict with {len(frame_features_dict)} video IDs -> data/{dataset_name}/fea/audio_features_frames_{model_mark}.pt")
            print(f"Global features: dict with {len(global_features_dict)} video IDs -> data/{dataset_name}/fea/audio_features_global_{model_mark}.pt")
        else:
            print("Audio features: Skipped (--skip_features enabled)")
        
        if not skip_transcripts:
            print(f"Frame transcripts: {len(frame_transcripts_data)} videos -> data/{dataset_name}/transcript_frames.jsonl")
        else:
            print("Frame transcripts: Skipped (--skip_transcripts enabled)")
        
        # Calculate transcript quality statistics (only if not skipping)
        if not skip_transcripts:
            total_transcripts = 0
            non_empty_transcripts = 0
            total_transcript_chars = 0
            
            for transcripts in all_frame_transcripts:
                for transcript in transcripts:
                    total_transcripts += 1
                    if transcript.strip():
                        non_empty_transcripts += 1
                        total_transcript_chars += len(transcript)
            
            transcript_success_rate = (non_empty_transcripts / total_transcripts * 100) if total_transcripts > 0 else 0
            avg_transcript_length = (total_transcript_chars / non_empty_transcripts) if non_empty_transcripts > 0 else 0
            
            print(f"\n=== TRANSCRIPT QUALITY ===")
            print(f"Total frame segments: {total_transcripts}")
            print(f"Successful transcripts: {non_empty_transcripts} ({transcript_success_rate:.1f}%)")
            print(f"Average transcript length: {avg_transcript_length:.1f} characters")
            
            if transcript_success_rate < 10:
                print(f"WARNING: Very low transcript success rate ({transcript_success_rate:.1f}%) - check audio quality")
            elif transcript_success_rate < 30:
                print(f"WARNING: Low transcript success rate ({transcript_success_rate:.1f}%) - many silent segments")
        
    else:
        print("No videos were successfully processed!")

def main():
    parser = argparse.ArgumentParser(description='Process audio features for video dataset')
    parser.add_argument('--dataset', type=str, default='FakeSV', 
                        choices=['FakeSV', 'FakeTT', 'FVC', 'TwitterVideo'],
                        help='Dataset name (default: FakeSV)')
    parser.add_argument('--wav2vec_model', type=str, default='CAiRE/SER-wav2vec2-large-xlsr-53-eng-zho-all-age',
                        help='Audio model name - supports both Wav2Vec2 and CLAP models (e.g., laion/clap-htsat-fused)')
    parser.add_argument('--skip_transcripts', action='store_true',
                        help='Skip transcript generation and only extract audio features')
    parser.add_argument('--skip_features', action='store_true',
                        help='Skip feature extraction and only generate transcripts')
    
    args = parser.parse_args()
    
    model_mark = get_model_mark(args.wav2vec_model)
    model_type = "CLAP" if is_clap_model(args.wav2vec_model) else "Wav2Vec2"
    print(f"Processing dataset: {args.dataset}")
    print(f"Using {model_type} model: {args.wav2vec_model}")
    print(f"Model mark for files: {model_mark}")
    print(f"Expected directory structure:")
    print(f"  - data/{args.dataset}/frames_16/ (video frame directories with timestamps)")
    print(f"  - data/{args.dataset}/audios/ (audio files)")
    print(f"Will create:")
    print(f"  - data/{args.dataset}/fea/audio_features_frames_{model_mark}.pt")
    print(f"  - data/{args.dataset}/fea/audio_features_global_{model_mark}.pt")
    print(f"  - data/{args.dataset}/transcript_frames.jsonl")
    print()
    
    # Check for conflicting arguments
    if args.skip_features and args.skip_transcripts:
        print("ERROR: Cannot skip both features and transcripts. Nothing would be processed!")
        return
    
    process_dataset(args.dataset, args.wav2vec_model, args.skip_transcripts, args.skip_features)

if __name__ == "__main__":
    main()
