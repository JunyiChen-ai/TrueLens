from transformers import ViTImageProcessor, ViTModel, CLIPVisionModel, CLIPImageProcessor
from transformers import ChineseCLIPProcessor, ChineseCLIPModel, ChineseCLIPImageProcessor, ChineseCLIPVisionModel
from transformers import AutoFeatureExtractor, ResNetModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from PIL import Image
import re
from tqdm import tqdm
import os
import numpy as np
import torch
import av
import argparse
from PIL import Image

NUM_FRAMES = 16

def robust_frame_extraction(video_path, num_frames):
    pil_images = []
    decoded_count = 0
    
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            total_frames = stream.frames
            duration = stream.duration * stream.time_base
            
            if total_frames == 0 or duration <= 0:
                raise ValueError(f"Video has no valid frames or duration: {video_path}")
            
            target_timestamps = [t * duration / num_frames for t in range(num_frames)]
            
            for timestamp in target_timestamps:
                container.seek(int(timestamp * stream.time_base.denominator), stream=stream)
                for frame in container.decode(video=0):
                    pil_image = frame.to_image()
                    pil_images.append(pil_image)
                    decoded_count += 1
                    break  # We only need the first frame at each timestamp
    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
    
    # If the extracted frames are insufficient, fill by duplicating the last frame
    if len(pil_images) < num_frames:
        last_frame = pil_images[-1] if pil_images else Image.new('RGB', (224, 224), color='black')
        pil_images.extend([last_frame] * (num_frames - len(pil_images)))

    # Minimal quality log for FVC only: how many frames were filled (i.e., not decoded)
    if '/FVC/' in video_path:
        filled = max(0, num_frames - decoded_count)
        if filled > 0:
            print(f"[AV INFO] video={video_path} decoded={decoded_count}/{num_frames}, filled={filled}/{num_frames}")
    
    return pil_images[:num_frames]  # Ensure the correct number of frames is returned

class VideoDataset(Dataset):
    def __init__(self, src_file, video_dir):
        self.data = pd.read_json(src_file, lines=True, dtype={'video_id': str})
        self.video_dir = video_dir
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid = self.data.iloc[index]['video_id']
        video_path = os.path.join(self.video_dir, f'{vid}.mp4')
        pil_images = robust_frame_extraction(video_path, NUM_FRAMES)
        return vid, pil_images

def customed_collate_fn(batch, processor):
    # Preprocess and merge to one list
    vids, pil_images = zip(*batch)
    images = []
    for images_list in pil_images:
        images.extend(images_list)
    inputs = processor(images, return_tensors="pt")
    return vids, inputs

def process_dataset(dataset_name):
    print(f"Processing {dataset_name} dataset...")
    
    # Dataset-specific configurations
    configs = {
        "FakeSV": {
            "src_file": f"data/{dataset_name}/data_complete.jsonl",
            "output_dir": f"data/{dataset_name}/fea",
            "video_dir": f"data/{dataset_name}/video",
            "model_id": "OFA-Sys/chinese-clip-vit-large-patch14",
            "use_chinese_clip": True,
            "output_file": "vit_tensor.pt"
        },
        "FakeTT": {
            "src_file": f"data/{dataset_name}/data_complete.jsonl",
            "output_dir": f"data/{dataset_name}/fea",
            "video_dir": f"data/{dataset_name}/video",
            "model_id": "openai/clip-vit-large-patch14",
            "use_chinese_clip": False,
            "output_file": "vit_tensor.pt"
        },
        "FVC": {
            "src_file": f"data/{dataset_name}/data_complete.jsonl",
            "output_dir": f"data/{dataset_name}/fea",
            "video_dir": f"data/{dataset_name}/video",
            "model_id": "openai/clip-vit-large-patch14",
            "use_chinese_clip": False,
            "output_file": "vit_tensor.pt"
        },
        "TwitterVideo": {
            "src_file": f"data/{dataset_name}/data.json",
            "output_dir": f"data/{dataset_name}/fea",
            "video_dir": f"data/{dataset_name}/video",
            "model_id": "openai/clip-vit-large-patch14",
            "use_chinese_clip": False,
            "output_file": "vit_tensor.pt"
        }
    }
    
    config = configs[dataset_name]
    
    # Load appropriate model and processor based on dataset
    if config["use_chinese_clip"]:
        processor = ChineseCLIPImageProcessor.from_pretrained(config["model_id"])
        model = ChineseCLIPVisionModel.from_pretrained(config["model_id"], device_map='cuda')
    else:
        processor = CLIPImageProcessor.from_pretrained(config["model_id"])
        model = CLIPVisionModel.from_pretrained(config["model_id"], device_map='cuda')
    
    # Create dataset and dataloader
    dataset = VideoDataset(config["src_file"], config["video_dir"])
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        collate_fn=lambda batch: customed_collate_fn(batch, processor), 
        num_workers=16
    )
    
    save_dict = {}
    model.eval()
    
    for batch in tqdm(dataloader):
        with torch.no_grad():
            vids, inputs = batch
            inputs = inputs.to('cuda')
            batch_size = len(vids)
            
            # Extract features differently depending on the model
            if config["use_chinese_clip"]:
                # Chinese-CLIP uses pooler_output
                pooler_output = model(**inputs).pooler_output
            else:
                # Regular CLIP uses last_hidden_state
                pooler_output = model(**inputs)['last_hidden_state'][:, 0, :]
            
            # Reshape to get per-frame features
            pooler_output = pooler_output.view(batch_size, NUM_FRAMES, -1)
            pooler_output = pooler_output.detach().cpu()
            
            # Save outputs by video ID
            for i, vid in enumerate(vids):
                save_dict[vid] = pooler_output[i]
    
    # Save features to file
    output_path = os.path.join(config["output_dir"], config["output_file"])
    os.makedirs(config["output_dir"], exist_ok=True)
    torch.save(save_dict, output_path)
    print(f"Saved features to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract video features using CLIP models')
    parser.add_argument('--datasets', nargs='+', default=['FakeSV', 'FakeTT', 'FVC'],
                       choices=['FakeSV', 'FakeTT', 'FVC', 'TwitterVideo'],
                       help='Datasets to process (default: FakeSV FakeTT FVC)')
    args = parser.parse_args()
    
    datasets = args.datasets
    
    print(f"Processing datasets: {datasets}")
    
    processed_count = 0
    failed_datasets = []
    
    for dataset_name in datasets:
        try:
            print(f"\n=== Starting {dataset_name} ===")
            process_dataset(dataset_name)
            processed_count += 1
            print(f"=== Completed {dataset_name} successfully ===")
        except Exception as e:
            print(f"=== Failed to process {dataset_name}: {str(e)} ===")
            failed_datasets.append(dataset_name)
    
    print(f"\n=== Feature Extraction Complete ===")
    print(f"Successfully processed: {processed_count}/{len(datasets)} datasets")
    
    if failed_datasets:
        print(f"Failed datasets: {failed_datasets}")
    else:
        print("All datasets processed successfully!")

if __name__ == "__main__":
    main()
