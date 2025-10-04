import os
import subprocess
import argparse
from tqdm import tqdm

def convert_mp4_to_wav(input_folder, output_folder, suffix="_full"):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all mp4 files
    mp4_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    if not mp4_files:
        print(f"No MP4 files found in {input_folder}")
        return []
    
    # Store names of files that couldn't be converted
    unconverted_files = []
    
    print(f"Found {len(mp4_files)} MP4 files to convert")
    
    # Create progress bar using tqdm
    for filename in tqdm(mp4_files, desc="Converting videos to audio"):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + suffix + '.wav'
        output_path = os.path.join(output_folder, output_filename)
        
        # Check if already converted
        if os.path.exists(output_path):
            continue
        
        # Build FFmpeg command
        ffmpeg_command = [
            'ffmpeg',
            '-i', input_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-loglevel', 'error',  # Reduce FFmpeg output
            output_path
        ]
        
        try:
            # Execute FFmpeg command
            subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f'\nError converting {filename}: {e.stderr.decode()}')
            unconverted_files.append(filename)

    # Print names of files that couldn't be converted
    if unconverted_files:
        print(f"\n{len(unconverted_files)} files could not be converted:")
        for file in unconverted_files:
            print(f"  - {file}")
        return unconverted_files
    else:
        print(f"\nAll {len(mp4_files)} files have been successfully converted!")
        return []

def process_dataset(dataset_name, suffix="_full"):
    """Process a single dataset"""
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Handle different video directory naming conventions
    video_dirs = [f'data/{dataset_name}/videos', f'data/{dataset_name}/video']
    input_folder = None
    
    for video_dir in video_dirs:
        if os.path.exists(video_dir):
            input_folder = video_dir
            break
    
    if input_folder is None:
        print(f"Error: No video folder found for {dataset_name}. Tried: {video_dirs}")
        return False
    
    output_folder = f'data/{dataset_name}/audios'
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    unconverted_files = convert_mp4_to_wav(input_folder, output_folder, suffix)
    
    if not unconverted_files:
        print(f"{dataset_name} conversion completed successfully!")
        return True
    else:
        print(f"{dataset_name} conversion completed with {len(unconverted_files)} errors")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert MP4 videos to WAV audio files')
    parser.add_argument('--datasets', nargs='+', default=['FakeSV'],
                       choices=['FakeSV', 'FakeTT', 'FVC', 'TwitterVideo'],
                       help='Datasets to process (default: FakeSV)')
    parser.add_argument('--suffix', default='_full',
                       help='Suffix to add to output filenames (default: _full)')
    args = parser.parse_args()
    
    datasets = args.datasets
    suffix = args.suffix
    
    print(f"Processing datasets: {datasets}")
    print(f"Output suffix: {suffix}")
    
    processed_count = 0
    failed_datasets = []
    
    for dataset in datasets:
        if process_dataset(dataset, suffix):
            processed_count += 1
        else:
            failed_datasets.append(dataset)
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {processed_count}/{len(datasets)} datasets")
    
    if failed_datasets:
        print(f"Failed datasets: {failed_datasets}")
    else:
        print("All datasets processed successfully!")

if __name__ == "__main__":
    main()