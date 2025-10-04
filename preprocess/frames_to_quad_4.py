import os
import cv2
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm

def process_frames(input_folder, output_folder, dataset_name):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        return False
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_count = 0
    error_count = 0

    video_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    
    if not video_folders:
        print(f"No video folders found in {input_folder}")
        return False
    
    print(f"Found {len(video_folders)} video folders to process")
    
    for vid_folder in tqdm(video_folders, desc=f"Processing {dataset_name}", unit="videos"):
        frames_folder = os.path.join(input_folder, vid_folder)
        
        # Check if all quad files already exist (checkpoint mechanism)
        all_quads_exist = True
        for quad_index in range(4):
            output_path = os.path.join(output_folder, f"{vid_folder}_quad_{quad_index}.jpg")
            if not os.path.exists(output_path):
                all_quads_exist = False
                break
        
        if all_quads_exist:
            tqdm.write(f"Skipping {vid_folder}: All quad images already exist")
            processed_count += 1
            continue
        
        # Load available frames
        frames = []
        for i in range(16):
            frame_path = os.path.join(frames_folder, f"frame_{i:03d}.jpg")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path)
                frames.append(frame)
            else:
                tqdm.write(f"Cannot find frame: {frame_path}")
        
        # Handle insufficient frames by padding with last frame
        if len(frames) == 0:
            tqdm.write(f"ERROR: No frames found for {vid_folder}")
            error_count += 1
            continue
        elif len(frames) < 16:
            tqdm.write(f"Padding {vid_folder}: {len(frames)} frames -> 16 frames (duplicating last frame)")
            last_frame = frames[-1]
            frames.extend([last_frame] * (16 - len(frames)))

        # Create 4 images with 2x2 grid layout
        for quad_index in range(4):
            output_path = os.path.join(output_folder, f"{vid_folder}_quad_{quad_index}.jpg")
            
            # Skip if this quad already exists
            if os.path.exists(output_path):
                continue
                
            start_index = quad_index * 4
            quad_frames = frames[start_index:start_index+4]
            
            min_width = min(frame.width for frame in quad_frames)
            min_height = min(frame.height for frame in quad_frames)
            quad_frames = [frame.resize((min_width, min_height)) for frame in quad_frames]

            grid = Image.new('RGB', (min_width * 2, min_height * 2))
            grid.paste(quad_frames[0], (0, 0))
            grid.paste(quad_frames[1], (min_width, 0))
            grid.paste(quad_frames[2], (0, min_height))
            grid.paste(quad_frames[3], (min_width, min_height))

            grid.save(output_path, 'JPEG')

        processed_count += 1

    print(f"Successfully processed: {processed_count} videos")
    print(f"Failed: {error_count} videos")
    
    return error_count == 0

def process_dataset(dataset_name):
    """Process a single dataset"""
    print(f"\nProcessing {dataset_name} dataset...")
    input_folder = f'data/{dataset_name}/frames_16'
    output_folder = f'data/{dataset_name}/quads_4'
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    
    success = process_frames(input_folder, output_folder, dataset_name)
    
    if success:
        print(f"{dataset_name} processing completed successfully!")
    else:
        print(f"{dataset_name} processing completed with errors")
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Create quad images from video frames')
    parser.add_argument('--datasets', nargs='+', default=['FakeSV', 'FakeTT', 'FVC'],
                       choices=['FakeSV', 'FakeTT', 'FVC', 'TwitterVideo'],
                       help='Datasets to process (default: FakeSV FakeTT FVC)')
    args = parser.parse_args()
    
    datasets = args.datasets
    
    print(f"Processing datasets: {datasets}")
    
    processed_count = 0
    failed_datasets = []
    
    for dataset in datasets:
        if process_dataset(dataset):
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