import os
import sys
import subprocess
import glob
import shutil
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_video_duration(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get video duration {video_path}: {e}")
        return None
    except ValueError:
        logging.error(f"Failed to parse video duration {video_path}")
        return None

def get_video_framerate(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
            capture_output=True, text=True, check=True
        )
        framerate = result.stdout.strip().split('/')
        return float(framerate[0]) / float(framerate[1])
    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError) as e:
        logging.warning(f"Failed to get video framerate {video_path}, using default value 30: {e}")
        return 30

def extract_frames(video_path, output_folder, num_frames, save_timestamps=True):
    duration = get_video_duration(video_path)
    if duration is None or duration <= 0:
        logging.error(f"Invalid video file or duration: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)

    if os.path.exists(video_output_folder):
        existing_frames = glob.glob(os.path.join(video_output_folder, "frame_*.jpg"))
        if len(existing_frames) == num_frames:
            logging.info(f"Skipping video {video_name}: Already have {num_frames} frames")
            return
        shutil.rmtree(video_output_folder)
    
    os.makedirs(video_output_folder, exist_ok=True)

    interval = duration / num_frames
    timestamps = [i * interval for i in range(num_frames)]
    framerate = get_video_framerate(video_path)

    for i, ts in enumerate(timestamps):
        output_file = os.path.join(video_output_folder, f"frame_{i:03d}.jpg")
        try:
            subprocess.run(
                ['ffmpeg', '-loglevel', 'error', '-ss', str(ts), '-i', video_path, '-frames:v', '1', '-q:v', '2', output_file],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Error processing frame {i} of video {video_name}: {e}")
            logging.error(f"ffmpeg error: {e.stderr}")

    # Save frame timestamps for audio alignment
    if save_timestamps:
        timestamp_file = os.path.join(video_output_folder, "frame_timestamps.txt")
        with open(timestamp_file, 'w') as f:
            for i, ts in enumerate(timestamps):
                # Calculate end timestamp for audio segment
                end_ts = ts + interval if i < num_frames - 1 else duration
                f.write(f"frame_{i:03d}.jpg,{ts:.6f},{end_ts:.6f}\n")

    created_frames = glob.glob(os.path.join(video_output_folder, "frame_*.jpg"))
    if len(created_frames) != num_frames:
        logging.warning(f"Warning: Video {video_name} only created {len(created_frames)} frames instead of expected {num_frames} frames")

def process_dataset(dataset_name, num_frames):
    # Handle different video directory naming conventions
    video_dirs = [f'data/{dataset_name}/videos', f'data/{dataset_name}/video']
    input_folder = None
    
    for video_dir in video_dirs:
        if os.path.exists(video_dir):
            input_folder = video_dir
            break
    
    if input_folder is None:
        logging.error(f"Error: No video folder found for {dataset_name}. Tried: {video_dirs}")
        return False

    output_folder = f'data/{dataset_name}/frames_{num_frames}'
    os.makedirs(output_folder, exist_ok=True)

    video_paths = glob.glob(os.path.join(input_folder, '*.mp4'))
    
    if not video_paths:
        logging.warning(f"No videos found in {input_folder}")
        return False
    
    logging.info(f"Processing {len(video_paths)} videos from {dataset_name}")
    for video_path in tqdm(video_paths, desc=f"Processing {dataset_name}"):
        extract_frames(video_path, output_folder, num_frames, save_timestamps=True)
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--datasets', nargs='+', default=['FakeSV'], 
                       choices=['FakeSV', 'FakeTT', 'FVC', 'TwitterVideo'],
                       help='Datasets to process (default: FakeSV)')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to extract (default: 16)')
    args = parser.parse_args()
    
    datasets = args.datasets
    num_frames = args.num_frames
    
    logging.info(f"Processing datasets: {datasets}")
    logging.info(f"Extracting {num_frames} frames per video")
    
    processed_count = 0
    for dataset in datasets:
        logging.info(f"Starting processing of {dataset} dataset...")
        if process_dataset(dataset, num_frames):
            processed_count += 1
            logging.info(f"Successfully completed {dataset} dataset")
        else:
            logging.error(f"Failed to process {dataset} dataset")
    
    if processed_count == 0:
        logging.error("No datasets were processed successfully")
        sys.exit(1)
    else:
        logging.info(f"Successfully processed {processed_count}/{len(datasets)} datasets")

if __name__ == "__main__":
    main()