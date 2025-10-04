#!/usr/bin/env python3
"""
Extract video descriptions and temporal evolution from videos using LLM
Supports multiple datasets with flexible field mapping
"""

import os
import json
import base64
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from tqdm import tqdm
from PIL import Image
import io
import signal
import sys
import atexit
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_description_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class VideoDescriptionExtractor:
    def __init__(self, 
                 data_dir: str,
                 dataset: str = None,
                 api_key: str = None,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.3,
                 max_tokens: int = 800,
                 rate_limit_delay: float = 0.5,
                 resume: bool = True,
                 max_concurrent: int = 10):
        """
        Initialize the video description extractor
        
        Args:
            data_dir: Path to the data directory containing video data and images
            dataset: Dataset name (e.g., 'FakeSV', 'FakeTT', 'TwitterVideo')
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            rate_limit_delay: Delay between API calls in seconds
            resume: Whether to resume from checkpoint
            max_concurrent: Maximum number of concurrent API calls
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limit_delay = rate_limit_delay
        self.resume = resume
        self.max_concurrent = max_concurrent
        
        # Dataset and language settings
        self.dataset = dataset
        self.use_chinese = dataset and dataset.lower() == 'fakesv'
        
        # Data paths
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        # Output file path (single source of truth)
        self.output_file = self.data_dir / "llm_video_descriptions.jsonl"
        # Deprecated: no separate checkpoint/temp; we resume by reading output_file
        self.checkpoint_file = self.data_dir / "llm_checkpoint.json"
        self.temp_file = self.data_dir / "llm_temp_results.jsonl"
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Processing state
        self.processed_videos = set()
        self.all_results = []  # kept for backward compatibility; not required now
        # Async write lock for safe concurrent appends
        self._write_lock = asyncio.Lock()
        
        # Shutdown flag
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Resume by reading existing output file to find already processed IDs
        if self.resume:
            self.load_existing_outputs()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\n=== Shutdown requested, saving progress... ===")
        self.shutdown_requested = True
    
    def load_existing_outputs(self):
        """Initialize processed_videos by scanning the existing output file."""
        if not self.output_file.exists():
            logger.info("No existing output file found; starting fresh")
            return
        count = 0
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    vid = item.get('video_id') or item.get('vid')
                    if vid:
                        self.processed_videos.add(vid)
                        count += 1
            logger.info(f"Resuming from output file: {count} videos already present")
        except Exception as e:
            logger.warning(f"Failed to read existing outputs: {e}. Starting fresh.")
    
    async def append_result(self, result: Dict):
        """Append one result to output file immediately (write-as-you-go)."""
        # Avoid duplicates
        vid = result.get('video_id')
        if vid in self.processed_videos:
            return
        async with self._write_lock:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            self.processed_videos.add(vid)
        # Stats
        self.stats["successful"] += 1
    
    # Deprecated method retained for compatibility; no-op now
    def save_temp_result(self, result: Dict):
        pass
    
    def load_video_data(self, data_filename: str = "data.json") -> List[Dict]:
        """Load video data from data file (supports both .json and .jsonl)"""
        
        # Try different possible data files
        possible_files = [
            self.data_dir / data_filename,
            self.data_dir / "data_complete.jsonl", 
            self.data_dir / "data.json",
            self.data_dir / "data.jsonl"
        ]
        
        data_file = None
        for file_path in possible_files:
            if file_path.exists():
                data_file = file_path
                break
        
        if not data_file:
            raise FileNotFoundError(f"No data file found. Tried: {[str(f) for f in possible_files]}")
        
        video_data = []
        logger.info(f"Loading video data from: {data_file}")
        
        # Handle both JSON and JSONL formats
        if data_file.suffix == '.jsonl':
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        video_data.append(item)
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('['):
                    # JSON array format
                    video_data = json.loads(content)
                else:
                    # JSONL format in .json file (each line is a JSON object)
                    for line in content.split('\n'):
                        if line.strip():
                            video_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(video_data)} videos")
        return video_data
    
    def load_transcript(self, video_id: str) -> str:
        """Load transcript for a video"""
        transcript_file = self.data_dir / "transcript.jsonl"
        
        if not transcript_file.exists():
            return ""
        
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Handle both 'vid' and 'video_id' field names
                    vid_key = 'vid' if 'vid' in item else 'video_id'
                    if item.get(vid_key) == video_id:
                        return item.get('transcript', '')
        
        return ""
    
    def prepare_composite_frames(self, video_id: str) -> List[str]:
        """Load and encode composite frames as base64"""
        # Try different possible frame directories
        frame_dirs = [
            self.data_dir / "quads_4",
            self.data_dir / "frames_4", 
            self.data_dir / "composite_frames"
        ]
        
        quads_dir = None
        for dir_path in frame_dirs:
            if dir_path.exists():
                quads_dir = dir_path
                break
        
        if not quads_dir:
            logger.warning(f"No composite frames directory found for {video_id}")
            return []
        
        encoded_images = []
        
        # Try different naming conventions
        for i in range(4):
            possible_names = [
                f"{video_id}_quad_{i}.jpg",
                f"{video_id}_frame_{i}.jpg", 
                f"{video_id}_{i}.jpg"
            ]
            
            image_path = None
            for name in possible_names:
                path = quads_dir / name
                if path.exists():
                    image_path = path
                    break
            
            if image_path:
                try:
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                        # Optionally resize if too large
                        img = Image.open(io.BytesIO(img_data))
                        if img.width > 1024 or img.height > 1024:
                            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                            buffer = io.BytesIO()
                            img.save(buffer, format='JPEG', quality=85)
                            img_data = buffer.getvalue()
                        
                        encoded = base64.b64encode(img_data).decode('utf-8')
                        encoded_images.append(encoded)
                except Exception as e:
                    logger.warning(f"Failed to load image {image_path}: {e}")
        
        return encoded_images
    
    def format_timestamp(self, timestamp_ms: Optional[int]) -> str:
        """Convert millisecond timestamp to YYYY-MM-DD format"""
        if timestamp_ms:
            try:
                return datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d')
            except:
                pass
        return ""
    
    def create_prompt(self, video_data: Dict, transcript: str) -> str:
        """Create the prompt for video content analysis"""
        title = video_data.get('title', '') or video_data.get('description', '')
        
        # Handle both keywords and event fields flexibly
        keywords_or_event = ""
        if 'keywords' in video_data:
            keywords_or_event = video_data.get('keywords', '')
        elif 'event' in video_data:
            keywords_or_event = video_data.get('event', '')
        
        publish_time = self.format_timestamp(video_data.get('publish_time', 0) or video_data.get('publish_time_norm', 0))
        
        # Use Chinese prompt for FakeSV dataset, English prompt for all others
        if self.use_chinese:
            prompt = f"""分析这个短视频的内容并提取视频描述信息。

视频信息：
- 标题：{title}
- 关键词：{keywords_or_event if keywords_or_event else "未提供关键词"}
- 音频转录：{transcript if transcript else "无音频转录"}
- 发布时间：{publish_time}

视频帧时间信息：
此视频包含4个复合帧图像，每个复合帧包含4个连续帧（2x2网格布局）：
- 第1个复合帧（帧0-3，视频开始部分）
- 第2个复合帧（帧4-7，视频前中部分）
- 第3个复合帧（帧8-11，视频后中部分）
- 第4个复合帧（帧12-15，视频结束部分）

请提取以下信息（用中文回答）：

1. 视频描述：
   简要描述这个视频的内容（50-100字）

2. 时序演变：
   描述内容如何在4个复合帧中演变

以JSON格式返回：
{{
  "description": "详细的视频描述",
  "temporal_evolution": "时序演变描述"
}}"""
        else:
            prompt = f"""Analyze the content of this short video and extract video description information.

Video Information:
- Title: {title}
- Keywords: {keywords_or_event if keywords_or_event else "No keywords/event provided"}
- Audio Transcript: {transcript if transcript else "No audio transcript"}
- Publish Time: {publish_time}

Video Frame Temporal Information:
This video contains 4 composite frame images, each containing 4 consecutive frames (2x2 grid layout):
- 1st composite frame (frames 0-3, beginning part of video)
- 2nd composite frame (frames 4-7, early-middle part of video)
- 3rd composite frame (frames 8-11, late-middle part of video)
- 4th composite frame (frames 12-15, ending part of video)

Please extract the following information (respond in English):

1. Video Description:
   Briefly describe what this video is about (50-100 words)

2. Temporal Evolution:
   Describe how the content evolves across the 4 composite frames

Return in JSON format:
{{
  "description": "Detailed description of the video",
  "temporal_evolution": "Description of temporal evolution"
}}"""
        
        return prompt
    
    async def extract_from_video(self, video_data: Dict, images: List[str]) -> Optional[Dict]:
        """Extract description and temporal evolution from a single video"""
        video_id = video_data.get('video_id')
        
        try:
            # Load transcript
            transcript = self.load_transcript(video_id)
            
            # Create prompt
            prompt = self.create_prompt(video_data, transcript)
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Add images if available
            for i, img_base64 in enumerate(images):
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                })
            
            # Call API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Create extraction result
            extraction = {
                "video_id": video_id,
                "annotation": video_data.get('annotation'),
                "title": video_data.get('title', '') or video_data.get('description', ''),
                "description": result.get("description", ""),
                "temporal_evolution": result.get("temporal_evolution", ""),
                "metadata": {
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "has_images": len(images) > 0,
                    "has_transcript": bool(transcript)
                }
            }
            
            # Add keywords/event field if present
            if 'keywords' in video_data:
                extraction["keywords"] = video_data.get('keywords', '')
            elif 'event' in video_data:
                extraction["event"] = video_data.get('event', '')
            
            # Update statistics
            self.stats["successful"] += 1
            
            return extraction
            
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            self.stats["failed"] += 1
            return None
    
    async def process_videos_concurrent(self, videos: List[Dict]) -> List[Dict]:
        """Process videos with controlled concurrency"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        # Write-as-you-go; no checkpoints
        
        async def process_with_semaphore(video_data):
            async with semaphore:
                video_id = video_data.get('video_id')
                
                # Skip if already processed
                if video_id in self.processed_videos:
                    return None
                
                # Check for shutdown
                if self.shutdown_requested:
                    return None
                
                images = self.prepare_composite_frames(video_id)
                result = await self.extract_from_video(video_data, images)
                
                # Save immediately if successful
                if result:
                    await self.append_result(result)
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
                # Update progress
                self.stats["total_processed"] += 1
                
                return result
        
        # Filter videos that haven't been processed
        videos_to_process = [v for v in videos if v.get('video_id') not in self.processed_videos]
        logger.info(f"Processing {len(videos_to_process)} videos (skipping {len(videos) - len(videos_to_process)} already processed)")
        
        if not videos_to_process:
            logger.info("All videos already processed!")
            return self.all_results
        
        # Process in batches for progress control
        batch_size = 50
        all_results = []
        
        for batch_start in range(0, len(videos_to_process), batch_size):
            if self.shutdown_requested:
                logger.info("Processing interrupted. Partial results already appended to output.")
                break
                
            batch_end = min(batch_start + batch_size, len(videos_to_process))
            batch_videos = videos_to_process[batch_start:batch_end]
            
            actual_processed = len(self.processed_videos)
            logger.info(f"Processing batch {batch_start//batch_size + 1}: {batch_end - batch_start} videos | Progress: {actual_processed}/{len(videos)} total")
            
            # Create tasks for this batch
            with tqdm(total=len(batch_videos), desc=f"Batch {batch_start//batch_size + 1}") as pbar:
                tasks = []
                for video in batch_videos:
                    tasks.append(process_with_semaphore(video))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and update progress bar
                for video, result in zip(batch_videos, results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process video {video['video_id']}: {result}")
                        self.stats["failed"] += 1
                    elif result:
                        all_results.append(result)
                    
                    pbar.update(1)
            
            logger.info(f"Batch complete. Processed {len(self.processed_videos)}/{len(videos)} total videos")
        
        return self.all_results
    
    # Deprecated: results are appended as they are produced
    def save_final_results(self):
        logger.info("Write-as-you-go enabled: results already saved to output file")
    
    def cleanup_temp_files(self):
        """No temp/checkpoint files in write-as-you-go mode"""
        return
    
    async def run(self, sample_size: Optional[int] = None, data_filename: str = "data.json"):
        """Run the video description extraction pipeline"""
        logger.info("Starting video description extraction pipeline")
        logger.info(f"Using model: {self.model}")
        logger.info(f"Max concurrent API calls: {self.max_concurrent}")
        logger.info(f"Rate limit delay: {self.rate_limit_delay}s")
        
        try:
            # Load video data
            videos = self.load_video_data(data_filename)
            
            # Sample if requested
            if sample_size:
                videos = videos[:sample_size]
                logger.info(f"Processing sample of {sample_size} videos")
            else:
                logger.info(f"Processing all {len(videos)} videos")
            
            # Update stats
            self.stats["total_videos"] = len(videos)
            
            # Process videos with concurrency
            results = await self.process_videos_concurrent(videos)
            
            # Check if user requested shutdown
            if self.shutdown_requested:
                logger.info("Processing interrupted by user. Partial results saved to temporary file.")
                logger.info("Run again to continue processing from where it stopped.")
                return
            
            # Finalization
            logger.info("All videos processed successfully! Results already saved to output file.")
            self.cleanup_temp_files()
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            logger.info("Run again to continue; processed items are persisted in output file.")
            raise
        
        # Print final statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print processing statistics"""
        logger.info("\n=== Processing Statistics ===")
        logger.info(f"Total videos: {self.stats.get('total_videos', 0)}")
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f"Successful: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        
        if self.stats['successful'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        # Calculate processing time
        start_time = datetime.fromisoformat(self.stats['start_time'])
        duration = datetime.now() - start_time
        logger.info(f"Processing time: {duration}")
        
        logger.info(f"Output file: {self.output_file}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract video descriptions and temporal evolution using LLM")
    parser.add_argument("--dataset", type=str, default="FakeTT", help="Dataset name (default: FakeTT)")
    parser.add_argument("--sample", type=int, help="Process only N samples for testing")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature (default: 0)")
    parser.add_argument("--max-tokens", type=int, default=800, help="Maximum tokens in response (default: 800)")
    parser.add_argument("--rate-limit", type=float, default=0.5, help="Delay between API calls in seconds (default: 0.5)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum concurrent API calls (default: 10)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoint")
    parser.add_argument("--data-file", type=str, default="data.json", help="Data filename to load (default: data.json)")
    
    args = parser.parse_args()
    
    # Auto-construct data directory path
    script_dir = Path(__file__).parent.parent  # Go up from preprocess/ to WWW-2025/
    data_dir = script_dir / "data" / args.dataset
    
    logger.info(f"Using dataset: {args.dataset}")
    logger.info(f"Data directory: {data_dir}")
    
    # Create extractor
    try:
        extractor = VideoDescriptionExtractor(
            data_dir=str(data_dir),
            dataset=args.dataset,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            rate_limit_delay=args.rate_limit,
            resume=not args.no_resume,
            max_concurrent=args.max_concurrent
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        return
    
    # Run extraction
    try:
        await extractor.run(sample_size=args.sample, data_filename=args.data_file)
        logger.info("Extraction completed successfully!")
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
