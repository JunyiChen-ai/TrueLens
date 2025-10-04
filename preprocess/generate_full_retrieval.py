#!/usr/bin/env python3
"""
Generate full dataset retrieval results for gating mechanism.
All samples retrieve from train+valid memory bank, with self-exclusion for train/valid samples.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy
import logging
import sys
from datetime import datetime
import time

# Add src to path
sys.path.append('src')
from data.baseline_data import FakeSVDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FullDatasetRetrieval:
    def __init__(self, dataset: str = "FakeSV", model_name: str = None, filter_k: int = None, 
                 use_pool: bool = False, txt_only: bool = False, no_audio: bool = False,
                 use_uncertainty_weighted: bool = True, top_k: int = 10):
        """
        Initialize full dataset retrieval system
        
        Args:
            dataset: Dataset name (e.g., 'FakeSV', 'FakeTT')
            model_name: Hugging Face model name for text embedding (auto-selected if None)
            filter_k: Number of most similar training samples to remove per test sample (None for no filtering)
            use_pool: Use pooled features instead of frame-level similarity calculation
            txt_only: Use only text modality (skip visual/audio)
            no_audio: Skip audio processing (keep text + visual)
        """
        self.dataset = dataset
        
        # Auto-select model if not provided
        if model_name is None:
            model_name = self.get_default_model(dataset)
        
        self.model_name = model_name
        self.model_short_name = model_name.split("/")[-1]
        self.filter_k = filter_k
        self.use_pool = use_pool
        self.txt_only = txt_only
        self.no_audio = no_audio
        self.use_uncertainty_weighted = use_uncertainty_weighted
        self.top_k = top_k
        
        # Dataset-specific paths
        self.data_dir = Path(f"data/{dataset}")
        self.entity_dir = self.data_dir  # All datasets now use the same structure
        self.output_dir = self.data_dir / "text_similarity_results"
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initializing {dataset} dataset retrieval with model: {model_name}")
        if filter_k is not None:
            logger.info(f"Will filter top-{filter_k} similar training samples per test sample")
        if use_uncertainty_weighted:
            logger.info(f"Using uncertainty weighted log pooling for sample selection (top_k={top_k})")
        if txt_only:
            logger.info("Text-only mode: skipping visual and audio processing")
        elif no_audio:
            logger.info("No-audio mode: processing text + visual only")
        elif use_pool:
            logger.info("Using pooled features for visual/audio similarity (fast mode)")
        else:
            logger.info("Using frame-level similarity for visual/audio (detailed mode)")
    
    def get_default_model(self, dataset: str) -> str:
        """Get default text encoding model based on dataset"""
        if dataset.lower() == 'fakesv':
            return 'OFA-Sys/chinese-clip-vit-large-patch14'  # Chinese model
        else:
            return 'zer0int/LongCLIP-GmP-ViT-L-14'  # English model for FakeTT and others
        
    def load_model(self):
        """Load the embedding model"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
        
    def encode_texts(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode texts using the loaded model
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings
        """
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding with {self.model_short_name}"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize - LongCLIP supports longer sequences, standard CLIP is 77
                if 'longclip' in self.model_name.lower():
                    max_len = 248  # LongCLIP supports longer sequences
                elif 'chinese-clip' in self.model_name.lower():
                    max_len = 512  # Chinese-CLIP supports longer sequences
                elif 'clip' in self.model_name.lower():
                    max_len = 77   # Standard CLIP token limit
                else:
                    max_len = 512  # Other models
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=max_len,
                    return_tensors="pt"
                )
                
                # Debug: Show tokenized content for first batch
                if i == 0 and len(batch_texts) > 0:
                    decoded_first = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
                    logger.info(f"DEBUG: First tokenized text after truncation:\n       Original length: {len(batch_texts[0])} chars\n       After tokenization: {len(decoded_first)} chars\n       Content: {decoded_first}")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                if 'chinese-clip' in self.model_name.lower():
                    # For Chinese-CLIP, use text_model
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                elif 'longclip' in self.model_name.lower():
                    # For LongCLIP, use text_model
                    outputs = self.model.text_model(**inputs)
                    # Use mean pooling instead of CLS token
                    last_hidden_state = outputs.last_hidden_state
                    embeddings = last_hidden_state.mean(dim=1)  # Mean pooling over sequence
                    # embeddings = self.model.get_text_features(**inputs)
                elif 'openai/clip' in self.model_name.lower() or 'clip-vit' in self.model_name.lower():
                    # For OpenAI CLIP, use text_model (same as Chinese-CLIP)
                    outputs = self.model.text_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                else:
                    # For other models
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0]  # CLS token
                
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f"Encoded {len(texts)} texts into shape {embeddings.shape}")
        return embeddings
    
    def load_entity_data(self):
        """Load entity/video data based on dataset"""
        # Both FakeSV and other datasets now use the same llm_video_descriptions.jsonl format
        llm_file = self.entity_dir / "llm_video_descriptions.jsonl"
        
        if llm_file.exists():
            # Load LLM-generated descriptions
            all_data = []
            with open(llm_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line))
            logger.info(f"Loaded {len(all_data)} video descriptions")
            
            # Separate by annotation (handles both Chinese and English)
            self.true_data = [item for item in all_data if item.get('annotation') in ['real', '真']]
            self.fake_data = [item for item in all_data if item.get('annotation') in ['fake', '假']]
            logger.info(f"Separated into {len(self.true_data)} true and {len(self.fake_data)} fake videos")
        else:
            logger.error(f"Data file not found: {llm_file}")
            self.true_data = []
            self.fake_data = []
        
        # Load original data for timestamps
        self.load_original_timestamps()
    
    def load_original_timestamps(self):
        """Load original data to get publish timestamps and event info"""
        from datetime import datetime
        
        # Dataset-specific original data files
        if self.dataset == "FakeSV":
            orig_file = self.data_dir / "data_complete.jsonl"
            publish_time_field = 'publish_time_norm'
        else:
            # For FakeTT and other datasets
            orig_file = self.data_dir / "data.json"
            publish_time_field = 'publish_time'
        
        self.timestamp_lookup = {}
        self.event_lookup = {}
        
        if orig_file.exists():
            with open(orig_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        video_id = item.get('video_id')
                        publish_time = item.get(publish_time_field)
                        event = item.get('event', '')
                        
                        # Store event info
                        if video_id:
                            self.event_lookup[video_id] = event
                        
                        if video_id and publish_time:
                            try:
                                # Convert timestamp to readable format
                                if len(str(publish_time)) > 10:  # Milliseconds
                                    timestamp = datetime.fromtimestamp(publish_time / 1000.0)
                                else:  # Seconds
                                    timestamp = datetime.fromtimestamp(publish_time)
                                
                                self.timestamp_lookup[video_id] = {
                                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                    'raw_timestamp': publish_time
                                }
                            except (ValueError, OSError):
                                # Handle invalid timestamps
                                self.timestamp_lookup[video_id] = {
                                    'timestamp': 'Unknown',
                                    'raw_timestamp': publish_time
                                }
            
            logger.info(f"Loaded timestamps for {len(self.timestamp_lookup)} videos")
            logger.info(f"Loaded event info for {len(self.event_lookup)} videos")
        else:
            logger.warning(f"Original data file not found: {orig_file}")
    
    def load_multimodal_features(self):
        """Load visual and audio features with optimization (conditional based on mode)"""
        if self.txt_only:
            logger.info("Text-only mode: skipping multimodal feature loading")
            self.visual_features = {}
            self.audio_features = {}
            self.visual_features_normalized = {}
            self.audio_features_normalized = {}
            return
        
        feature_dir = self.data_dir / "fea"
        
        # Load visual features (unless text-only)
        if not self.txt_only:
            vit_file = feature_dir / "vit_tensor.pt"
            if vit_file.exists():
                self.visual_features = torch.load(vit_file, map_location='cpu')
                logger.info(f"Loaded visual features for {len(self.visual_features)} videos")
            else:
                self.visual_features = {}
                logger.warning(f"Visual features file not found: {vit_file}")
        else:
            self.visual_features = {}
        
        # Load audio features (unless text-only or no-audio)
        if not self.txt_only and not self.no_audio:
            # Choose audio feature file based on use_pool setting
            if self.use_pool:
                # Use global features for pooled mode
                audio_files_to_try = [
                    # "audio_features_global.pt",  # Default for FakeSV, FakeTT
                    "audio_features_global_laion-clap-htsat-fused.pt",  # TwitterVideo
                ]
            else:
                # Use frame features for detailed mode
                audio_files_to_try = [
                    "audio_features_frames.pt",  # Default for FakeSV, FakeTT
                    "audio_features_frames_laion-clap-htsat-fused.pt",  # TwitterVideo
                ]
            
            self.audio_features = {}
            for audio_filename in audio_files_to_try:
                audio_file = feature_dir / audio_filename
                if audio_file.exists():
                    self.audio_features = torch.load(audio_file, map_location='cpu')
                    logger.info(f"Loaded audio features from {audio_filename} for {len(self.audio_features)} videos")
                    break
            
            if not self.audio_features:
                logger.warning(f"No audio features file found in {feature_dir}. Tried: {audio_files_to_try}")
        else:
            self.audio_features = {}
            if self.no_audio:
                logger.info("No-audio mode: skipping audio feature loading")
        
        # Convert to numpy and ensure float32 for efficiency
        if self.visual_features or self.audio_features:
            logger.info("Converting and optimizing features...")
            for video_id in self.visual_features:
                features = self.visual_features[video_id].numpy().astype(np.float32)
                # Ensure contiguous memory layout for faster access
                self.visual_features[video_id] = np.ascontiguousarray(features)
            
            for video_id in self.audio_features:
                features = self.audio_features[video_id].numpy().astype(np.float32)
                # Ensure contiguous memory layout for faster access
                self.audio_features[video_id] = np.ascontiguousarray(features)
        
        # Initialize caches for normalized features
        self.visual_features_normalized = {}
        self.audio_features_normalized = {}
        
        logger.info("Multimodal features loaded, converted and optimized")
    
    def get_normalized_features(self, video_id: str, modality: str) -> np.ndarray:
        """Get normalized features with caching"""
        if modality == 'V':
            if video_id not in self.visual_features_normalized:
                if video_id in self.visual_features:
                    normalized = self.l2_normalize(self.visual_features[video_id], axis=-1)
                    self.visual_features_normalized[video_id] = normalized
                else:
                    return None
            return self.visual_features_normalized[video_id]
        
        elif modality == 'A':
            if video_id not in self.audio_features_normalized:
                if video_id in self.audio_features:
                    normalized = self.l2_normalize(self.audio_features[video_id], axis=-1)
                    self.audio_features_normalized[video_id] = normalized
                else:
                    return None
            return self.audio_features_normalized[video_id]
        
        return None
    
    def precompute_bank_features(self, memory_true_data: List[Dict], memory_fake_data: List[Dict], 
                                true_embeddings: np.ndarray, fake_embeddings: np.ndarray):
        """Pre-compute and cache normalized memory bank features (mode-dependent)"""
        logger.info("Pre-computing normalized memory bank features...")
        
        # Pre-compute true bank features
        self.true_bank_features = {'T': []}
        if not self.txt_only:
            self.true_bank_features['V'] = []
            if not self.no_audio:
                self.true_bank_features['A'] = []
        
        self.true_available_indices = []
        
        for i, memory_item in enumerate(memory_true_data):
            video_id = memory_item['video_id']
            
            if self.txt_only:
                # Text-only mode: include all samples
                self.true_available_indices.append(i)
            else:
                # Multimodal mode: require visual features
                visual_features = self.get_normalized_features(video_id, 'V')
                audio_features = self.get_normalized_features(video_id, 'A') if not self.no_audio else np.array([])
                
                # Check required features are available
                has_required = visual_features is not None and (self.no_audio or audio_features is not None)
                
                if has_required:
                    self.true_bank_features['V'].append(visual_features)
                    if not self.no_audio:
                        self.true_bank_features['A'].append(audio_features)
                    self.true_available_indices.append(i)
        
        if len(self.true_available_indices) > 0:
            # Stack into arrays (already normalized from cache)
            self.true_bank_features['T'] = true_embeddings[self.true_available_indices]
            if not self.txt_only:
                self.true_bank_features['V'] = np.stack(self.true_bank_features['V'], axis=0)
                if not self.no_audio:
                    self.true_bank_features['A'] = np.stack(self.true_bank_features['A'], axis=0)
            
            logger.info(f"Pre-computed true bank features: text shape: {self.true_bank_features['T'].shape}")
            if not self.txt_only:
                logger.info(f"                                  visual shape: {self.true_bank_features['V'].shape}")
                if not self.no_audio:
                    logger.info(f"                                  audio shape: {self.true_bank_features['A'].shape}")
        
        # Pre-compute fake bank features
        self.fake_bank_features = {'T': []}
        if not self.txt_only:
            self.fake_bank_features['V'] = []
            if not self.no_audio:
                self.fake_bank_features['A'] = []
        
        self.fake_available_indices = []
        
        for i, memory_item in enumerate(memory_fake_data):
            video_id = memory_item['video_id']
            
            if self.txt_only:
                # Text-only mode: include all samples
                self.fake_available_indices.append(i)
            else:
                # Multimodal mode: require visual features
                visual_features = self.get_normalized_features(video_id, 'V')
                audio_features = self.get_normalized_features(video_id, 'A') if not self.no_audio else np.array([])
                
                # Check required features are available
                has_required = visual_features is not None and (self.no_audio or audio_features is not None)
                
                if has_required:
                    self.fake_bank_features['V'].append(visual_features)
                    if not self.no_audio:
                        self.fake_bank_features['A'].append(audio_features)
                    self.fake_available_indices.append(i)
        
        if len(self.fake_available_indices) > 0:
            # Stack into arrays (already normalized from cache)
            self.fake_bank_features['T'] = fake_embeddings[self.fake_available_indices]
            if not self.txt_only:
                self.fake_bank_features['V'] = np.stack(self.fake_bank_features['V'], axis=0)
                if not self.no_audio:
                    self.fake_bank_features['A'] = np.stack(self.fake_bank_features['A'], axis=0)
            
            logger.info(f"Pre-computed fake bank features: text shape: {self.fake_bank_features['T'].shape}")
            if not self.txt_only:
                logger.info(f"                                  visual shape: {self.fake_bank_features['V'].shape}")
                if not self.no_audio:
                    logger.info(f"                                  audio shape: {self.fake_bank_features['A'].shape}")
        
        logger.info("Bank features pre-computation complete")
        
    def get_temporal_splits(self):
        """Get temporal split video IDs (dataset-specific)"""
        if self.dataset == "FakeSV":
            # FakeSV has predefined splits
            vid_dir = self.data_dir / "vids"
            
            splits = {}
            for split in ['train', 'valid', 'test']:
                split_file = vid_dir / f"vid_time3_{split}.txt"
                if split_file.exists():
                    with open(split_file, 'r') as f:
                        splits[split] = set(line.strip() for line in f)
                    logger.info(f"{split.capitalize()} split: {len(splits[split])} videos")
                else:
                    splits[split] = set()
                    logger.warning(f"Split file not found: {split_file}")
        else:
            # For other datasets (like FakeTT), create simple splits based on available data
            all_video_ids = set()
            for item in self.true_data + self.fake_data:
                all_video_ids.add(item['video_id'])
            
            # Simple 70/15/15 split for non-FakeSV datasets
            video_list = sorted(list(all_video_ids))
            n_total = len(video_list)
            n_train = int(0.7 * n_total)
            n_valid = int(0.15 * n_total)
            
            splits = {
                'train': set(video_list[:n_train]),
                'valid': set(video_list[n_train:n_train + n_valid]),
                'test': set(video_list[n_train + n_valid:])
            }
            
            logger.info(f"Created splits for {self.dataset}: Train: {len(splits['train'])}, Valid: {len(splits['valid'])}, Test: {len(splits['test'])}")
        
        return splits

    def get_fold_splits(self, fold: int) -> Dict[str, Set[str]]:
        """Get five-fold split video IDs from vids directory.
        Memory bank = vid_fold_no_{fold}.txt; Test = vid_fold_{fold}.txt; Valid = empty set.
        """
        vid_dir = self.data_dir / "vids"
        train_file = vid_dir / f"vid_fold_no_{fold}.txt"
        test_file = vid_dir / f"vid_fold_{fold}.txt"

        splits = {'train': set(), 'valid': set(), 'test': set()}
        if train_file.exists():
            with open(train_file, 'r') as f:
                splits['train'] = set(line.strip() for line in f if line.strip())
            logger.info(f"Fold {fold}: Train (fold_no) split: {len(splits['train'])} videos")
        else:
            logger.warning(f"Fold {fold}: train file not found: {train_file}")

        if test_file.exists():
            with open(test_file, 'r') as f:
                splits['test'] = set(line.strip() for line in f if line.strip())
            logger.info(f"Fold {fold}: Test (fold) split: {len(splits['test'])} videos")
        else:
            logger.warning(f"Fold {fold}: test file not found: {test_file}")

        # Valid remains empty in 5-fold usage
        return splits

    def _run_retrieval_for_fold(self, fold: int, subfolder_dir: Path):
        """Run retrieval for a specific fold and save to subfolder/fold_{fold}.json"""
        start_time = time.time()
        logger.info(f"Starting 5-fold retrieval generation for fold {fold}...")

        # Use five-fold splits
        splits = self.get_fold_splits(fold)

        # Apply filtering to training set if specified (uses text encoder)
        filtered_train_ids = self.filter_training_set(splits)

        # Prepare data with filtered training set
        self.prepare_data(splits, filtered_train_ids)

        # Prepare texts for encoding
        true_texts = [self.create_text_representation(item) for item in self.memory_true_data]
        fake_texts = [self.create_text_representation(item) for item in self.memory_fake_data]
        query_texts = [self.create_text_representation(item) for item in self.query_data]

        # Encode texts
        logger.info("Encoding queries (5-fold)...")
        query_embeddings = self.encode_texts(query_texts)
        logger.info("Encoding true memory bank (5-fold)...")
        true_embeddings = self.encode_texts(true_texts) if true_texts else np.array([])
        logger.info("Encoding fake memory bank (5-fold)...")
        fake_embeddings = self.encode_texts(fake_texts) if fake_texts else np.array([])

        # Compute similarities
        results = self.find_similar_videos(
            query_embeddings, true_embeddings, fake_embeddings,
            self.query_data,
            self.memory_true_data, self.memory_fake_data,
            batch_size=64
        )

        # Save results to subfolder/fold_{fold}.json (overwrite)
        output_file = subfolder_dir / f"fold_{fold}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved 5-fold retrieval results to {output_file}")

        # Generate concise stats file: stats_f{fold}.json in same folder
        stats = {
            'total_queries': len(results),
            'memory_bank_size': len(self.memory_true_data) + len(self.memory_fake_data),
            'memory_true_count': len(self.memory_true_data),
            'memory_fake_count': len(self.memory_fake_data),
            'excluded_self_count': sum(1 for r in results if r['excluded_self']),
            'splits': {}
        }
        for result in results:
            split = result['query_video']['split']
            stats['splits'][split] = stats['splits'].get(split, 0) + 1

        true_similarities = []
        fake_similarities = []
        test_to_memory_true_similarities = []
        test_to_memory_fake_similarities = []
        test_to_memory_train_true_similarities = []
        test_to_memory_train_fake_similarities = []

        memory_bank_splits = {}
        for item in self.memory_true_data + self.memory_fake_data:
            video_id = item['video_id']
            if hasattr(self, 'query_splits') and video_id in self.query_splits:
                memory_bank_splits[video_id] = self.query_splits[video_id]

        for result in results:
            if result['similar_true']:
                true_similarities.append(result['similar_true']['similarity_score'])
                if result['query_video']['split'] == 'test':
                    test_to_memory_true_similarities.append(result['similar_true']['similarity_score'])
                    vid = result['similar_true']['video_id']
                    if vid in memory_bank_splits and memory_bank_splits[vid] == 'train':
                        test_to_memory_train_true_similarities.append(result['similar_true']['similarity_score'])
            if result['similar_fake']:
                fake_similarities.append(result['similar_fake']['similarity_score'])
                if result['query_video']['split'] == 'test':
                    test_to_memory_fake_similarities.append(result['similar_fake']['similarity_score'])
                    vid = result['similar_fake']['video_id']
                    if vid in memory_bank_splits and memory_bank_splits[vid] == 'train':
                        test_to_memory_train_fake_similarities.append(result['similar_fake']['similarity_score'])

        stats['avg_true_similarity'] = float(np.mean(true_similarities)) if true_similarities else 0.0
        stats['avg_fake_similarity'] = float(np.mean(fake_similarities)) if fake_similarities else 0.0
        stats['test_to_memory_avg_true_similarity'] = float(np.mean(test_to_memory_true_similarities)) if test_to_memory_true_similarities else 0.0
        stats['test_to_memory_avg_fake_similarity'] = float(np.mean(test_to_memory_fake_similarities)) if test_to_memory_fake_similarities else 0.0
        stats['test_to_memory_train_avg_true_similarity'] = float(np.mean(test_to_memory_train_true_similarities)) if test_to_memory_train_true_similarities else 0.0
        stats['test_to_memory_train_avg_fake_similarity'] = float(np.mean(test_to_memory_train_fake_similarities)) if test_to_memory_train_fake_similarities else 0.0

        stats_file = subfolder_dir / f"stats_f{fold}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("="*60)
        logger.info(f"FIVE-FOLD RETRIEVAL SUMMARY (fold {fold})")
        logger.info("="*60)
        logger.info(f"Total queries: {stats['total_queries']}")
        logger.info(f"Memory bank size: {stats['memory_bank_size']} (True: {stats['memory_true_count']}, Fake: {stats['memory_fake_count']})")
        logger.info(f"Self-exclusions: {stats['excluded_self_count']}")
        logger.info(f"Split distribution: {stats['splits']}")
        logger.info(f"Average similarities - True: {stats['avg_true_similarity']:.4f}, Fake: {stats['avg_fake_similarity']:.4f}")
        logger.info(f"Test-to-Memory average similarities - True: {stats['test_to_memory_avg_true_similarity']:.4f}, Fake: {stats['test_to_memory_avg_fake_similarity']:.4f}")
        logger.info(f"Test-to-Memory-Train average similarities - True: {stats['test_to_memory_train_avg_true_similarity']:.4f}, Fake: {stats['test_to_memory_train_avg_fake_similarity']:.4f}")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Statistics saved to: {stats_file}")

        total_time = time.time() - start_time
        logger.info(f"Fold {fold} execution time: {total_time:.2f} seconds")
    
    def l2_normalize(self, x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
        """L2 normalization for features"""
        n = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (n + eps)
    
    def _softmax(self, x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
        """Softmax function with numerical stability"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + eps)
    
    def _entropy_top_k(self, r: np.ndarray, k: int = None, eps: float = 1e-12) -> float:
        """Compute entropy for top-k elements"""
        if k is None:
            k = self.top_k
        
        if len(r) > k:
            top_k_idx = np.argpartition(r, -k)[-k:]
            top_k_probs = r[top_k_idx]
        else:
            top_k_probs = r
        
        top_k_probs = top_k_probs / (top_k_probs.sum() + eps)
        # Compute entropy
        top_k_probs = top_k_probs + eps
        return -np.sum(top_k_probs * np.log(top_k_probs))
    
    def compute_frame_similarity(self, query_frames: np.ndarray, candidate_frames: np.ndarray) -> float:
        """
        Compute frame-level similarity between query and candidate videos
        
        Args:
            query_frames: [T_q, D] frames for query video
            candidate_frames: [T_c, D] frames for candidate video
            
        Returns:
            Scalar similarity score
        """
        # L2 normalize frames
        query_frames = self.l2_normalize(query_frames)
        candidate_frames = self.l2_normalize(candidate_frames)
        
        # Compute similarity matrix [T_q, T_c]
        similarity_matrix = np.dot(query_frames, candidate_frames.T)
        
        # For each query frame, find max similarity across candidate frames
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Average across all query frames
        return float(np.mean(max_similarities))
    
    def compute_batch_frame_similarity(self, query_frames: np.ndarray, candidate_frames_batch: np.ndarray) -> np.ndarray:
        """
        Vectorized batch computation of frame-level similarity
        
        Args:
            query_frames: [T_q, D] frames for query video (already normalized)
            candidate_frames_batch: [N, T_c, D] frames for N candidate videos (already normalized)
            
        Returns:
            Array of N similarity scores
        """
        # Handle edge case: empty candidate batch
        if candidate_frames_batch.shape[0] == 0:
            return np.array([], dtype=np.float32)
        
        # Ensure all arrays are float32 for better performance
        query_frames = query_frames.astype(np.float32)
        candidate_frames_batch = candidate_frames_batch.astype(np.float32)
        
        N, T_c, D = candidate_frames_batch.shape
        T_q = query_frames.shape[0]
        
        # Handle edge case: query has no frames
        if T_q == 0:
            return np.zeros(N, dtype=np.float32)
        
        # Reshape for batch matrix multiplication
        # query_frames: [T_q, D] -> [1, T_q, D] -> [N, T_q, D]
        query_batch = np.broadcast_to(query_frames[None, :, :], (N, T_q, D))
        
        # Batch matrix multiplication: [N, T_q, D] @ [N, D, T_c] -> [N, T_q, T_c]
        similarity_matrices = np.einsum('nqd,ndc->nqc', query_batch, candidate_frames_batch.transpose(0, 2, 1))
        
        # For each candidate, find max similarity for each query frame, then average
        max_similarities = np.max(similarity_matrices, axis=2)  # [N, T_q]
        similarity_scores = np.mean(max_similarities, axis=1)   # [N]
        
        return similarity_scores.astype(np.float32)
    
    def fused_scores_sample_gating(self, q: Dict[str, np.ndarray], bank: Dict[str, np.ndarray], 
                                  eps: float = 1e-12) -> np.ndarray:
        """
        Multimodal fusion using sample-adaptive gating or uncertainty weighted log pooling
        Supports text-only mode for simplified similarity computation
        
        Args:
            q: Query features {'T': text_emb, 'V': visual_frames, 'A': audio_frames}
            bank: Bank features {'T': text_bank, 'V': visual_bank, 'A': audio_bank}
            
        Returns:
            Fused scores or probabilities for all samples in bank
        """
        # For text-only mode, return simple cosine similarity
        if self.txt_only:
            if self.use_uncertainty_weighted:
                logger.warning("Uncertainty weighted pooling requires multiple modalities. Falling back to regular similarity for text-only mode.")
            
            q_text = self.l2_normalize(q['T'])
            bank_text = self.l2_normalize(bank['T'], axis=1)
            return np.dot(bank_text, q_text)
        
        # Step 1: L2 normalize features (assume they come from cache and are already normalized)
        q_norm = {}
        bank_norm = {}
        
        for m in q:
            # For query features, normalize text but visual/audio should already be normalized from cache
            if m == 'T':
                q_norm[m] = self.l2_normalize(q[m])
            else:
                q_norm[m] = q[m]  # Already normalized from cache
            
            # For bank features, they should already be normalized from precompute_bank_features
            if m == 'T':
                bank_norm[m] = self.l2_normalize(bank[m], axis=1)
            else:
                bank_norm[m] = bank[m]  # Already normalized from cache
        
        # Step 2: Compute similarities for each modality
        S = {}  # Raw similarities per modality
        for m in bank_norm:
            if m == 'T':
                # Text: simple dot product (already normalized)
                S[m] = np.dot(bank_norm[m], q_norm[m])
            else:
                # Visual/Audio: choose between pooled and frame-level similarity
                if self.use_pool:
                    # Check if features are already global (1D) or need pooling (2D)
                    if q_norm[m].ndim == 1:
                        # Global features: direct cosine similarity (audio when use_pool=True)
                        S[m] = np.dot(bank_norm[m], q_norm[m])
                    else:
                        # Frame features: average pool features, re-normalize, then compute cosine similarity
                        q_pooled = np.mean(q_norm[m], axis=0)  # [D]
                        q_pooled = q_pooled / (np.linalg.norm(q_pooled) + eps)  # Re-normalize after pooling
                        
                        bank_pooled = np.mean(bank_norm[m], axis=1)  # [N, D]
                        bank_pooled = bank_pooled / (np.linalg.norm(bank_pooled, axis=1, keepdims=True) + eps)  # Re-normalize
                        
                        S[m] = np.dot(bank_pooled, q_pooled)  # Now it's cosine similarity
                else:
                    # Frame-level mode: vectorized frame-level similarity
                    S[m] = self.compute_batch_frame_similarity(q_norm[m], bank_norm[m])
        
        # Choose fusion strategy
        if self.use_uncertainty_weighted:
            return self._uncertainty_weighted_log_pool_fusion(S, eps)
        else:
            return self._sample_adaptive_gating_fusion(S, eps)
    
    def _uncertainty_weighted_log_pool_fusion(self, S: Dict[str, np.ndarray], eps: float = 1e-12) -> np.ndarray:
        """
        Uncertainty Weighted Log Pool fusion implementation
        Strictly following analyze_multimodal_choice.py implementation
        
        Args:
            S: Raw similarities per modality {'T': [N], 'V': [N], 'A': [N]}
            eps: Small constant for numerical stability
            
        Returns:
            Fused probabilities for all samples in bank
        """
        # Note: In analyze_multimodal_choice.py, this operates on [Q, N] matrices where Q is queries
        # Here we're processing a single query at a time, so our arrays are [N] (candidates)
        # We need to treat this as a [1, N] case and compute query-specific weights
        
        # Step 1: Convert similarities to probabilities using softmax
        temp = 0.1  # Temperature parameter for softmax
        P = {}
        for m in S:
            P[m] = self._softmax(S[m] / temp)  # [N]
        
        # Step 2: Compute certainty scores (inverse of entropy) using top-k elements
        # Following analyze_multimodal_choice.py: compute entropy for this specific query's distribution
        C = {}
        for m in P:
            # Compute entropy of top-k probabilities for this query's distribution
            ent = self._entropy_top_k(P[m])
            # Certainty is inverse of entropy (scalar for this query)
            C[m] = 1.0 / (ent + eps)
        
        # Step 3: Normalize weights across modalities for this query
        total_certainty = sum(C.values())
        W = {m: C[m] / (total_certainty + eps) for m in C}
        
        # Step 4: Log-linear pooling (same as original)
        fused_log = np.zeros_like(P[list(P.keys())[0]])
        for m in P:
            fused_log += W[m] * np.log(P[m] + eps)
        
        # Step 5: Convert back to probabilities
        fused = np.exp(fused_log)
        fused = fused / (fused.sum() + eps)  # Normalize
        
        return fused
    
    def _sample_adaptive_gating_fusion(self, S: Dict[str, np.ndarray], eps: float = 1e-12) -> np.ndarray:
        """
        Original sample-adaptive gating fusion (fallback)
        
        Args:
            S: Raw similarities per modality
            eps: Small constant for numerical stability
            
        Returns:
            Fused scores for all samples in bank
        """
        # Z-standardization across samples (handle case where std is 0)
        Z = {}
        for m in S:
            std_val = S[m].std()
            if std_val < eps:
                # If all similarities are the same, just center them
                Z[m] = S[m] - S[m].mean()
            else:
                Z[m] = (S[m] - S[m].mean()) / (std_val + eps)
        
        # Sample-adaptive gating (vectorized)
        pos = [np.maximum(Z[m], 0.0) for m in Z]  # [·]₊
        
        # Vectorized computation
        pos_array = np.stack(pos, axis=0)  # [num_modalities, num_samples]
        num = np.sum(pos_array * pos_array, axis=0)  # Element-wise square and sum
        den = np.sum(pos_array, axis=0)  # Sum across modalities
            
        return num / (den + eps)  # Final fused scores
        
    def prepare_data(self, splits: Dict[str, Set[str]], filtered_train_ids: Set[str] = None):
        """Prepare data for retrieval"""
        # Create lookup for entity data
        entity_lookup = {}
        for item in self.true_data + self.fake_data:
            entity_lookup[item['video_id']] = item
        
        # Use filtered train IDs if provided, otherwise use original
        if filtered_train_ids is None:
            filtered_train_ids = splits['train']
        
        # Memory bank consists of (filtered) train + valid samples
        memory_bank_video_ids = filtered_train_ids | splits['valid']
        logger.info(f"Memory bank size: {len(memory_bank_video_ids)} videos "
                   f"(filtered train: {len(filtered_train_ids)}, valid: {len(splits['valid'])})")
        
        # Prepare memory bank data (train + valid)
        self.memory_true_data = []
        self.memory_fake_data = []
        
        for video_id in memory_bank_video_ids:
            if video_id in entity_lookup:
                item = entity_lookup[video_id]
                # Handle both Chinese and English annotations
                annotation = item.get('annotation', '').lower()
                if annotation in ['真', 'real']:
                    self.memory_true_data.append(item)
                elif annotation in ['假', 'fake']:
                    self.memory_fake_data.append(item)
        
        # Prepare all query samples (filtered train + valid + test)
        all_video_ids = filtered_train_ids | splits['valid'] | splits['test']
        self.query_data = []
        self.query_splits = {}  # Track which split each sample belongs to
        
        for video_id in all_video_ids:
            if video_id in entity_lookup:
                item = entity_lookup[video_id]
                self.query_data.append(item)
                
                # Determine which split this video belongs to
                if video_id in filtered_train_ids:
                    self.query_splits[video_id] = 'train'
                elif video_id in splits['valid']:
                    self.query_splits[video_id] = 'valid'
                elif video_id in splits['test']:
                    self.query_splits[video_id] = 'test'
                else:
                    self.query_splits[video_id] = 'unknown'
        
        logger.info(f"Memory bank: {len(self.memory_true_data)} true, {len(self.memory_fake_data)} fake")
        logger.info(f"Query set: {len(self.query_data)} videos")
        
    def create_text_representation(self, item: Dict) -> str:
        """Create text representation for embedding (dataset-flexible)"""
        parts = []
        
        # Handle title/description field
        title = item.get('title', '') or item.get('description', '')
        if title:
            parts.append(str(title))
        
        # Handle keywords vs event field (FakeTT uses 'event', FakeSV uses 'keywords')
        keywords_or_event = item.get('keywords', '') or item.get('event', '')
        if keywords_or_event:
            parts.append(str(keywords_or_event))
        
        # Add description and temporal_evolution
        for field in ['description', 'temporal_evolution']:
            if field in item and item[field] and field != 'title':  # Avoid duplicate if title == description
                parts.append(str(item[field]))
        
        # Add entity_claims only if available (FakeSV has this, FakeTT doesn't)
        if 'entity_claims' in item and item['entity_claims']:
            claims_text = []
            for entity, claims in item['entity_claims'].items():
                for claim in claims:
                    claims_text.append(f"{entity}: {claim}")
            if claims_text:
                parts.append(" ".join(claims_text))
        
        text_repr = " ".join(parts)
        if not text_repr.strip():
            # Fallback if no text found
            text_repr = item.get('video_id', 'unknown')
            logger.warning(f"No text content found for video {item.get('video_id')}, using ID as fallback")
        
        # Debug: print first few text representations to check content
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
        
        if self._debug_count <= 3:
            logger.info(f"DEBUG: Text representation {self._debug_count} (video_id: {item.get('video_id', 'unknown')}):")
            logger.info(f"       Content (first 200 chars): {text_repr[:200]}...")
            logger.info(f"       Total length: {len(text_repr)} characters")
        
        return text_repr
    
    def filter_training_set(self, splits: Dict[str, Set[str]]) -> Set[str]:
        """
        Filter training set by removing top-k most similar samples to test set
        
        Args:
            splits: Dictionary containing train/valid/test splits
            
        Returns:
            Set of filtered training video IDs
        """
        if self.filter_k is None:
            return splits['train']
        
        logger.info(f"Filtering training set: removing top-{self.filter_k} similar samples per test sample...")
        
        # Create entity lookup for all samples
        entity_lookup = {}
        for item in self.true_data + self.fake_data:
            entity_lookup[item['video_id']] = item
        
        # Only work with video IDs that exist in entity data to ensure consistency
        available_video_ids = set(entity_lookup.keys())
        filtered_train_ids = splits['train'] & available_video_ids
        filtered_test_ids = splits['test'] & available_video_ids
        
        logger.info(f"Available train IDs: {len(filtered_train_ids)} (original: {len(splits['train'])})")
        logger.info(f"Available test IDs: {len(filtered_test_ids)} (original: {len(splits['test'])})")
        
        # Prepare training data for temporary memory bank (only available IDs)
        train_data = []
        for video_id in filtered_train_ids:
            train_data.append(entity_lookup[video_id])
        
        # Prepare test data (only available IDs)
        test_data = []
        for video_id in filtered_test_ids:
            test_data.append(entity_lookup[video_id])
        
        logger.info(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # Create text representations and encode
        train_texts = [self.create_text_representation(item) for item in train_data]
        test_texts = [self.create_text_representation(item) for item in test_data]
        
        logger.info("Encoding training samples for filtering...")
        train_embeddings = self.encode_texts(train_texts)
        
        logger.info("Encoding test samples for filtering...")
        test_embeddings = self.encode_texts(test_texts)
        
        # Compute similarities between test and training samples
        logger.info("Computing test-train similarities for filtering...")
        similarities = cosine_similarity(test_embeddings, train_embeddings)
        
        # Find top-k most similar training samples for each test sample
        training_samples_to_remove = set()
        for i, test_item in enumerate(test_data):
            test_similarities = similarities[i]
            
            # Get top-k most similar training sample indices
            top_k_indices = np.argsort(test_similarities)[-self.filter_k:]
            
            for idx in top_k_indices:
                similar_train_id = train_data[idx]['video_id']
                training_samples_to_remove.add(similar_train_id)
                
                logger.debug(f"Test {test_item['video_id']} -> Remove train {similar_train_id} "
                           f"(similarity: {test_similarities[idx]:.4f})")
        
        # Create filtered training set from the available train IDs
        final_filtered_train_ids = filtered_train_ids - training_samples_to_remove
        
        logger.info(f"Filtering complete: removed {len(training_samples_to_remove)} training samples")
        logger.info(f"Filtered training set: {len(final_filtered_train_ids)} samples "
                   f"(original: {len(splits['train'])})")
        
        # Save filtered training IDs to file (dataset-specific)
        if self.dataset == "FakeSV":
            vid_dir = self.data_dir / "vids"
            filtered_file = vid_dir / f"vid_time3_train_k{self.filter_k}.txt"
        else:
            # For other datasets, save in the dataset directory
            filtered_file = self.data_dir / f"train_filtered_k{self.filter_k}.txt"
        
        with open(filtered_file, 'w') as f:
            for vid_id in sorted(final_filtered_train_ids):
                f.write(f"{vid_id}\n")
        
        logger.info(f"Saved filtered training IDs to {filtered_file}")
        
        return final_filtered_train_ids
    
    def find_similar_videos(self, query_embeddings: np.ndarray, 
                           true_embeddings: np.ndarray, fake_embeddings: np.ndarray,
                           query_data: List[Dict], 
                           memory_true_data: List[Dict], memory_fake_data: List[Dict],
                           top_k: int = 1, batch_size: int = 64) -> List[Dict]:
        """Find most similar videos using optimized multimodal fusion"""
        logger.info(f"Computing multimodal similarities for {len(query_data)} query videos (batch_size={batch_size})...")
        
        # Pre-compute and cache normalized memory bank features
        self.precompute_bank_features(memory_true_data, memory_fake_data, true_embeddings, fake_embeddings)
        
        results = []
        
        # Process queries in batches for better performance  
        for batch_start in tqdm(range(0, len(query_data), batch_size), desc="Processing query batches"):
            batch_end = min(batch_start + batch_size, len(query_data))
            batch_queries = query_data[batch_start:batch_end]
            batch_embeddings = query_embeddings[batch_start:batch_end]
            
            batch_results = self._process_query_batch(
                batch_queries, batch_embeddings, memory_true_data, memory_fake_data
            )
            results.extend(batch_results)
            
        return results
    
    def _process_query_batch(self, batch_queries: List[Dict], batch_embeddings: np.ndarray,
                            memory_true_data: List[Dict], memory_fake_data: List[Dict]) -> List[Dict]:
        """Process a batch of queries efficiently"""
        batch_results = []
        
        # Process each query in the batch
        for i, query_item in enumerate(batch_queries):
            query_video_id = query_item['video_id']
            query_split = self.query_splits.get(query_video_id, 'unknown')
            
            # Prepare query features based on mode
            query_features = {'T': batch_embeddings[i]}  # Always include text
            
            if not self.txt_only:
                # Skip if multimodal features not available for query
                visual_features = self.get_normalized_features(query_video_id, 'V')
                audio_features = self.get_normalized_features(query_video_id, 'A') if not self.no_audio else None
                
                if visual_features is None or (not self.no_audio and audio_features is None):
                    logger.warning(f"Missing multimodal features for query {query_video_id}, skipping")
                    continue
                    
                # Add multimodal features
                query_features['V'] = visual_features
                if not self.no_audio and audio_features is not None:
                    query_features['A'] = audio_features
            
            # Process true and fake candidates in parallel using optimized method
            true_candidate, fake_candidate = self._find_best_candidates_parallel(
                query_features, query_split, query_video_id, memory_true_data, memory_fake_data
            )
            
            # Get event from original data if available (for LongCLIP/English datasets)
            event_info = self.event_lookup.get(query_video_id, '')
            
            result = {
                'query_video': {
                    'video_id': query_video_id,
                    'annotation': query_item.get('annotation', ''),
                    'split': query_split,
                    'title': query_item.get('title', ''),
                    'keywords': query_item.get('keywords', '') or query_item.get('event', ''),
                    'event': event_info,  # Add event field from original data
                    'description': query_item.get('description', ''),
                    'temporal_evolution': query_item.get('temporal_evolution', ''),
                    'entity_claims': query_item.get('entity_claims', {}),
                    'publish_time': self.timestamp_lookup.get(query_video_id, {}).get('timestamp', 'Unknown')
                },
                'similar_true': true_candidate,
                'similar_fake': fake_candidate,
                'excluded_self': query_split in ['train', 'valid']
            }
            
            batch_results.append(result)
        
        return batch_results
    
    def _find_best_candidates_parallel(self, query_features: Dict[str, np.ndarray], query_split: str, 
                                     query_video_id: str, memory_true_data: List[Dict], memory_fake_data: List[Dict]):
        """Find best true and fake candidates using parallel processing"""
        import concurrent.futures
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both true and fake candidate searches in parallel
            future_true = executor.submit(
                self._find_best_candidate, query_features, query_split, query_video_id,
                self.true_bank_features, self.true_available_indices, memory_true_data, 'true'
            )
            future_fake = executor.submit(
                self._find_best_candidate, query_features, query_split, query_video_id,
                self.fake_bank_features, self.fake_available_indices, memory_fake_data, 'fake'  
            )
            
            # Get results
            true_candidate = future_true.result()
            fake_candidate = future_fake.result()
            
        return true_candidate, fake_candidate
    
    def _find_best_candidate(self, query_features: Dict[str, np.ndarray], query_split: str,
                           query_video_id: str, bank_features: Dict[str, np.ndarray], available_indices: List[int],
                           memory_data: List[Dict], candidate_type: str):
        """Find best candidate from a specific bank (true or fake)"""
        # Check if bank has any features
        if len(available_indices) == 0:
            return None
        
        # For text-only mode, only check text features
        if self.txt_only:
            if 'T' not in bank_features or len(bank_features['T']) == 0:
                return None
        else:
            # For multimodal mode, check visual features (audio is optional based on no_audio flag)
            if 'V' not in bank_features or bank_features['V'] is None or len(bank_features['V']) == 0:
                return None
        
        # Handle self-exclusion for train/valid
        exclude_indices = set()
        if query_split in ['train', 'valid'] and query_video_id:
            for j, idx in enumerate(available_indices):
                if memory_data[idx]['video_id'] == query_video_id:
                    exclude_indices.add(j)
        
        # Create filtered bank features
        if exclude_indices:
            keep_indices = [j for j in range(len(available_indices)) if j not in exclude_indices]
            if not keep_indices:
                return None
            filtered_bank = {'T': bank_features['T'][keep_indices]}
            # Only include V and A if they exist in bank_features
            if 'V' in bank_features:
                filtered_bank['V'] = bank_features['V'][keep_indices]
            if 'A' in bank_features:
                filtered_bank['A'] = bank_features['A'][keep_indices]
            filtered_indices = [available_indices[j] for j in keep_indices]
        else:
            filtered_bank = bank_features
            filtered_indices = available_indices
            
        # Compute multimodal similarities
        fused_scores = self.fused_scores_sample_gating(query_features, filtered_bank)
        
        # Always use argmax to select best candidate (both uncertainty weighted and traditional)
        best_idx = np.argmax(fused_scores)
            
        memory_idx = filtered_indices[best_idx]
        
        # Compute individual modality similarities for output
        text_sim = float(np.dot(self.l2_normalize(filtered_bank['T'][best_idx]), 
                               self.l2_normalize(query_features['T'])))
        
        # Compute visual/audio similarities only if available
        if 'V' in query_features and 'V' in filtered_bank:
            visual_sim = self.compute_frame_similarity(query_features['V'], filtered_bank['V'][best_idx])
        else:
            visual_sim = 0.0
            
        if 'A' in query_features and 'A' in filtered_bank:
            if self.use_pool and query_features['A'].ndim == 1:
                # Global audio features: direct cosine similarity
                query_audio_norm = self.l2_normalize(query_features['A'])
                bank_audio_norm = self.l2_normalize(filtered_bank['A'][best_idx])
                audio_sim = float(np.dot(bank_audio_norm, query_audio_norm))
            else:
                # Frame-level audio features: use frame similarity
                audio_sim = self.compute_frame_similarity(query_features['A'], filtered_bank['A'][best_idx])
        else:
            audio_sim = 0.0
        
        best_item = memory_data[memory_idx]
        candidate = {
            'video_id': best_item['video_id'],
            'similarity_score': float(fused_scores[best_idx]),  # Fused score
            'text_similarity': text_sim,
            'visual_similarity': visual_sim,
            'audio_similarity': audio_sim,
            'annotation': best_item['annotation'],
            'title': best_item.get('title', ''),
            'keywords': best_item.get('keywords', '') or best_item.get('event', ''),
            'event': self.event_lookup.get(best_item['video_id'], ''),  # Add event field
            'description': best_item.get('description', ''),
            'temporal_evolution': best_item.get('temporal_evolution', ''),
            'entity_claims': best_item.get('entity_claims', {}),
            'publish_time': self.timestamp_lookup.get(best_item['video_id'], {}).get('timestamp', 'Unknown')
        }
        
        return candidate
    
    def run_retrieval(self):
        """Run the full retrieval process with performance monitoring"""
        start_time = time.time()
        logger.info("Starting optimized full dataset retrieval...")
        
        # Load data
        load_start = time.time()
        self.load_entity_data()
        splits = self.get_temporal_splits()
        logger.info(f"Data loading took {time.time() - load_start:.2f} seconds")
        
        # Load multimodal features
        features_start = time.time()
        self.load_multimodal_features()
        logger.info(f"Multimodal features loading took {time.time() - features_start:.2f} seconds")
        
        # Load model first (required for filtering)
        model_start = time.time()
        self.load_model()
        logger.info(f"Model loading took {time.time() - model_start:.2f} seconds")
        
        # Apply filtering to training set if specified (needs model loaded)
        filter_start = time.time()
        filtered_train_ids = self.filter_training_set(splits)
        logger.info(f"Training set filtering took {time.time() - filter_start:.2f} seconds")
        
        # Prepare data with filtered training set
        prepare_start = time.time()
        self.prepare_data(splits, filtered_train_ids)
        logger.info(f"Data preparation took {time.time() - prepare_start:.2f} seconds")
        
        # Prepare texts for encoding
        text_prep_start = time.time()
        logger.info("Preparing texts for encoding...")
        
        # Separate memory bank texts
        true_texts = [self.create_text_representation(item) for item in self.memory_true_data]
        fake_texts = [self.create_text_representation(item) for item in self.memory_fake_data]
        
        # Query texts (all data)
        query_texts = [self.create_text_representation(item) for item in self.query_data]
        logger.info(f"Text preparation took {time.time() - text_prep_start:.2f} seconds")
        
        # Encode texts separately
        encoding_start = time.time()
        logger.info("Encoding queries...")
        query_embeddings = self.encode_texts(query_texts)
        
        logger.info("Encoding true memory bank...")
        true_embeddings = self.encode_texts(true_texts) if true_texts else np.array([])
        
        logger.info("Encoding fake memory bank...")
        fake_embeddings = self.encode_texts(fake_texts) if fake_texts else np.array([])
        logger.info(f"Text encoding took {time.time() - encoding_start:.2f} seconds")
        
        # Find similar videos with optimized batch processing
        retrieval_start = time.time()
        results = self.find_similar_videos(
            query_embeddings, true_embeddings, fake_embeddings,
            self.query_data, 
            self.memory_true_data, self.memory_fake_data,
            batch_size=64  # Use larger batch size for better performance
        )
        logger.info(f"Similarity computation took {time.time() - retrieval_start:.2f} seconds")
        
        # Build output filename with appropriate suffixes
        filename_parts = []
        if self.use_uncertainty_weighted:
            filename_parts.append("uncertainty")
        filename_parts.append(f"full_dataset_retrieval_{self.model_short_name}")
        if self.filter_k is not None:
            filename_parts.append(f"k{self.filter_k}")
        if self.use_pool:
            filename_parts.append("pool")
        
        output_file = self.output_dir / f"{'_'.join(filename_parts)}.json"
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved retrieval results to {output_file}")
        
        # Generate summary statistics
        summary_start = time.time()
        self.generate_summary(results, output_file)
        logger.info(f"Summary generation took {time.time() - summary_start:.2f} seconds")
        
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Processed {len(results)} queries at {len(results)/total_time:.2f} queries/sec")
        logger.info("="*60)
        
        return results
    
    def generate_summary(self, results: List[Dict], output_file: Path):
        """Generate summary statistics"""
        stats = {
            'total_queries': len(results),
            'memory_bank_size': len(self.memory_true_data) + len(self.memory_fake_data),
            'memory_true_count': len(self.memory_true_data),
            'memory_fake_count': len(self.memory_fake_data),
            'excluded_self_count': sum(1 for r in results if r['excluded_self']),
            'splits': {}
        }
        
        # Count by splits
        for result in results:
            split = result['query_video']['split']
            if split not in stats['splits']:
                stats['splits'][split] = 0
            stats['splits'][split] += 1
        
        # Calculate average similarities
        true_similarities = []
        fake_similarities = []
        test_to_memory_true_similarities = []
        test_to_memory_fake_similarities = []
        test_to_memory_train_true_similarities = []
        test_to_memory_train_fake_similarities = []
        
        # Create lookup for memory bank splits (train vs valid)
        memory_bank_splits = {}
        for item in self.memory_true_data + self.memory_fake_data:
            video_id = item['video_id']
            # Determine if this memory bank item is from train or valid
            if hasattr(self, 'query_splits') and video_id in self.query_splits:
                memory_bank_splits[video_id] = self.query_splits[video_id]
        
        for result in results:
            if result['similar_true']:
                true_similarities.append(result['similar_true']['similarity_score'])
                # Track test set to memory bank similarities separately
                if result['query_video']['split'] == 'test':
                    test_to_memory_true_similarities.append(result['similar_true']['similarity_score'])
                    # Check if the similar true sample is from train split in memory bank
                    similar_video_id = result['similar_true']['video_id']
                    if similar_video_id in memory_bank_splits and memory_bank_splits[similar_video_id] == 'train':
                        test_to_memory_train_true_similarities.append(result['similar_true']['similarity_score'])
                    
            if result['similar_fake']:
                fake_similarities.append(result['similar_fake']['similarity_score'])
                # Track test set to memory bank similarities separately  
                if result['query_video']['split'] == 'test':
                    test_to_memory_fake_similarities.append(result['similar_fake']['similarity_score'])
                    # Check if the similar fake sample is from train split in memory bank
                    similar_video_id = result['similar_fake']['video_id']
                    if similar_video_id in memory_bank_splits and memory_bank_splits[similar_video_id] == 'train':
                        test_to_memory_train_fake_similarities.append(result['similar_fake']['similarity_score'])
        
        stats['avg_true_similarity'] = float(np.mean(true_similarities)) if true_similarities else 0.0
        stats['avg_fake_similarity'] = float(np.mean(fake_similarities)) if fake_similarities else 0.0
        stats['test_to_memory_avg_true_similarity'] = float(np.mean(test_to_memory_true_similarities)) if test_to_memory_true_similarities else 0.0
        stats['test_to_memory_avg_fake_similarity'] = float(np.mean(test_to_memory_fake_similarities)) if test_to_memory_fake_similarities else 0.0
        stats['test_to_memory_train_avg_true_similarity'] = float(np.mean(test_to_memory_train_true_similarities)) if test_to_memory_train_true_similarities else 0.0
        stats['test_to_memory_train_avg_fake_similarity'] = float(np.mean(test_to_memory_train_fake_similarities)) if test_to_memory_train_fake_similarities else 0.0
        
        # Build stats filename with same suffixes
        stats_parts = []
        if self.use_uncertainty_weighted:
            stats_parts.append("uncertainty")
        stats_parts.append(f"full_dataset_stats_{self.model_short_name}")
        if self.filter_k is not None:
            stats_parts.append(f"k{self.filter_k}")
        if self.use_pool:
            stats_parts.append("pool")
        
        stats_file = output_file.parent / f"{'_'.join(stats_parts)}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Print summary
        logger.info("="*60)
        logger.info("FULL DATASET RETRIEVAL SUMMARY")
        logger.info("="*60)
        logger.info(f"Total queries: {stats['total_queries']}")
        logger.info(f"Memory bank size: {stats['memory_bank_size']} (True: {stats['memory_true_count']}, Fake: {stats['memory_fake_count']})")
        logger.info(f"Self-exclusions: {stats['excluded_self_count']}")
        logger.info(f"Split distribution: {stats['splits']}")
        logger.info(f"Average similarities - True: {stats['avg_true_similarity']:.4f}, Fake: {stats['avg_fake_similarity']:.4f}")
        logger.info(f"Test-to-Memory average similarities - True: {stats['test_to_memory_avg_true_similarity']:.4f}, Fake: {stats['test_to_memory_avg_fake_similarity']:.4f}")
        logger.info(f"Test-to-Memory-Train average similarities - True: {stats['test_to_memory_train_avg_true_similarity']:.4f}, Fake: {stats['test_to_memory_train_avg_fake_similarity']:.4f}")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Statistics saved to: {stats_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate full dataset retrieval results')
    parser.add_argument('--dataset', type=str, default='FakeSV',
                       help='Dataset name (default: FakeSV). Examples: FakeSV, FakeTT')
    parser.add_argument('--model', type=str, default=None,
                       help='Hugging Face model name for text embedding (auto-selected based on dataset if not provided)')
    parser.add_argument('--filter-k', type=int, default=None,
                       help='Number of most similar training samples to remove per test sample')
    parser.add_argument('--use-pool', action='store_true',
                       help='Use pooled features instead of frame-level similarity (faster)')
    parser.add_argument('--txt-only', action='store_true',
                       help='Use text-only mode (skip visual and audio processing)')
    parser.add_argument('--no-audio', action='store_true',
                       help='Skip audio processing (keep text + visual)')
    parser.add_argument('--disable-uncertainty-weighted', action='store_true',
                       help='Disable uncertainty weighted log pooling (use original sample-adaptive gating)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Top-k elements for entropy computation in uncertainty weighting (default: 10)')
    # Five-fold generation toggle (temporal behavior remains unchanged when this is not set)
    parser.add_argument('--5-fold', dest='five_fold', action='store_true',
                       help='Enable five-fold retrieval generation (writes fold_{n}.json under mechanism subfolders)')
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.txt_only and args.no_audio:
        logger.warning("Both --txt-only and --no-audio specified. --txt-only takes precedence.")
    
    retriever = FullDatasetRetrieval(
        dataset=args.dataset,
        model_name=args.model, 
        filter_k=args.filter_k, 
        use_pool=args.use_pool,
        txt_only=args.txt_only,
        no_audio=args.no_audio,
        use_uncertainty_weighted=not args.disable_uncertainty_weighted,
        top_k=args.top_k
    )

    if args.five_fold:
        # Five-fold generation: do not alter temporal outputs
        # Preload data/model/features once for efficiency
        retriever.load_entity_data()
        retriever.load_multimodal_features()
        retriever.load_model()

        # Determine mechanism subfolder name: uw, k{filter_k}, default (exclude pool token as requested)
        sub_tokens = []
        if not args.disable_uncertainty_weighted:
            sub_tokens.append('uw')
        if args.filter_k is not None:
            sub_tokens.append(f'k{args.filter_k}')
        subfolder = 'default' if not sub_tokens else '_'.join(sub_tokens)

        # Create subfolder
        mech_dir = retriever.output_dir / subfolder
        mech_dir.mkdir(parents=True, exist_ok=True)

        # Check fold files existence to ensure 5-fold is defined for the dataset
        vids_dir = retriever.data_dir / 'vids'
        has_folds = all((vids_dir / f'vid_fold_{i}.txt').exists() and (vids_dir / f'vid_fold_no_{i}.txt').exists() for i in range(1, 6))
        if not has_folds:
            logger.error(f"Five-fold files not found under {vids_dir}. Expected vid_fold_{n}.txt and vid_fold_no_{n}.txt for n=1..5")
            sys.exit(1)

        for fold in range(1, 6):
            logger.info(f"Generating 5-fold retrieval for fold {fold} -> {mech_dir / f'fold_{fold}.json'}")
            retriever._run_retrieval_for_fold(fold=fold, subfolder_dir=mech_dir)
        logger.info("Completed five-fold retrieval generation.")
    else:
        # Original temporal behavior
        retriever.run_retrieval()
