import torch
from torch.utils.data import Dataset, DataLoader
from .baseline_data import FakeSVDataset, FakeTTDataset, FVCDataset
import numpy as np
import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from loguru import logger

class FakeSVDataset_ExMRD(FakeSVDataset):
    def __init__(self, fold: int, split: str, lm: str='gpt-4o', ablation_no_cot: bool=False, 
                 description: bool=False, temp_evolution: bool=False,
                 use_text: bool=True, use_image: bool=True, use_audio: bool=True, 
                 filter_k: int=None, **kwargs):
        super(FakeSVDataset_ExMRD, self).__init__(filter_k=filter_k, **kwargs)
        self.data = self._get_data(fold, split)
        self.ablation_no_cot = ablation_no_cot
        self.description = description
        self.temp_evolution = temp_evolution
        self.use_text = use_text
        self.use_image = use_image
        self.use_audio = use_audio
        
        # Load features based on modality controls
        self.fea_frames = None
        self.audio_features = None
        self.audio_dim = 768
        self.entity_data = None
        
        if self.use_image:
            self.fea_frames = torch.load('data/FakeSV/fea/vit_tensor.pt', weights_only=True)
        if self.use_audio:
            self.audio_features = torch.load('data/FakeSV/fea/audio_features_frames.pt', weights_only=True)
        
        # Load LLM video descriptions for text features if needed
        if self.use_text:
            # Load LLM-generated descriptions similar to FakeTT
            try:
                import json
                llm_data = []
                with open('data/FakeSV/llm_video_descriptions.jsonl', 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            # Add empty entity_claims for compatibility
                            item['entity_claims'] = {}
                            llm_data.append(item)
                
                self.entity_data = pd.DataFrame(llm_data)
                print(f"Loaded LLM descriptions: {len(self.entity_data)} samples")
            except Exception as e:
                raise FileNotFoundError(f"Could not load LLM descriptions: {e}")
        
        # Get available video IDs based on enabled modalities
        available_ids = set(self.data['video_id'])  # Start with all video IDs
        
        if self.use_image and self.fea_frames is not None:
            visual_ids = set(self.fea_frames.keys())
            available_ids &= visual_ids
            
        if self.use_audio and self.audio_features is not None:
            audio_ids = set(self.audio_features.keys())
            available_ids &= audio_ids
            
        if self.use_text and self.entity_data is not None:
            entity_ids = set(self.entity_data['video_id'])
            available_ids &= entity_ids
        
        # Filter dataset to only include videos with required features
        original_length = len(self.data)
        self.data = self.data[self.data['video_id'].isin(available_ids)].reset_index(drop=True)
        
        print(f"Filtered dataset: {original_length} -> {len(self.data)} samples (removed {original_length - len(self.data)} incomplete samples)")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = torch.tensor(item['label'], dtype=torch.long)
        
        # Initialize features as None
        visual_features = None
        audio_features = None
        concatenated_text = None
        
        # Get visual features if enabled
        if self.use_image:
            if self.fea_frames is None or vid not in self.fea_frames:
                print(f"Warning: Video features not found for {vid}, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
            visual_features = torch.Tensor(self.fea_frames[vid])  # (16, 1024)
            
        # Get audio features if enabled
        if self.use_audio:
            if self.audio_features is None or vid not in self.audio_features:
                print(f"Warning: Audio features not found for {vid}, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
            audio_features = torch.Tensor(self.audio_features[vid])  # (seq_len, 768)
        
        # Get text features if enabled
        if self.use_text:
            if self.entity_data is None:
                print(f"Warning: Entity data not loaded, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
                
            entity_row = self.entity_data[self.entity_data['video_id'] == vid]
            if len(entity_row) == 0:
                print(f"Warning: Entity data not found for {vid}, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
            
            entity_row = entity_row.iloc[0]
        
        # Process text features if enabled
        if self.use_text:
            # Concatenate text features based on description and temp_evolution settings
            text_parts = []
            
            # Always include title and keywords
            if 'title' in entity_row and pd.notna(entity_row['title']):
                text_parts.append(str(entity_row['title']))
            if 'keywords' in entity_row and pd.notna(entity_row['keywords']):
                text_parts.append(str(entity_row['keywords']))
                
            # Include description if enabled
            if self.description and 'description' in entity_row and pd.notna(entity_row['description']):
                text_parts.append(str(entity_row['description']))
                
            # Include temporal_evolution if enabled  
            if self.temp_evolution and 'temporal_evolution' in entity_row and pd.notna(entity_row['temporal_evolution']):
                text_parts.append(str(entity_row['temporal_evolution']))
            
            concatenated_text = ' '.join(text_parts)
        
        return {
            'vid': vid,
            'label': label,
            'concatenated_text': concatenated_text,
            'visual_features': visual_features,
            'audio_features': audio_features,
            'use_text': self.use_text,
            'use_image': self.use_image,
            'use_audio': self.use_audio,
        }
    
class FakeSVCollator_ExMRD:
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = self._infer_max_len(tokenizer_name)
        logger.info(f"[{self.__class__.__name__}] tokenizer={self.tokenizer_name} max_text_len={self.max_text_len}")

    def _infer_max_len(self, name):
        # Keep FakeSV setting: force max_len to 512
        return 512
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        
        # Get modality flags from first item (should be same for all)
        use_text = batch[0]['use_text']
        use_image = batch[0]['use_image'] 
        use_audio = batch[0]['use_audio']
        
        # Initialize return dict
        result = {
            'vids': vids,
            'labels': labels,
            'use_text': use_text,
            'use_image': use_image,
            'use_audio': use_audio,
        }
        
        # Process text features if enabled
        if use_text:
            concatenated_texts = [item['concatenated_text'] for item in batch]
            max_len = self.max_text_len
            entity_text_input = self.tokenizer(
                concatenated_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_len
            )
            result['entity_text_input'] = entity_text_input
            
        # Process visual features if enabled
        if use_image:
            visual_features = torch.stack([item['visual_features'] for item in batch])
            result['visual_features'] = visual_features
            
        # Process audio features if enabled
        if use_audio:
            audio_features = torch.stack([item['audio_features'] for item in batch])
            result['audio_features'] = audio_features
        
        return result

class FakeTTDataset_ExMRD(FakeTTDataset):
    def __init__(self, fold: int, split: str, lm: str='gpt-4o', **kwargs):
        super(FakeTTDataset_ExMRD, self).__init__()
        self.data = self._get_data(fold, split)
        
        self.ocr = pd.read_json('data/FakeTT/ocr.jsonl', lines=True, dtype={'vid': 'str'})
        self.lm_ocr = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_text_refine.jsonl', lines=True, dtype={'vid': 'str'})
        self.caption = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_visual_refine.jsonl', lines=True, dtype={'vid': 'str'})
     
        self.lm_comsense = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_retrieve.jsonl', lines=True, dtype={'vid': 'str'})
        self.lm_causal = pd.read_json(f'data/FakeTT/CoT/{lm}/lm_reason.jsonl', lines=True, dtype={'vid': 'str'})
    
        self.fea_frames = torch.load('data/FakeTT/fea/vit_tensor.pt', weights_only=True)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        # response
        text_caption = self.caption[self.caption['vid']==vid]['text'].iloc[0]
        text_lm_ocr = self.lm_ocr[self.lm_ocr['vid']==vid]['text'].iloc[0]
        text_lm_ocr = f'{item['description']}\n{text_lm_ocr}'
        text_lm_comsense = self.lm_comsense[self.lm_comsense['vid']==vid]['text'].iloc[0]
        text_lm_causal = self.lm_causal[self.lm_causal['vid']==vid]['text'].iloc[0]
        
        label = torch.tensor(item['label'], dtype=torch.long)

        fea_frames = self.fea_frames[vid]
        fea_frames = torch.Tensor(fea_frames)

        
        return {
            'vid': vid,
            'label': label,
            'text_lm_ocr': text_lm_ocr,
            'text_caption': text_caption,
            'text_comsense': text_lm_comsense,
            'text_causal': text_lm_causal,
            'fea_frames': fea_frames,
        }
    
class FakeTTCollator_ExMRD:
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = self._infer_max_len(tokenizer_name)
        logger.info(f"[{self.__class__.__name__}] tokenizer={self.tokenizer_name} max_text_len={self.max_text_len}")

    def _infer_max_len(self, name):
        max_len = None
        try:
            cfg = AutoConfig.from_pretrained(name)
            max_len = getattr(cfg, 'max_position_embeddings', None)
        except Exception:
            pass
        if max_len is None:
            max_len = getattr(self.tokenizer, 'model_max_length', None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 100000:
            nm = str(name).lower()
            max_len = 77 if 'clip' in nm else 512
        return max_len
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        texts_lm_ocr = [item['text_lm_ocr'] for item in batch]
        texts_caption = [item['text_caption'] for item in batch]
        texts_comsense = [item['text_comsense'] for item in batch]
        texts_causal = [item['text_causal'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        fea_frames = torch.stack([item['fea_frames'] for item in batch])

        max_len = self.max_text_len
        # texts_input = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        lm_ocr_input = self.tokenizer(list(texts_lm_ocr), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        caption_input = self.tokenizer(list(texts_caption), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        comsense_input = self.tokenizer(list(texts_comsense), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        causal_input = self.tokenizer(list(texts_causal), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {
            'vids': vids,
            'fea_frames': fea_frames,
            'labels': labels,
            'lm_ocr_input': lm_ocr_input,
            'caption_input': caption_input,
            'comsense_input': comsense_input,
            'causal_input': causal_input,
        }


class FVCDataset_ExMRD(FVCDataset):
    def __init__(self, fold: int, split: str, lm: str='gpt-4o', **kwargs):
        super(FVCDataset_ExMRD, self).__init__()
        self.data = self._get_data(fold, split)
        
        self.lm_ocr = pd.read_json(f'data/FVC/CoT/{lm}/lm_text_refine.jsonl', lines=True, dtype={'vid': 'str'})
        self.caption = pd.read_json(f'data/FVC/CoT/{lm}/lm_visual_refine.jsonl', lines=True, dtype={'vid': 'str'})
         
        self.lm_comsense = pd.read_json(f'data/FVC/CoT/{lm}/lm_retrieve.jsonl', lines=True, dtype={'vid': 'str'})
        self.lm_causal = pd.read_json(f'data/FVC/CoT/{lm}/lm_reason.jsonl', lines=True, dtype={'vid': 'str'})
    
        self.fea_frames = torch.load('data/FVC/fea/vit_tensor.pt', weights_only=True)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['vid']
        # response
        text_caption = self.caption[self.caption['vid']==vid]['text'].iloc[0]
        text_lm_ocr = self.lm_ocr[self.lm_ocr['vid']==vid]['text'].iloc[0]
        text_lm_ocr = f'{item['title']}\n{item['description']}\n{text_lm_ocr}'
        
        text_lm_comsense = self.lm_comsense[self.lm_comsense['vid']==vid]['text'].iloc[0]
        text_lm_causal = self.lm_causal[self.lm_causal['vid']==vid]['text'].iloc[0]
        
        label = torch.tensor(item['label'], dtype=torch.long)

        fea_frames = self.fea_frames[vid]
        fea_frames = torch.Tensor(fea_frames)
        
        return {
            'vid': vid,
            'label': label,
            'text_lm_ocr': text_lm_ocr,
            'text_caption': text_caption,
            'text_comsense': text_lm_comsense,
            'text_causal': text_lm_causal,
            'fea_frames': fea_frames,
        }
    
class FVCCollator_ExMRD:
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = self._infer_max_len(tokenizer_name)
        logger.info(f"[{self.__class__.__name__}] tokenizer={self.tokenizer_name} max_text_len={self.max_text_len}")

    def _infer_max_len(self, name):
        max_len = None
        try:
            cfg = AutoConfig.from_pretrained(name)
            max_len = getattr(cfg, 'max_position_embeddings', None)
        except Exception:
            pass
        if max_len is None:
            max_len = getattr(self.tokenizer, 'model_max_length', None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 100000:
            nm = str(name).lower()
            max_len = 77 if 'clip' in nm else 512
        return max_len
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        texts_lm_ocr = [item['text_lm_ocr'] for item in batch]
        texts_caption = [item['text_caption'] for item in batch]
        texts_comsense = [item['text_comsense'] for item in batch]
        texts_causal = [item['text_causal'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        fea_frames = torch.stack([item['fea_frames'] for item in batch])

        max_len = self.max_text_len
        lm_ocr_input = self.tokenizer(list(texts_lm_ocr), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        caption_input = self.tokenizer(list(texts_caption), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        comsense_input = self.tokenizer(list(texts_comsense), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        causal_input = self.tokenizer(list(texts_causal), padding=True, truncation=True, return_tensors='pt', max_length=max_len)
        return {
            'vids': vids,
            'fea_frames': fea_frames,
            'labels': labels,
            'lm_ocr_input': lm_ocr_input,
            'caption_input': caption_input,
            'comsense_input': comsense_input,
            'causal_input': causal_input,
        }


class FakeSVDataset_Defer(FakeSVDataset_ExMRD):
    """Dataset for learn-to-defer training with LLM predictions and label smoothing"""
    
    def __init__(self, fold: int, split: str, llm_predictions_path: str = None, 
                 label_smoothing_epsilon: float = 0.01, filter_k: int = None, 
                 llm_independent: bool = False, **kwargs):
        super(FakeSVDataset_Defer, self).__init__(fold, split, filter_k=filter_k, **kwargs)
        
        self.epsilon = label_smoothing_epsilon
        self.filter_k = filter_k
        self.llm_independent = llm_independent
        
        # Load LLM predictions - use k-specific and independent-specific path if specified
        if llm_predictions_path is None:
            base_path = "data/FakeSV/entity_claims/gating_predictions"
            if filter_k is not None:
                base_path += f"_k{filter_k}"
            if llm_independent:
                base_path += "_independent"
            llm_predictions_path = f"{base_path}/gating_predictions.json"
        
        llm_path = Path(llm_predictions_path)
        if not llm_path.exists():
            raise FileNotFoundError(f"LLM predictions file not found: {llm_path}")
        
        with open(llm_path, 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        
        # Create mapping from video_id to LLM predictions
        self.llm_predictions = {}
        for sample in llm_data:
            video_id = sample['video_id']
            
            if self.llm_independent:
                # For independent mode, both predictions are the same
                llm_pred = sample['predictions']['slm_predicts_true']['prediction']
                slm_true_pred = llm_pred
                slm_fake_pred = llm_pred
            else:
                # Get LLM predictions for both scenarios (SLM predicts true/fake)
                slm_true_pred = sample['predictions']['slm_predicts_true']['prediction']
                slm_fake_pred = sample['predictions']['slm_predicts_fake']['prediction']
                
                # Convert Chinese labels to numeric and then to probabilities
                # Use the better of the two LLM predictions (the one that matches ground truth if available)
                ground_truth = sample['ground_truth']
                
                # Choose the LLM prediction that matches ground truth, or slm_true_pred as default
                if slm_true_pred == ground_truth:
                    llm_pred = slm_true_pred
                elif slm_fake_pred == ground_truth:
                    llm_pred = slm_fake_pred
                else:
                    # If neither matches, choose slm_true_pred (could also be slm_fake_pred)
                    llm_pred = slm_true_pred
            
            # Convert to numeric label
            llm_label = 0 if llm_pred == '真' else 1
            
            # Apply label smoothing: convert hard label to probability distribution
            if llm_label == 0:  # Real
                p_dm = torch.tensor([1.0 - self.epsilon, self.epsilon], dtype=torch.float32)
            else:  # Fake
                p_dm = torch.tensor([self.epsilon, 1.0 - self.epsilon], dtype=torch.float32)
            
            self.llm_predictions[video_id] = {
                'p_dm': p_dm,
                'hard_label': llm_label,
                'slm_true_pred': slm_true_pred,
                'slm_fake_pred': slm_fake_pred
            }
        
        # Filter dataset to only include samples with LLM predictions
        available_llm_ids = set(self.llm_predictions.keys())
        original_length = len(self.data)
        self.data = self.data[self.data['video_id'].isin(available_llm_ids)].reset_index(drop=True)
        
        print(f"Defer dataset: {original_length} -> {len(self.data)} samples "
              f"(removed {original_length - len(self.data)} without LLM predictions)")
        print(f"Loaded LLM predictions for {len(self.llm_predictions)} videos with ε={self.epsilon}")
    
    def __getitem__(self, idx):
        # Get base sample from parent class
        sample = super(FakeSVDataset_Defer, self).__getitem__(idx)
        vid = sample['vid']
        
        # Add LLM prediction probabilities
        if vid in self.llm_predictions:
            sample['p_dm'] = self.llm_predictions[vid]['p_dm']
        else:
            # This shouldn't happen due to filtering, but add fallback
            print(f"Warning: No LLM prediction for {vid}, using uniform distribution")
            sample['p_dm'] = torch.tensor([0.5, 0.5], dtype=torch.float32)
        
        return sample


class FakeSVCollator_Defer(FakeSVCollator_ExMRD):
    """Collator for defer dataset that includes LLM predictions"""
    
    def __call__(self, batch):
        # Get base batch from parent collator
        result = super(FakeSVCollator_Defer, self).__call__(batch)
        
        # Add LLM prediction probabilities
        p_dm = torch.stack([item['p_dm'] for item in batch])  # (batch_size, 2)
        result['p_dm'] = p_dm
        
        return result


class FVCDataset_Retrieval(Dataset):
    """Dataset for retrieval-augmented training with positive/negative samples for FVC (FakeTT-like format)."""

    def __init__(self, fold: int, split: str, retrieval_path: str = None,
                 retrieval_path_5fold: str = None,
                 filter_k: int = None, description: bool = True, temp_evolution: bool = True,
                 use_text: bool = True, use_image: bool = True, use_audio: bool = True, **kwargs):
        from pathlib import Path
        import json as _json
        super().__init__()

        self.filter_k = filter_k
        self.description = description
        self.temp_evolution = temp_evolution
        self.use_text = use_text
        self.use_image = use_image
        self.use_audio = use_audio

        # Load FVC base data in FakeTT-like format
        self.data = self._get_data(fold, split)

        # Load retrieval data (5-fold preference when fold in [1..5])
        if retrieval_path is None:
            retrieval_path = "text_similarity_results/uncertainty_full_dataset_retrieval_LongCLIP-GmP-ViT-L-14.json"
        # Prefer 5-fold path when fold in [1..5]
        if isinstance(fold, int) and 1 <= fold <= 5 and retrieval_path_5fold:
            retrieval_path = retrieval_path_5fold
        retrieval_file = None
        if isinstance(fold, int) and 1 <= fold <= 5:
            dataset_root = Path('data/FVC')
            rp = Path(retrieval_path)
            base_dir = None
            if rp.is_dir():
                base_dir = rp
            elif not rp.is_absolute() and (dataset_root / rp).is_dir():
                base_dir = (dataset_root / rp)
            if base_dir is not None:
                candidate = base_dir / f"fold_{fold}.json"
                if candidate.exists():
                    retrieval_file = candidate
                else:
                    logger.warning(f"\x1b[31m[5-fold] fold_{fold}.json not found under directory: {base_dir}. Will search fallbacks.\x1b[0m")
            else:
                parent = rp.parent if rp.name else rp
                candidate = (parent / f"fold_{fold}.json") if parent else None
                if candidate and candidate.exists():
                    retrieval_file = candidate
                else:
                    tsr = dataset_root / 'text_similarity_results'
                    if tsr.exists():
                        matches = list(tsr.rglob(f"*/fold_{fold}.json"))
                        if matches:
                            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                            retrieval_file = matches[0]
                            logger.warning(f"\x1b[31m[5-fold] Using auto-matched fold file: {retrieval_file}\x1b[0m")
                        else:
                            legacy = list(tsr.rglob(f"ret_*_f{fold}*.json"))
                            if legacy:
                                legacy.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                retrieval_file = legacy[0]
                                logger.warning(f"\x1b[31m[5-fold] Using legacy-matched file: {retrieval_file}\x1b[0m")
            if retrieval_file is None:
                direct = Path(retrieval_path)
                if not direct.exists():
                    direct = dataset_root / retrieval_path
                if direct.exists():
                    retrieval_file = direct
                    logger.warning(f"\x1b[31m[5-fold] Fold file not found. Falling back to: {retrieval_file}\x1b[0m")
                else:
                    raise FileNotFoundError(f"Retrieval file not found for 5-fold (fold {fold}). Checked directory and fallbacks.")
        else:
            retrieval_file = Path(retrieval_path)
            if not retrieval_file.exists():
                retrieval_file = Path(f"data/FVC/{retrieval_path}")
            if not retrieval_file.exists():
                raise FileNotFoundError(f"Retrieval file not found: {retrieval_path}")
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            retrieval_data = _json.load(f)
        self.retrieval_mapping = {}
        for item in retrieval_data:
            video_id = str(item['query_video']['video_id'])
            self.retrieval_mapping[video_id] = {
                'positive_video_id': str(item['similar_true']['video_id']),
                'negative_video_id': str(item['similar_fake']['video_id']),
                'positive_data': item['similar_true'],
                'negative_data': item['similar_fake']
            }

        # Load features based on modality controls
        self.fea_frames = None
        self.audio_features = None
        if self.use_image:
            self.fea_frames = torch.load('data/FVC/fea/vit_tensor.pt', weights_only=True)
        if self.use_audio:
            # Prefer most recent audio_features_frames*.pt (model-mark suffix), fall back if none
            fea_dir = Path('data/FVC/fea')
            candidates = sorted(fea_dir.glob('audio_features_frames*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
            audio_path = candidates[0] if candidates else fea_dir / 'audio_features_frames.pt'
            if audio_path.exists():
                self.audio_features = torch.load(str(audio_path), weights_only=True)
                # Infer audio feature dim from first entry
                try:
                    first_key = next(iter(self.audio_features.keys()))
                    first_val = self.audio_features[first_key]
                    if hasattr(first_val, 'shape') and len(first_val.shape) == 2:
                        self.audio_dim = int(first_val.shape[1])
                except Exception:
                    pass
                print(f"Loaded FVC audio features from: {audio_path} (dim={self.audio_dim})")
            else:
                print("Warning: Audio features not found for FVC, using dummy features")
                self.audio_features = None

        # Load LLM video descriptions (REQUIRED)
        try:
            llm_data = []
            with open('data/FVC/llm_video_descriptions.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = _json.loads(line)
                        item['video_id'] = str(item['video_id'])
                        llm_data.append(item)
            self.entity_data = pd.DataFrame(llm_data)
            print(f"Loaded LLM descriptions: {len(self.entity_data)} samples")
        except Exception as e:
            raise FileNotFoundError(f"REQUIRED: Could not load LLM descriptions for FVC: {e}.")

        # Filter dataset to only include samples with retrieval data and required features
        available_retrieval_ids = set(self.retrieval_mapping.keys())
        data_ids = set(self.data['video_id'])
        available_ids = data_ids & available_retrieval_ids
        if self.use_image and self.fea_frames is not None:
            available_ids &= set(self.fea_frames.keys())
        if self.use_audio and self.audio_features is not None:
            available_ids &= set(self.audio_features.keys())

        print(f"Using {len(available_ids)} samples with retrieval data and features (FVC)")
        original_length = len(self.data)
        self.data = self.data[self.data['video_id'].isin(available_ids)].reset_index(drop=True)
        print(f"FVC Retrieval dataset: {original_length} -> {len(self.data)} samples (removed {original_length - len(self.data)})")

    def _get_data(self, fold, split):
        if fold in ['temporal']:
            return self._get_temporal_data(split)
        elif fold in [1, 2, 3, 4, 5]:
            return self._get_fold_data(fold, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")

    def _get_complete_data(self):
        # FVC aligned to FakeTT format; data.json is JSONL lines
        data = pd.read_json('data/FVC/data.json', orient='records', lines=True, dtype={'video_id': 'str'})
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['annotation'].replace(replace_values)
        if 'event' in data.columns:
            data['event'], _ = pd.factorize(data['event'])
        else:
            data['event'] = 0
        return data

    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'data/FVC/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'data/FVC/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, 'r') as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = f'data/FVC/vids/vid_time3_{split}.txt'
        with open(vid_path, 'r') as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data

    def __len__(self):
        return len(self.data)

    def _construct_text_representation(self, row_data):
        # Reuse FakeTT-style text assembly
        text_parts = []
        if 'title' in row_data and pd.notna(row_data['title']) and row_data['title']:
            text_parts.append(str(row_data['title']))
        if 'event' in row_data and pd.notna(row_data['event']) and row_data['event']:
            text_parts.append(str(row_data['event']))
        elif 'keywords' in row_data and pd.notna(row_data['keywords']) and row_data['keywords']:
            text_parts.append(str(row_data['keywords']))
        if self.description and 'description' in row_data and pd.notna(row_data['description']) and row_data['description']:
            text_parts.append(str(row_data['description']))
        if self.temp_evolution and 'temporal_evolution' in row_data and pd.notna(row_data['temporal_evolution']) and row_data['temporal_evolution']:
            text_parts.append(str(row_data['temporal_evolution']))
        return ' '.join(text_parts)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = torch.tensor(item['label'], dtype=torch.long)

        visual_features = None
        audio_features = None
        concatenated_text = None

        if self.use_image:
            if self.fea_frames is None or vid not in self.fea_frames:
                print(f"Warning: Video features not found for {vid}, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
            visual_features = torch.Tensor(self.fea_frames[vid])

        if self.use_audio:
            if self.audio_features is None or vid not in self.audio_features:
                audio_features = torch.zeros(16, self.audio_dim)
            else:
                audio_features = torch.Tensor(self.audio_features[vid])

        if self.use_text:
            entity_row = self.entity_data[self.entity_data['video_id'] == vid]
            if len(entity_row) == 0:
                raise ValueError(f"No LLM description found for video {vid}. LLM descriptions are mandatory.")
            entity_row = entity_row.iloc[0]
            concatenated_text = self._construct_text_representation(entity_row)

        # Retrieval samples
        positive_features = {}
        negative_features = {}
        positive_video_id = None
        negative_video_id = None
        if vid in self.retrieval_mapping:
            retrieval_info = self.retrieval_mapping[vid]
            pos_vid = retrieval_info['positive_video_id']
            neg_vid = retrieval_info['negative_video_id']
            positive_video_id = pos_vid
            negative_video_id = neg_vid
            if self.use_image and self.fea_frames is not None:
                if pos_vid in self.fea_frames:
                    positive_features['visual'] = torch.Tensor(self.fea_frames[pos_vid])
                if neg_vid in self.fea_frames:
                    negative_features['visual'] = torch.Tensor(self.fea_frames[neg_vid])
            if self.use_audio and self.audio_features is not None:
                if pos_vid in self.audio_features:
                    positive_features['audio'] = torch.Tensor(self.audio_features[pos_vid])
                if neg_vid in self.audio_features:
                    negative_features['audio'] = torch.Tensor(self.audio_features[neg_vid])
            if self.use_text:
                pos_data = retrieval_info['positive_data']
                neg_data = retrieval_info['negative_data']
                positive_features['text'] = self._construct_text_representation(pos_data)
                negative_features['text'] = self._construct_text_representation(neg_data)

        return {
            'vid': vid,
            'label': label,
            'concatenated_text': concatenated_text,
            'visual_features': visual_features,
            'audio_features': audio_features,
            'use_text': self.use_text,
            'use_image': self.use_image,
            'use_audio': self.use_audio,
            'positive_features': positive_features,
            'negative_features': negative_features,
            'positive_video_id': positive_video_id,
            'negative_video_id': negative_video_id,
        }


class FVCCollator_Retrieval:
    """Collator for FVC retrieval dataset that includes positive/negative samples"""
    
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = self._infer_max_len(tokenizer_name)
        logger.info(f"[{self.__class__.__name__}] tokenizer={self.tokenizer_name} max_text_len={self.max_text_len}")

    def _infer_max_len(self, name):
        max_len = None
        try:
            cfg = AutoConfig.from_pretrained(name)
            max_len = getattr(cfg, 'max_position_embeddings', None)
        except Exception:
            pass
        if max_len is None:
            max_len = getattr(self.tokenizer, 'model_max_length', None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 100000:
            nm = str(name).lower()
            max_len = 77 if 'clip' in nm else 512
        return max_len
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        
        # Get modality flags from first item (should be same for all)
        use_text = batch[0]['use_text']
        use_image = batch[0]['use_image'] 
        use_audio = batch[0]['use_audio']
        
        # Initialize return dict
        result = {
            'vids': vids,
            'labels': labels,
            'use_text': use_text,
            'use_image': use_image,
            'use_audio': use_audio,
        }
        
        # Process text features if enabled
        if use_text:
            concatenated_texts = [item['concatenated_text'] for item in batch]
            max_len = self.max_text_len
            entity_text_input = self.tokenizer(
                concatenated_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_len
            )
            result['entity_text_input'] = entity_text_input
            
        # Process visual features if enabled
        if use_image:
            visual_features = torch.stack([item['visual_features'] for item in batch])
            result['visual_features'] = visual_features
            
        # Process audio features if enabled
        if use_audio:
            audio_features = torch.stack([item['audio_features'] for item in batch])
            result['audio_features'] = audio_features
        
        # Process positive and negative text features if enabled
        if use_text:
            positive_texts = []
            negative_texts = []
            
            for item in batch:
                pos_text = item['positive_features'].get('text', '')
                neg_text = item['negative_features'].get('text', '')
                positive_texts.append(pos_text)
                negative_texts.append(neg_text)
            
            # Tokenize positive texts
            positive_text_input = self.tokenizer(
                positive_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=self.max_text_len
            )
            result['positive_text_input'] = positive_text_input
            
            # Tokenize negative texts
            negative_text_input = self.tokenizer(
                negative_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=self.max_text_len
            )
            result['negative_text_input'] = negative_text_input
        
        # Process positive and negative visual features if enabled
        if use_image:
            positive_visual_features = []
            negative_visual_features = []
            
            for item in batch:
                pos_visual = item['positive_features'].get('visual')
                neg_visual = item['negative_features'].get('visual')
                
                if pos_visual is not None:
                    positive_visual_features.append(pos_visual)
                else:
                    positive_visual_features.append(torch.zeros(16, 1024))
                    
                if neg_visual is not None:
                    negative_visual_features.append(neg_visual)
                else:
                    negative_visual_features.append(torch.zeros(16, 1024))
            
            result['positive_visual_features'] = torch.stack(positive_visual_features)
            result['negative_visual_features'] = torch.stack(negative_visual_features)
        
        # Process positive and negative audio features if enabled
        if use_audio:
            positive_audio_features = []
            negative_audio_features = []
            # Infer audio dim from batch audio_features if available; else default 768
            inferred_dim = 768
            try:
                if 'audio_features' in batch[0] and isinstance(batch[0]['audio_features'], torch.Tensor):
                    inferred_dim = int(batch[0]['audio_features'].shape[1])
            except Exception:
                pass
            
            for item in batch:
                pos_audio = item['positive_features'].get('audio')
                neg_audio = item['negative_features'].get('audio')
                
                if pos_audio is not None:
                    positive_audio_features.append(pos_audio)
                else:
                    positive_audio_features.append(torch.zeros(16, inferred_dim))
                    
                if neg_audio is not None:
                    negative_audio_features.append(neg_audio)
                else:
                    negative_audio_features.append(torch.zeros(16, inferred_dim))
            
            result['positive_audio_features'] = torch.stack(positive_audio_features)
            result['negative_audio_features'] = torch.stack(negative_audio_features)
        
        # Add video IDs for debugging
        result['positive_video_ids'] = [item['positive_video_id'] for item in batch]
        result['negative_video_ids'] = [item['negative_video_id'] for item in batch]
        
        return result


class FakeSVDataset_Retrieval(FakeSVDataset_ExMRD):
    """Dataset for retrieval-augmented training with positive/negative samples"""
    
    def __init__(self, fold: int, split: str, retrieval_path: str = None,
                 retrieval_path_5fold: str = None,
                 filter_k: int = None, **kwargs):
        super(FakeSVDataset_Retrieval, self).__init__(fold, split, filter_k=filter_k, **kwargs)
        
        self.filter_k = filter_k
        
        # Load retrieval data (5-fold preference when fold in [1..5])
        if retrieval_path is None:
            retrieval_path = "text_similarity_results/full_dataset_retrieval_chinese-clip-vit-large-patch14.json"
        # Prefer 5-fold path when fold in [1..5]
        if isinstance(fold, int) and 1 <= fold <= 5 and retrieval_path_5fold:
            retrieval_path = retrieval_path_5fold

        retrieval_file = None
        if isinstance(fold, int) and 1 <= fold <= 5:
            dataset_root = Path('data/FakeSV')
            rp = Path(retrieval_path)
            base_dir = None
            if rp.is_dir():
                base_dir = rp
            elif not rp.is_absolute() and (dataset_root / rp).is_dir():
                base_dir = (dataset_root / rp)
            if base_dir is not None:
                candidate = base_dir / f"fold_{fold}.json"
                if candidate.exists():
                    retrieval_file = candidate
                else:
                    logger.warning(f"\x1b[31m[5-fold] fold_{fold}.json not found under directory: {base_dir}. Will search fallbacks.\x1b[0m")
            else:
                parent = rp.parent if rp.name else rp
                candidate = (parent / f"fold_{fold}.json") if parent else None
                if candidate and candidate.exists():
                    retrieval_file = candidate
                else:
                    tsr = dataset_root / 'text_similarity_results'
                    if tsr.exists():
                        matches = list(tsr.rglob(f"*/fold_{fold}.json"))
                        if matches:
                            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                            retrieval_file = matches[0]
                            logger.warning(f"\x1b[31m[5-fold] Using auto-matched fold file: {retrieval_file}\x1b[0m")
                        else:
                            legacy = list(tsr.rglob(f"ret_*_f{fold}*.json"))
                            if legacy:
                                legacy.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                retrieval_file = legacy[0]
                                logger.warning(f"\x1b[31m[5-fold] Using legacy-matched file: {retrieval_file}\x1b[0m")
            if retrieval_file is None:
                direct = Path(retrieval_path)
                if not direct.exists():
                    direct = dataset_root / retrieval_path
                if direct.exists():
                    retrieval_file = direct
                    logger.warning(f"\x1b[31m[5-fold] Fold file not found. Falling back to: {retrieval_file}\x1b[0m")
                else:
                    raise FileNotFoundError(f"Retrieval file not found for 5-fold (fold {fold}). Checked directory and fallbacks.")
        else:
            retrieval_file = Path(retrieval_path)
            if not retrieval_file.exists():
                retrieval_file = Path(f"data/FakeSV/{retrieval_path}")
            if not retrieval_file.exists():
                raise FileNotFoundError(f"Retrieval file not found: {retrieval_path}")
        
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        
        # Create mapping from video_id to retrieval data
        self.retrieval_mapping = {}
        for item in retrieval_data:
            video_id = item['query_video']['video_id']
            self.retrieval_mapping[video_id] = {
                'positive_video_id': item['similar_true']['video_id'],
                'negative_video_id': item['similar_fake']['video_id'],
                'positive_data': item['similar_true'],
                'negative_data': item['similar_fake']
            }
        
        # Filter dataset to only include samples with retrieval data
        available_retrieval_ids = set(self.retrieval_mapping.keys())
        original_length = len(self.data)
        self.data = self.data[self.data['video_id'].isin(available_retrieval_ids)].reset_index(drop=True)
        
        print(f"Retrieval dataset: {original_length} -> {len(self.data)} samples "
              f"(removed {original_length - len(self.data)} without retrieval data)")
        print(f"Loaded retrieval data for {len(self.retrieval_mapping)} videos")
    
    def __getitem__(self, idx):
        # Get base sample from parent class
        sample = super(FakeSVDataset_Retrieval, self).__getitem__(idx)
        vid = sample['vid']
        
        if vid not in self.retrieval_mapping:
            print(f"Warning: No retrieval data for {vid}")
            return sample
        
        retrieval_info = self.retrieval_mapping[vid]
        pos_vid = retrieval_info['positive_video_id']
        neg_vid = retrieval_info['negative_video_id']
        
        # Add positive sample features
        pos_features = {}
        neg_features = {}
        
        # Visual features
        if self.use_image and self.fea_frames is not None:
            if pos_vid in self.fea_frames:
                pos_features['visual'] = torch.Tensor(self.fea_frames[pos_vid])  # (16, 1024)
            if neg_vid in self.fea_frames:
                neg_features['visual'] = torch.Tensor(self.fea_frames[neg_vid])  # (16, 1024)
        
        # Audio features  
        if self.use_audio and self.audio_features is not None:
            if pos_vid in self.audio_features:
                pos_features['audio'] = torch.Tensor(self.audio_features[pos_vid])  # (16, 768)
            if neg_vid in self.audio_features:
                neg_features['audio'] = torch.Tensor(self.audio_features[neg_vid])  # (16, 768)
        
        # Text features - construct from retrieval data
        if self.use_text:
            pos_data = retrieval_info['positive_data']
            neg_data = retrieval_info['negative_data']
            
            # Construct text for positive sample
            pos_text_parts = []
            if 'title' in pos_data and pd.notna(pos_data['title']):
                pos_text_parts.append(str(pos_data['title']))
            if 'keywords' in pos_data and pd.notna(pos_data['keywords']):
                pos_text_parts.append(str(pos_data['keywords']))
            if self.description and 'description' in pos_data and pd.notna(pos_data['description']):
                pos_text_parts.append(str(pos_data['description']))
            if self.temp_evolution and 'temporal_evolution' in pos_data and pd.notna(pos_data['temporal_evolution']):
                pos_text_parts.append(str(pos_data['temporal_evolution']))
            
            # Construct text for negative sample
            neg_text_parts = []
            if 'title' in neg_data and pd.notna(neg_data['title']):
                neg_text_parts.append(str(neg_data['title']))
            if 'keywords' in neg_data and pd.notna(neg_data['keywords']):
                neg_text_parts.append(str(neg_data['keywords']))
            if self.description and 'description' in neg_data and pd.notna(neg_data['description']):
                neg_text_parts.append(str(neg_data['description']))
            if self.temp_evolution and 'temporal_evolution' in neg_data and pd.notna(neg_data['temporal_evolution']):
                neg_text_parts.append(str(neg_data['temporal_evolution']))
            
            pos_features['text'] = ' '.join(pos_text_parts)
            neg_features['text'] = ' '.join(neg_text_parts)
        
        # Add retrieval features to sample
        sample['positive_features'] = pos_features
        sample['negative_features'] = neg_features
        sample['positive_video_id'] = pos_vid
        sample['negative_video_id'] = neg_vid
        
        return sample


class FakeSVCollator_Retrieval(FakeSVCollator_ExMRD):
    """Collator for retrieval dataset that includes positive/negative samples"""
    
    def __call__(self, batch):
        # Get base batch from parent collator
        result = super(FakeSVCollator_Retrieval, self).__call__(batch)
        
        # Get modality flags from first item (should be same for all)
        use_text = batch[0]['use_text']
        use_image = batch[0]['use_image'] 
        use_audio = batch[0]['use_audio']
        
        # Process positive and negative text features if enabled
        if use_text:
            positive_texts = []
            negative_texts = []
            
            for item in batch:
                pos_text = item['positive_features'].get('text', '')
                neg_text = item['negative_features'].get('text', '')
                positive_texts.append(pos_text)
                negative_texts.append(neg_text)
            
            max_len = self.max_text_len
            
            # Tokenize positive texts
            positive_text_input = self.tokenizer(
                positive_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_len
            )
            result['positive_text_input'] = positive_text_input
            
            # Tokenize negative texts
            negative_text_input = self.tokenizer(
                negative_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_len
            )
            result['negative_text_input'] = negative_text_input
        
        # Process positive and negative visual features if enabled
        if use_image:
            positive_visual_features = []
            negative_visual_features = []
            
            for item in batch:
                pos_visual = item['positive_features'].get('visual')
                neg_visual = item['negative_features'].get('visual')
                
                if pos_visual is not None:
                    positive_visual_features.append(pos_visual)
                else:
                    # Create dummy features if missing
                    positive_visual_features.append(torch.zeros(16, 1024))
                    
                if neg_visual is not None:
                    negative_visual_features.append(neg_visual)
                else:
                    # Create dummy features if missing  
                    negative_visual_features.append(torch.zeros(16, 1024))
            
            result['positive_visual_features'] = torch.stack(positive_visual_features)
            result['negative_visual_features'] = torch.stack(negative_visual_features)
        
        # Process positive and negative audio features if enabled
        if use_audio:
            positive_audio_features = []
            negative_audio_features = []
            
            for item in batch:
                pos_audio = item['positive_features'].get('audio')
                neg_audio = item['negative_features'].get('audio')
                
                if pos_audio is not None:
                    positive_audio_features.append(pos_audio)
                else:
                    # Create dummy features if missing
                    positive_audio_features.append(torch.zeros(16, 768))
                    
                if neg_audio is not None:
                    negative_audio_features.append(neg_audio)
                else:
                    # Create dummy features if missing
                    negative_audio_features.append(torch.zeros(16, 768))
            
            result['positive_audio_features'] = torch.stack(positive_audio_features)
            result['negative_audio_features'] = torch.stack(negative_audio_features)
        
        # Add video IDs for debugging
        result['positive_video_ids'] = [item['positive_video_id'] for item in batch]
        result['negative_video_ids'] = [item['negative_video_id'] for item in batch]
        
        return result


class FakeTTDataset_Retrieval(FakeTTDataset):
    """Dataset for retrieval-augmented training with positive/negative samples for FakeTT"""
    
    def __init__(self, fold: int, split: str, retrieval_path: str = None,
                 retrieval_path_5fold: str = None,
                 filter_k: int = None, description: bool = True, temp_evolution: bool = True,
                 use_text: bool = True, use_image: bool = True, use_audio: bool = True, **kwargs):
        super(FakeTTDataset_Retrieval, self).__init__(**kwargs)
        
        self.filter_k = filter_k
        self.data = self._get_data(fold, split)
        
        self.description = description
        self.temp_evolution = temp_evolution
        self.use_text = use_text
        self.use_image = use_image
        self.use_audio = use_audio
        
        # Load retrieval data (5-fold preference when fold in [1..5])
        if retrieval_path is None:
            retrieval_path = "text_similarity_results/full_dataset_retrieval_LongCLIP-GmP-ViT-L-14.json"
        # Prefer 5-fold path when fold in [1..5]
        if isinstance(fold, int) and 1 <= fold <= 5 and retrieval_path_5fold:
            retrieval_path = retrieval_path_5fold

        # Resolve fold-specific file if 5-fold; otherwise keep temporal behavior
        retrieval_file = None
        if isinstance(fold, int) and 1 <= fold <= 5:
            # 1) If retrieval_path is a directory, prefer <dir>/fold_{fold}.json
            rp = Path(retrieval_path)
            dataset_root = Path('data/FakeTT')
            # Resolve directory-style base
            base_dir = None
            if rp.is_dir():
                base_dir = rp
            elif not rp.is_absolute() and (dataset_root / rp).is_dir():
                base_dir = (dataset_root / rp)
            if base_dir is not None:
                candidate = base_dir / f"fold_{fold}.json"
                if candidate.exists():
                    retrieval_file = candidate
                else:
                    logger.warning(f"\x1b[31m[5-fold] fold_{fold}.json not found under directory: {base_dir}. Will search fallbacks.\x1b[0m")
            else:
                # 2) If retrieval_path is a file, check its parent for fold_{fold}.json
                parent = rp.parent if rp.name else rp
                candidate = (parent / f"fold_{fold}.json") if parent else None
                if candidate and candidate.exists():
                    retrieval_file = candidate
                else:
                    # Also try under dataset default 5fold subfolders
                    tsr = dataset_root / 'text_similarity_results'
                    if tsr.exists():
                        matches = list(tsr.rglob(f"*/fold_{fold}.json"))
                        if matches:
                            # pick latest by mtime and warn
                            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                            retrieval_file = matches[0]
                            logger.warning(f"\x1b[31m[5-fold] Using auto-matched fold file: {retrieval_file}\x1b[0m")
                        else:
                            # Try legacy pattern ret_*_f{fold}*.json
                            legacy = list(tsr.rglob(f"ret_*_f{fold}*.json"))
                            if legacy:
                                legacy.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                                retrieval_file = legacy[0]
                                logger.warning(f"\x1b[31m[5-fold] Using legacy-matched file: {retrieval_file}\x1b[0m")

            if retrieval_file is None:
                # Final fallback: original retrieval_path (warn)
                direct = Path(retrieval_path)
                if not direct.exists():
                    # Try with dataset prefix
                    direct = dataset_root / retrieval_path
                if direct.exists():
                    retrieval_file = direct
                    logger.warning(f"\x1b[31m[5-fold] Fold file not found. Falling back to: {retrieval_file}\x1b[0m")
                else:
                    raise FileNotFoundError(f"Retrieval file not found for 5-fold (fold {fold}). Checked directory and fallbacks.")
        else:
            # Temporal/other: keep original behavior
            retrieval_file = Path(retrieval_path)
            if not retrieval_file.exists():
                retrieval_file = Path(f"data/FakeTT/{retrieval_path}")
            if not retrieval_file.exists():
                raise FileNotFoundError(f"Retrieval file not found: {retrieval_path}")
        
        import json
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        
        # Create mapping from video_id to retrieval data
        self.retrieval_mapping = {}
        for item in retrieval_data:
            video_id = item['query_video']['video_id']
            self.retrieval_mapping[video_id] = {
                'positive_video_id': item['similar_true']['video_id'],
                'negative_video_id': item['similar_fake']['video_id'],
                'positive_data': item['similar_true'],
                'negative_data': item['similar_fake']
            }
        
        # Load features based on modality controls
        self.fea_frames = None
        self.audio_features = None
        
        if self.use_image:
            self.fea_frames = torch.load('data/FakeTT/fea/vit_tensor.pt', weights_only=True)
        if self.use_audio:
            # Check if audio features exist for FakeTT
            audio_path = Path('data/FakeTT/fea/audio_features_frames.pt')
            if audio_path.exists():
                self.audio_features = torch.load('data/FakeTT/fea/audio_features_frames.pt', weights_only=True)
            else:
                print("Warning: Audio features not found for FakeTT, using dummy features")
                self.audio_features = None
        
        # Load LLM video descriptions for retrieval text construction (REQUIRED)
        # Use manual JSON loading to avoid pandas precision issues with large integers
        try:
            import json
            llm_data = []
            with open('data/FakeTT/llm_video_descriptions.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Ensure video_id is treated as string from the start
                        item['video_id'] = str(item['video_id'])
                        llm_data.append(item)
            
            self.entity_data = pd.DataFrame(llm_data)
            print(f"Loaded LLM descriptions: {len(self.entity_data)} samples")
        except Exception as e:
            raise FileNotFoundError(f"REQUIRED: Could not load LLM descriptions: {e}. LLM descriptions are mandatory.")
            
        # Filter dataset to only include samples with retrieval data and required features
        available_retrieval_ids = set(self.retrieval_mapping.keys())
        data_ids = set(self.data['video_id'])
        available_ids = data_ids & available_retrieval_ids
        
        if self.use_image and self.fea_frames is not None:
            visual_ids = set(self.fea_frames.keys())
            available_ids &= visual_ids
            
        if self.use_audio and self.audio_features is not None:
            audio_ids = set(self.audio_features.keys())
            available_ids &= audio_ids
            
        print(f"Using {len(available_ids)} samples with retrieval data and features")
        
        original_length = len(self.data)
        self.data = self.data[self.data['video_id'].isin(available_ids)].reset_index(drop=True)
        
        print(f"FakeTT Retrieval dataset: {original_length} -> {len(self.data)} samples "
              f"(removed {original_length - len(self.data)} without complete features)")
        print(f"Loaded retrieval data for {len(self.retrieval_mapping)} videos")
    
    def _get_complete_data(self):
        """Override to use the correct FakeTT data file"""
        # Use the same data source as the LLM descriptions to ensure video ID consistency
        data = pd.read_json('data/FakeTT/data.json', orient='records', lines=True, dtype={'video_id': 'str'})
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['annotation'].replace(replace_values)
        data['event'], _ = pd.factorize(data['event'])
        return data
    
    def _get_temporal_data(self, split: str):
        """Override to use filtered training file if available, otherwise use standard temporal splits"""
        # Use filtered training file if filter_k is specified and split is 'train'
        if split == 'train' and self.filter_k is not None:
            vid_path = f'data/FakeTT/train_filtered_k{self.filter_k}.txt'
            if Path(vid_path).exists():
                with open(vid_path, "r") as fr:
                    vids = [line.strip() for line in fr.readlines()]
                data = self._get_complete_data()
                data = data[data['video_id'].isin(vids)]
                return data
            else:
                print(f"Warning: Filtered training file not found: {vid_path}")
                # Fallback to regular temporal data
        
        # Use standard temporal split files
        vid_path = f'data/FakeTT/vids/vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data
    
    def __len__(self):
        return len(self.data)
    
    def _construct_text_representation(self, row_data):
        """Construct text representation from FakeTT data fields"""
        text_parts = []
        
        # Handle both base data and LLM description data formats
        
        # Always include title if available (from LLM descriptions)
        if 'title' in row_data and pd.notna(row_data['title']) and row_data['title']:
            text_parts.append(str(row_data['title']))
            
        # Include event field (available in both base data and LLM descriptions)
        if 'event' in row_data and pd.notna(row_data['event']) and row_data['event']:
            text_parts.append(str(row_data['event']))
        elif 'keywords' in row_data and pd.notna(row_data['keywords']) and row_data['keywords']:
            text_parts.append(str(row_data['keywords']))
            
        # Include description (available in both formats)
        if self.description and 'description' in row_data and pd.notna(row_data['description']) and row_data['description']:
            text_parts.append(str(row_data['description']))
            
        # Include temporal_evolution if enabled (only in LLM descriptions)
        if self.temp_evolution and 'temporal_evolution' in row_data and pd.notna(row_data['temporal_evolution']) and row_data['temporal_evolution']:
            text_parts.append(str(row_data['temporal_evolution']))
        
        return ' '.join(text_parts)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = torch.tensor(item['label'], dtype=torch.long)
        
        # Initialize features as None
        visual_features = None
        audio_features = None
        concatenated_text = None
        
        # Get visual features if enabled
        if self.use_image:
            if self.fea_frames is None or vid not in self.fea_frames:
                print(f"Warning: Video features not found for {vid}, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
            visual_features = torch.Tensor(self.fea_frames[vid])  # (16, 1024)
            
        # Get audio features if enabled
        if self.use_audio:
            if self.audio_features is None or vid not in self.audio_features:
                # Create dummy audio features if not available
                audio_features = torch.zeros(16, 768)  # Standard audio feature shape
            else:
                audio_features = torch.Tensor(self.audio_features[vid])
        
        # Get text features if enabled (MUST use LLM descriptions)
        if self.use_text:
            entity_row = self.entity_data[self.entity_data['video_id'] == vid]
            if len(entity_row) == 0:
                raise ValueError(f"No LLM description found for video {vid}. LLM descriptions are mandatory.")
            
            entity_row = entity_row.iloc[0]
            concatenated_text = self._construct_text_representation(entity_row)
        
        # Initialize retrieval features
        positive_features = {}
        negative_features = {}
        positive_video_id = None
        negative_video_id = None
        
        if vid in self.retrieval_mapping:
            retrieval_info = self.retrieval_mapping[vid]
            pos_vid = retrieval_info['positive_video_id']
            neg_vid = retrieval_info['negative_video_id']
            positive_video_id = pos_vid
            negative_video_id = neg_vid
            
            # Visual features for retrieval samples
            if self.use_image and self.fea_frames is not None:
                if pos_vid in self.fea_frames:
                    positive_features['visual'] = torch.Tensor(self.fea_frames[pos_vid])  # (16, 1024)
                if neg_vid in self.fea_frames:
                    negative_features['visual'] = torch.Tensor(self.fea_frames[neg_vid])  # (16, 1024)
            
            # Audio features for retrieval samples
            if self.use_audio and self.audio_features is not None:
                if pos_vid in self.audio_features:
                    positive_features['audio'] = torch.Tensor(self.audio_features[pos_vid])  # (16, 768)
                if neg_vid in self.audio_features:
                    negative_features['audio'] = torch.Tensor(self.audio_features[neg_vid])  # (16, 768)
            
            # Text features for retrieval samples - construct from retrieval data
            if self.use_text:
                pos_data = retrieval_info['positive_data']
                neg_data = retrieval_info['negative_data']
                
                positive_features['text'] = self._construct_text_representation(pos_data)
                negative_features['text'] = self._construct_text_representation(neg_data)
        
        return {
            'vid': vid,
            'label': label,
            'concatenated_text': concatenated_text,
            'visual_features': visual_features,
            'audio_features': audio_features,
            'use_text': self.use_text,
            'use_image': self.use_image,
            'use_audio': self.use_audio,
            'positive_features': positive_features,
            'negative_features': negative_features,
            'positive_video_id': positive_video_id,
            'negative_video_id': negative_video_id,
        }


class FakeTTCollator_Retrieval:
    """Collator for FakeTT retrieval dataset that includes positive/negative samples"""
    
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = self._infer_max_len(tokenizer_name)
        logger.info(f"[{self.__class__.__name__}] tokenizer={self.tokenizer_name} max_text_len={self.max_text_len}")

    def _infer_max_len(self, name):
        max_len = None
        try:
            cfg = AutoConfig.from_pretrained(name)
            max_len = getattr(cfg, 'max_position_embeddings', None)
        except Exception:
            pass
        if max_len is None:
            max_len = getattr(self.tokenizer, 'model_max_length', None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 100000:
            nm = str(name).lower()
            max_len = 77 if 'clip' in nm else 512
        return max_len
    
    def __call__(self, batch):
        vids = [item['vid'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        
        # Get modality flags from first item (should be same for all)
        use_text = batch[0]['use_text']
        use_image = batch[0]['use_image'] 
        use_audio = batch[0]['use_audio']
        
        # Initialize return dict
        result = {
            'vids': vids,
            'labels': labels,
            'use_text': use_text,
            'use_image': use_image,
            'use_audio': use_audio,
        }
        
        # Process text features if enabled
        if use_text:
            concatenated_texts = [item['concatenated_text'] for item in batch]
            max_len = self.max_text_len
            entity_text_input = self.tokenizer(
                concatenated_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_len
            )
            result['entity_text_input'] = entity_text_input
            
        # Process visual features if enabled
        if use_image:
            visual_features = torch.stack([item['visual_features'] for item in batch])
            result['visual_features'] = visual_features
            
        # Process audio features if enabled
        if use_audio:
            audio_features = torch.stack([item['audio_features'] for item in batch])
            result['audio_features'] = audio_features
        
        # Process positive and negative text features if enabled
        if use_text:
            positive_texts = []
            negative_texts = []
            
            for item in batch:
                pos_text = item['positive_features'].get('text', '')
                neg_text = item['negative_features'].get('text', '')
                positive_texts.append(pos_text)
                negative_texts.append(neg_text)
            
            # Tokenize positive texts
            positive_text_input = self.tokenizer(
                positive_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=self.max_text_len
            )
            result['positive_text_input'] = positive_text_input
            
            # Tokenize negative texts
            negative_text_input = self.tokenizer(
                negative_texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=self.max_text_len
            )
            result['negative_text_input'] = negative_text_input
        
        # Process positive and negative visual features if enabled
        if use_image:
            positive_visual_features = []
            negative_visual_features = []
            
            for item in batch:
                pos_visual = item['positive_features'].get('visual')
                neg_visual = item['negative_features'].get('visual')
                
                if pos_visual is not None:
                    positive_visual_features.append(pos_visual)
                else:
                    # Create dummy features if missing
                    positive_visual_features.append(torch.zeros(16, 1024))
                    
                if neg_visual is not None:
                    negative_visual_features.append(neg_visual)
                else:
                    # Create dummy features if missing  
                    negative_visual_features.append(torch.zeros(16, 1024))
            
            result['positive_visual_features'] = torch.stack(positive_visual_features)
            result['negative_visual_features'] = torch.stack(negative_visual_features)
        
        # Process positive and negative audio features if enabled
        if use_audio:
            positive_audio_features = []
            negative_audio_features = []
            
            for item in batch:
                pos_audio = item['positive_features'].get('audio')
                neg_audio = item['negative_features'].get('audio')
                
                if pos_audio is not None:
                    positive_audio_features.append(pos_audio)
                else:
                    # Create dummy features if missing
                    positive_audio_features.append(torch.zeros(16, 768))
                    
                if neg_audio is not None:
                    negative_audio_features.append(neg_audio)
                else:
                    # Create dummy features if missing
                    negative_audio_features.append(torch.zeros(16, 768))
            
            result['positive_audio_features'] = torch.stack(positive_audio_features)
            result['negative_audio_features'] = torch.stack(negative_audio_features)
        
        # Add video IDs for debugging
        result['positive_video_ids'] = [item['positive_video_id'] for item in batch]
        result['negative_video_ids'] = [item['negative_video_id'] for item in batch]
        
        return result


class TwitterVideoDataset_Retrieval(FakeTTDataset_Retrieval):
    """TwitterVideo dataset using same format as FakeTT - inherits all functionality"""
    
    def __init__(self, fold: int, split: str, retrieval_path: str = None, 
                 filter_k: int = None, description: bool = True, temp_evolution: bool = True,
                 use_text: bool = True, use_image: bool = True, use_audio: bool = True, **kwargs):
        # Initialize base Dataset class directly (bypass FakeTTDataset_Retrieval.__init__)
        super(FakeTTDataset, self).__init__()
        
        self.filter_k = filter_k
        self.data = self._get_data(fold, split)
        
        self.description = description
        self.temp_evolution = temp_evolution
        self.use_text = use_text
        self.use_image = use_image
        self.use_audio = use_audio
        
        # Load retrieval data - use TwitterVideo specific path if not provided
        if retrieval_path is None:
            retrieval_path = "text_similarity_results/uncertainty_full_dataset_retrieval_LongCLIP-GmP-ViT-L-14_pool.json"
        
        retrieval_file = Path(retrieval_path)
        if not retrieval_file.exists():
            # Try with data/TwitterVideo/ prefix
            retrieval_file = Path(f"data/TwitterVideo/{retrieval_path}")
        
        if not retrieval_file.exists():
            raise FileNotFoundError(f"Retrieval file not found: {retrieval_path}")
        
        import json
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
        
        # Create mapping from video_id to retrieval data
        self.retrieval_mapping = {}
        for item in retrieval_data:
            video_id = item['query_video']['video_id']
            self.retrieval_mapping[video_id] = {
                'positive_video_id': item['similar_true']['video_id'],
                'negative_video_id': item['similar_fake']['video_id'],
                'positive_data': item['similar_true'],
                'negative_data': item['similar_fake']
            }
        
        # Load features based on modality controls - use TwitterVideo paths
        self.fea_frames = None
        self.audio_features = None
        
        if self.use_image:
            self.fea_frames = torch.load('data/TwitterVideo/fea/vit_tensor.pt', weights_only=True)
        if self.use_audio:
            # Check if audio features exist for TwitterVideo (laion-clap-htsat fused)
            audio_path = Path('data/TwitterVideo/fea/audio_features_frames_laion-clap-htsat-fused.pt')
            if audio_path.exists():
                self.audio_features = torch.load(str(audio_path), weights_only=True)
                # Get audio feature dimension dynamically from the first sample
                first_key = next(iter(self.audio_features.keys()))
                first_audio = self.audio_features[first_key]
                self.audio_dim = first_audio.shape[-1] if hasattr(first_audio, 'shape') else 768
                print(f"Detected audio feature dimension: {self.audio_dim}")
            else:
                print("Warning: Audio features not found for TwitterVideo, using dummy features")
                self.audio_features = None
                self.audio_dim = 768  # Default fallback
        
        # Load LLM video descriptions for retrieval text construction (REQUIRED)
        # Use manual JSON loading to avoid pandas precision issues with large integers
        try:
            import json
            llm_data = []
            with open('data/TwitterVideo/llm_video_descriptions.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        # Ensure video_id is treated as string from the start
                        item['video_id'] = str(item['video_id'])
                        llm_data.append(item)
            
            self.entity_data = pd.DataFrame(llm_data)
            print(f"Loaded LLM descriptions: {len(self.entity_data)} samples")
        except Exception as e:
            raise FileNotFoundError(f"REQUIRED: Could not load LLM descriptions: {e}. LLM descriptions are mandatory.")
            
        # Filter dataset to only include samples with retrieval data and required features
        available_retrieval_ids = set(self.retrieval_mapping.keys())
        data_ids = set(self.data['video_id'])
        available_ids = data_ids & available_retrieval_ids
        
        if self.use_image and self.fea_frames is not None:
            visual_ids = set(self.fea_frames.keys())
            available_ids &= visual_ids
            
        if self.use_audio and self.audio_features is not None:
            audio_ids = set(self.audio_features.keys())
            available_ids &= audio_ids
            
        print(f"Using {len(available_ids)} samples with retrieval data and features")
        
        original_length = len(self.data)
        self.data = self.data[self.data['video_id'].isin(available_ids)].reset_index(drop=True)
        
        print(f"TwitterVideo Retrieval dataset: {original_length} -> {len(self.data)} samples "
              f"(removed {original_length - len(self.data)} without complete features)")
        print(f"Loaded retrieval data for {len(self.retrieval_mapping)} videos")
    
    def _get_complete_data(self):
        """Override to use TwitterVideo data file"""
        # Use the same data source as the LLM descriptions to ensure video ID consistency
        data = pd.read_json('data/TwitterVideo/data.json', orient='records', lines=True, dtype={'video_id': 'str'})
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['annotation'].replace(replace_values)
        # TwitterVideo doesn't have event field, use categories or default
        if 'event' in data.columns:
            data['event'], _ = pd.factorize(data['event'])
        else:
            data['event'] = 0  # Default value
        return data
    
    def _get_temporal_data(self, split: str):
        """Override to use TwitterVideo temporal splits or filtered training file if available"""
        # Use filtered training file if filter_k is specified and split is 'train'
        if split == 'train' and self.filter_k is not None:
            vid_path = f'data/TwitterVideo/train_filtered_k{self.filter_k}.txt'
            if Path(vid_path).exists():
                with open(vid_path, "r") as fr:
                    vids = [line.strip() for line in fr.readlines()]
                data = self._get_complete_data()
                data = data[data['video_id'].isin(vids)]
                return data
            else:
                print(f"Warning: Filtered training file not found: {vid_path}")
                # Fallback to regular temporal data
        
        # Use standard temporal split files if they exist
        vid_path = f'data/TwitterVideo/vids/vid_time3_{split}.txt'
        if Path(vid_path).exists():
            with open(vid_path, "r") as fr:
                vids = [line.strip() for line in fr.readlines()]
            data = self._get_complete_data()
            data = data[data['video_id'].isin(vids)]
            return data
        else:
            # If no temporal splits exist, use complete data for now
            print(f"Warning: Temporal split file not found: {vid_path}, using complete dataset")
            return self._get_complete_data()

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']
        label = torch.tensor(item['label'], dtype=torch.long)
        
        # Initialize features as None
        visual_features = None
        audio_features = None
        concatenated_text = None
        
        # Get visual features if enabled
        if self.use_image:
            if self.fea_frames is None or vid not in self.fea_frames:
                print(f"Warning: Video features not found for {vid}, skipping...")
                return self.__getitem__((idx + 1) % len(self.data))
            visual_features = torch.Tensor(self.fea_frames[vid])  # (16, 1024)
            
        # Get audio features if enabled (use dynamic dimensions)
        if self.use_audio:
            if self.audio_features is None or vid not in self.audio_features:
                # Create dummy audio features with detected dimension
                audio_features = torch.zeros(16, getattr(self, 'audio_dim', 768))
                print(f"Warning: Audio features not found for {vid}, using dummy features")
            else:
                audio_features = torch.Tensor(self.audio_features[vid])
        
        # Get text features if enabled (MUST use LLM descriptions)
        if self.use_text:
            entity_row = self.entity_data[self.entity_data['video_id'] == vid]
            if len(entity_row) == 0:
                raise ValueError(f"No LLM description found for video {vid}. LLM descriptions are mandatory.")
            
            entity_row = entity_row.iloc[0]
            concatenated_text = self._construct_text_representation(entity_row)
        
        # Initialize retrieval features
        positive_features = {}
        negative_features = {}
        positive_video_id = None
        negative_video_id = None
        
        if vid in self.retrieval_mapping:
            retrieval_info = self.retrieval_mapping[vid]
            pos_vid = retrieval_info['positive_video_id']
            neg_vid = retrieval_info['negative_video_id']
            positive_video_id = pos_vid
            negative_video_id = neg_vid
            
            # Visual features for retrieval samples
            if self.use_image and self.fea_frames is not None:
                if pos_vid in self.fea_frames:
                    positive_features['visual'] = torch.Tensor(self.fea_frames[pos_vid])  # (16, 1024)
                if neg_vid in self.fea_frames:
                    negative_features['visual'] = torch.Tensor(self.fea_frames[neg_vid])  # (16, 1024)
            
            # Audio features for retrieval samples (use dynamic dimensions)
            if self.use_audio and self.audio_features is not None:
                if pos_vid in self.audio_features:
                    positive_features['audio'] = torch.Tensor(self.audio_features[pos_vid])
                else:
                    positive_features['audio'] = torch.zeros(16, getattr(self, 'audio_dim', 768))
                if neg_vid in self.audio_features:
                    negative_features['audio'] = torch.Tensor(self.audio_features[neg_vid])
                else:
                    negative_features['audio'] = torch.zeros(16, getattr(self, 'audio_dim', 768))
            
            # Text features for retrieval samples - construct from retrieval data
            if self.use_text:
                pos_data = retrieval_info['positive_data']
                neg_data = retrieval_info['negative_data']
                
                positive_features['text'] = self._construct_text_representation(pos_data)
                negative_features['text'] = self._construct_text_representation(neg_data)
        
        return {
            'vid': vid,
            'label': label,
            'concatenated_text': concatenated_text,
            'visual_features': visual_features,
            'audio_features': audio_features,
            'use_text': self.use_text,
            'use_image': self.use_image,
            'use_audio': self.use_audio,
            'positive_features': positive_features,
            'negative_features': negative_features,
            'positive_video_id': positive_video_id,
            'negative_video_id': negative_video_id,
        }


class TwitterVideoCollator_Retrieval(FakeTTCollator_Retrieval):
    """TwitterVideo collator using same format as FakeTT - inherits all functionality"""
    
    def __call__(self, batch):
        # Get base result from parent
        result = super(TwitterVideoCollator_Retrieval, self).__call__(batch)
        
        # Override audio feature processing to handle dynamic dimensions
        use_audio = batch[0]['use_audio']
        if use_audio:
            # Get audio dimension from first sample that has audio
            audio_dim = None
            for item in batch:
                if item['audio_features'] is not None:
                    audio_dim = item['audio_features'].shape[-1]
                    break
            
            if audio_dim is None:
                audio_dim = 768  # Fallback
            
            # Re-process audio features with correct dimensions
            audio_features = []
            for item in batch:
                if item['audio_features'] is not None:
                    audio_features.append(item['audio_features'])
                else:
                    audio_features.append(torch.zeros(16, audio_dim))
            
            result['audio_features'] = torch.stack(audio_features)
            
            # Re-process positive and negative audio features
            positive_audio_features = []
            negative_audio_features = []
            
            for item in batch:
                pos_audio = item['positive_features'].get('audio')
                neg_audio = item['negative_features'].get('audio')
                
                if pos_audio is not None:
                    positive_audio_features.append(pos_audio)
                else:
                    positive_audio_features.append(torch.zeros(16, audio_dim))
                    
                if neg_audio is not None:
                    negative_audio_features.append(neg_audio)
                else:
                    negative_audio_features.append(torch.zeros(16, audio_dim))
            
            result['positive_audio_features'] = torch.stack(positive_audio_features)
            result['negative_audio_features'] = torch.stack(negative_audio_features)
        
        return result
