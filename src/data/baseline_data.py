from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


class FakeSVDataset(Dataset):
    def __init__(self, include_piyao=False, filter_k=None, **kwargs):
        super(FakeSVDataset, self).__init__()
        self.include_piyao = include_piyao
        self.filter_k = filter_k
    
    def _get_complete_data(self):
        # Resolve repository root and prefer data.jsonl; fallback to data_complete.jsonl
        repo_root = Path(__file__).resolve().parent.parent
        candidate_files = [
            repo_root / 'data/FakeSV/data.jsonl',
            repo_root / 'data/FakeSV/data_complete.jsonl',
        ]
        data_file = None
        for f in candidate_files:
            if f.exists():
                data_file = f
                break
        if data_file is None:
            raise FileNotFoundError(f"Neither data.jsonl nor data_complete.jsonl found under {repo_root / 'data/FakeSV'}")

        data = pd.read_json(str(data_file), orient='records', dtype=False, lines=True)

        # Map annotation to numeric labels; drop '辟谣' if present
        if 'annotation' in data.columns:
            replace_values = {'辟谣': 2, '假': 1, '真': 0}
            data['label'] = data['annotation'].replace(replace_values)
            # Remove 辟谣 rows if any
            if (data['label'] == 2).any():
                data = data[data['label'] != 2]
        elif 'label' in data.columns:
            # Already numeric or string labels
            if data['label'].dtype == object:
                replace_values = {'fake': 1, 'real': 0, '假': 1, '真': 0}
                data['label'] = data['label'].replace(replace_values)
        else:
            raise KeyError("FakeSV data file must contain 'annotation' or 'label' column")

        # Factorize event/category; prefer 'keywords'
        if 'keywords' in data.columns:
            data['event'], _ = pd.factorize(data['keywords'])
        elif 'event' in data.columns:
            data['event'], _ = pd.factorize(data['event'])
        else:
            # Fallback: create a single event
            data['event'] = 0

        return data
    
    def _get_data(self, fold, split):
        if fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        elif fold in ['temporal']:
            data = self._get_temporal_data(split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        base = Path(__file__).resolve().parent.parent
        if split == 'train':
            vid_path = base / f'data/FakeSV/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = base / f'data/FakeSV/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        base = Path(__file__).resolve().parent.parent
        # Use filtered training file if filter_k is specified and split is 'train'
        if split == 'train' and self.filter_k is not None:
            vid_path = base / f'data/FakeSV/vids/vid_time3_train_k{self.filter_k}.txt'
        else:
            vid_path = base / f'data/FakeSV/vids/vid_time3_{split}.txt'
            
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data


class FakeTTDataset(Dataset):
    def __init__(self):
        super(FakeTTDataset, self).__init__()
    
    def _get_complete_data(self):
        data = pd.read_json('data/FakeTT/data.jsonl', orient='records', lines=True, dtype={'video_id': 'str'})
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['annotation'].replace(replace_values)
        data['event'], _ = pd.factorize(data['event'])
        # set type of video_id to str
        return data
    
    def _get_data(self, fold, split):
        if fold in ['temporal']:
            data = self._get_temporal_data(split)
        elif fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'data/FakeTT/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'data/FakeTT/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = f'data/FakeTT/vids/vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data
    
class FVCDataset(Dataset):
    def __init__(self):
        super(FVCDataset, self).__init__()
    
    def _get_complete_data(self):
        data = pd.read_json('data/FVC/data.jsonl', orient='records', lines=True, dtype={'vid': 'str'})
        data = data[data['label'].isin(['fake', 'real'])]
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['label'].replace(replace_values)
        data['event'], _ = pd.factorize(data['event_id'])
        data['video_id'] = data['vid']
        return data
    
    def _get_data(self, fold, split):
        if fold in ['temporal']:
            data = self._get_temporal_data(split)
        elif fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'data/FVC/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'data/FVC/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = f'data/FVC/vids/vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data
