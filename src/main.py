import sys
import json
import os
import time
from datetime import datetime
import math
import sys
import hydra
import subprocess
import re

def get_gpu_memory_info():
    """Get GPU memory information using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, check=True)
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                gpu_id, free_memory = map(int, line.split(', '))
                gpu_info.append((gpu_id, free_memory))
        return gpu_info
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: nvidia-smi not available, using GPU 0")
        return [(0, 0)]

def select_best_gpu():
    """Select GPU with most free memory"""
    gpu_info = get_gpu_memory_info()
    best_gpu = max(gpu_info, key=lambda x: x[1])
    print(f"Available GPUs: {gpu_info}")
    print(f"Selected GPU {best_gpu[0]} with {best_gpu[1]} MiB free memory")
    return str(best_gpu[0])

# Set CUDA device - use GPU with most available memory to avoid multi-GPU conflicts
selected_gpu = select_best_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorama
from colorama import Back, Fore, Style
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import wandb
import torch.multiprocessing as mp

from utils.core_utils import (
    get_data_collator,
    get_dataset,
    load_model,
    set_seed,
    set_worker_seed,
    get_optimizer,
    get_scheduler,
    BinaryClassificationMetric,
    EarlyStopping
)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



# Initialize log_path globally (will be updated in main())
log_path = None


class Trainer():
    def __init__(self,
                 cfg: DictConfig):
        self.cfg = cfg
        
        self.device = 'cuda'
        self.evaluator = BinaryClassificationMetric(self.device)
        self.type = cfg.type
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.generator = torch.Generator().manual_seed(cfg.seed)
        # Base save path; per-fold subfolders will be set in run()
        # Root path for this run; do NOT mutate across folds
        self.run_root = log_path
        # Current save path (may point to per-fold dir during training/eval)
        self.save_path = log_path
        self.global_step = 0  # Add global step counter that persists across epochs
        
        if cfg.type == '5-fold':
            self.dataset_range = [1, 2, 3, 4, 5]
        elif cfg.type == 'temporal':
            self.dataset_range = ['temporal']
        else:
            raise ValueError('experiment type not supported')
        
        self.collator = get_data_collator(cfg.model, cfg.dataset, **cfg.data)
    
    def _reset(self, cfg, fold, type):
        drop_last = True if cfg.model == 'CAFE' else False
        train_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='train', **cfg.data)
        test_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='test', **cfg.data)
        test_collator = self.collator
        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=True, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed), drop_last=drop_last)
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=test_collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed), drop_last=drop_last)
        if type == 'temporal':
            valid_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='test', **cfg.data)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed), drop_last=drop_last)

        # Print label distribution right before training starts
        logger.info(f"{Fore.CYAN}=== FINAL DATASET LABEL DISTRIBUTION (before training) ===")
        logger.info(f"{Fore.CYAN}Configuration: include_piyao={cfg.data.get('include_piyao', False)}, ablation_no_cot={cfg.data.get('ablation_no_cot', False)}")
        
        if hasattr(train_dataset, 'data'):
            train_labels = train_dataset.data['label'].value_counts().sort_index()
            logger.info(f"{Fore.CYAN}TRAIN set - Label distribution: {dict(train_labels)} (Total: {len(train_dataset.data)})")
        
        if hasattr(test_dataset, 'data'):
            test_labels = test_dataset.data['label'].value_counts().sort_index()
            logger.info(f"{Fore.CYAN}TEST set - Label distribution: {dict(test_labels)} (Total: {len(test_dataset.data)})")
        
        if type == 'temporal' and hasattr(valid_dataset, 'data'):
            valid_labels = valid_dataset.data['label'].value_counts().sort_index()
            logger.info(f"{Fore.CYAN}VALID set - Label distribution: {dict(valid_labels)} (Total: {len(valid_dataset.data)})")
        
        logger.info(f"{Fore.CYAN}Label mapping: 0=Real/真/辟谣, 1=Fake/假")
        logger.info(f"{Fore.CYAN}==========================================================")

        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        # Sync ablation parameters from data config to model parameters
        model_params = dict(cfg.para)
        if 'ablation_no_cot' in cfg.data:
            model_params['ablation_no_cot'] = cfg.data.ablation_no_cot
        if 'ablation_no_vision' in cfg.data:
            model_params['ablation_no_vision'] = cfg.data.ablation_no_vision
        
        # Handle cross-modal configuration
        if 'cross_modal' in model_params:
            # Always remove individual cross_modal params from model_params
            cross_modal_params = ['cross_modal_layers', 'cross_modal_heads', 'cross_modal_dim_ff', 'cross_modal_dropout']
            
            if model_params['cross_modal']:
                # If cross_modal is True, create config dict
                cross_modal_config = {
                    'n_layers': model_params.get('cross_modal_layers', 4),
                    'n_heads': model_params.get('cross_modal_heads', 8),
                    'd_ff': model_params.get('cross_modal_dim_ff', 1024),
                    'dropout': model_params.get('cross_modal_dropout', 0.1)
                }
                model_params['cross_modal_config'] = cross_modal_config
            
            # Remove individual params regardless of cross_modal value
            for key in cross_modal_params:
                model_params.pop(key, None)
            
        self.model = load_model(cfg.model, **model_params)
        self.model.to(self.device)

        # Warm-up a dummy forward to initialize any Lazy layers (e.g., audio proj)
        try:
            dummy_batch = next(iter(self.train_dataloader))
            
            def to_device(obj):
                import torch
                from collections.abc import Mapping
                if isinstance(obj, torch.Tensor):
                    return obj.to(self.device)
                if isinstance(obj, Mapping):
                    return {k: to_device(v) for k, v in obj.items()}
                return obj
            # Prepare inputs by moving tensors to device; exclude labels and ID lists
            inputs = {k: v for k, v in dummy_batch.items() if k not in ['labels', 'vids', 'positive_video_ids', 'negative_video_ids']}
            inputs = to_device(inputs)
            with torch.no_grad():
                _ = self.model(**inputs)
        except Exception as e:
            # If warm-up fails, continue; parameters will initialize on first real forward
            logger.warning(f"Warm-up forward failed (lazy init may happen later): {e}")

        # Print full model parameters after warm-up
        from model.TrueLens import print_model_params
        print_model_params(self.model)

        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        num_epoch = cfg.num_epoch
        self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, num_epoch=num_epoch, **dict(cfg.sche))
        self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path/'best_model.pth')
        self._first_forward_done = False
        
    def run(self):
        acc_list, f1_list, prec_list, rec_list = [], [], [], []
        f1_real_list, f1_fake_list = [], []
        prec_real_list, prec_fake_list = [], []
        rec_real_list, rec_fake_list = [], []
        
        for fold in self.dataset_range:
            # Always derive from immutable run_root to avoid nested cv/fold_* repetition
            if self.type == '5-fold':
                fold_dir = Path(self.run_root) / 'cv' / f'fold_{fold}'
            else:
                fold_dir = Path(self.run_root) / 'temporal'
            fold_dir.mkdir(parents=True, exist_ok=True)
            self.save_path = fold_dir
            # Reinitialize early stopper with fold-specific path
            self.earlystopping = EarlyStopping(patience=self.cfg.patience, path=self.save_path / 'best_model.pth')

            self._reset(self.cfg, fold, self.type)
            logger.info(f'Current fold: {fold}')
            for epoch in range(self.num_epoch):
                logger.info(f'Current Epoch: {epoch}')
                self._train()
                if self.type == 'temporal':
                    self._valid(split='valid', use_earlystop=True)
                elif self.type == '5-fold':
                    self._valid(split='test', use_earlystop=True)
                if self.earlystopping.early_stop:
                    logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                    break
                if self.type == 'temporal':
                    self._valid(split='test')
            logger.info(f'{Fore.RED}Best of Acc in fold {fold}:')
            self.model.load_state_dict(torch.load(self.save_path/'best_model.pth', weights_only=False))
            best_metrics = self._valid(split='test', final=True)
            
            # Collect macro metrics
            acc_list.append(best_metrics['acc'])
            f1_list.append(best_metrics['f1'])
            prec_list.append(best_metrics['prec'])
            rec_list.append(best_metrics['rec'])
            
            # Collect per-class metrics
            f1_real_list.append(best_metrics['f1_real'])
            f1_fake_list.append(best_metrics['f1_fake'])
            prec_real_list.append(best_metrics['prec_real'])
            prec_fake_list.append(best_metrics['prec_fake'])
            rec_real_list.append(best_metrics['rec_real'])
            rec_fake_list.append(best_metrics['rec_fake'])
            
        logger.info(f"{Fore.MAGENTA}=== FINAL RESULTS SUMMARY ===")
        logger.info(f"{Fore.MAGENTA}Macro Metrics: Acc: {np.mean(acc_list):.4f}, F1: {np.mean(f1_list):.4f}, Precision: {np.mean(prec_list):.4f}, Recall: {np.mean(rec_list):.4f}")
        logger.info(f"{Fore.MAGENTA}Real(0) Metrics: F1: {np.mean(f1_real_list):.4f}, P: {np.mean(prec_real_list):.4f}, R: {np.mean(rec_real_list):.4f}")
        logger.info(f"{Fore.MAGENTA}Fake(1) Metrics: F1: {np.mean(f1_fake_list):.4f}, P: {np.mean(prec_fake_list):.4f}, R: {np.mean(rec_fake_list):.4f}")
        logger.info(f"{Fore.MAGENTA}===============================")
    
    def run_eval_only(self):
        """Run evaluation only mode - load model and evaluate on test set.
        Besides saving human-readable predictions, also save arrays for analysis:
        - prob: (n, K) probability matrix (rows sum to 1)
        - y: (n,) ground-truth labels
        And derived:
        - confidence: top-1 probability per sample
        - y_hat: argmax over classes
        - p_true: probability assigned to the true class
        """
        logger.info(f"{Fore.GREEN}=== EVALUATION ONLY MODE ===")
        # Find best_model.pth according to split type and provided checkpoint path
        eval_targets = []  # list of tuples: (fold_id, best_model_path)
        base = self.save_path
        if self.type == '5-fold':
            # Case 1: checkpoint points directly to a fold directory
            if base.name.startswith('fold_') and (base / 'best_model.pth').exists():
                try:
                    fold_id = int(base.name.split('_')[-1])
                except Exception:
                    fold_id = base.name
                eval_targets.append((fold_id, base / 'best_model.pth'))
            else:
                # Case 2: checkpoint is run root -> scan cv/fold_*/best_model.pth
                cv_dir = base / 'cv'
                if cv_dir.exists():
                    for fd in sorted(cv_dir.glob('fold_*')):
                        bmp = fd / 'best_model.pth'
                        if bmp.exists():
                            try:
                                fid = int(fd.name.split('_')[-1])
                            except Exception:
                                fid = fd.name
                            eval_targets.append((fid, bmp))
                # Recursive fallback for previously nested structure
                if not eval_targets:
                    for bmp in sorted(base.rglob('fold_*/best_model.pth')):
                        fd = bmp.parent
                        try:
                            fid = int(fd.name.split('_')[-1])
                        except Exception:
                            fid = fd.name
                        eval_targets.append((fid, bmp))
        else:
            # temporal split
            if (base / 'temporal' / 'best_model.pth').exists():
                eval_targets.append(('temporal', base / 'temporal' / 'best_model.pth'))
            elif (base / 'best_model.pth').exists():
                eval_targets.append(('temporal', base / 'best_model.pth'))

        if not eval_targets:
            # Last resort: direct best_model under provided path
            direct_bmp = base / 'best_model.pth'
            if direct_bmp.exists():
                eval_targets.append((self.dataset_range[0], direct_bmp))
        if not eval_targets:
            raise ValueError(f"Best model not found under {base}. Expected one of: {base/'cv/fold_*/best_model.pth'} or {base/'temporal/best_model.pth'}")

        # Evaluate one or multiple folds; aggregate outputs into single files (no fold suffix)
        all_metrics = []
        combined_predictions = []
        combined_prob_list = []
        combined_y_list = []
        for fold_id, best_model_path in eval_targets:
            # Prepare loaders for this fold
            fold_to_use = fold_id if fold_id != 'temporal' else 'temporal'
            self._reset(self.cfg, fold_to_use, self.type)
            logger.info(f"Loading model from: {best_model_path}")
            self.model.load_state_dict(torch.load(best_model_path, weights_only=False))
            # Run evaluation on test set with detailed predictions and arrays
            predictions, prob, y = self._eval_with_predictions()
            combined_predictions.extend(predictions)
            combined_prob_list.append(np.asarray(prob))
            combined_y_list.append(np.asarray(y))
            # Also compute and store metrics via _valid for consistency
            m = self._valid(split='test', final=True)
            all_metrics.append((fold_id, m))
        # Save aggregated outputs once (no fold suffix)
        result_dir = Path(f"result/{self.dataset_name}")
        result_dir.mkdir(parents=True, exist_ok=True)
        model_mark = f"_{getattr(self, 'model_name', 'ExMRD')}"
        if hasattr(self.model, 'name') and self.model.name == 'ExMRD_Defer':
            predictions_file = result_dir / f"defer_predictions{model_mark}.json"
        else:
            predictions_file = result_dir / f"slm_prediction{model_mark}.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(combined_predictions, f, ensure_ascii=False, indent=2)
        logger.info(f"{Fore.GREEN}Aggregated predictions saved to: {predictions_file}")

        try:
            project_root = Path(__file__).resolve().parent.parent
            draw_dir = project_root / 'draw' / self.dataset_name
            draw_dir.mkdir(parents=True, exist_ok=True)
            if combined_prob_list:
                prob_all = np.concatenate(combined_prob_list, axis=0)
                y_all = np.concatenate(combined_y_list, axis=0)
            else:
                prob_all = np.empty((0, 2))
                y_all = np.empty((0,))
            np.save(draw_dir / 'prob.npy', prob_all)
            np.save(draw_dir / 'y.npy', y_all)
            if prob_all.size:
                conf = prob_all.max(axis=1)
                y_hat = prob_all.argmax(axis=1)
                p_true = prob_all[np.arange(len(y_all)), y_all]
            else:
                conf = np.array([])
                y_hat = np.array([])
                p_true = np.array([])
            np.save(draw_dir / 'confidence.npy', conf)
            np.save(draw_dir / 'y_hat.npy', y_hat)
            np.save(draw_dir / 'p_true.npy', p_true)
            logger.info(f"{Fore.GREEN}Aggregated eval arrays saved to: {draw_dir}")
        except Exception as e:
            logger.warning(f"{Fore.YELLOW}Failed to save aggregated eval arrays: {e}")
        
        # Save predictions to result folder (include model mark for clarity)
        result_dir = Path(f"result/{self.dataset_name}")
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename based on model type
        model_mark = f"_{getattr(self, 'model_name', 'ExMRD')}"
        if hasattr(self.model, 'name') and self.model.name == 'ExMRD_Defer':
            predictions_file = result_dir / f"defer_predictions{model_mark}.json"
        else:
            predictions_file = result_dir / f"slm_prediction{model_mark}.json"
            
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        logger.info(f"{Fore.GREEN}Predictions saved to: {predictions_file}")
        logger.info(f"{Fore.GREEN}Total predictions: {len(predictions)}")

        # Also save prob/y and derived arrays to draw/<dataset>/ for downstream analysis
        try:
            project_root = Path(__file__).resolve().parent.parent
            draw_dir = project_root / 'draw' / self.dataset_name
            draw_dir.mkdir(parents=True, exist_ok=True)

            # Ensure numpy arrays
            prob_np = np.asarray(prob)
            y_np = np.asarray(y)
            confidence_np = prob_np.max(axis=1)
            y_hat_np = prob_np.argmax(axis=1)
            p_true_np = prob_np[np.arange(len(y_np)), y_np]

            # Save as .npy files for compactness and fidelity
            np.save(draw_dir / 'prob.npy', prob_np)
            np.save(draw_dir / 'y.npy', y_np)
            np.save(draw_dir / 'confidence.npy', confidence_np)
            np.save(draw_dir / 'y_hat.npy', y_hat_np)
            np.save(draw_dir / 'p_true.npy', p_true_np)

            logger.info(f"{Fore.GREEN}Saved eval arrays to: {draw_dir}")
        except Exception as e:
            logger.warning(f"{Fore.YELLOW}Failed to save eval arrays to draw directory: {e}")
        
        # Run defer analysis if this is a defer model
        defer_analysis = {}
        if hasattr(self.model, 'name') and self.model.name == 'ExMRD_Defer':
            defer_analysis = self._eval_defer_analysis(predictions)
            
            # Save defer analysis results
            analysis_file = result_dir / "defer_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(defer_analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"{Fore.GREEN}Defer analysis saved to: {analysis_file}")
        
        # Generate confidence distribution plot if confidence scores are available
        confidence_available = any('confidence' in p for p in predictions)
        if confidence_available:
            logger.info(f"{Fore.CYAN}Generating confidence distribution plot...")
            try:
                import subprocess
                import sys
                
                # Run the plotting script
                plot_script = Path(__file__).parent.parent / "plot_confidence_distribution.py"
                confidence_output_dir = result_dir / "confidence_analysis"
                
                cmd = [
                    sys.executable, str(plot_script),
                    "--prediction_file", str(predictions_file),
                    "--output_dir", str(confidence_output_dir),
                    "--dataset_name", self.dataset_name,
                    "--plot_type", "both"
                ]
                
                subprocess.run(cmd, check=True)
                logger.info(f"{Fore.CYAN}Confidence analysis plots saved to: {confidence_output_dir}")
                
            except Exception as e:
                logger.warning(f"{Fore.YELLOW}Could not generate confidence plots: {e}")
        else:
            logger.info(f"{Fore.YELLOW}No confidence values found in predictions - skipping confidence analysis")
        
    def _eval_with_predictions(self):
        """Evaluate model on test set and return:
        - predictions: list of dicts (for human-readable inspection)
        - prob: numpy array of shape (n, K)
        - y: numpy array of shape (n,)
        """
        self.model.eval()
        predictions = []
        prob_list = []
        y_list = []
        
        logger.info(f"{Fore.YELLOW}Running detailed evaluation on test set...")
        
        for batch in tqdm(self.test_dataloader, bar_format=f"{Fore.YELLOW}{{l_bar}}{{bar}}{{r_bar}}"):
            vids = batch.pop('vids')
            
            # Separate tensor and non-tensor inputs
            inputs = {}
            for key, value in batch.items():
                if hasattr(value, 'to'):  # Check if it's a tensor
                    inputs[key] = value.to(self.device)
                else:  # Keep non-tensor values as is (like boolean flags)
                    inputs[key] = value
                    
            labels = inputs.pop('labels')
        
            with torch.no_grad():
                output = self.model(**inputs)
                
                # Handle model-specific outputs
                if hasattr(self.model, 'name') and self.model.name == 'ExMRD_Defer':
                    # Defer model: get detailed predictions for analysis
                    predict_output = self.model.predict(inputs['p_dm'], **{k:v for k,v in inputs.items() if k != 'p_dm'})
                    
                    pred = predict_output['final_probs']  # Final predictions after deferral
                    defer_info = {
                        'p_slm': predict_output['p_slm'],         # SLM predictions
                        'p_dm': predict_output['p_dm'],           # LLM predictions  
                        'pi': predict_output['pi'],               # Defer probabilities
                        'defer_decisions': predict_output['defer_decisions'],  # Boolean defer decisions
                        'defer_threshold': self.model.defer_threshold
                    }
                    
                elif hasattr(self.model, 'name') and self.model.name == 'ExMRD_Evidential':
                    # Check if using evidential or traditional classifier
                    if hasattr(self.model, 'use_evidential') and self.model.use_evidential:
                        # Evidential classifier: use fused predictions
                        pred = output['p_fused']  # Predictive probabilities from fused evidence
                    else:
                        # Traditional classifier: output is logits
                        pred = F.softmax(output, dim=1)
                    defer_info = None
                    
                elif hasattr(self.model, 'name') and self.model.name == 'TrueLens':
                    # Retrieval-augmented evidential classifier
                    pred = output['p_fused']  # Predictive probabilities from fused evidence
                    defer_info = None
                else:
                    # Regular model handling
                    if type(output) is dict:
                        pred = output['pred']
                    else:
                        pred = output
                    defer_info = None
            
            # Accumulate batch probabilities and labels
            prob_list.append(pred.detach().cpu().float().numpy())
            y_list.append(labels.detach().cpu().long().numpy())

            _, preds = torch.max(pred, 1)
            
            # Store predictions for each sample in the batch
            for i in range(len(vids)):
                video_id = vids[i]
                ground_truth = labels[i].item()
                prediction = preds[i].item()
                
                # Convert labels to meaningful strings
                gt_label = "真" if ground_truth == 0 else "假"
                pred_label = "真" if prediction == 0 else "假"
                
                # Extract confidence (max prediction probability)
                confidence = float(torch.max(pred[i]).item())
                
                # Add model type information for analysis
                if hasattr(self.model, 'name') and self.model.name == 'ExMRD_Defer':
                    model_type = "defer"
                elif hasattr(self.model, 'use_evidential') and self.model.use_evidential:
                    model_type = "evidential"
                else:
                    model_type = "traditional"
                
                pred_entry = {
                    "video_id": video_id,
                    "annotation": gt_label,
                    "prediction": pred_label,
                    "ground_truth": gt_label,  # For compatibility with plotting script
                    "ground_truth_numeric": ground_truth,
                    "prediction_numeric": prediction,
                    "confidence": confidence,
                    "model_type": model_type
                }
                
                # Add defer-specific information if available
                if defer_info is not None:
                    # SLM predictions
                    slm_pred_probs = defer_info['p_slm'][i]
                    _, slm_pred_idx = torch.max(slm_pred_probs, 0)
                    slm_pred_label = "真" if slm_pred_idx.item() == 0 else "假"
                    
                    # LLM predictions  
                    llm_pred_probs = defer_info['p_dm'][i]
                    _, llm_pred_idx = torch.max(llm_pred_probs, 0)
                    llm_pred_label = "真" if llm_pred_idx.item() == 0 else "假"
                    
                    # Defer information
                    pred_entry.update({
                        "slm_prediction": slm_pred_label,
                        "slm_prediction_numeric": slm_pred_idx.item(),
                        "slm_confidence": float(torch.max(slm_pred_probs).item()),
                        "llm_prediction": llm_pred_label,
                        "llm_prediction_numeric": llm_pred_idx.item(), 
                        "llm_confidence": float(torch.max(llm_pred_probs).item()),
                        "defer_probability": float(defer_info['pi'][i].item()),
                        "is_deferred": bool(defer_info['defer_decisions'][i].item()),
                        "defer_threshold": float(defer_info['defer_threshold'])
                    })
                    
                predictions.append(pred_entry)
        
        # Calculate and log metrics
        ground_truth = [p["ground_truth_numeric"] for p in predictions]
        pred_values = [p["prediction_numeric"] for p in predictions]
        
        self.evaluator._reset()
        self.evaluator.update(torch.tensor(pred_values).to(self.device), torch.tensor(ground_truth).to(self.device))
        metrics = self.evaluator.compute()
        
        logger.info(f"{Fore.GREEN}=== EVALUATION RESULTS ===")
        logger.info(f"{Fore.GREEN}Total samples: {len(predictions)}")
        logger.info(f"{Fore.GREEN}Accuracy: {metrics['acc']:.4f}")
        logger.info(f"{Fore.GREEN}Macro F1: {metrics['f1']:.4f}")
        logger.info(f"{Fore.GREEN}Macro Precision: {metrics['prec']:.4f}")
        logger.info(f"{Fore.GREEN}Macro Recall: {metrics['rec']:.4f}")
        logger.info(f"{Fore.GREEN}Real(0) F1: {metrics['f1_real']:.4f}, P: {metrics['prec_real']:.4f}, R: {metrics['rec_real']:.4f}")
        logger.info(f"{Fore.GREEN}Fake(1) F1: {metrics['f1_fake']:.4f}, P: {metrics['prec_fake']:.4f}, R: {metrics['rec_fake']:.4f}")
        logger.info(f"{Fore.GREEN}===========================")
        
        # Stack arrays
        try:
            prob_all = np.concatenate(prob_list, axis=0)
            y_all = np.concatenate(y_list, axis=0)
        except Exception as e:
            logger.warning(f"{Fore.YELLOW}Failed to concatenate prob/y arrays: {e}")
            prob_all = np.vstack(prob_list) if prob_list else np.empty((0, 0))
            y_all = np.hstack(y_list) if y_list else np.empty((0,), dtype=int)

        return predictions, prob_all, y_all
        
    def _eval_defer_analysis(self, predictions):
        """Analyze defer model predictions with detailed statistics"""
        if not any(p.get('model_type') == 'defer' for p in predictions):
            logger.info(f"{Fore.YELLOW}No defer model predictions found - skipping defer analysis")
            return {}
        
        logger.info(f"{Fore.CYAN}=== DEFER MODEL DETAILED ANALYSIS ===")
        
        # Separate deferred and non-deferred samples
        deferred_samples = [p for p in predictions if p.get('is_deferred', False)]
        non_deferred_samples = [p for p in predictions if not p.get('is_deferred', False)]
        
        total_samples = len(predictions)
        defer_count = len(deferred_samples)
        non_defer_count = len(non_deferred_samples)
        defer_rate = defer_count / total_samples if total_samples > 0 else 0
        
        # Calculate defer probability statistics
        defer_probs = [p.get('defer_probability', 0) for p in predictions]
        avg_defer_prob = np.mean(defer_probs) if defer_probs else 0
        std_defer_prob = np.std(defer_probs) if defer_probs else 0
        threshold = predictions[0].get('defer_threshold', 0.5) if predictions else 0.5
        
        # Overall metrics (using final predictions after deferral)
        overall_gt = [p["ground_truth_numeric"] for p in predictions]
        overall_pred = [p["prediction_numeric"] for p in predictions]
        overall_acc = accuracy_score(overall_gt, overall_pred)
        overall_f1_macro = f1_score(overall_gt, overall_pred, average='macro')
        overall_prec_macro = precision_score(overall_gt, overall_pred, average='macro')
        overall_rec_macro = recall_score(overall_gt, overall_pred, average='macro')
        
        # Per-class metrics
        overall_f1_per_class = f1_score(overall_gt, overall_pred, average=None)
        overall_prec_per_class = precision_score(overall_gt, overall_pred, average=None)
        overall_rec_per_class = recall_score(overall_gt, overall_pred, average=None)
        
        logger.info(f"{Fore.CYAN}Overall Performance (with deferral):")
        logger.info(f"{Fore.CYAN}  Accuracy: {overall_acc:.4f} ({len(overall_gt)} samples)")
        logger.info(f"{Fore.CYAN}  Macro F1: {overall_f1_macro:.4f}, Precision: {overall_prec_macro:.4f}, Recall: {overall_rec_macro:.4f}")
        logger.info(f"{Fore.CYAN}  Real(0) F1: {overall_f1_per_class[0]:.4f}, P: {overall_prec_per_class[0]:.4f}, R: {overall_rec_per_class[0]:.4f}")
        logger.info(f"{Fore.CYAN}  Fake(1) F1: {overall_f1_per_class[1]:.4f}, P: {overall_prec_per_class[1]:.4f}, R: {overall_rec_per_class[1]:.4f}")
        
        # Pure SLM baseline (no deferral)
        slm_only_pred = [p["slm_prediction_numeric"] for p in predictions]
        slm_acc = accuracy_score(overall_gt, slm_only_pred)
        slm_f1_macro = f1_score(overall_gt, slm_only_pred, average='macro')
        slm_prec_macro = precision_score(overall_gt, slm_only_pred, average='macro')
        slm_rec_macro = recall_score(overall_gt, slm_only_pred, average='macro')
        
        # Per-class metrics for SLM
        slm_f1_per_class = f1_score(overall_gt, slm_only_pred, average=None)
        slm_prec_per_class = precision_score(overall_gt, slm_only_pred, average=None)
        slm_rec_per_class = recall_score(overall_gt, slm_only_pred, average=None)
        
        logger.info(f"{Fore.CYAN}Pure SLM Baseline (no deferral):")
        logger.info(f"{Fore.CYAN}  Accuracy: {slm_acc:.4f} ({len(overall_gt)} samples)")
        logger.info(f"{Fore.CYAN}  Macro F1: {slm_f1_macro:.4f}, Precision: {slm_prec_macro:.4f}, Recall: {slm_rec_macro:.4f}")
        logger.info(f"{Fore.CYAN}  Real(0) F1: {slm_f1_per_class[0]:.4f}, P: {slm_prec_per_class[0]:.4f}, R: {slm_rec_per_class[0]:.4f}")
        logger.info(f"{Fore.CYAN}  Fake(1) F1: {slm_f1_per_class[1]:.4f}, P: {slm_prec_per_class[1]:.4f}, R: {slm_rec_per_class[1]:.4f}")
        
        # Defer vs Pure SLM comparison
        acc_improvement = overall_acc - slm_acc
        f1_improvement = overall_f1_macro - slm_f1_macro
        logger.info(f"{Fore.CYAN}Deferral vs Pure SLM:")
        logger.info(f"{Fore.CYAN}  Accuracy improvement: {acc_improvement:+.4f}")
        logger.info(f"{Fore.CYAN}  Macro F1 improvement: {f1_improvement:+.4f}")
        
        logger.info(f"{Fore.CYAN}Defer Statistics:")
        logger.info(f"{Fore.CYAN}  Defer Rate: {defer_rate:.3f} ({defer_count}/{total_samples} samples)")
        logger.info(f"{Fore.CYAN}  Average π: {avg_defer_prob:.3f} ± {std_defer_prob:.3f}")
        logger.info(f"{Fore.CYAN}  Threshold: τ = {threshold}")
        
        analysis_results = {
            'total_samples': total_samples,
            'defer_rate': defer_rate,
            'defer_count': defer_count,
            'non_defer_count': non_defer_count,
            'avg_defer_probability': avg_defer_prob,
            'std_defer_probability': std_defer_prob,
            'defer_threshold': threshold,
            'overall_metrics': {
                'accuracy': overall_acc,
                'f1_macro': overall_f1_macro,
                'precision_macro': overall_prec_macro,
                'recall_macro': overall_rec_macro,
                'f1_per_class': overall_f1_per_class.tolist(),
                'precision_per_class': overall_prec_per_class.tolist(),
                'recall_per_class': overall_rec_per_class.tolist()
            },
            'slm_baseline': {
                'accuracy': slm_acc,
                'f1_macro': slm_f1_macro,
                'precision_macro': slm_prec_macro,
                'recall_macro': slm_rec_macro,
                'f1_per_class': slm_f1_per_class.tolist(),
                'precision_per_class': slm_prec_per_class.tolist(),
                'recall_per_class': slm_rec_per_class.tolist()
            },
            'improvement_over_slm': {
                'accuracy': acc_improvement,
                'f1_macro': f1_improvement
            }
        }
        
        # Analyze deferred samples
        if deferred_samples:
            # SLM-only performance on deferred samples
            deferred_gt = [p["ground_truth_numeric"] for p in deferred_samples]
            deferred_slm_pred = [p["slm_prediction_numeric"] for p in deferred_samples]
            deferred_llm_pred = [p["llm_prediction_numeric"] for p in deferred_samples]
            
            # SLM metrics on deferred
            slm_def_acc = accuracy_score(deferred_gt, deferred_slm_pred)
            slm_def_f1_macro = f1_score(deferred_gt, deferred_slm_pred, average='macro')
            slm_def_prec_macro = precision_score(deferred_gt, deferred_slm_pred, average='macro')
            slm_def_rec_macro = recall_score(deferred_gt, deferred_slm_pred, average='macro')
            
            slm_def_f1_per_class = f1_score(deferred_gt, deferred_slm_pred, average=None)
            slm_def_prec_per_class = precision_score(deferred_gt, deferred_slm_pred, average=None)
            slm_def_rec_per_class = recall_score(deferred_gt, deferred_slm_pred, average=None)
            
            # LLM metrics on deferred
            llm_def_acc = accuracy_score(deferred_gt, deferred_llm_pred)
            llm_def_f1_macro = f1_score(deferred_gt, deferred_llm_pred, average='macro')
            llm_def_prec_macro = precision_score(deferred_gt, deferred_llm_pred, average='macro')
            llm_def_rec_macro = recall_score(deferred_gt, deferred_llm_pred, average='macro')
            
            llm_def_f1_per_class = f1_score(deferred_gt, deferred_llm_pred, average=None)
            llm_def_prec_per_class = precision_score(deferred_gt, deferred_llm_pred, average=None)
            llm_def_rec_per_class = recall_score(deferred_gt, deferred_llm_pred, average=None)
            
            improvement_acc = llm_def_acc - slm_def_acc
            improvement_f1 = llm_def_f1_macro - slm_def_f1_macro
            
            logger.info(f"{Fore.CYAN}Deferred Samples Analysis ({defer_count} samples):")
            logger.info(f"{Fore.CYAN}  SLM-only:")
            logger.info(f"{Fore.CYAN}    Accuracy: {slm_def_acc:.4f}")
            logger.info(f"{Fore.CYAN}    Macro F1: {slm_def_f1_macro:.4f}, Precision: {slm_def_prec_macro:.4f}, Recall: {slm_def_rec_macro:.4f}")
            logger.info(f"{Fore.CYAN}    Real(0) F1: {slm_def_f1_per_class[0]:.4f}, P: {slm_def_prec_per_class[0]:.4f}, R: {slm_def_rec_per_class[0]:.4f}")
            logger.info(f"{Fore.CYAN}    Fake(1) F1: {slm_def_f1_per_class[1]:.4f}, P: {slm_def_prec_per_class[1]:.4f}, R: {slm_def_rec_per_class[1]:.4f}")
            
            logger.info(f"{Fore.CYAN}  LLM:")
            logger.info(f"{Fore.CYAN}    Accuracy: {llm_def_acc:.4f}")
            logger.info(f"{Fore.CYAN}    Macro F1: {llm_def_f1_macro:.4f}, Precision: {llm_def_prec_macro:.4f}, Recall: {llm_def_rec_macro:.4f}")
            logger.info(f"{Fore.CYAN}    Real(0) F1: {llm_def_f1_per_class[0]:.4f}, P: {llm_def_prec_per_class[0]:.4f}, R: {llm_def_rec_per_class[0]:.4f}")
            logger.info(f"{Fore.CYAN}    Fake(1) F1: {llm_def_f1_per_class[1]:.4f}, P: {llm_def_prec_per_class[1]:.4f}, R: {llm_def_rec_per_class[1]:.4f}")
            
            logger.info(f"{Fore.CYAN}  LLM vs SLM Improvement:")
            logger.info(f"{Fore.CYAN}    Accuracy: {improvement_acc:+.4f}, Macro F1: {improvement_f1:+.4f}")
            
            analysis_results['deferred_analysis'] = {
                'count': defer_count,
                'slm_metrics': {
                    'accuracy': slm_def_acc,
                    'f1_macro': slm_def_f1_macro,
                    'precision_macro': slm_def_prec_macro,
                    'recall_macro': slm_def_rec_macro,
                    'f1_per_class': slm_def_f1_per_class.tolist(),
                    'precision_per_class': slm_def_prec_per_class.tolist(),
                    'recall_per_class': slm_def_rec_per_class.tolist()
                },
                'llm_metrics': {
                    'accuracy': llm_def_acc,
                    'f1_macro': llm_def_f1_macro,
                    'precision_macro': llm_def_prec_macro,
                    'recall_macro': llm_def_rec_macro,
                    'f1_per_class': llm_def_f1_per_class.tolist(),
                    'precision_per_class': llm_def_prec_per_class.tolist(),
                    'recall_per_class': llm_def_rec_per_class.tolist()
                },
                'improvement': {
                    'accuracy': improvement_acc,
                    'f1_macro': improvement_f1
                }
            }
        
        # Analyze non-deferred samples
        if non_deferred_samples:
            non_deferred_gt = [p["ground_truth_numeric"] for p in non_deferred_samples]
            non_deferred_slm_pred = [p["slm_prediction_numeric"] for p in non_deferred_samples]
            
            slm_nondef_acc = accuracy_score(non_deferred_gt, non_deferred_slm_pred)
            slm_nondef_f1_macro = f1_score(non_deferred_gt, non_deferred_slm_pred, average='macro')
            slm_nondef_prec_macro = precision_score(non_deferred_gt, non_deferred_slm_pred, average='macro')
            slm_nondef_rec_macro = recall_score(non_deferred_gt, non_deferred_slm_pred, average='macro')
            
            slm_nondef_f1_per_class = f1_score(non_deferred_gt, non_deferred_slm_pred, average=None)
            slm_nondef_prec_per_class = precision_score(non_deferred_gt, non_deferred_slm_pred, average=None)
            slm_nondef_rec_per_class = recall_score(non_deferred_gt, non_deferred_slm_pred, average=None)
            
            logger.info(f"{Fore.CYAN}Non-deferred Samples ({non_defer_count} samples):")
            logger.info(f"{Fore.CYAN}  SLM-only:")
            logger.info(f"{Fore.CYAN}    Accuracy: {slm_nondef_acc:.4f}")
            logger.info(f"{Fore.CYAN}    Macro F1: {slm_nondef_f1_macro:.4f}, Precision: {slm_nondef_prec_macro:.4f}, Recall: {slm_nondef_rec_macro:.4f}")
            logger.info(f"{Fore.CYAN}    Real(0) F1: {slm_nondef_f1_per_class[0]:.4f}, P: {slm_nondef_prec_per_class[0]:.4f}, R: {slm_nondef_rec_per_class[0]:.4f}")
            logger.info(f"{Fore.CYAN}    Fake(1) F1: {slm_nondef_f1_per_class[1]:.4f}, P: {slm_nondef_prec_per_class[1]:.4f}, R: {slm_nondef_rec_per_class[1]:.4f}")
            
            analysis_results['non_deferred_analysis'] = {
                'count': non_defer_count,
                'slm_metrics': {
                    'accuracy': slm_nondef_acc,
                    'f1_macro': slm_nondef_f1_macro,
                    'precision_macro': slm_nondef_prec_macro,
                    'recall_macro': slm_nondef_rec_macro,
                    'f1_per_class': slm_nondef_f1_per_class.tolist(),
                    'precision_per_class': slm_nondef_prec_per_class.tolist(),
                    'recall_per_class': slm_nondef_rec_per_class.tolist()
                }
            }
        
        # Breakdown by ground truth labels
        real_samples = [p for p in predictions if p["ground_truth_numeric"] == 0]
        fake_samples = [p for p in predictions if p["ground_truth_numeric"] == 1]
        
        real_deferred = len([p for p in real_samples if p.get('is_deferred', False)])
        fake_deferred = len([p for p in fake_samples if p.get('is_deferred', False)])
        
        real_defer_rate = real_deferred / len(real_samples) if real_samples else 0
        fake_defer_rate = fake_deferred / len(fake_samples) if fake_samples else 0
        
        logger.info(f"{Fore.CYAN}Breakdown by Ground Truth:")
        logger.info(f"{Fore.CYAN}  Real news: {real_defer_rate:.3f} deferred ({real_deferred}/{len(real_samples)})")
        logger.info(f"{Fore.CYAN}  Fake news: {fake_defer_rate:.3f} deferred ({fake_deferred}/{len(fake_samples)})")
        
        analysis_results['ground_truth_breakdown'] = {
            'real_defer_rate': real_defer_rate,
            'fake_defer_rate': fake_defer_rate,
            'real_total': len(real_samples),
            'fake_total': len(fake_samples),
            'real_deferred': real_deferred,
            'fake_deferred': fake_deferred
        }
        
        logger.info(f"{Fore.CYAN}======================================")
        
        return analysis_results
        
    def _train(self):
        loss_list =  []
        self.model.train()
        pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
        for batch in pbar:
            _ = batch.pop('vids')
            
            # Separate tensor and non-tensor inputs
            inputs = {}
            for key, value in batch.items():
                if hasattr(value, 'to'):  # Check if it's a tensor
                    inputs[key] = value.to(self.device)
                else:  # Keep non-tensor values as is (like boolean flags)
                    inputs[key] = value
                    
            labels = inputs.pop('labels')
            
            if hasattr(self.model, 'name') and self.model.name == 'CAFE':
                output = self.model(**inputs, task='similarity')
                loss = self.model.calculate_loss(*output)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            output = self.model(**inputs)
            
            # Print full model parameters after first forward pass initializes lazy layers
            if not self._first_forward_done:
                try:
                    from model.TrueLens import print_model_params
                    print_model_params(self.model)
                except ImportError:
                    logger.info("TrueLens print_model_params not available, skipping parameter count")
                self._first_forward_done = True
            
            # Handle model-specific outputs
            if hasattr(self.model, 'name') and self.model.name in ['ExMRD_Evidential', 'ExMRD_Defer', 'TrueLens']:
                if self.model.name == 'ExMRD_Defer':
                    # Defer model: uses combined defer + evidential loss
                    losses_dict = self.model.compute_losses(output, labels, self.global_step)
                    loss = losses_dict['loss_total']
                    
                    # Use SLM predictions for evaluation during training
                    pred = output['p_slm']  # SLM predictions for training metrics
                    
                    # Log defer losses
                    if self.global_step % 100 == 0:
                        logger.debug(f"Defer losses - Total: {loss.item():.4f}, Defer: {losses_dict['loss_defer'].item():.4f}, "
                                   f"Evid: {losses_dict['loss_evid'].item():.4f}, π_mean: {losses_dict['pi_mean'].item():.3f}, "
                                   f"π_std: {losses_dict['pi_std'].item():.3f}, CE_SLM: {losses_dict['ce_slm'].item():.4f}, "
                                   f"CE_DM: {losses_dict['ce_dm'].item():.4f}")
                        
                elif self.model.name == 'ExMRD_Evidential':
                    # Check if using evidential or traditional classifier
                    if hasattr(self.model, 'use_evidential') and self.model.use_evidential:
                        # Evidential classifier: output is a dict with alpha values
                        losses_dict = self.model.compute_losses(output, labels, self.global_step)
                        loss = losses_dict['loss_total']
                        
                        # Use fused predictions for evaluation
                        pred = output['p_fused']  # Predictive probabilities from fused evidence
                        
                        # Log evidential losses
                        if self.global_step % 100 == 0:
                            logger.debug(f"Evidential losses - Total: {loss.item():.4f}, Fused: {losses_dict['loss_fused'].item():.4f}, "
                                       f"Text: {losses_dict['loss_text'].item():.4f}, Audio: {losses_dict['loss_audio'].item():.4f}, "
                                       f"Image: {losses_dict['loss_image'].item():.4f}, KL_weight: {losses_dict['kl_weight'].item():.3f}")
                    else:
                        # Traditional classifier: output is logits
                        losses_dict = self.model.compute_losses(output, labels, self.global_step)
                        loss = losses_dict['loss_total']
                        
                        # Convert logits to probabilities for evaluation
                        pred = F.softmax(output, dim=1)
                        
                        # Log traditional losses
                        if self.global_step % 100 == 0:
                            logger.debug(f"Traditional loss - Total: {loss.item():.4f}")
                            
                elif self.model.name == 'TrueLens':
                    # Retrieval-augmented evidential classifier
                    losses_dict = self.model.compute_losses(output, labels, self.global_step)
                    loss = losses_dict['loss_total']
                    
                    # Use fused predictions for evaluation
                    pred = output['p_fused']  # Predictive probabilities from fused evidence
                    
                    # Log retrieval losses
                    if self.global_step % 100 == 0:
                        logger.debug(f"Retrieval losses - Total: {loss.item():.4f}, Fused: {losses_dict['loss_fused'].item():.4f}, "
                                   f"Text: {losses_dict['loss_text'].item():.4f}, Audio: {losses_dict['loss_audio'].item():.4f}, "
                                   f"Image: {losses_dict['loss_image'].item():.4f}, KL_weight: {losses_dict['kl_weight'].item():.3f}")
            else:
                # Regular model handling
                if type(output) is dict:
                    pred = output['pred']
                else:
                    pred = output
                
                loss = None
                if hasattr(self.model, 'name'):
                    match self.model.name:
                        case _:
                            loss = F.cross_entropy(pred, labels)
                else:
                    loss = F.cross_entropy(pred, labels)

            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())

            loss.backward()
            
            # Optional gradient clipping
            if hasattr(self.model, 'gradient_clip_norm') and self.model.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.gradient_clip_norm)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1
            
        metrics = self.evaluator.compute()
        # print
        logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}")
        logger.info(f'{Fore.BLUE}Train: Acc: {metrics["acc"]:.4f}, Macro F1: {metrics["f1"]:.4f}, Macro Precision: {metrics["prec"]:.4f}, Macro Recall: {metrics["rec"]:.4f}')
        logger.info(f'{Fore.BLUE}Train Per-class: Real(0) F1: {metrics["f1_real"]:.4f}, P: {metrics["prec_real"]:.4f}, R: {metrics["rec_real"]:.4f} | Fake(1) F1: {metrics["f1_fake"]:.4f}, P: {metrics["prec_fake"]:.4f}, R: {metrics["rec_fake"]:.4f}')
    
    def _valid(self, split, use_earlystop=False, final=False):
        loss_list = []
        self.model.eval()
        if split == 'valid' and final:
            raise ValueError('print_wrong only support test split')
        if split == 'valid':
            dataloader = self.valid_dataloader
            split_name = 'Valid'
            fcolor = Fore.YELLOW
        elif split == 'test':
            dataloader = self.test_dataloader
            split_name = 'Test'
            fcolor = Fore.RED
        else:
            raise ValueError('split not supported')
        for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
            vids = batch.pop('vids')
            
            # Separate tensor and non-tensor inputs
            inputs = {}
            for key, value in batch.items():
                if hasattr(value, 'to'):  # Check if it's a tensor
                    inputs[key] = value.to(self.device)
                else:  # Keep non-tensor values as is (like boolean flags)
                    inputs[key] = value
                    
            labels = inputs.pop('labels')
        
            with torch.no_grad():
                output = self.model(**inputs)
                
                # Handle model-specific outputs
                if hasattr(self.model, 'name') and self.model.name in ['ExMRD_Evidential', 'ExMRD_Defer', 'TrueLens']:
                    if self.model.name == 'ExMRD_Defer':
                        # Defer model: use threshold-based deferral for final predictions
                        predict_output = self.model.predict(inputs['p_dm'], **{k:v for k,v in inputs.items() if k != 'p_dm'})
                        pred = predict_output['final_probs']  # Final predictions after deferral
                        losses_dict = self.model.compute_losses(output, labels)
                        loss = losses_dict['loss_total']
                        
                        # Log defer statistics during validation
                        defer_rate = predict_output['defer_rate'].item()
                        if hasattr(self, '_defer_stats'):
                            self._defer_stats.append(defer_rate)
                        else:
                            self._defer_stats = [defer_rate]
                            
                    elif self.model.name == 'ExMRD_Evidential':
                        # Check if using evidential or traditional classifier
                        if hasattr(self.model, 'use_evidential') and self.model.use_evidential:
                            # Evidential classifier: use fused predictions
                            pred = output['p_fused']  # Predictive probabilities from fused evidence
                            losses_dict = self.model.compute_losses(output, labels)
                            loss = losses_dict['loss_total']
                        else:
                            # Traditional classifier: output is logits
                            losses_dict = self.model.compute_losses(output, labels)
                            loss = losses_dict['loss_total']
                            pred = F.softmax(output, dim=1)
                            
                    elif self.model.name == 'TrueLens':
                        # Retrieval-augmented evidential classifier
                        pred = output['p_fused']  # Predictive probabilities from fused evidence
                        losses_dict = self.model.compute_losses(output, labels)
                        loss = losses_dict['loss_total']
                else:
                    # Regular model handling
                    if type(output) is dict:
                        pred = output['pred']
                    else:
                        pred = output
                    loss = F.cross_entropy(pred, labels)
            
            _, preds = torch.max(pred, 1)

            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
        metrics = self.evaluator.compute()
        
        logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list)}")
        logger.info(f"{fcolor}{split_name}: Acc: {metrics['acc']:.4f}, Macro F1: {metrics['f1']:.4f}, Macro Precision: {metrics['prec']:.4f}, Macro Recall: {metrics['rec']:.4f}")
        logger.info(f"{fcolor}{split_name} Per-class: Real(0) F1: {metrics['f1_real']:.4f}, P: {metrics['prec_real']:.4f}, R: {metrics['rec_real']:.4f} | Fake(1) F1: {metrics['f1_fake']:.4f}, P: {metrics['prec_fake']:.4f}, R: {metrics['rec_fake']:.4f}")
        
        # Report defer statistics if available
        if hasattr(self, '_defer_stats') and self._defer_stats:
            defer_rate_mean = np.mean(self._defer_stats)
            defer_rate_std = np.std(self._defer_stats)
            logger.info(f"{fcolor}{split_name} Defer: Rate: {defer_rate_mean:.3f}±{defer_rate_std:.3f} (threshold: {getattr(self.model, 'defer_threshold', 'N/A')})")
            # Reset defer stats for next validation
            self._defer_stats = []
        
        if use_earlystop:
            self.earlystopping(metrics['acc'], self.model)
        return metrics

@hydra.main(version_base=None, config_path="config", config_name="ExMRD_FakeSV")
def main(cfg: DictConfig):
    global log_path
    
    # Check if eval_only mode
    if cfg.get('eval_only', False):
        if not cfg.get('checkpoint'):
            raise ValueError("checkpoint parameter is required when eval_only=True")
        
        # Load configuration from checkpoint
        checkpoint_path = Path(cfg.checkpoint)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
        
        checkpoint_config_path = checkpoint_path / 'config.yaml'
        if not checkpoint_config_path.exists():
            raise ValueError(f"Config file not found in checkpoint: {checkpoint_config_path}")
        
        # Load the original configuration
        original_cfg = OmegaConf.load(checkpoint_config_path)
        
        # Override eval_only and checkpoint from current config
        original_cfg.eval_only = True
        original_cfg.checkpoint = cfg.checkpoint
        
        # Use checkpoint path as log_path for this evaluation
        log_path = checkpoint_path
        
        logger.remove()
        logger.add(log_path / 'eval_log.log', retention="10 days", level="DEBUG")
        logger.add(sys.stdout, level="INFO")
        logger.info("=== EVALUATION ONLY MODE ===")
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        logger.info(OmegaConf.to_yaml(original_cfg))
        
        pd.set_option('future.no_silent_downcasting', True)
        colorama.init()
        set_seed(original_cfg.seed)
        
        # Run evaluation with original config
        trainer = Trainer(original_cfg)
        trainer.run_eval_only()
        
    else:
        # Normal training mode
        dataset_name = cfg.dataset if 'dataset' in cfg else 'unknown'
        split_suffix = '_cv' if cfg.type == '5-fold' else '_temporal'
        piyao_suffix = '_with_piyao' if cfg.data.get('include_piyao', False) else ''
        ablation_suffix = '_no_cot' if cfg.data.get('ablation_no_cot', False) else ''
        filter_k_suffix = f'_k{cfg.data.filter_k}' if cfg.data.get('filter_k') is not None else ''
        log_path = Path(f'log/{dataset_name}{split_suffix}{piyao_suffix}{ablation_suffix}{filter_k_suffix}_{datetime.now().strftime("%m%d-%H%M%S")}')
        log_path.mkdir(parents=True, exist_ok=True)
        
        logger.remove()
        logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
        logger.add(sys.stdout, level="INFO")
        logger.info(OmegaConf.to_yaml(cfg))
        
        # Save configuration to YAML file in log directory
        config_save_path = log_path / 'config.yaml'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(config=cfg, f=f)
        logger.info(f"Configuration saved to: {config_save_path}")
        pd.set_option('future.no_silent_downcasting', True)
        colorama.init()
        set_seed(cfg.seed)
        # mp.set_sharing_strategy('file_system')
        
        trainer = Trainer(cfg)
        trainer.run()

if  __name__ == '__main__':
    main()
