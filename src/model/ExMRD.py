import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel
import torch
from loguru import logger
import math


def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Convert to human-readable format
    def human_readable(num):
        for unit in ['', 'K', 'M', 'B', 'T']:
            if num < 1000:
                return f"{num:.1f}{unit}"
            num /= 1000
    
    print(f"Total parameters: {human_readable(total_params)}")
    print(f"Trainable parameters: {human_readable(trainable_params)}")


def print_full_model_params(model):
    """Print parameters for full model, handling lazy layers"""
    try:
        initialized_params = 0
        trainable_params = 0
        lazy_layers = 0
        
        for name, param in model.named_parameters():
            if hasattr(param, '_is_lazy') and param._is_lazy:
                lazy_layers += 1
            else:
                param_count = param.numel()
                initialized_params += param_count
                if param.requires_grad:
                    trainable_params += param_count
        
        # Convert to human-readable format
        def human_readable(num):
            for unit in ['', 'K', 'M', 'B', 'T']:
                if num < 1000:
                    return f"{num:.1f}{unit}"
                num /= 1000
        
        logger.info(f"Full ExMRD model - Total parameters: {human_readable(initialized_params)}")
        logger.info(f"Full ExMRD model - Trainable parameters: {human_readable(trainable_params)}")
        if lazy_layers > 0:
            logger.info(f"Full ExMRD model - Uninitialized lazy layers: {lazy_layers}")
            
    except Exception as e:
        logger.warning(f"Could not count full model parameters: {e}")


class BERT_FT(nn.Module):
    def __init__(self, text_encoder, num_frozen_layers):
        super(BERT_FT, self).__init__()
        self.bert = AutoModel.from_pretrained(text_encoder).text_model

        for name, param in self.bert.named_parameters():
            # print(name)
            if 'embeddings' in name or 'final_layer_norm' in name:
                param.requires_grad = False
            elif 'encoder.layer' in name:
                # get the layer index
                layer_idx = int(name.split('.')[2])
                if layer_idx >= num_frozen_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, **inputs):
        return self.bert(**inputs)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super(LearnablePositionalEncoding, self).__init__()
        
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
        
        self.init_weights()
    def init_weights(self):
        position = torch.arange(0, self.positional_encoding.size(0)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.positional_encoding.size(1), 2) * 
                             -(math.log(10000.0) / self.positional_encoding.size(1)))
        
        pe = torch.zeros_like(self.positional_encoding)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.positional_encoding.data.copy_(pe)
    def forward(self, x):
        return x + self.positional_encoding[:x.size(1), :]


class CrossModalTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=4, d_ff=1024, dropout=0.1):
        super(CrossModalTransformer, self).__init__()
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Modality embeddings (0 for visual, 1 for text)
        self.modality_embeddings = nn.Embedding(2, d_model)
        
        # Position embeddings for 21 tokens (CLS + 16 visual + 4 text)
        self.position_embeddings = nn.Parameter(torch.zeros(21, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize parameters
        self.init_weights()
    
    def init_weights(self):
        # Initialize CLS token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize position embeddings with sinusoidal pattern
        position = torch.arange(0, self.position_embeddings.size(0)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.position_embeddings.size(1), 2) * 
                             -(math.log(10000.0) / self.position_embeddings.size(1)))
        
        pe = torch.zeros_like(self.position_embeddings)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        
        self.position_embeddings.data.copy_(pe)
        
        # Initialize modality embeddings
        nn.init.normal_(self.modality_embeddings.weight, std=0.02)
    
    def forward(self, visual_tokens, text_tokens):
        batch_size = visual_tokens.size(0)
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate CLS token, visual tokens, and text tokens
        # Shape: [batch_size, 21, d_model]
        all_tokens = torch.cat([cls_tokens, visual_tokens, text_tokens], dim=1)
        
        # Add position embeddings
        all_tokens = all_tokens + self.position_embeddings.unsqueeze(0)
        
        # Create modality embeddings
        # 0 for CLS and visual tokens, 1 for text tokens
        modality_ids = torch.cat([
            torch.zeros(batch_size, 17, dtype=torch.long, device=visual_tokens.device),  # CLS + 16 visual
            torch.ones(batch_size, 4, dtype=torch.long, device=visual_tokens.device)      # 4 text
        ], dim=1)
        
        # Add modality embeddings
        all_tokens = all_tokens + self.modality_embeddings(modality_ids)
        
        # Pass through transformer
        encoded_tokens = self.transformer_encoder(all_tokens)
        
        # Apply layer normalization
        encoded_tokens = self.layer_norm(encoded_tokens)
        
        # Extract CLS token (first token)
        cls_output = encoded_tokens[:, 0, :]
        
        return cls_output

class ExMRD(nn.Module):
    def __init__(self, hid_dim, dropout, text_encoder, num_frozen_layers=12, ablation='No', ablation_no_cot=False,
                 ablation_no_vision=False, cross_modal=False, cross_modal_config=None):
        super(ExMRD, self).__init__()
        self.name = 'ExMRD'
        self.text_encoder = text_encoder
        self.ablation_no_cot = ablation_no_cot
        self.ablation_no_vision = ablation_no_vision
        self.cross_modal = cross_modal
        
        # Check for conflicting ablations
        if self.ablation_no_cot and self.ablation_no_vision:
            raise ValueError("Cannot have both ablation_no_cot and ablation_no_vision enabled simultaneously")
        
        if ablation == 'w/o-finetune':
            num_frozen_layers = 12
        
        if not self.ablation_no_cot:
            # Initialize BERT and text encoders if not in text ablation mode
            self.bert = BERT_FT(text_encoder, num_frozen_layers)
            print_model_params(self.bert)
            self.linear_ocr = nn.Sequential(nn.LazyLinear(hid_dim))
            self.linear_caption = nn.Sequential(nn.LazyLinear(hid_dim))
            self.linear_comsense = nn.Sequential(nn.LazyLinear(hid_dim))
            self.linear_causal = nn.Sequential(nn.LazyLinear(hid_dim))

        if not self.ablation_no_vision:
            # Initialize video processing components if not in vision ablation mode
            self.linear_video = nn.Sequential(nn.LazyLinear(hid_dim))
            self.temporal_pe = LearnablePositionalEncoding(1024, 16)
        
        self.classifier = nn.LazyLinear(2)
        
        # Initialize cross-modal transformer if enabled
        if self.cross_modal and not self.ablation_no_cot:
            cross_modal_config = cross_modal_config or {}
            self.cross_modal_transformer = CrossModalTransformer(
                d_model=hid_dim,
                n_heads=cross_modal_config.get('n_heads', 8),
                n_layers=cross_modal_config.get('n_layers', 4),
                d_ff=cross_modal_config.get('d_ff', 1024),
                dropout=cross_modal_config.get('dropout', 0.1)
            )
            # Count CrossModalTransformer parameters
            transformer_params = sum(p.numel() for p in self.cross_modal_transformer.parameters())
            transformer_trainable = sum(p.numel() for p in self.cross_modal_transformer.parameters() if p.requires_grad)
            logger.info(f"Cross-modal transformer initialized with config: {cross_modal_config}")
            logger.info(f"CrossModalTransformer parameters: {transformer_params:,} total, {transformer_trainable:,} trainable")

        self.ablation = ablation
        
    def forward(self, **inputs):
        fea_frames = inputs.get('fea_frames', None)
        
        if self.ablation_no_cot:
            # Text ablation: use only video features, no CoT text features
            if fea_frames is None:
                raise ValueError("fea_frames required for ablation_no_cot mode")
            fea_frames = self.temporal_pe(fea_frames)
            fea_frames = self.linear_video(fea_frames)
            fea = torch.mean(fea_frames, dim=1)
        elif self.ablation_no_vision:
            # Vision ablation: use only text features, no visual features
            # Note: BERT will use the mode set by the trainer (train/eval)
            
            lm_ocr_input = inputs['lm_ocr_input']
            caption_input = inputs['caption_input']
            comsense_input = inputs['comsense_input']
            causal_input = inputs['causal_input']
            
            if 'chinese' in self.text_encoder:
                fea_ocr = self.bert(**lm_ocr_input)['last_hidden_state'][:,0,:]
                fea_caption = self.bert(**caption_input)['last_hidden_state'][:,0,:]
                fea_comsense = self.bert(**comsense_input)['last_hidden_state'][:,0,:]
                fea_causal = self.bert(**causal_input)['last_hidden_state'][:,0,:]
            else:
                fea_ocr = self.bert(**lm_ocr_input)['pooler_output']
                fea_caption = self.bert(**caption_input)['pooler_output']
                fea_comsense = self.bert(**comsense_input)['pooler_output']
                fea_causal = self.bert(**causal_input)['pooler_output']
            
            fea_ocr = self.linear_ocr(fea_ocr)
            fea_caption = self.linear_caption(fea_caption)
            fea_comsense = self.linear_comsense(fea_comsense)
            fea_causal = self.linear_causal(fea_causal)
            
            # Average text features only (no video features)
            fea_x = torch.concat((fea_ocr.unsqueeze(1), fea_caption.unsqueeze(1), fea_comsense.unsqueeze(1), fea_causal.unsqueeze(1)), 1)
            fea = torch.mean(fea_x, dim=1)
        else:
            # Normal mode: use both video features and CoT explanations
            # Note: BERT will use the mode set by the trainer (train/eval)
            
            lm_ocr_input = inputs['lm_ocr_input']
            caption_input = inputs['caption_input']
            comsense_input = inputs['comsense_input']
            causal_input = inputs['causal_input']
            
            if 'chinese' in self.text_encoder:
                fea_ocr = self.bert(**lm_ocr_input)['last_hidden_state'][:,0,:]
                fea_caption = self.bert(**caption_input)['last_hidden_state'][:,0,:]
                fea_comsense = self.bert(**comsense_input)['last_hidden_state'][:,0,:]
                fea_causal = self.bert(**causal_input)['last_hidden_state'][:,0,:]
            else:
                fea_ocr = self.bert(**lm_ocr_input)['pooler_output']
                fea_caption = self.bert(**caption_input)['pooler_output']
                fea_comsense = self.bert(**comsense_input)['pooler_output']
                fea_causal = self.bert(**causal_input)['pooler_output']
            
            fea_ocr = self.linear_ocr(fea_ocr)
            fea_caption = self.linear_caption(fea_caption)
            fea_comsense = self.linear_comsense(fea_comsense)
            fea_causal = self.linear_causal(fea_causal)

            # Apply temporal PE to video features
            fea_frames = self.temporal_pe(fea_frames)
            fea_frames = self.linear_video(fea_frames)
            
            if self.cross_modal:
                # Cross-modal transformer fusion
                # Keep visual tokens as sequence [B, 16, 256]
                visual_tokens = fea_frames
                
                # Stack text tokens as sequence [B, 4, 256]
                text_tokens = torch.stack([fea_ocr, fea_caption, fea_comsense, fea_causal], dim=1)
                
                # Pass through cross-modal transformer
                fea = self.cross_modal_transformer(visual_tokens, text_tokens)
            else:
                # Original averaging approach
                fea_x = torch.concat((fea_ocr.unsqueeze(1), fea_caption.unsqueeze(1), fea_comsense.unsqueeze(1), fea_causal.unsqueeze(1)), 1)
                fea_x = torch.mean(fea_x, dim=1)
                fea_frames = torch.mean(fea_frames, dim=1)
                fea = torch.stack((fea_frames, fea_x), 1)
                fea = torch.mean(fea, dim=1)
        
        output = self.classifier(fea)
        return output