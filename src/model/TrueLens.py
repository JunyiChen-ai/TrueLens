import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from transformers import AutoModel
from loguru import logger
from typing import Dict, Tuple, Optional
from pathlib import Path


def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def human_readable(num):
        for unit in ['', 'K', 'M', 'B', 'T']:
            if num < 1000:
                return f"{num:.1f}{unit}"
            num /= 1000
    
    print(f"Total parameters: {human_readable(total_params)}")
    print(f"Trainable parameters: {human_readable(trainable_params)}")


# -------------------------
# Utilities: evidential math (copied from ExMRD_Evidential)
# -------------------------

def alpha_to_belief_u(alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    alpha: (B, K) Dirichlet parameters (>=1)
    returns:
      b: (B, K) belief masses per class
      u: (B, 1) uncertainty mass
    """
    eps = 1e-8
    S = torch.clamp(alpha.sum(dim=-1, keepdim=True), min=eps)  # (B,1)
    b = (alpha - 1.0) / torch.clamp(S, min=eps)                # (B,K)
    u = alpha.size(-1) / torch.clamp(S, min=eps)               # (B,1)
    # Numerical safety: keep within [0,1] and make sums consistent
    b = torch.clamp(b, min=0.0)
    u = torch.clamp(u, min=eps, max=1.0 - eps)
    # Renormalize b to sum to (1 - u) if mild drift occurs
    b_sum = b.sum(dim=-1, keepdim=True)
    target = (1.0 - u).clamp(min=eps)
    b = b * (target / torch.clamp(b_sum, min=eps))
    return b, u


def combine_two_opinions(b1: torch.Tensor, u1: torch.Tensor,
                         b2: torch.Tensor, u2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reduced DST combination for subjective logic opinions with singletons + uncertainty.
    b1,b2: (B,K), u1,u2: (B,1)
    returns:
      b: (B,K), u: (B,1)
    """
    eps = 1e-8
    # conflict C = sum_{i != j} b1_i b2_j
    # Compute via totals to avoid explicit double sum:
    # sum_i sum_j b1_i b2_j = (sum_i b1_i) * (sum_j b2_j)
    # conflict = total_pair - sum_i b1_i b2_i
    total_pair = (b1.sum(dim=-1, keepdim=True) * b2.sum(dim=-1, keepdim=True))  # (B,1)
    dot_same = (b1 * b2).sum(dim=-1, keepdim=True)                               # (B,1)
    C = total_pair - dot_same                                                    # (B,1)
    S = torch.clamp(1.0 - C, min=eps)

    b = (b1 * b2 + b1 * u2 + b2 * u1) / S
    u = (u1 * u2) / S

    # Clamp & tidy
    b = torch.clamp(b, min=0.0)
    u = torch.clamp(u, min=eps, max=1.0 - eps)
    # Renormalize so that sum(b) + u = 1
    b_sum = b.sum(dim=-1, keepdim=True)
    scale = (1.0 - u) / torch.clamp(b_sum, min=eps)
    b = b * scale
    return b, u


def opinion_to_alpha(b: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Map opinion (b, u) back to Dirichlet parameters alpha = e + 1 with
    S = K / u, e_k = b_k * S.
    """
    eps = 1e-8
    K = b.size(-1)
    u = torch.clamp(u, min=eps, max=1.0 - eps)
    S = K / u
    e = b * S
    alpha = e + 1.0
    return torch.clamp(alpha, min=1.0 + eps)


def dirichlet_mean(alpha: torch.Tensor) -> torch.Tensor:
    """E[p] under Dir(alpha): (B,K)"""
    S = alpha.sum(dim=-1, keepdim=True)
    return alpha / torch.clamp(S, min=1e-8)


def nll_dirichlet(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for Dir(alpha) given one-hot targets.
    alpha: (B, K), y: (B,) integer labels
    returns: (B,)
    """
    S = alpha.sum(dim=-1)  # (B,)
    y_alpha = alpha.gather(1, y.unsqueeze(1)).squeeze(1)  # (B,)
    return torch.digamma(S) - torch.digamma(y_alpha)


def kl_dirichlet_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL(Dir(alpha) || Dir(1,...,1))
    alpha: (B, K)
    returns: (B,)
    """
    # K = alpha.size(-1)
    # S = alpha.sum(dim=-1)  # (B,)
    # kl = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(S) - K * torch.lgamma(torch.tensor(1.0, device=alpha.device))
    # kl += (alpha - 1.0).sum(dim=-1) * (torch.digamma(alpha) - torch.digamma(S).unsqueeze(-1)).sum(dim=-1)
    # return kl
    eps = 1e-8
    K = alpha.size(-1)
    S = torch.clamp(alpha.sum(dim=-1), min=eps)  # (B,)
    term1 = torch.lgamma(S)
    term2 = torch.lgamma(alpha).sum(dim=-1)
    term3 = torch.lgamma(torch.tensor(K, dtype=alpha.dtype, device=alpha.device))
    term4 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(-1)))).sum(dim=-1)
    return term1 - term2 - term3 + term4


def anneal_factor(global_step: int, anneal_steps: int) -> float:
    """Annealing factor for KL term: starts at 0, goes to 1."""
    if anneal_steps <= 0:
        return 1.0
    return min(1.0, global_step / anneal_steps)


# -------------------------
# BERT Fine-tuning module (copied from ExMRD_Evidential)
# -------------------------

class BERT_FT(nn.Module):
    def __init__(self, bert_name, num_frozen_layers=0):
        super(BERT_FT, self).__init__()
        self.num_frozen_layers = num_frozen_layers
        
        # Load only text model to save memory and parameters
        self.bert = AutoModel.from_pretrained(bert_name).text_model
        print(f"Loaded language model from {bert_name}")
        # Comprehensive freezing strategy
        trainable_params = 0
        total_params = 0
        frozen_params = 0
        
        for name, param in self.bert.named_parameters():
            total_params += param.numel()
            
            # Always freeze embeddings and final layer norm
            if 'embeddings' in name or 'final_layer_norm' in name:
                param.requires_grad = False
                frozen_params += param.numel()
            elif 'encoder.layer' in name:
                # Extract layer index from parameter name
                layer_idx = int(name.split('.')[2])
                if layer_idx >= num_frozen_layers:
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
                    frozen_params += param.numel()
            else:
                # For other parameters (pooler, etc), keep trainable
                param.requires_grad = True
                trainable_params += param.numel()
        
        logger.info(f"BERT_FT: Frozen layers 0-{num_frozen_layers-1}, trainable layers {num_frozen_layers}-11")
        logger.info(f"BERT_FT: Total {total_params/1e6:.1f}M params, Trainable {trainable_params/1e6:.1f}M, Frozen {frozen_params/1e6:.1f}M")
        
    def forward(self, **inputs):
        # Request hidden states from Chinese CLIP text model
        outputs = self.bert(**inputs, output_hidden_states=True)
        # Create a compatible output format
        return {
            'last_hidden_state': outputs.hidden_states[-1],  # Last layer hidden states
            'pooler_output': outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        }


# -------------------------
# Simple Modality-Specific FFN modules
# -------------------------

class ModalityFFN(nn.Module):
    """
    Simple Feed-Forward Network for processing single modality sequences
    """
    def __init__(self, d_model=256, hidden_dim=512, dropout=0.1):
        super(ModalityFFN, self).__init__()
        self.d_model = d_model
        
        self.input_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, seq_len, d_model) input sequence
            mask: (B, seq_len) padding mask (True for padding positions)
        Returns:
            (B, seq_len, d_model) processed sequence
        """
        # Input normalization
        x_norm = self.input_norm(x)
        
        # Apply FFN with residual connection
        out = self.ffn(x_norm) + x
        
        # Output normalization
        out = self.output_norm(out)
        
        # Apply mask if provided (set padding positions to zero)
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)
            
        return out


class TextFFN(ModalityFFN):
    """Text-specific FFN"""
    def __init__(self, d_model=256, hidden_dim=512, dropout=0.1):
        super(TextFFN, self).__init__(d_model, hidden_dim, dropout)


class AudioFFN(ModalityFFN):
    """Audio-specific FFN"""
    def __init__(self, d_model=256, hidden_dim=512, dropout=0.1):
        super(AudioFFN, self).__init__(d_model, hidden_dim, dropout)


class VisualFFN(ModalityFFN):
    """Visual-specific FFN"""
    def __init__(self, d_model=256, hidden_dim=512, dropout=0.1):
        super(VisualFFN, self).__init__(d_model, hidden_dim, dropout)


# -------------------------
# Retrieval Attention
# -------------------------

class RetrievalAttention(nn.Module):
    """
    Cross-attention using retrieval samples as K,V and current sample as Q
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super(RetrievalAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for output
        self.output_norm = nn.LayerNorm(d_model)
        
    def forward(self, query_seq, key_seq, value_seq, key_padding_mask=None):
        """
        Args:
            query_seq: (B, T_q, d_model) - current sample sequence
            key_seq: (B, T_k, d_model) - retrieval sample sequence  
            value_seq: (B, T_v, d_model) - retrieval sample sequence (same as key_seq)
            key_padding_mask: (B, T_k) - True for positions to be masked (padded tokens)
        Returns:
            attended_seq: (B, T_q, d_model)
        """
        attended_seq, _ = self.multihead_attn(
            query=query_seq,
            key=key_seq,
            value=value_seq,
            key_padding_mask=key_padding_mask
        )
        # Normalize output to prevent scale explosion
        attended_seq = self.output_norm(attended_seq)
        return attended_seq


# -------------------------
# Evidential Classifier (copied and modified from ExMRD_Evidential)
# -------------------------

class MultiViewEvidentialClassifier(nn.Module):
    def __init__(self, d: int, num_classes: int = 2, hidden: int = 256, dropout: float = 0.1):
        super(MultiViewEvidentialClassifier, self).__init__()
        self.K = num_classes

        # Input normalization for each modality
        self.text_input_norm = nn.LayerNorm(d)
        self.audio_input_norm = nn.LayerNorm(d)
        self.image_input_norm = nn.LayerNorm(d)

        # Per-modality evidential heads
        def make_head():
            return nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, self.K),
                nn.Softplus()  # Ensures alpha >= 0, then we add 1
            )

        self.text_head = make_head()
        self.audio_head = make_head()
        self.image_head = make_head()

    def _fuse_opinions(self, alpha_t: torch.Tensor, alpha_a: torch.Tensor, alpha_i: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fuse three Dirichlet opinions using subjective logic."""
        # Convert to beliefs + uncertainties
        b_t, u_t = alpha_to_belief_u(alpha_t)
        b_a, u_a = alpha_to_belief_u(alpha_a)
        b_i, u_i = alpha_to_belief_u(alpha_i)

        # Combine text + audio
        b_ta, u_ta = combine_two_opinions(b_t, u_t, b_a, u_a)
        # Combine (text+audio) + image
        b_f, u_f = combine_two_opinions(b_ta, u_ta, b_i, u_i)

        # Convert back to alpha
        alpha_f = opinion_to_alpha(b_f, u_f)

        return {"alpha_f": alpha_f, "b_f": b_f, "u_f": u_f}

    def forward(self, x_text=None, x_audio=None, x_image=None):
        """
        x_*: (B, d) or None if modality disabled
        returns dict with per-view & fused alphas, uncertainties, and predictive means.
        """
        # Initialize alphas
        batch_size = None
        device = None
        
        # Determine batch size and device from available modalities
        for x in [x_text, x_audio, x_image]:
            if x is not None:
                batch_size = x.shape[0]
                device = x.device
                break
        
        if batch_size is None:
            raise ValueError("At least one modality must be provided")
        
        # Process each modality or create dummy alpha (with input normalization)
        if x_text is not None:
            x_text_norm = self.text_input_norm(x_text)
            out_t = self.text_head(x_text_norm)
        else:
            # Create neutral/uniform alpha for missing text modality
            out_t = torch.zeros(batch_size, self.K, device=device)
            
        if x_audio is not None:
            x_audio_norm = self.audio_input_norm(x_audio)
            out_a = self.audio_head(x_audio_norm)
        else:
            # Create neutral/uniform alpha for missing audio modality
            out_a = torch.zeros(batch_size, self.K, device=device)
            
        if x_image is not None:
            x_image_norm = self.image_input_norm(x_image)
            out_i = self.image_head(x_image_norm)
        else:
            # Create neutral/uniform alpha for missing image modality
            out_i = torch.zeros(batch_size, self.K, device=device)

        # Convert to alpha format if needed
        alpha_t = out_t if isinstance(out_t, torch.Tensor) else out_t["alpha"]
        alpha_a = out_a if isinstance(out_a, torch.Tensor) else out_a["alpha"]  
        alpha_i = out_i if isinstance(out_i, torch.Tensor) else out_i["alpha"]

        # Ensure alpha >= 1
        alpha_t = alpha_t + 1.0
        alpha_a = alpha_a + 1.0
        alpha_i = alpha_i + 1.0

        # Fused (no gradients through opinion algebra for stability)
        with torch.no_grad():
            fuse = self._fuse_opinions(alpha_t, alpha_a, alpha_i)

        # Predictive means
        p_t = dirichlet_mean(alpha_t)
        p_a = dirichlet_mean(alpha_a)
        p_i = dirichlet_mean(alpha_i)
        p_f = dirichlet_mean(fuse["alpha_f"])

        # Uncertainty masses
        _, u_t = alpha_to_belief_u(alpha_t)
        _, u_a = alpha_to_belief_u(alpha_a)
        _, u_i = alpha_to_belief_u(alpha_i)
        u_f = fuse["u_f"]

        return {
            # per-view
            "alpha_text": alpha_t, "p_text": p_t, "u_text": u_t,
            "alpha_audio": alpha_a, "p_audio": p_a, "u_audio": u_a,
            "alpha_image": alpha_i, "p_image": p_i, "u_image": u_i,
            # fused
            "alpha_fused": fuse["alpha_f"], "p_fused": p_f, "u_fused": u_f
        }

    def compute_losses(self,
                       outputs: Dict[str, torch.Tensor],
                       y: torch.Tensor,
                       global_step: int = 0,
                       anneal_steps: int = 1000,
                       weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """
        y: (B,) ground-truth labels [0..K-1]
        Returns dict of individual losses and total:
          total = w_fused*L_fused + w_text*L_text + w_audio*L_audio + w_image*L_image
        """
        if weights is None:
            weights = {"fused": 1.0, "text": 1.0, "audio": 1.0, "image": 1.0}

        beta = anneal_factor(global_step, anneal_steps)

        # Per-view losses
        def loss_for(alpha):
            nll = nll_dirichlet(alpha, y)                  # (B,)
            kl  = kl_dirichlet_to_uniform(alpha)           # (B,)
            return (nll + beta * kl).mean()                # scalar

        L_text  = loss_for(outputs["alpha_text"])
        L_audio = loss_for(outputs["alpha_audio"])
        L_image = loss_for(outputs["alpha_image"])
        L_fused = loss_for(outputs["alpha_fused"])

        total = (weights["fused"] * L_fused +
                 weights["text"]  * L_text +
                 weights["audio"] * L_audio +
                 weights["image"] * L_image)

        return {
            "loss_total": total,
            "loss_fused": L_fused,
            "loss_text": L_text,
            "loss_audio": L_audio,
            "loss_image": L_image,
            "kl_weight": torch.tensor(beta, device=outputs["alpha_fused"].device)
        }


# -------------------------
# Main TrueLens Model
# -------------------------

class TrueLens(nn.Module):
    def __init__(self, hid_dim, dropout, text_encoder, num_frozen_layers=4, 
                 evidential_hidden=256, evidential_dropout=0.1, 
                 loss_weights=None, anneal_steps=1000,
                 transformer_layers=4, transformer_heads=8, retrieval_alpha=0.5,
                 retrieval_path=None, gradient_clip_norm=0.0):
        super(TrueLens, self).__init__()
        self.name = 'TrueLens'
        self.text_encoder = text_encoder
        self.hid_dim = hid_dim
        self.anneal_steps = anneal_steps
        self.retrieval_alpha = retrieval_alpha
        self.gradient_clip_norm = gradient_clip_norm
        
        # Text encoder (Chinese CLIP BERT)
        self.bert = BERT_FT(text_encoder, num_frozen_layers)
        print_model_params(self.bert)
        
        # Feature projection layers
        self.text_proj = nn.Linear(768, hid_dim)  # Chinese CLIP BERT hidden size is 768
        self.audio_proj = nn.LazyLinear(hid_dim)  # Audio features dimension detected dynamically
        self.visual_proj = nn.Linear(1024, hid_dim)  # ViT features are 1024-dim
        
        # Modality-specific FFN modules
        ffn_hidden_dim = hid_dim * 2  # Hidden dimension for FFNs
        self.text_ffn = TextFFN(d_model=hid_dim, hidden_dim=ffn_hidden_dim, dropout=evidential_dropout)
        self.audio_ffn = AudioFFN(d_model=hid_dim, hidden_dim=ffn_hidden_dim, dropout=evidential_dropout)
        self.visual_ffn = VisualFFN(d_model=hid_dim, hidden_dim=ffn_hidden_dim, dropout=evidential_dropout)
        
        # Retrieval attention modules
        self.text_pos_attention = RetrievalAttention(hid_dim, transformer_heads, evidential_dropout)
        self.text_neg_attention = RetrievalAttention(hid_dim, transformer_heads, evidential_dropout)
        self.audio_pos_attention = RetrievalAttention(hid_dim, transformer_heads, evidential_dropout)
        self.audio_neg_attention = RetrievalAttention(hid_dim, transformer_heads, evidential_dropout)
        self.visual_pos_attention = RetrievalAttention(hid_dim, transformer_heads, evidential_dropout)
        self.visual_neg_attention = RetrievalAttention(hid_dim, transformer_heads, evidential_dropout)
        
        # Evidential classifier
        self.evidential_classifier = MultiViewEvidentialClassifier(
            d=hid_dim, 
            num_classes=2, 
            hidden=evidential_hidden,
            dropout=evidential_dropout
        )
        
        # Loss weights for different modalities
        self.loss_weights = loss_weights or {"fused": 1.0, "text": 0.5, "audio": 0.5, "image": 0.5}
        logger.info(f"Retrieval loss weights: {self.loss_weights}")
        logger.info(f"Retrieval alpha: {self.retrieval_alpha}")
        
        # Normalization for weighted combinations and final features
        self.text_combination_norm = nn.LayerNorm(hid_dim)
        self.audio_combination_norm = nn.LayerNorm(hid_dim)  
        self.visual_combination_norm = nn.LayerNorm(hid_dim)
        self.final_feature_norm = nn.LayerNorm(hid_dim)
        
        logger.info(f"Initialized TrueLens with hidden_dim={hid_dim}")

    def forward(self, **inputs):
        """
        Forward pass with retrieval-augmented features:
        - entity_text_input: tokenized entity claims text (if use_text=True)
        - audio_features: (B, seq_len, audio_dim) audio features (if use_audio=True)
        - visual_features: (B, 16, 1024) visual features (if use_image=True)
        - positive_text_features: positive retrieval text features
        - negative_text_features: negative retrieval text features
        - positive_audio_features: positive retrieval audio features  
        - negative_audio_features: negative retrieval audio features
        - positive_visual_features: positive retrieval visual features
        - negative_visual_features: negative retrieval visual features
        """
        # Get modality flags
        use_text = inputs.get('use_text', True)
        use_image = inputs.get('use_image', True)
        use_audio = inputs.get('use_audio', True)
        
        # Initialize features as None
        text_seq = None
        audio_seq = None
        visual_seq = None
        
        # Text encoding if enabled - get full sequence, not just CLS
        text_mask = None
        if use_text and 'entity_text_input' in inputs:
            entity_text_input = inputs['entity_text_input']
            bert_output = self.bert(**entity_text_input)
            
            # Extract attention mask for padding (0 = padding, 1 = valid token)
            # Convert to padding mask (True = padding, False = valid token)
            if 'attention_mask' in entity_text_input:
                text_mask = ~entity_text_input['attention_mask'].bool()  # Invert: 0->True, 1->False
            
            # Handle different model architectures for extracting hidden states
            if 'last_hidden_state' in bert_output:
                # Standard BERT or Chinese CLIP text model output
                text_features = bert_output['last_hidden_state']  # (B, seq_len, 768)
            elif 'text_embeds' in bert_output:
                # Chinese CLIP model output
                text_features = bert_output['text_embeds']  # (B, seq_len, 768)
            else:
                raise ValueError("Unknown BERT output format")
            
            # Debug logging for potential data issues
            if torch.isnan(text_features).any():
                logger.warning(f"NaN detected in text_features: {torch.isnan(text_features).sum()} values")
            if torch.isinf(text_features).any():
                logger.warning(f"Inf detected in text_features: {torch.isinf(text_features).sum()} values")
            
            text_seq = self.text_proj(text_features)  # (B, seq_len, hid_dim)
            
            # Also encode positive and negative text if available
            if 'positive_text_input' in inputs:
                pos_text_output = self.bert(**inputs['positive_text_input'])
                if 'last_hidden_state' in pos_text_output:
                    inputs['positive_text_features'] = pos_text_output['last_hidden_state']  # (B, seq_len, 768)
                elif 'text_embeds' in pos_text_output:
                    inputs['positive_text_features'] = pos_text_output['text_embeds']  # (B, seq_len, 768)
                
            if 'negative_text_input' in inputs:
                neg_text_output = self.bert(**inputs['negative_text_input'])
                if 'last_hidden_state' in neg_text_output:
                    inputs['negative_text_features'] = neg_text_output['last_hidden_state']  # (B, seq_len, 768)
                elif 'text_embeds' in neg_text_output:
                    inputs['negative_text_features'] = neg_text_output['text_embeds']  # (B, seq_len, 768)
        
        # Audio encoding if enabled - keep as sequence
        if use_audio and 'audio_features' in inputs:
            audio_feats = inputs['audio_features']  # (B, seq_len, audio_dim)
            audio_seq = self.audio_proj(audio_feats)  # (B, seq_len, hid_dim)
        
        # Visual encoding if enabled - keep as sequence
        if use_image and 'visual_features' in inputs:
            visual_feats = inputs['visual_features']  # (B, 16, 1024)
            visual_seq = self.visual_proj(visual_feats)  # (B, 16, hid_dim)
        
        # Apply modality-specific FFNs
        ffn_outputs = {}
        if text_seq is not None:
            ffn_outputs['text'] = self.text_ffn(text_seq, text_mask)
        if audio_seq is not None:
            ffn_outputs['audio'] = self.audio_ffn(audio_seq, None)  # Audio typically doesn't have padding
        if visual_seq is not None:
            ffn_outputs['visual'] = self.visual_ffn(visual_seq, None)  # Visual frames don't have padding
        
        # Apply retrieval attention if retrieval features are available
        final_text_features = None
        final_audio_features = None  
        final_visual_features = None
        
        # Debug: Log retrieval data availability
        has_pos_text = 'positive_text_features' in inputs
        has_neg_text = 'negative_text_features' in inputs
        has_pos_audio = 'positive_audio_features' in inputs
        has_neg_audio = 'negative_audio_features' in inputs
        has_pos_visual = 'positive_visual_features' in inputs
        has_neg_visual = 'negative_visual_features' in inputs
        
        retrieval_counts = {
            'text_pos': has_pos_text, 'text_neg': has_neg_text,
            'audio_pos': has_pos_audio, 'audio_neg': has_neg_audio,
            'visual_pos': has_pos_visual, 'visual_neg': has_neg_visual
        }
        retrieval_available = sum(retrieval_counts.values())
        
        if retrieval_available == 0:
            logger.debug("No retrieval features available for this batch")
        elif retrieval_available < 6:
            logger.debug(f"Partial retrieval data: {retrieval_counts}")
        
        # Text retrieval attention
        if 'text' in ffn_outputs:
            text_seq_transformed = ffn_outputs['text']
            
            # Apply positive and negative attention if retrieval data available
            if 'positive_text_features' in inputs and 'negative_text_features' in inputs:
                # Process retrieved features through projection and FFN (same as current sample)
                pos_text_proj = self.text_proj(inputs['positive_text_features'])  # (B, seq_len, hid_dim)
                neg_text_proj = self.text_proj(inputs['negative_text_features'])  # (B, seq_len, hid_dim)
                
                pos_text_features = self.text_ffn(pos_text_proj, mask=None)  # Apply FFN to retrieved features
                neg_text_features = self.text_ffn(neg_text_proj, mask=None)  # Apply FFN to retrieved features
                
                # For retrieval features, we assume no padding (use None for key_padding_mask)
                pos_attended = self.text_pos_attention(text_seq_transformed, pos_text_features, pos_text_features, key_padding_mask=None)
                neg_attended = self.text_neg_attention(text_seq_transformed, neg_text_features, neg_text_features, key_padding_mask=None)
                
                attended_text = (self.retrieval_alpha * pos_attended + 
                                (1 - self.retrieval_alpha) * neg_attended + 
                                text_seq_transformed)
                # Normalize to prevent scale explosion
                attended_text = self.text_combination_norm(attended_text)
              
            else:
                attended_text = text_seq_transformed
                
            # Average pooling and normalize
            final_text_features = attended_text.mean(dim=1)  # (B, hid_dim)
            final_text_features = self.final_feature_norm(final_text_features)
        
        # Audio retrieval attention
        if 'audio' in ffn_outputs:
            audio_seq_transformed = ffn_outputs['audio']
            
            # Apply positive and negative attention if retrieval data available
            if 'positive_audio_features' in inputs and 'negative_audio_features' in inputs:
                # Process retrieved features through projection and FFN (same as current sample)
                pos_audio_proj = self.audio_proj(inputs['positive_audio_features'])  # (B, seq_len, hid_dim)
                neg_audio_proj = self.audio_proj(inputs['negative_audio_features'])  # (B, seq_len, hid_dim)
                
                pos_audio_features = self.audio_ffn(pos_audio_proj, mask=None)  # Apply FFN to retrieved features
                neg_audio_features = self.audio_ffn(neg_audio_proj, mask=None)  # Apply FFN to retrieved features
                
                pos_attended = self.audio_pos_attention(audio_seq_transformed, pos_audio_features, pos_audio_features, key_padding_mask=None)
                neg_attended = self.audio_neg_attention(audio_seq_transformed, neg_audio_features, neg_audio_features, key_padding_mask=None)
                
                # Weighted combination
                attended_audio = (self.retrieval_alpha * pos_attended + 
                                (1 - self.retrieval_alpha) * neg_attended + 
                                audio_seq_transformed)
                # Normalize to prevent scale explosion
                attended_audio = self.audio_combination_norm(attended_audio)
            else:
                attended_audio = audio_seq_transformed
                
            # Average pooling and normalize
            final_audio_features = attended_audio.mean(dim=1)  # (B, hid_dim)
            final_audio_features = self.final_feature_norm(final_audio_features)
        
        # Visual retrieval attention
        if 'visual' in ffn_outputs:
            visual_seq_transformed = ffn_outputs['visual']
            
            # Apply positive and negative attention if retrieval data available
            if 'positive_visual_features' in inputs and 'negative_visual_features' in inputs:
                # Process retrieved features through projection and FFN (same as current sample)
                pos_visual_proj = self.visual_proj(inputs['positive_visual_features'])  # (B, seq_len, hid_dim)
                neg_visual_proj = self.visual_proj(inputs['negative_visual_features'])  # (B, seq_len, hid_dim)
                
                pos_visual_features = self.visual_ffn(pos_visual_proj, mask=None)  # Apply FFN to retrieved features
                neg_visual_features = self.visual_ffn(neg_visual_proj, mask=None)  # Apply FFN to retrieved features
                
                pos_attended = self.visual_pos_attention(visual_seq_transformed, pos_visual_features, pos_visual_features, key_padding_mask=None)
                neg_attended = self.visual_neg_attention(visual_seq_transformed, neg_visual_features, neg_visual_features, key_padding_mask=None)
                
                # Weighted combination
                attended_visual = (self.retrieval_alpha * pos_attended + 
                                 (1 - self.retrieval_alpha) * neg_attended + 
                                 visual_seq_transformed)
                # Normalize to prevent scale explosion
                attended_visual = self.visual_combination_norm(attended_visual)
            else:
                attended_visual = visual_seq_transformed
                
            # Average pooling and normalize
            final_visual_features = attended_visual.mean(dim=1)  # (B, hid_dim)
            final_visual_features = self.final_feature_norm(final_visual_features)
        
        # Apply evidential classification
        evidential_outputs = self.evidential_classifier(
            final_text_features if use_text else None,
            final_audio_features if use_audio else None, 
            final_visual_features if use_image else None
        )
        
        return evidential_outputs
    
    def compute_losses(self, outputs, labels, global_step=0):
        """Compute evidential losses"""
        return self.evidential_classifier.compute_losses(
            outputs, labels, global_step, self.anneal_steps, self.loss_weights
        )
