import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from loguru import logger
from typing import Dict, Tuple


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
# Utilities: evidential math
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
    Negative expected log-likelihood for class y under Dir(alpha):
      NLL = -(E[log p_y]) = -(psi(alpha_y) - psi(sum alpha))
    alpha: (B,K), y: (B,) long
    returns: (B,)
    """
    eps = 1e-8
    S = torch.clamp(alpha.sum(dim=-1), min=eps)                      # (B,)
    alpha_y = alpha.gather(dim=-1, index=y.view(-1,1)).squeeze(-1)   # (B,)
    return -(torch.digamma(alpha_y) - torch.digamma(S))


def kl_dirichlet_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL(Dir(alpha) || Dir(1)), where Dir(1) is the uniform Dirichlet.
    Closed form:
      KL = logGamma(S) - sum logGamma(alpha_k) - logGamma(K)
           + sum (alpha_k - 1) * (psi(alpha_k) - psi(S))
    returns: (B,)
    """
    eps = 1e-8
    K = alpha.size(-1)
    S = torch.clamp(alpha.sum(dim=-1), min=eps)  # (B,)
    term1 = torch.lgamma(S)
    term2 = torch.lgamma(alpha).sum(dim=-1)
    term3 = torch.lgamma(torch.tensor(K, dtype=alpha.dtype, device=alpha.device))
    term4 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(-1)))).sum(dim=-1)
    return term1 - term2 - term3 + term4


def anneal_factor(global_step: int, anneal_steps: int = 1000) -> float:
    """Linear ramp of KL weight from 0→1."""
    return float(min(1.0, max(0.0, global_step / float(max(1, anneal_steps)))))


# -------------------------
# Model components
# -------------------------

class BERT_FT(nn.Module):
    def __init__(self, text_encoder, num_frozen_layers):
        super(BERT_FT, self).__init__()
        self.bert = AutoModel.from_pretrained(text_encoder).text_model

        for name, param in self.bert.named_parameters():
            if 'embeddings' in name or 'final_layer_norm' in name:
                param.requires_grad = False
            elif 'encoder.layer' in name:
                layer_idx = int(name.split('.')[2])
                if layer_idx >= num_frozen_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, **inputs):
        return self.bert(**inputs)


class EvidenceHead(nn.Module):
    def __init__(self, d_in: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e = self.softplus(self.net(x))              # (B,K) evidence >= 0
        alpha = e + 1.0                             # (B,K) Dirichlet params >= 1
        return {"e": e, "alpha": alpha}


class MultiViewEvidentialClassifier(nn.Module):
    def __init__(self, d: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.K = num_classes
        self.text_head  = EvidenceHead(d, num_classes, hidden, dropout)
        self.audio_head = EvidenceHead(d, num_classes, hidden, dropout)
        self.image_head = EvidenceHead(d, num_classes, hidden, dropout)

    @torch.no_grad()
    def _fuse_opinions(self, alpha_text, alpha_audio, alpha_image) -> Dict[str, torch.Tensor]:
        # Map each to opinions (b,u)
        b_t, u_t = alpha_to_belief_u(alpha_text)
        b_a, u_a = alpha_to_belief_u(alpha_audio)
        b_i, u_i = alpha_to_belief_u(alpha_image)

        # DST fold: ((text ⨝ audio) ⨝ image)
        b_ta, u_ta = combine_two_opinions(b_t, u_t, b_a, u_a)
        b_f,  u_f  = combine_two_opinions(b_ta, u_ta, b_i, u_i)

        # Opinion -> Dirichlet
        alpha_f = opinion_to_alpha(b_f, u_f)  # (B,K)
        return {"b_f": b_f, "u_f": u_f, "alpha_f": alpha_f}

    def forward(self, x_text: torch.Tensor = None, x_audio: torch.Tensor = None, x_image: torch.Tensor = None) -> Dict[str, torch.Tensor]:
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
        
        # Process each modality or create dummy alpha
        if x_text is not None:
            out_t = self.text_head(x_text)
        else:
            # Create neutral/uniform alpha for missing text modality
            out_t = torch.ones(batch_size, self.K, device=device)
            
        if x_audio is not None:
            out_a = self.audio_head(x_audio)
        else:
            # Create neutral/uniform alpha for missing audio modality
            out_a = torch.ones(batch_size, self.K, device=device)
            
        if x_image is not None:
            out_i = self.image_head(x_image)
        else:
            # Create neutral/uniform alpha for missing image modality
            out_i = torch.ones(batch_size, self.K, device=device)

        # Convert to alpha format if needed
        alpha_t = out_t if isinstance(out_t, torch.Tensor) else out_t["alpha"]
        alpha_a = out_a if isinstance(out_a, torch.Tensor) else out_a["alpha"]  
        alpha_i = out_i if isinstance(out_i, torch.Tensor) else out_i["alpha"]

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
# Main model
# -------------------------

class ExMRD_Evidential(nn.Module):
    def __init__(self, hid_dim, dropout, text_encoder, num_frozen_layers=4, 
                 evidential_hidden=256, evidential_dropout=0.1, 
                 loss_weights=None, anneal_steps=1000, use_evidential=True):
        super(ExMRD_Evidential, self).__init__()
        self.name = 'ExMRD_Evidential'
        self.text_encoder = text_encoder
        self.hid_dim = hid_dim
        self.anneal_steps = anneal_steps
        self.use_evidential = use_evidential
        
        # Text encoder (Chinese CLIP BERT)
        self.bert = BERT_FT(text_encoder, num_frozen_layers)
        print_model_params(self.bert)
        
        # Feature projection layers
        self.text_proj = nn.Linear(768, hid_dim)  # Chinese CLIP BERT hidden size is 768
        self.audio_proj = nn.LazyLinear(hid_dim)  # Audio features dimension detected dynamically
        self.visual_proj = nn.Linear(1024, hid_dim)  # ViT features are 1024-dim
        
        if self.use_evidential:
            # Evidential classifier
            self.evidential_classifier = MultiViewEvidentialClassifier(
                d=hid_dim, 
                num_classes=2, 
                hidden=evidential_hidden,
                dropout=evidential_dropout
            )
            # Loss weights for different modalities
            self.loss_weights = loss_weights or {"fused": 1.0, "text": 0.5, "audio": 0.5, "image": 0.5}
            logger.info(f"Evidential loss weights: {self.loss_weights}")
        else:
            # Traditional concatenation + classifier
            self.concat_classifier = nn.Sequential(
                nn.Linear(hid_dim * 3, evidential_hidden),  # Concatenate 3 modalities
                nn.GELU(),
                nn.Dropout(evidential_dropout),
                nn.Linear(evidential_hidden, 2)
            )
        
        logger.info(f"Initialized ExMRD_Evidential with hidden_dim={hid_dim}, use_evidential={self.use_evidential}")

    def forward(self, **inputs):
        """
        Forward pass with optional modalities:
        - entity_text_input: tokenized entity claims text (if use_text=True)
        - audio_features: (B, seq_len, audio_dim) audio features (if use_audio=True)
        - visual_features: (B, 16, 1024) visual features (if use_image=True)
        """
        # Get modality flags
        use_text = inputs.get('use_text', True)
        use_image = inputs.get('use_image', True)
        use_audio = inputs.get('use_audio', True)
        
        # Initialize features as None
        text_features = None
        audio_features = None
        visual_features = None
        
        # Text encoding if enabled
        if use_text and 'entity_text_input' in inputs:
            entity_text_input = inputs['entity_text_input']
            if 'chinese' in self.text_encoder:
                text_features = self.bert(**entity_text_input)['last_hidden_state'][:, 0, :]  # CLS token
            else:
                text_features = self.bert(**entity_text_input)['pooler_output']
            text_features = self.text_proj(text_features)  # (B, hid_dim)
        
        # Audio encoding if enabled
        if use_audio and 'audio_features' in inputs:
            audio_feats = inputs['audio_features']  # (B, seq_len, audio_dim)
            audio_feats = torch.mean(audio_feats, dim=1)  # (B, audio_dim)
            audio_features = self.audio_proj(audio_feats)  # (B, hid_dim)
        
        # Visual encoding if enabled
        if use_image and 'visual_features' in inputs:
            visual_feats = inputs['visual_features']  # (B, 16, 1024)
            visual_feats = torch.mean(visual_feats, dim=1)  # (B, 1024)
            visual_features = self.visual_proj(visual_feats)  # (B, hid_dim)
        
        # Create gate_x by concatenating available modality features
        # For missing modalities, use zero vectors of appropriate size
        batch_size = None
        device = None
        
        # Determine batch size and device from available features
        for feat in [text_features, audio_features, visual_features]:
            if feat is not None:
                batch_size = feat.shape[0]
                device = feat.device
                break
        
        if batch_size is None:
            raise ValueError("At least one modality must be provided")
        
        # Create feature representations for gate_x (use zero padding for missing modalities)
        if text_features is None:
            text_features = torch.zeros(batch_size, self.hid_dim, device=device)
        if audio_features is None:
            audio_features = torch.zeros(batch_size, self.hid_dim, device=device)
        if visual_features is None:
            visual_features = torch.zeros(batch_size, self.hid_dim, device=device)
        
        # Concatenate all modalities for gate_x (B, 3*hid_dim)
        gate_x = torch.cat([text_features, audio_features, visual_features], dim=1)
        
        if self.use_evidential:
            # Evidential classification
            evidential_outputs = self.evidential_classifier(
                text_features if use_text else None,
                audio_features if use_audio else None, 
                visual_features if use_image else None
            )
            # Add gate_x to the output for defer mechanism
            evidential_outputs['gate_x'] = gate_x
            return evidential_outputs
        else:
            # Traditional concatenation + classification
            logits = self.concat_classifier(gate_x)  # Use gate_x for consistency
            return {
                'logits': logits,
                'gate_x': gate_x
            }
    
    def compute_losses(self, outputs, labels, global_step=0):
        """Compute evidential or traditional losses"""
        if self.use_evidential:
            return self.evidential_classifier.compute_losses(
                outputs, labels, global_step, self.anneal_steps, self.loss_weights
            )
        else:
            # Traditional cross-entropy loss
            import torch.nn.functional as F
            loss = F.cross_entropy(outputs, labels)
            return {
                "loss_total": loss,
                "loss_concat": loss
            }