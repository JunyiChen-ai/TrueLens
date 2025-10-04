"""
Learn-to-Defer Components for ExMRD_Evidential

Implements the basic end-to-end differentiable learn-to-defer setup (NeurIPS'18)
for binary fake-news detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from loguru import logger


class GateNet(nn.Module):
    """
    Small MLP that outputs defer probability π(x) ∈ (0,1).
    Takes concatenation [p_slm, gate_x] as input.
    """
    
    def __init__(self, p_slm_dim: int = 2, gate_x_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super(GateNet, self).__init__()
        
        input_dim = p_slm_dim + gate_x_dim  # 2 + 3*256 = 770 by default
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output π ∈ (0,1)
        )
        
        logger.info(f"GateNet initialized: input_dim={input_dim}, hidden_dim={hidden_dim}, dropout={dropout}")
    
    def forward(self, p_slm: torch.Tensor, gate_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p_slm: (B, 2) SLM predictive probabilities
            gate_x: (B, D) concatenated modality features from SLM
        
        Returns:
            pi: (B, 1) defer probabilities
        """
        # Concatenate inputs
        gate_input = torch.cat([p_slm, gate_x], dim=1)  # (B, 2+D)
        
        # Pass through MLP
        pi = self.net(gate_input)  # (B, 1)
        
        # Clamp to avoid numerical issues at boundaries
        pi = torch.clamp(pi, min=1e-6, max=1.0 - 1e-6)
        
        return pi


class DeferWrapper(nn.Module):
    """
    Wrapper that orchestrates SLM, Gate, and DM probabilities.
    Computes the combined defer loss: L_defer + L_evid.
    """
    
    def __init__(self, slm_model, gate_net, gate_x_dim: int = 768):
        super(DeferWrapper, self).__init__()
        
        self.slm_model = slm_model
        self.gate_net = gate_net
        self.gate_x_dim = gate_x_dim
        
        logger.info(f"DeferWrapper initialized with gate_x_dim={gate_x_dim}")
    
    def forward(self, p_dm: torch.Tensor, **slm_inputs) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SLM and Gate.
        
        Args:
            p_dm: (B, 2) LLM decision maker probabilities (label smoothed)
            **slm_inputs: inputs to the SLM model
            
        Returns:
            dict with:
                - p_slm: (B, 2) SLM predictive probabilities  
                - gate_x: (B, D) SLM intermediate features
                - pi: (B, 1) defer probabilities
                - p_dm: (B, 2) DM probabilities (passed through)
                - all other SLM outputs
        """
        # Run SLM forward pass
        slm_outputs = self.slm_model(**slm_inputs)
        
        # Extract SLM probabilities and gate features
        if 'p_fused' in slm_outputs:
            # Evidential model: use fused probabilities
            p_slm = slm_outputs['p_fused']  # (B, 2)
        elif 'logits' in slm_outputs:
            # Traditional model: convert logits to probabilities
            p_slm = F.softmax(slm_outputs['logits'], dim=1)  # (B, 2)
        else:
            raise ValueError("SLM outputs must contain either 'p_fused' or 'logits'")
        
        gate_x = slm_outputs['gate_x']  # (B, D)
        
        # Compute defer probability
        pi = self.gate_net(p_slm, gate_x)  # (B, 1)
        
        # Add defer-specific outputs to SLM outputs
        outputs = slm_outputs.copy() if isinstance(slm_outputs, dict) else {}
        outputs.update({
            'p_slm': p_slm,
            'pi': pi,
            'p_dm': p_dm,
            'gate_x': gate_x
        })
        
        return outputs
    
    def compute_losses(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, 
                      global_step: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute combined defer + evidential losses.
        
        Args:
            outputs: forward pass outputs
            labels: (B,) ground truth labels [0, 1]
            global_step: for KL annealing
            
        Returns:
            dict with individual and total losses
        """
        p_slm = outputs['p_slm']  # (B, 2)
        p_dm = outputs['p_dm']    # (B, 2)
        pi = outputs['pi']        # (B, 1)
        
        # Cross-entropy losses (binary CE on log-probabilities)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        p_slm_safe = torch.clamp(p_slm, min=eps, max=1.0-eps)
        p_dm_safe = torch.clamp(p_dm, min=eps, max=1.0-eps)
        
        # Manual cross-entropy computation for more control
        log_p_slm = torch.log(p_slm_safe)  # (B, 2)
        log_p_dm = torch.log(p_dm_safe)    # (B, 2)
        
        # CE_slm and CE_dm: (B,)
        CE_slm = F.nll_loss(log_p_slm, labels, reduction='none')  # (B,)
        CE_dm = F.nll_loss(log_p_dm, labels, reduction='none')    # (B,)
        
        # Defer loss: L_defer = (1-π) * CE_slm + π * CE_dm
        pi_flat = pi.squeeze(1)  # (B,) squeeze out the last dimension
        L_defer = (1.0 - pi_flat) * CE_slm + pi_flat * CE_dm  # (B,)
        
        L_defer = L_defer.mean()  # scalar
        
        # Evidential loss (if applicable)
        L_evid = torch.tensor(0.0, device=labels.device)
        evid_losses = {}
        
        if hasattr(self.slm_model, 'use_evidential') and self.slm_model.use_evidential:
            # Compute evidential losses using the SLM model's method
            evid_losses = self.slm_model.compute_losses(outputs, labels, global_step)
            L_evid = evid_losses['loss_total']
        
        # Total loss: L_total = L_defer + L_evid
        L_total = L_defer + L_evid
        # L_defer = (1.0 - pi_flat) * L_evid + pi_flat * CE_dm  # (B,)
        # L_defer = L_defer.mean()  # scalar
        # L_total = L_defer 
        # Prepare output dictionary
        loss_dict = {
            'loss_total': L_total,
            'loss_defer': L_defer,
            'loss_evid': L_evid,
            'ce_slm': CE_slm.mean(),
            'ce_dm': CE_dm.mean(),
            'pi_mean': pi_flat.mean(),  # Average defer probability
            'pi_std': pi_flat.std(),    # Defer probability standard deviation
        }
        
        # Add evidential losses if available
        if evid_losses:
            for key, value in evid_losses.items():
                if key != 'loss_total':  # Avoid overwriting
                    loss_dict[f'evid_{key}'] = value
        
        return loss_dict
    
    def predict(self, p_dm: torch.Tensor, threshold: float = 0.5, **slm_inputs) -> Dict[str, torch.Tensor]:
        """
        Inference with threshold-based deferral.
        
        Args:
            p_dm: (B, 2) LLM decision maker probabilities
            threshold: defer threshold τ
            **slm_inputs: inputs to SLM
            
        Returns:
            dict with:
                - final_probs: (B, 2) final prediction probabilities
                - defer_decisions: (B,) boolean defer decisions
                - defer_rate: scalar defer rate
        """
        # Forward pass
        outputs = self.forward(p_dm, **slm_inputs)
        
        p_slm = outputs['p_slm']  # (B, 2)
        pi = outputs['pi'].squeeze(1)  # (B,)
        
        # Defer decisions: defer if π > τ
        defer_decisions = pi > threshold  # (B,) boolean
        defer_rate = defer_decisions.float().mean()  # scalar
        
        # Final predictions: use DM if defer, else use SLM
        final_probs = torch.where(
            defer_decisions.unsqueeze(1),  # (B, 1)
            p_dm,    # Use DM probabilities
            p_slm    # Use SLM probabilities
        )
        
        return {
            'final_probs': final_probs,
            'defer_decisions': defer_decisions,
            'defer_rate': defer_rate,
            'p_slm': p_slm,
            'p_dm': p_dm,
            'pi': pi
        }