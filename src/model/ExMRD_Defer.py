"""
ExMRD_Defer: Learn-to-defer version of ExMRD_Evidential

This model wraps the ExMRD_Evidential model with a gating mechanism
that can defer to an LLM (frozen decision maker).
"""

import torch
import torch.nn as nn
from loguru import logger
from .ExMRD_Evidential import ExMRD_Evidential
from .defer_components import GateNet, DeferWrapper


class ExMRD_Defer(nn.Module):
    """
    Learn-to-defer wrapper around ExMRD_Evidential.
    
    Combines:
    - SLM: ExMRD_Evidential (with KL regularization)
    - Gate: Small MLP outputting defer probability
    - DM: Frozen LLM predictions (loaded from data)
    """
    
    def __init__(self, 
                 # ExMRD_Evidential parameters
                 hid_dim=256, dropout=0.3, text_encoder="OFA-Sys/chinese-clip-vit-large-patch14",
                 num_frozen_layers=4, evidential_hidden=256, evidential_dropout=0.1,
                 loss_weights=None, anneal_steps=1000, use_evidential=True,
                 # Defer-specific parameters
                 gate_hidden_dim=256, gate_dropout=0.1,
                 defer_threshold=0.5, label_smoothing_epsilon=0.01,
                 **kwargs):
        super(ExMRD_Defer, self).__init__()
        
        self.name = 'ExMRD_Defer'
        self.defer_threshold = defer_threshold
        self.epsilon = label_smoothing_epsilon
        
        # Initialize SLM (ExMRD_Evidential)
        self.slm_model = ExMRD_Evidential(
            hid_dim=hid_dim, dropout=dropout, text_encoder=text_encoder,
            num_frozen_layers=num_frozen_layers, evidential_hidden=evidential_hidden,
            evidential_dropout=evidential_dropout, loss_weights=loss_weights,
            anneal_steps=anneal_steps, use_evidential=use_evidential, **kwargs
        )
        
        # Initialize GateNet
        gate_x_dim = hid_dim * 3  # 3 modalities concatenated
        self.gate_net = GateNet(
            p_slm_dim=2,  # Binary classification
            gate_x_dim=gate_x_dim,
            hidden_dim=gate_hidden_dim,
            dropout=gate_dropout
        )
        
        # Initialize DeferWrapper
        self.defer_wrapper = DeferWrapper(
            slm_model=self.slm_model,
            gate_net=self.gate_net,
            gate_x_dim=gate_x_dim
        )
        
        logger.info(f"ExMRD_Defer initialized:")
        logger.info(f"  - SLM: ExMRD_Evidential (hid_dim={hid_dim}, use_evidential={use_evidential})")
        logger.info(f"  - Gate: hidden_dim={gate_hidden_dim}, dropout={gate_dropout}")
        logger.info(f"  - Defer threshold: {defer_threshold}")
        logger.info(f"  - Label smoothing ε: {label_smoothing_epsilon}")
    
    def forward(self, p_dm=None, **inputs):
        """
        Forward pass through defer wrapper.
        
        Args:
            p_dm: (B, 2) LLM decision maker probabilities (from data loader)
            **inputs: SLM inputs (modality features, etc.)
        
        Returns:
            Dict with SLM outputs + defer-specific outputs
        """
        if p_dm is None:
            raise ValueError("p_dm (LLM probabilities) must be provided for defer model")
        
        return self.defer_wrapper(p_dm, **inputs)
    
    def compute_losses(self, outputs, labels, global_step=0):
        """
        Compute combined defer + evidential losses.
        """
        return self.defer_wrapper.compute_losses(outputs, labels, global_step)
    
    def predict(self, p_dm, threshold=None, **inputs):
        """
        Inference with threshold-based deferral.
        
        Args:
            p_dm: (B, 2) LLM probabilities
            threshold: defer threshold (uses self.defer_threshold if None)
            **inputs: SLM inputs
        """
        if threshold is None:
            threshold = self.defer_threshold
        
        return self.defer_wrapper.predict(p_dm, threshold, **inputs)
    
    def get_slm_model(self):
        """Get the underlying SLM model for evaluation/analysis."""
        return self.slm_model
    
    def get_gate_net(self):
        """Get the gate network for analysis."""
        return self.gate_net
    
    def set_defer_threshold(self, threshold):
        """Update the defer threshold."""
        self.defer_threshold = threshold
        logger.info(f"Updated defer threshold to {threshold}")