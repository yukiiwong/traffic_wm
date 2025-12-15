"""
Decoder for World Model (MLP + optional (x,y) residual head)

Outputs:
- states: [B, T, K, F]  (absolute state predictions in normalized space)
- existence_logits: [B, T, K]  (logits; use sigmoid to get probabilities)
- residual_xy: [B, T, K, 2] (optional; small residual for (x,y) to be added on top of a kinematic prior)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class StateDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 12,
        continuous_dim: int = 9,  # Number of continuous features to output
        max_agents: int = 50,
        dropout: float = 0.1,
        enable_xy_residual: bool = True,
        enable_angle_head: bool = True,  # Separate head for angle (periodic, non-normalized)
        binary_feature_indices: list = None,  # Indices of binary features (for sigmoid)
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # Full dimension (for reference)
        self.continuous_dim = continuous_dim  # Only continuous features
        self.max_agents = max_agents
        self.enable_xy_residual = enable_xy_residual
        self.enable_angle_head = enable_angle_head
        # Binary features: has_preceding (6), has_following (7) in continuous output (9 features)
        # After removing discrete (class_id, lane_id, site_id) and angle from full 12 features
        self.binary_feature_indices = binary_feature_indices if binary_feature_indices is not None else [6, 7]

        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Absolute state head - ONLY outputs continuous features
        self.state_head = nn.Linear(hidden_dim, max_agents * continuous_dim)

        # Existence logits head
        self.existence_head = nn.Linear(hidden_dim, max_agents)

        # Residual head for x,y (optional)
        if enable_xy_residual:
            self.residual_xy_head = nn.Linear(hidden_dim, max_agents * 2)
            # Start from pure kinematic prior: residual ≈ 0 at init
            nn.init.zeros_(self.residual_xy_head.weight)
            nn.init.zeros_(self.residual_xy_head.bias)
        else:
            self.residual_xy_head = None

        # Angle head (periodic, outputs raw radians, not normalized)
        if enable_angle_head:
            self.angle_head = nn.Linear(hidden_dim, max_agents)
            # Initialize with small weights for stability
            nn.init.normal_(self.angle_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.angle_head.bias)
        else:
            self.angle_head = None

    def forward(
        self,
        latent: torch.Tensor,
        return_residual_xy: bool = False,
        return_angle: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            latent: [B, T, D]
            return_residual_xy: whether to return residual_xy
            return_angle: whether to return angle predictions

        Returns:
            states: [B, T, K, F_cont] (continuous features excluding angle)
            existence_logits: [B, T, K]
            residual_xy: [B, T, K, 2] or None
            angle: [B, T, K] or None (raw radians, not normalized)
        """
        if latent.dim() != 3:
            raise ValueError(f"latent must be [B, T, D], got {tuple(latent.shape)}")

        B, T, D = latent.shape
        if D != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got D={D}")

        h = self.backbone(latent)  # [B, T, H]

        # Output only continuous features (excluding angle)
        states = self.state_head(h).view(B, T, self.max_agents, self.continuous_dim)  # [B, T, K, F_cont]
        
        # Apply sigmoid to binary features (has_preceding, has_following)
        if self.binary_feature_indices:
            for idx in self.binary_feature_indices:
                if 0 <= idx < self.continuous_dim:
                    states[..., idx] = torch.sigmoid(states[..., idx])
        
        existence_logits = self.existence_head(h)  # [B, T, K]

        residual_xy = None
        if return_residual_xy:
            if not self.enable_xy_residual or self.residual_xy_head is None:
                raise ValueError("return_residual_xy=True but enable_xy_residual=False")
            residual_xy = self.residual_xy_head(h).view(B, T, self.max_agents, 2)  # [B,T,K,2]

        angle = None
        if return_angle:
            if not self.enable_angle_head or self.angle_head is None:
                raise ValueError("return_angle=True but enable_angle_head=False")
            angle = self.angle_head(h).view(B, T, self.max_agents)  # [B,T,K] raw radians

        return states, existence_logits, residual_xy, angle
