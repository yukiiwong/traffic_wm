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
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim  # Full dimension (for reference)
        self.continuous_dim = continuous_dim  # Only continuous features
        self.max_agents = max_agents
        self.enable_xy_residual = enable_xy_residual

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

    def forward(
        self,
        latent: torch.Tensor,
        return_residual_xy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            latent: [B, T, D]
            return_residual_xy: whether to return residual_xy

        Returns:
            states: [B, T, K, F]
            existence_logits: [B, T, K]
            residual_xy: [B, T, K, 2] or None
        """
        if latent.dim() != 3:
            raise ValueError(f"latent must be [B, T, D], got {tuple(latent.shape)}")

        B, T, D = latent.shape
        if D != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got D={D}")

        h = self.backbone(latent)  # [B, T, H]

        # Output only continuous features
        states = self.state_head(h).view(B, T, self.max_agents, self.continuous_dim)  # [B, T, K, F_cont]
        existence_logits = self.existence_head(h)  # [B, T, K]

        residual_xy = None
        if return_residual_xy:
            if not self.enable_xy_residual or self.residual_xy_head is None:
                raise ValueError("return_residual_xy=True but enable_xy_residual=False")
            residual_xy = self.residual_xy_head(h).view(B, T, self.max_agents, 2)  # [B,T,K,2]

        return states, existence_logits, residual_xy
