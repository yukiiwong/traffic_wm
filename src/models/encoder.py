"""
Multi-Agent Encoder (per-frame agent interaction encoder)

- Projects continuous features with an MLP
- Embeds discrete IDs (lane_id / class_id / site_id) with embeddings
- Applies a TransformerEncoder over agents for each time step
- Pools over agents (masked mean) to produce a scene latent per time step: [B, T, D]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional


class MultiAgentEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        max_agents: int = 50,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        # discrete feature indices (must match your dataset feature order)
        lane_feature_idx: int = 8,
        class_feature_idx: int = 9,
        site_feature_idx: int = 11,
        # embedding configs
        num_lanes: int = 100,
        num_classes: int = 10,
        num_sites: int = 10,
        use_lane_embedding: bool = True,
        use_class_embedding: bool = True,
        use_site_id: bool = True,
        lane_embed_dim: int = 16,
        class_embed_dim: int = 8,
        site_embed_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_agents = max_agents

        self.lane_feature_idx = lane_feature_idx
        self.class_feature_idx = class_feature_idx
        self.site_feature_idx = site_feature_idx

        self.use_lane_embedding = use_lane_embedding
        self.use_class_embedding = use_class_embedding
        self.use_site_id = use_site_id

        self.num_lanes = num_lanes
        self.num_classes = num_classes
        self.num_sites = num_sites

        self.lane_embed_dim = lane_embed_dim if use_lane_embedding else 0
        self.class_embed_dim = class_embed_dim if use_class_embedding else 0
        self.site_embed_dim = site_embed_dim if use_site_id else 0

        # Identify continuous feature indices by excluding discrete ones
        discrete = {lane_feature_idx, class_feature_idx, site_feature_idx}
        self.continuous_indices: List[int] = [i for i in range(input_dim) if i not in discrete]
        self.n_cont = len(self.continuous_indices)

        self.continuous_projector = nn.Sequential(
            nn.Linear(self.n_cont, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if use_lane_embedding:
            self.lane_embedding = nn.Embedding(num_lanes, lane_embed_dim)
        else:
            self.lane_embedding = None

        if use_class_embedding:
            self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        else:
            self.class_embedding = None

        if use_site_id:
            self.site_embedding = nn.Embedding(num_sites, site_embed_dim)
        else:
            self.site_embedding = None

        fused_dim = hidden_dim + self.lane_embed_dim + self.class_embed_dim + self.site_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.agent_transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [B, T, K, F] (float, contains normalized continuous + discrete as float)
            masks:  [B, T, K]    (0/1)

        Returns:
            latent: [B, T, D]
        """
        if states.dim() != 4:
            raise ValueError(f"states must be [B,T,K,F], got {tuple(states.shape)}")
        if masks.dim() != 3:
            raise ValueError(f"masks must be [B,T,K], got {tuple(masks.shape)}")

        B, T, K, F = states.shape
        if K != self.max_agents:
            raise ValueError(f"Expected K={self.max_agents}, got {K}")
        if F != self.input_dim:
            raise ValueError(f"Expected F={self.input_dim}, got {F}")

        # Flatten B and T to process each frame: [B*T, K, F]
        states_flat = states.reshape(B * T, K, F)
        masks_flat = masks.reshape(B * T, K)  # [B*T, K]
        pad = (masks_flat == 0)

        # Continuous features
        cont = states_flat[..., self.continuous_indices]  # [B*T, K, n_cont]
        cont_emb = self.continuous_projector(cont)  # [B*T, K, H]

        embeddings = [cont_emb]

        # Discrete embeddings (make safe)
        if self.use_lane_embedding:
            lane_ids = states_flat[..., self.lane_feature_idx].long()
            lane_ids = lane_ids.clamp(min=0, max=self.num_lanes - 1)
            lane_ids = lane_ids.masked_fill(pad, 0)
            embeddings.append(self.lane_embedding(lane_ids))

        if self.use_class_embedding:
            class_ids = states_flat[..., self.class_feature_idx].long()
            class_ids = class_ids.clamp(min=0, max=self.num_classes - 1)
            class_ids = class_ids.masked_fill(pad, 0)
            embeddings.append(self.class_embedding(class_ids))

        if self.use_site_id:
            site_ids = states_flat[..., self.site_feature_idx].long()
            site_ids = site_ids.clamp(min=0, max=self.num_sites - 1)
            site_ids = site_ids.masked_fill(pad, 0)
            embeddings.append(self.site_embedding(site_ids))

        agent_feats = torch.cat(embeddings, dim=-1)  # [B*T, K, fused_dim]
        agent_feats = self.fusion(agent_feats)       # [B*T, K, H]

        # Agent transformer (per-frame). Mask padded agents.
        # src_key_padding_mask: True for positions to ignore
        src_key_padding_mask = pad  # [B*T, K] bool
        agent_feats = self.agent_transformer(agent_feats, src_key_padding_mask=src_key_padding_mask)

        # Masked mean pool over agents to scene embedding per frame
        weights = masks_flat.unsqueeze(-1)  # [B*T, K, 1]
        pooled = (agent_feats * weights).sum(dim=1) / (weights.sum(dim=1).clamp(min=1e-6))  # [B*T, H]

        latent = self.to_latent(pooled).view(B, T, self.latent_dim)  # [B,T,D]
        return latent
