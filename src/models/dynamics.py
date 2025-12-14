"""
Latent Dynamics Model (Transformer-only)

This module models temporal evolution in latent space.

Key changes vs. old version:
- Transformer-only (no GRU/LSTM/RSSM).
- Causal masking: output[t] can only attend to <= t.
- Positional encoding for temporal order.
- Optional time padding mask to ignore padded timesteps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LatentDynamics(nn.Module):
    """
    Temporal dynamics model in latent space.

    Input:
        latent: [B, T, D]  (encoded latents)
    Output:
        predicted_latent: [B, T, D]  where predicted_latent[:, t] predicts latent at time t+1
                                     (the last timestep is usually ignored in the one-step loss).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 512,
        use_learned_pos_emb: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.use_learned_pos_emb = use_learned_pos_emb

        if use_learned_pos_emb:
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len, latent_dim))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        else:
            self.register_buffer("pos_emb", self._build_sinusoidal_pos_emb(max_len, latent_dim), persistent=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Small output projection (helps stability / capacity)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )

    @staticmethod
    def _build_sinusoidal_pos_emb(max_len: int, d_model: int) -> torch.Tensor:
        """[1, max_len, d_model] sinusoidal position encoding."""
        position = torch.arange(max_len).float().unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    @staticmethod
    def _causal_mask(T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Causal mask for TransformerEncoder: shape [T, T].
        Masked positions are -inf (additive mask).
        """
        # Upper triangular (excluding diagonal) should be masked
        mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        latent: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None,  # kept for API compatibility
        time_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            latent: [B, T, D]
            time_padding_mask: [B, T] with True for padded timesteps (ignored by attention)

        Returns:
            predicted_latent: [B, T, D]
            None: (no hidden state for transformer)
        """
        del hidden  # not used

        if latent.dim() != 3:
            raise ValueError(f"latent must be [B, T, D], got {tuple(latent.shape)}")

        B, T, D = latent.shape
        if D != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got D={D}")
        if T > self.max_len:
            raise ValueError(f"T={T} exceeds max_len={self.max_len}. Increase max_len.")

        x = latent + self.pos_emb[:, :T, :].to(latent.dtype)

        causal = self._causal_mask(T, device=latent.device, dtype=latent.dtype)

        out = self.transformer(
            x,
            mask=causal,
            src_key_padding_mask=time_padding_mask,
        )  # [B, T, D]

        out = self.output_proj(out)  # [B, T, D]
        return out, None

    @torch.no_grad()
    def step(
        self,
        latent_history: torch.Tensor,
        time_padding_mask: Optional[torch.Tensor] = None,
        max_context: Optional[int] = None,
    ) -> torch.Tensor:
        """
        One-step prediction using full (or truncated) history.

        Args:
            latent_history: [B, T, D] latents up to current time t (inclusive)
            time_padding_mask: [B, T] optional padding mask
            max_context: if set, only use last max_context steps for efficiency

        Returns:
            next_latent: [B, D] predicted latent for time t+1
        """
        if latent_history.dim() != 3:
            raise ValueError(f"latent_history must be [B, T, D], got {tuple(latent_history.shape)}")

        if max_context is not None and latent_history.size(1) > max_context:
            latent_history = latent_history[:, -max_context:, :]
            if time_padding_mask is not None:
                time_padding_mask = time_padding_mask[:, -max_context:]

        pred, _ = self.forward(latent_history, time_padding_mask=time_padding_mask)
        return pred[:, -1, :]  # output at last token predicts next latent
