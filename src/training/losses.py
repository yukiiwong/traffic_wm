"""
Losses for World Model

- Reconstruction loss: Huber (SmoothL1) on continuous features only
- Prediction loss:     Huber on continuous features only (one-step ahead)
- Existence loss:      BCEWithLogits on existence logits (optional for prediction branch too)

Note:
Discrete IDs (lane_id, class_id, site_id) should NOT be included in regression losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, List


class WorldModelLoss(nn.Module):
    def __init__(
        self,
        recon_weight: float = 1.0,
        pred_weight: float = 1.0,
        exist_weight: float = 0.1,
        huber_beta: float = 1.0,
        continuous_indices: Optional[List[int]] = None,
        use_pred_existence_loss: bool = True,
    ):
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.pred_weight = float(pred_weight)
        self.exist_weight = float(exist_weight)
        self.huber_beta = float(huber_beta)
        self.continuous_indices = continuous_indices
        self.use_pred_existence_loss = bool(use_pred_existence_loss)

        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")

    def _masked_huber_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, T, K, F_cont] - predictions (already continuous-only from decoder)
        target: [B, T, K, F_full] - targets (full features, needs filtering)
        mask:   [B, T, K]  (0/1)
        """
        # Target needs filtering to continuous features only
        # Pred is already continuous-only from decoder (no filtering needed)
        if self.continuous_indices is not None:
            target = target[..., self.continuous_indices]

        # SmoothL1 in "beta" form (PyTorch SmoothL1Loss uses beta)
        diff = pred - target
        abs_diff = diff.abs()
        beta = self.huber_beta
        loss = torch.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)

        # apply mask
        loss = loss * mask.unsqueeze(-1)
        denom = mask.sum() * loss.shape[-1]
        return loss.sum() / (denom.clamp(min=1.0))

    def _existence_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, T, K]
        mask:   [B, T, K] (0/1)  (targets: 1 if agent exists)
        """
        loss = self.bce_logits(logits, mask)
        return loss.mean()

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        targets must include:
          - states: [B,T,K,F_full] (full features including discrete, will be filtered to continuous)
          - masks:  [B,T,K]
        predictions must include:
          - reconstructed_states: [B,T,K,F_cont] (continuous features only)
          - predicted_states:     [B,T,K,F_cont] (continuous features only)
          - existence_logits:     [B,T,K]
          - predicted_existence_logits: [B,T,K] (optional)

        Note: Predictions are continuous-only. Targets will be filtered to continuous features using continuous_indices.
        """
        states = targets["states"]
        masks = targets["masks"]

        recon_states = predictions["reconstructed_states"]
        pred_states = predictions["predicted_states"]
        exist_logits = predictions["existence_logits"]

        # Reconstruction: align t with t
        recon_loss = self._masked_huber_loss(recon_states, states, masks)

        # Prediction: pred at t predicts target at t+1 (ignore last)
        pred_loss = self._masked_huber_loss(pred_states[:, :-1], states[:, 1:], masks[:, :-1])

        exist_loss = self._existence_loss(exist_logits, masks)

        pred_exist_loss = torch.tensor(0.0, device=states.device)
        if self.use_pred_existence_loss and "predicted_existence_logits" in predictions:
            pred_exist_loss = self._existence_loss(predictions["predicted_existence_logits"][:, :-1], masks[:, 1:])

        total = (
            self.recon_weight * recon_loss
            + self.pred_weight * pred_loss
            + self.exist_weight * (exist_loss + pred_exist_loss)
        )

        return {
            "total_loss": total,
            "recon_loss": recon_loss.detach(),
            "pred_loss": pred_loss.detach(),
            "exist_loss": exist_loss.detach(),
            "pred_exist_loss": pred_exist_loss.detach(),
        }
