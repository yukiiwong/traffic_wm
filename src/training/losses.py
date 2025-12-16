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
        angle_weight: float = 0.5,  # Weight for angle loss (angular distance)
        velocity_direction_weight: float = 0.3,  # Weight for velocity direction loss
        huber_beta: float = 1.0,
        continuous_indices: Optional[List[int]] = None,
        angle_idx: Optional[int] = None,  # Index of angle in full state (default 6)
        use_pred_existence_loss: bool = True,
        velocity_threshold: float = 0.5,  # m/s threshold for moving vehicles
    ):
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.pred_weight = float(pred_weight)
        self.exist_weight = float(exist_weight)
        self.angle_weight = float(angle_weight)
        self.velocity_direction_weight = float(velocity_direction_weight)
        self.huber_beta = float(huber_beta)
        self.continuous_indices = continuous_indices
        self.angle_idx = angle_idx if angle_idx is not None else 6  # Default angle at index 6
        self.use_pred_existence_loss = bool(use_pred_existence_loss)
        self.velocity_threshold = float(velocity_threshold)

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

    @staticmethod
    def _angular_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute angular distance between two angles (handles periodicity).
        
        Args:
            pred: [B, T, K] predicted angles in radians
            target: [B, T, K] target angles in radians
        
        Returns:
            [B, T, K] angular distance in [0, pi]
        """
        # Using atan2(sin, cos) is numerically stable for periodic differences
        return torch.abs(torch.atan2(torch.sin(pred - target), torch.cos(pred - target)))

    def _angular_loss(self, pred_angle: torch.Tensor, target_angle: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute angular loss (angular distance with mask).
        
        Args:
            pred_angle: [B, T, K] predicted angle in radians
            target_angle: [B, T, K] target angle in radians (from full state)
            mask: [B, T, K] validity mask
        
        Returns:
            scalar loss
        """
        angular_dist = self._angular_distance(pred_angle, target_angle)  # [B,T,K] in [0, pi]
        masked = angular_dist * mask
        return (masked.sum() / mask.sum().clamp(min=1.0))

    def _velocity_direction_loss(
        self,
        pred_vel: torch.Tensor,
        target_vel: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity direction loss for moving vehicles.
        
        This directly supervises the DIRECTION of velocity prediction,
        which is what appears as arrows in visualization and is measured
        by velocity_direction_error metric.
        
        Args:
            pred_vel: [B, T, K, 2] predicted velocity (vx, vy) in continuous feature order
            target_vel: [B, T, K, 2] target velocity (vx, vy) from full state
            mask: [B, T, K] validity mask
        
        Returns:
            scalar loss (angular error in radians)
        """
        # Filter to moving vehicles only (avoid noise from stationary vehicles)
        target_speed = torch.norm(target_vel, dim=-1)  # [B, T, K]
        moving_mask = mask * (target_speed > self.velocity_threshold).float()
        
        if moving_mask.sum() < 1:
            return torch.tensor(0.0, device=pred_vel.device)
        
        # Compute velocity directions (angles)
        pred_dir = torch.atan2(pred_vel[..., 1], pred_vel[..., 0])  # [B, T, K]
        target_dir = torch.atan2(target_vel[..., 1], target_vel[..., 0])  # [B, T, K]
        
        # Compute angular difference (periodic, in [0, pi])
        dir_diff = torch.abs(
            torch.atan2(
                torch.sin(pred_dir - target_dir),
                torch.cos(pred_dir - target_dir)
            )
        )  # [B, T, K]
        
        # Average over moving vehicles
        masked_diff = dir_diff * moving_mask
        return masked_diff.sum() / moving_mask.sum()


    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        targets must include:
          - states: [B,T,K,F_full] (full features including discrete and angle)
          - masks:  [B,T,K]
        predictions must include:
          - reconstructed_states: [B,T,K,F_cont] (continuous features only, excluding angle)
          - predicted_states:     [B,T,K,F_cont] (continuous features only, excluding angle)
          - reconstructed_angle:  [B,T,K] (angle in radians)
          - predicted_angle:      [B,T,K] (angle in radians with prior)
          - existence_logits:     [B,T,K]
          - predicted_existence_logits: [B,T,K] (optional)

        Note: Predictions are continuous-only (excluding angle). Angle is predicted separately.
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

        # Angle losses (use angular distance for periodic feature)
        recon_angle_loss = torch.tensor(0.0, device=states.device)
        pred_angle_loss = torch.tensor(0.0, device=states.device)
        
        if "reconstructed_angle" in predictions and "predicted_angle" in predictions:
            # Extract ground truth angle from full states
            gt_angle = states[..., self.angle_idx]  # [B,T,K] angle is NOT normalized
            
            recon_angle = predictions["reconstructed_angle"]  # [B,T,K]
            pred_angle = predictions["predicted_angle"]        # [B,T,K]
            
            # Reconstruction angle loss
            recon_angle_loss = self._angular_loss(recon_angle, gt_angle, masks)
            
            # Prediction angle loss (one-step ahead)
            pred_angle_loss = self._angular_loss(pred_angle[:, :-1], gt_angle[:, 1:], masks[:, :-1])

        # Velocity direction losses (supervise motion direction for moving vehicles)
        recon_vel_dir_loss = torch.tensor(0.0, device=states.device)
        pred_vel_dir_loss = torch.tensor(0.0, device=states.device)
        
        if self.continuous_indices is not None and 2 in self.continuous_indices and 3 in self.continuous_indices:
            # Find vx, vy in continuous features
            try:
                vx_idx_cont = self.continuous_indices.index(2)
                vy_idx_cont = self.continuous_indices.index(3)
                
                # Extract predicted velocities (already in continuous order)
                recon_vel = recon_states[..., [vx_idx_cont, vy_idx_cont]]  # [B,T,K,2]
                pred_vel = pred_states[..., [vx_idx_cont, vy_idx_cont]]    # [B,T,K,2]
                
                # Extract GT velocities from full states (features 2,3)
                gt_vel = states[..., [2, 3]]  # [B,T,K,2]
                
                # Reconstruction velocity direction loss
                recon_vel_dir_loss = self._velocity_direction_loss(recon_vel, gt_vel, masks)
                
                # Prediction velocity direction loss (one-step ahead)
                pred_vel_dir_loss = self._velocity_direction_loss(
                    pred_vel[:, :-1], 
                    gt_vel[:, 1:], 
                    masks[:, :-1]
                )
            except (ValueError, IndexError):
                # vx or vy not in continuous_indices, skip velocity direction loss
                pass

        total = (
            self.recon_weight * recon_loss
            + self.pred_weight * pred_loss
            + self.exist_weight * (exist_loss + pred_exist_loss)
            + self.angle_weight * (recon_angle_loss + pred_angle_loss)
            + self.velocity_direction_weight * (recon_vel_dir_loss + pred_vel_dir_loss)
        )

        return {
            "total_loss": total,
            "recon_loss": recon_loss.detach(),
            "pred_loss": pred_loss.detach(),
            "exist_loss": exist_loss.detach(),
            "pred_exist_loss": pred_exist_loss.detach(),
            "recon_angle_loss": recon_angle_loss.detach(),
            "pred_angle_loss": pred_angle_loss.detach(),
            "recon_vel_dir_loss": recon_vel_dir_loss.detach(),
            "pred_vel_dir_loss": pred_vel_dir_loss.detach(),
        }
