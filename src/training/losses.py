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
        velocity_direction_weight: float = 1.0,  # Weight for velocity direction loss (increased from 0.3)
        huber_beta: float = 1.0,
        continuous_indices: Optional[List[int]] = None,
        angle_idx: Optional[int] = None,  # Index of angle in full state (default 6)
        use_pred_existence_loss: bool = True,
        velocity_threshold: float = 2.0,  # Speed threshold in PHYSICAL units (px/s)
        normalization_mean: Optional[torch.Tensor] = None,  # [n_continuous] mean for denormalization
        normalization_std: Optional[torch.Tensor] = None,   # [n_continuous] std for denormalization
        kinematic_weight: float = 0.0,  # Weight for kinematic consistency loss (start with 0.0, warmup to 0.1-0.2)
        dt: float = 1.0 / 30.0,  # seconds per frame; used for dx/dt consistency with vx/vy
        disabled_full_state_indices: Optional[List[int]] = None,  # full-state indices to exclude from Huber losses
    ):
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.pred_weight = float(pred_weight)
        self.exist_weight = float(exist_weight)
        self.angle_weight = float(angle_weight)
        self.velocity_direction_weight = float(velocity_direction_weight)
        self.kinematic_weight = float(kinematic_weight)
        self.huber_beta = float(huber_beta)
        self.continuous_indices = continuous_indices
        self.angle_idx = angle_idx if angle_idx is not None else 6  # Default angle at index 6
        self.use_pred_existence_loss = bool(use_pred_existence_loss)
        self.velocity_threshold = float(velocity_threshold)  # Physical units (px/s)
        self.dt = float(dt)
        
        # Store normalization stats for denormalization
        self.register_buffer('norm_mean', normalization_mean if normalization_mean is not None else torch.zeros(1))
        self.register_buffer('norm_std', normalization_std if normalization_std is not None else torch.ones(1))
        self._stats_initialized = (normalization_mean is not None and normalization_std is not None)

        # Optional per-feature weights over continuous outputs (used for recon/pred Huber losses).
        # Default: all 1s. If disabled_full_state_indices provided, those features get weight 0.
        feature_weights = None
        if continuous_indices is not None:
            feature_weights = torch.ones(len(continuous_indices), dtype=torch.float32)
            if disabled_full_state_indices:
                for full_idx in disabled_full_state_indices:
                    if full_idx in continuous_indices:
                        feature_weights[continuous_indices.index(full_idx)] = 0.0
        self.register_buffer("feature_weights", feature_weights if feature_weights is not None else torch.ones(1))
        
        # One-time debug flag for feature verification
        self._feature_verified = False
        
        # Find vx/vy indices in continuous features for denormalization
        self.vx_continuous_idx = None
        self.vy_continuous_idx = None
        if continuous_indices is not None:
            # Assume vx=2, vy=3 in full state
            if 2 in continuous_indices:
                self.vx_continuous_idx = continuous_indices.index(2)
            if 3 in continuous_indices:
                self.vy_continuous_idx = continuous_indices.index(3)

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

        # Optional feature weighting (e.g., disable vx/vy supervision)
        if self.feature_weights.numel() > 1:
            loss = loss * self.feature_weights.view(1, 1, 1, -1)

        # apply mask
        loss = loss * mask.unsqueeze(-1)
        if self.feature_weights.numel() > 1:
            eff_dim = self.feature_weights.sum().clamp(min=1.0)
        else:
            eff_dim = torch.tensor(float(loss.shape[-1]), device=loss.device)
        denom = mask.sum() * eff_dim
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
        Compute velocity direction loss for moving vehicles IN PHYSICAL SPACE.
        
        This denormalizes vx/vy back to px/frame before computing direction,
        avoiding anisotropic distortion from z-score normalization.
        
        Args:
            pred_vel: [B, T, K, 2] predicted velocity (vx, vy) in NORMALIZED continuous feature order
            target_vel: [B, T, K, 2] target velocity (vx, vy) in NORMALIZED full state
            mask: [B, T, K] validity mask
        
        Returns:
            scalar loss (angular error in radians)
        """
        # One-time feature verification (防止索引错位，确保真的是vx/vy)
        if not self._feature_verified:
            print("\n" + "="*70)
            print("🔍 [LOSS INTERNAL] Velocity Direction Loss Feature Verification")
            print("="*70)
            
            if self.continuous_indices is not None and self.vx_continuous_idx is not None and self.vy_continuous_idx is not None:
                # Map continuous index back to full state index
                full_vx_idx = self.continuous_indices[self.vx_continuous_idx]
                full_vy_idx = self.continuous_indices[self.vy_continuous_idx]
                print(f"  vx: continuous[{self.vx_continuous_idx}] <- full_state[{full_vx_idx}]")
                print(f"  vy: continuous[{self.vy_continuous_idx}] <- full_state[{full_vy_idx}]")
                
                if self._stats_initialized:
                    print(f"  vx normalization: mean={self.norm_mean[self.vx_continuous_idx].item():.4f}, std={self.norm_std[self.vx_continuous_idx].item():.4f}")
                    print(f"  vy normalization: mean={self.norm_mean[self.vy_continuous_idx].item():.4f}, std={self.norm_std[self.vy_continuous_idx].item():.4f}")
            else:
                print("  ⚠️  WARNING: continuous_indices or vx/vy indices not properly set!")
            
            # Verify actual data (first batch)
            valid_mask = mask > 0.5
            if valid_mask.sum() > 0:
                pred_vx_sample = pred_vel[..., 0][valid_mask][:100]  # Sample 100 valid entries
                pred_vy_sample = pred_vel[..., 1][valid_mask][:100]
                target_vx_sample = target_vel[..., 0][valid_mask][:100]
                target_vy_sample = target_vel[..., 1][valid_mask][:100]
                
                print(f"\n  Data sanity check (normalized, 100 valid samples):")
                print(f"    pred_vx:  mean={pred_vx_sample.mean():.4f}, std={pred_vx_sample.std():.4f}, min={pred_vx_sample.min():.4f}, max={pred_vx_sample.max():.4f}")
                print(f"    pred_vy:  mean={pred_vy_sample.mean():.4f}, std={pred_vy_sample.std():.4f}, min={pred_vy_sample.min():.4f}, max={pred_vy_sample.max():.4f}")
                print(f"    target_vx: mean={target_vx_sample.mean():.4f}, std={target_vx_sample.std():.4f}")
                print(f"    target_vy: mean={target_vy_sample.mean():.4f}, std={target_vy_sample.std():.4f}")
                
                # Check if data looks like velocity (not discrete 0/1)
                unique_vx = len(torch.unique(pred_vx_sample))
                unique_vy = len(torch.unique(pred_vy_sample))
                print(f"    unique values: vx={unique_vx}, vy={unique_vy} (should be >10 for continuous)")
                
                if unique_vx < 5 or unique_vy < 5:
                    print("  ❌ ERROR: Data looks DISCRETE (unique values < 5)! Wrong feature selected!")
                elif pred_vx_sample.abs().max() < 0.1 and pred_vy_sample.abs().max() < 0.1:
                    print("  ⚠️  WARNING: All values very close to zero - may be wrong feature")
                else:
                    print("  ✅ Data looks like continuous velocity")
            
            print("="*70 + "\n")
            self._feature_verified = True
        
        # Denormalize velocities to physical space (px/frame)
        if self._stats_initialized and self.vx_continuous_idx is not None and self.vy_continuous_idx is not None:
            # Extract mean/std for vx and vy
            vx_mean = self.norm_mean[self.vx_continuous_idx]
            vx_std = self.norm_std[self.vx_continuous_idx]
            vy_mean = self.norm_mean[self.vy_continuous_idx]
            vy_std = self.norm_std[self.vy_continuous_idx]
            
            # Denormalize: v_physical = v_norm * std + mean
            pred_vx_phys = pred_vel[..., 0] * vx_std + vx_mean
            pred_vy_phys = pred_vel[..., 1] * vy_std + vy_mean
            target_vx_phys = target_vel[..., 0] * vx_std + vx_mean
            target_vy_phys = target_vel[..., 1] * vy_std + vy_mean
        else:
            # Fallback: use normalized values (less accurate)
            pred_vx_phys = pred_vel[..., 0]
            pred_vy_phys = pred_vel[..., 1]
            target_vx_phys = target_vel[..., 0]
            target_vy_phys = target_vel[..., 1]
        
        # Compute PHYSICAL speed for thresholding
        target_speed = torch.sqrt(target_vx_phys**2 + target_vy_phys**2 + 1e-8)  # [B, T, K]
        moving_mask = mask * (target_speed > self.velocity_threshold).float()
        
        if moving_mask.sum() < 1:
            return torch.tensor(0.0, device=pred_vel.device)
        
        # Compute velocity directions in PHYSICAL space (no distortion)
        pred_dir = torch.atan2(pred_vy_phys, pred_vx_phys)  # [B, T, K]
        target_dir = torch.atan2(target_vy_phys, target_vx_phys)  # [B, T, K]
        
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


    def _kinematic_consistency_loss(
        self, 
        pred_states: torch.Tensor,  # [B, T, K, F_cont] predicted states
        masks: torch.Tensor,  # [B, T, K] validity masks
        pos_idx: tuple = (0, 1),  # (x_idx, y_idx) in continuous space
        vel_idx: tuple = (2, 3),  # (vx_idx, vy_idx) in continuous space
        disp_threshold: float = 0.5,  # Minimum displacement magnitude (px) to be valid
    ) -> torch.Tensor:
        """
        Kinematic consistency: enforce that predicted velocity matches displacement from positions.
        
        Key idea (unit + alignment):
        - In preprocessing, vx/vy are computed as backward differences divided by dt:
            v[t] = (x[t] - x[t-1]) / dt   (px/s)
        - So the displacement between frames i-1 -> i should align with the velocity at i.
        
        This prevents "position one way, velocity another way" and binds velocity to trajectory geometry.
        
        All computation in PHYSICAL space (after denormalization) to avoid anisotropic distortion.
        """
        if not self._stats_initialized:
            return torch.tensor(0.0, device=pred_states.device)

        # IMPORTANT: In this codebase, `predicted_states[t]` is a teacher-forced one-step prediction of
        # `states[t+1]` (see pred_loss alignment in forward()).
        # Therefore, callers must pass masks aligned to the same time axis as `pred_states`.
        # (We enforce this by shifting at the call site in forward()).
        
        px, py = pos_idx
        vx_i, vy_i = vel_idx
        
        # Extract normalized predictions
        pred_xn = pred_states[..., px]    # [B, T, K]
        pred_yn = pred_states[..., py]
        pred_vxn = pred_states[..., vx_i]
        pred_vyn = pred_states[..., vy_i]
        
        # Denormalize to PHYSICAL space
        # Find indices in continuous_indices
        if self.continuous_indices is None:
            return torch.tensor(0.0, device=pred_states.device)
        
        try:
            x_cont_idx = self.continuous_indices.index(0)  # x is feature 0
            y_cont_idx = self.continuous_indices.index(1)  # y is feature 1
            vx_cont_idx = self.continuous_indices.index(2)  # vx is feature 2
            vy_cont_idx = self.continuous_indices.index(3)  # vy is feature 3
        except ValueError:
            return torch.tensor(0.0, device=pred_states.device)
        
        x_mean, x_std = self.norm_mean[x_cont_idx], self.norm_std[x_cont_idx]
        y_mean, y_std = self.norm_mean[y_cont_idx], self.norm_std[y_cont_idx]
        vx_mean, vx_std = self.norm_mean[vx_cont_idx], self.norm_std[vx_cont_idx]
        vy_mean, vy_std = self.norm_mean[vy_cont_idx], self.norm_std[vy_cont_idx]
        
        # Denormalize: x_phys = x_norm * std + mean
        pred_x = pred_xn * x_std + x_mean    # [B, T, K]
        pred_y = pred_yn * y_std + y_mean
        pred_vx = pred_vxn * vx_std + vx_mean
        pred_vy = pred_vyn * vy_std + vy_mean
        
        # Implied velocity from predicted positions (px/s) to match preprocessing
        dx = pred_x[:, 1:] - pred_x[:, :-1]  # [B, T-1, K]
        dy = pred_y[:, 1:] - pred_y[:, :-1]
        dt = max(self.dt, 1e-8)
        v_from_pos_x = dx / dt
        v_from_pos_y = dy / dt
        
        # Compute PHYSICAL speed for moving mask
        speed_t = torch.sqrt(pred_vx**2 + pred_vy**2 + 1e-8)  # [B, T, K]
        moving = (speed_t > self.velocity_threshold).float()  # [B, T, K] as float for multiplication
        
        # Pair mask: both t and t+1 must be moving (displacement is from t to t+1)
        moving_pair = moving[:, :-1] * moving[:, 1:]  # [B, T-1, K]
        
        # Displacement magnitude filter (avoid noisy directions from tiny movements)
        disp_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)
        valid_disp = (disp_mag > disp_threshold).float()  # [B, T-1, K]
        
        # Validity mask: both frames valid + moving + sufficient displacement
        valid_pair = masks[:, :-1] * masks[:, 1:]  # [B, T-1, K]
        final_mask = valid_pair * moving_pair * valid_disp  # [B, T-1, K]
        
        if final_mask.sum() < 1:
            return torch.tensor(0.0, device=pred_states.device)
        
        # Vector consistency (time alignment):
        # In this codebase, pred_states[i] corresponds to a predicted state at a *future* time.
        # For the internal kinematic relation within the predicted sequence, displacement
        # between pred_states[i] and pred_states[i-1] should align with the velocity at pred_states[i].
        # Therefore, compare v_pred[:, 1:] with (x[:, 1:] - x[:, :-1]).
        err_vx = pred_vx[:, 1:] - v_from_pos_x  # [B, T-1, K]
        err_vy = pred_vy[:, 1:] - v_from_pos_y
        
        # Robust loss (Huber-like): quadratic for small errors, linear for large
        vec_err = torch.sqrt(err_vx**2 + err_vy**2 + 1e-6)  # [B, T-1, K]
        huber_delta = 1.0  # px/s
        loss_per_element = torch.where(
            vec_err < huber_delta,
            0.5 * vec_err**2,
            huber_delta * (vec_err - 0.5 * huber_delta)
        )
        
        # Average over valid pairs
        masked_loss = loss_per_element * final_mask
        return masked_loss.sum() / final_mask.sum()


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
        # Use AND mask so we supervise only stable existence (avoid appear/disappear boundary frames).
        pred_mask = masks[:, :-1] * masks[:, 1:]
        pred_loss = self._masked_huber_loss(pred_states[:, :-1], states[:, 1:], pred_mask)

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
        
        if (
            self.velocity_direction_weight > 0
            and self.continuous_indices is not None
            and 2 in self.continuous_indices
            and 3 in self.continuous_indices
        ):
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
                    pred_mask
                )
            except (ValueError, IndexError):
                # vx or vy not in continuous_indices, skip velocity direction loss
                pass

        # Kinematic consistency loss: bind velocity to trajectory geometry
        kinematic_loss = torch.tensor(0.0, device=states.device)
        if self.kinematic_weight > 0 and self.continuous_indices is not None:
            # `pred_states[:, :-1]` predicts `states[:, 1:]` (one-step teacher forcing).
            # Align masks to the same time axis as `pred_states[:, :-1]` to avoid off-by-one masking.
            kinematic_loss = self._kinematic_consistency_loss(
                pred_states[:, :-1],
                masks[:, 1:],
                pos_idx=(0, 1),  # x, y in continuous space
                vel_idx=(2, 3),  # vx, vy in continuous space
                disp_threshold=0.5  # Minimum 0.5 px displacement to compute direction
            )

        total = (
            self.recon_weight * recon_loss
            + self.pred_weight * pred_loss
            + self.exist_weight * (exist_loss + pred_exist_loss)
            + self.angle_weight * (recon_angle_loss + pred_angle_loss)
            + self.velocity_direction_weight * (recon_vel_dir_loss + pred_vel_dir_loss)
            + self.kinematic_weight * kinematic_loss
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
            "kinematic_loss": kinematic_loss.detach(),
        }
