"""
World Model: Encoder -> LatentDynamics (Transformer) -> Decoder

Key additions:
1) Transformer-only temporal dynamics with causal masking (see dynamics.py).
2) Kinematic prior + (x,y) residual:
   x_{t+1} = x_t + v_x_t * dt + 0.5 * a_x_t * dt^2 + r_x
   y_{t+1} = y_t + v_y_t * dt + 0.5 * a_y_t * dt^2 + r_y
   where r_x, r_y come from the decoder residual head.

Important:
- Because dataset uses z-score normalization on continuous features, the kinematic prior is computed
  in *raw* space (denormalize -> physics -> renormalize), using train-set mean/std.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from src.models.encoder import MultiAgentEncoder
from src.models.dynamics import LatentDynamics
from src.models.decoder import StateDecoder


class WorldModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 12,
        continuous_dim: int = 9,  # Number of continuous features (excluding discrete class/lane/site IDs)
        max_agents: int = 50,
        latent_dim: int = 256,
        dynamics_layers: int = 4,
        dynamics_heads: int = 8,
        dt: float = 1.0 / 30.0,
        max_dynamics_len: int = 512,
        max_dynamics_context: int = 128,
        # feature indices (must match your preprocess / dataset order)
        idx_x: int = 0,
        idx_y: int = 1,
        idx_vx: int = 2,
        idx_vy: int = 3,
        idx_ax: int = 4,
        idx_ay: int = 5,
        idx_angle: int = 6,  # angle feature index (periodic, not normalized)
        # discrete feature indices (must be provided from metadata at model construction)
        lane_feature_idx: int = 8,
        class_feature_idx: int = 9,
        site_feature_idx: int = 11,
        use_acceleration: bool = True,
        # embeddings (match your metadata)
        num_lanes: int = 100,
        num_sites: int = 10,
        num_classes: int = 10,
        use_lane_embedding: bool = True,
        use_site_id: bool = True,
        use_class_embedding: bool = True,
        lane_embed_dim: int = 16,
        site_embed_dim: int = 8,
        class_embed_dim: int = 8,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.continuous_dim = continuous_dim
        self.max_agents = max_agents
        self.latent_dim = latent_dim
        self.dt = float(dt)
        self.max_dynamics_context = int(max_dynamics_context)

        # If True, open-loop rollout kinematic prior derives velocity from (x,y) differences
        # instead of reading vx/vy channels (useful when vx/vy are not supervised).
        self.rollout_prior_velocity_from_positions = False

        self.idx_x, self.idx_y = idx_x, idx_y
        self.idx_vx, self.idx_vy = idx_vx, idx_vy
        self.idx_ax, self.idx_ay = idx_ax, idx_ay
        self.idx_angle = int(idx_angle)
        self.use_acceleration = bool(use_acceleration)
        # store discrete indices
        self.lane_feature_idx = int(lane_feature_idx)
        self.class_feature_idx = int(class_feature_idx)
        self.site_feature_idx = int(site_feature_idx)

        # Validate discrete indices early to avoid silent misconfiguration
        def _check_idx(name: str, idx: int, input_dim: int):
            if not (0 <= int(idx) < int(input_dim)):
                raise ValueError(f"{name}={idx} out of range [0, {input_dim})")

        _check_idx("lane_feature_idx", self.lane_feature_idx, self.input_dim)
        _check_idx("class_feature_idx", self.class_feature_idx, self.input_dim)
        _check_idx("site_feature_idx", self.site_feature_idx, self.input_dim)

        if len({self.lane_feature_idx, self.class_feature_idx, self.site_feature_idx}) != 3:
            raise ValueError(
                f"Discrete feature indices must be distinct, got "
                f"lane={self.lane_feature_idx}, class={self.class_feature_idx}, site={self.site_feature_idx}"
            )

        # Encoder: per-frame agent attention -> pooled scene latent [B,T,D]
        self.encoder = MultiAgentEncoder(
            input_dim=input_dim,
            hidden_dim=latent_dim,
            latent_dim=latent_dim,
            max_agents=max_agents,
            lane_feature_idx=lane_feature_idx,
            class_feature_idx=class_feature_idx,
            site_feature_idx=site_feature_idx,
            num_lanes=num_lanes,
            num_sites=num_sites,
            num_classes=num_classes,
            use_lane_embedding=use_lane_embedding,
            use_site_id=use_site_id,
            use_class_embedding=use_class_embedding,
            lane_embed_dim=lane_embed_dim,
            site_embed_dim=site_embed_dim,
            class_embed_dim=class_embed_dim,
        )

        # Dynamics: temporal transformer
        self.dynamics = LatentDynamics(
            latent_dim=latent_dim,
            n_layers=dynamics_layers,
            n_heads=dynamics_heads,
            max_len=max_dynamics_len,
        )

        # Decoder: MLP decoder + residual_xy head + angle head (outputs only continuous features excluding angle)
        self.decoder = StateDecoder(
            latent_dim=latent_dim,
            hidden_dim=latent_dim,
            output_dim=input_dim,
            continuous_dim=continuous_dim,
            max_agents=max_agents,
            enable_xy_residual=True,
            enable_angle_head=True,  # Separate head for angle (raw radians)
        )

        # Normalization stats (continuous features only)
        # We store mean/std for continuous dims and the mapping feature_idx -> continuous_idx
        self.register_buffer("norm_mean_cont", torch.zeros(1), persistent=False)
        self.register_buffer("norm_std_cont", torch.ones(1), persistent=False)
        self.register_buffer("cont_index_map", torch.full((input_dim,), -1, dtype=torch.long), persistent=False)

    def set_normalization_stats(self, mean_cont, std_cont, continuous_indices: List[int]) -> None:
        """
        Args:
            mean_cont, std_cont: arrays/tensors of shape [n_continuous]
            continuous_indices: list of feature indices that were normalized (same order as mean/std)
        """
        mean_t = torch.as_tensor(mean_cont, dtype=torch.float32)
        std_t = torch.as_tensor(std_cont, dtype=torch.float32).clamp(min=1e-6)

        self.norm_mean_cont = mean_t.to(self.norm_mean_cont.device)
        self.norm_std_cont = std_t.to(self.norm_std_cont.device)

        m = torch.full((self.input_dim,), -1, dtype=torch.long, device=self.cont_index_map.device)
        for j, feat_idx in enumerate(continuous_indices):
            if 0 <= feat_idx < self.input_dim:
                m[feat_idx] = j
        self.cont_index_map = m

    def _require_stats(self, feat_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) scalars for a normalized feature index."""
        if self.cont_index_map.numel() != self.input_dim:
            raise RuntimeError("cont_index_map not initialized correctly.")
        cont_j = int(self.cont_index_map[feat_idx].item())
        if cont_j < 0:
            raise RuntimeError(
                f"Feature idx {feat_idx} is not marked as continuous/normalized. "
                f"Check dataset continuous_indices."
            )
        mean = self.norm_mean_cont[cont_j]
        std = self.norm_std_cont[cont_j]
        return mean, std

    def _runtime_check_feature_dim(self, F: int) -> None:
        """Runtime check to ensure input feature dimension F is compatible with configured indices."""
        max_idx = max(
            self.idx_x,
            self.idx_y,
            self.idx_vx,
            self.idx_vy,
            self.idx_ax,
            self.idx_ay,
            self.lane_feature_idx,
            self.class_feature_idx,
            self.site_feature_idx,
        )
        if F <= max_idx:
            raise RuntimeError(
                f"Input feature dim F={F} is incompatible with configured indices (max_idx={max_idx}). "
                f"Check metadata/feature_layout and model init."
            )

    def _denorm(self, x_norm: torch.Tensor, feat_idx: int) -> torch.Tensor:
        mean, std = self._require_stats(feat_idx)
        return x_norm * std + mean

    def _renorm(self, x_raw: torch.Tensor, feat_idx: int) -> torch.Tensor:
        mean, std = self._require_stats(feat_idx)
        return (x_raw - mean) / std

    def _kinematic_prior_xy(self, prev_states: torch.Tensor) -> torch.Tensor:
        """
        prev_states: [B, T, K, F] in normalized space
        returns: xy_next_prior_norm: [B, T, K, 2] in normalized space
        """
        x = self._denorm(prev_states[..., self.idx_x], self.idx_x)
        y = self._denorm(prev_states[..., self.idx_y], self.idx_y)
        vx = self._denorm(prev_states[..., self.idx_vx], self.idx_vx)
        vy = self._denorm(prev_states[..., self.idx_vy], self.idx_vy)

        if self.use_acceleration:
            ax = self._denorm(prev_states[..., self.idx_ax], self.idx_ax)
            ay = self._denorm(prev_states[..., self.idx_ay], self.idx_ay)
            x_next = x + vx * self.dt + 0.5 * ax * (self.dt ** 2)
            y_next = y + vy * self.dt + 0.5 * ay * (self.dt ** 2)
        else:
            x_next = x + vx * self.dt
            y_next = y + vy * self.dt

        x_next_n = self._renorm(x_next, self.idx_x)
        y_next_n = self._renorm(y_next, self.idx_y)
        return torch.stack([x_next_n, y_next_n], dim=-1)  # [..., 2]

    def _kinematic_prior_xy_from_positions(
        self,
        prev_prev_states: torch.Tensor,
        prev_states: torch.Tensor,
        prev_prev_masks: Optional[torch.Tensor] = None,
        prev_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Kinematic prior that derives velocity from position differences.

        This avoids dependence on vx/vy channels during open-loop rollout.

        Args:
            prev_prev_states: [B,1,K,F] normalized
            prev_states:      [B,1,K,F] normalized
            prev_prev_masks:  [B,1,K] optional
            prev_masks:       [B,1,K] optional

        Returns:
            xy_next_prior_norm: [B,1,K,2]
        """
        x0 = self._denorm(prev_prev_states[..., self.idx_x], self.idx_x)
        y0 = self._denorm(prev_prev_states[..., self.idx_y], self.idx_y)
        x1 = self._denorm(prev_states[..., self.idx_x], self.idx_x)
        y1 = self._denorm(prev_states[..., self.idx_y], self.idx_y)

        dt = max(self.dt, 1e-8)
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt

        if prev_prev_masks is not None and prev_masks is not None:
            pair = (prev_prev_masks * prev_masks).to(vx.dtype)
            vx = vx * pair
            vy = vy * pair

        if self.use_acceleration:
            ax = self._denorm(prev_states[..., self.idx_ax], self.idx_ax)
            ay = self._denorm(prev_states[..., self.idx_ay], self.idx_ay)
            if prev_masks is not None:
                ax = ax * prev_masks.to(ax.dtype)
                ay = ay * prev_masks.to(ay.dtype)
            x_next = x1 + vx * dt + 0.5 * ax * (dt ** 2)
            y_next = y1 + vy * dt + 0.5 * ay * (dt ** 2)
        else:
            x_next = x1 + vx * dt
            y_next = y1 + vy * dt

        x_next_n = self._renorm(x_next, self.idx_x)
        y_next_n = self._renorm(y_next, self.idx_y)
        return torch.stack([x_next_n, y_next_n], dim=-1)

    def _kinematic_prior_angle(self, prev_states: torch.Tensor) -> torch.Tensor:
        """
        Compute angle prior based on velocity direction (physics-based).
        
        prev_states: [B, T, K, F] (may be normalized for continuous features, angle is raw)
        returns: angle_prior: [B, T, K] in radians (raw, not normalized)
        
        Note: angle is NOT normalized, so we extract it directly.
        Velocity needs denormalization if it was normalized.
        """
        # Extract velocities (need denormalization if they were normalized)
        vx = self._denorm(prev_states[..., self.idx_vx], self.idx_vx)
        vy = self._denorm(prev_states[..., self.idx_vy], self.idx_vy)
        
        # Angle prior: direction of velocity vector
        # atan2(vy, vx) gives angle in [-pi, pi]
        angle_prior = torch.atan2(vy, vx)  # [B, T, K]
        
        return angle_prior

    def forward(self, states: torch.Tensor, masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            states: [B, T, K, F] (normalized continuous + discrete as float) - FULL input including discrete features
            masks:  [B, T, K]     (0/1)

        Returns:
            dict with:
              - latent: [B,T,D]
              - reconstructed_states: [B,T,K,F_cont]  (CONTINUOUS FEATURES ONLY, F_cont=9)
              - predicted_states: [B,T,K,F_cont]  (CONTINUOUS FEATURES ONLY)
              - existence_logits: [B,T,K]    (for reconstructed states)
              - predicted_existence_logits: [B,T,K] (for predicted states)

        Note: Decoder now outputs ONLY continuous features. Discrete features (class_id, lane_id, site_id)
              are used as conditioning inputs via embeddings, not predicted as outputs.
        """
        B, T, K, F = states.shape
        # runtime sanity check for feature layout
        self._runtime_check_feature_dim(F)
        assert F == self.input_dim, f"Expected F={self.input_dim}, got {F}"

        latent = self.encoder(states, masks)  # [B,T,D]

        # time padding: True where all agents are absent at that timestep
        time_padding = (masks.sum(dim=-1) == 0)  # [B,T] bool

        predicted_latent, _ = self.dynamics(latent, time_padding_mask=time_padding)  # [B,T,D]

        # Decoder outputs: continuous states (excluding angle) + angle separately
        recon_states, exist_logits, _, recon_angle = self.decoder(latent, return_residual_xy=False, return_angle=True)
        pred_states_base, pred_exist_logits, residual_xy, pred_angle_base = self.decoder(
            predicted_latent, return_residual_xy=True, return_angle=True
        )

        # Apply kinematic prior + residual to (x,y) for prediction branch
        pred_states = pred_states_base.clone()
        if residual_xy is not None:
            prior_xy = self._kinematic_prior_xy(states)  # [B,T,K,2] prior for t+1 based on state_t
            # residual_xy is predicted in normalized space; also zero it out for padding agents
            residual_xy = residual_xy * masks.unsqueeze(-1)
            pred_states[..., self.idx_x] = prior_xy[..., 0] + residual_xy[..., 0]
            pred_states[..., self.idx_y] = prior_xy[..., 1] + residual_xy[..., 1]

        # Apply kinematic prior to angle (velocity direction)
        pred_angle = pred_angle_base  # Start with decoder prediction
        if pred_angle_base is not None:
            angle_prior = self._kinematic_prior_angle(states)  # [B,T,K] based on velocity
            # Blend: use prior as strong guidance, decoder learns residual
            # For now, simple weighted sum (can make this learnable later)
            pred_angle = 0.7 * angle_prior + 0.3 * pred_angle_base
            # Apply mask
            pred_angle = pred_angle * masks

        return {
            "latent": latent,
            "reconstructed_states": recon_states,
            "predicted_states": pred_states,
            "reconstructed_angle": recon_angle,  # [B,T,K] raw radians
            "predicted_angle": pred_angle,  # [B,T,K] raw radians with prior
            "existence_logits": exist_logits,
            "predicted_existence_logits": pred_exist_logits,
        }

    @torch.no_grad()
    def rollout(
        self,
        initial_states: torch.Tensor,
        initial_masks: torch.Tensor,
        continuous_indices: List[int],
        discrete_indices: List[int],
        n_steps: int = 20,
        threshold: float = 0.5,
        teacher_forcing: bool = False,
        ground_truth_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out future predictions.

        Args:
            initial_states: [B, T0, K, F] - FULL states (continuous + discrete)
            initial_masks:  [B, T0, K]
            continuous_indices: List of indices for continuous features
            discrete_indices: List of indices for discrete features
            n_steps: number of steps to predict forward
            threshold: existence threshold
            teacher_forcing: if True, use ground_truth_states as the "prev" for next step
            ground_truth_states: [B, T0+n_steps, K, F] (same normalization as input)

        Returns:
            predicted_states: [B, n_steps, K, F_cont] - CONTINUOUS features only
            predicted_masks:  [B, n_steps, K]

        Note:
            - Decoder outputs continuous features only [F_cont=9]
            - Discrete features (class_id, lane_id, site_id) are kept constant from initial_states
            - For kinematic prior, we reconstruct full states by combining continuous + discrete
        """
        B, T0, K, F = initial_states.shape
        # runtime sanity check for feature layout
        self._runtime_check_feature_dim(F)

        # Extract constant discrete features (site_id, class_id, lane_id) from initial states
        # These will remain constant throughout rollout
        discrete_template = initial_states[:, -1:, :, discrete_indices]  # [B, 1, K, n_discrete]

        latent_ctx = self.encoder(initial_states, initial_masks)  # [B,T0,D]
        time_padding = (initial_masks.sum(dim=-1) == 0)  # [B,T0]

        pred_latent_ctx, _ = self.dynamics(latent_ctx, time_padding_mask=time_padding)  # [B,T0,D]
        current_latent = pred_latent_ctx[:, -1:, :]  # latent for time T0 (1-step after last context)

        latent_hist = latent_ctx  # history up to T0-1
        prev_state_full = initial_states[:, -1:, :, :]  # FULL state at time T0-1 [B,1,K,F]
        if T0 >= 2:
            prev_prev_state_full = initial_states[:, -2:-1, :, :]
            prev_prev_mask = initial_masks[:, -2:-1, :]
        else:
            prev_prev_state_full = prev_state_full
            prev_prev_mask = initial_masks[:, -1:, :]
        prev_mask = initial_masks[:, -1:, :]

        out_states = []  # Will contain continuous features only [F_cont]
        out_masks = []

        for step in range(n_steps):
            # Decode latent - outputs CONTINUOUS features only
            base_states_cont, exist_logits, residual_xy, _ = self.decoder(current_latent, return_residual_xy=True)
            # base_states_cont: [B,1,K,F_cont=9], residual_xy: [B,1,K,2]
            pred_state_cont = base_states_cont.clone()

            # Reconstruct FULL state for kinematic prior (continuous + discrete)
            # Kinematic prior needs all features to compute physics properly
            pred_state_full = torch.zeros(B, 1, K, F, device=base_states_cont.device, dtype=base_states_cont.dtype)
            pred_state_full[..., continuous_indices] = pred_state_cont
            pred_state_full[..., discrete_indices] = discrete_template  # Keep discrete constant

            # Compute kinematic prior using FULL prev state
            if self.rollout_prior_velocity_from_positions:
                prior_xy = self._kinematic_prior_xy_from_positions(
                    prev_prev_state_full, prev_state_full, prev_prev_masks=prev_prev_mask, prev_masks=prev_mask
                )
            else:
                prior_xy = self._kinematic_prior_xy(prev_state_full)  # [B,1,K,2]

            # Apply residual to (x,y) in continuous prediction
            if residual_xy is not None:
                # Find x,y positions in continuous_indices
                cont_idx_x = continuous_indices.index(self.idx_x)
                cont_idx_y = continuous_indices.index(self.idx_y)
                pred_state_cont[..., cont_idx_x] = prior_xy[..., 0] + residual_xy[..., 0]
                pred_state_cont[..., cont_idx_y] = prior_xy[..., 1] + residual_xy[..., 1]

            # Existence mask
            exist_prob = torch.sigmoid(exist_logits)  # logits -> prob
            pred_mask = (exist_prob > threshold).float()  # [B,1,K]

            # Store CONTINUOUS predictions only
            out_states.append(pred_state_cont)
            out_masks.append(pred_mask)

            # Decide "prev" for next step
            if teacher_forcing and ground_truth_states is not None:
                # Use ground-truth at this predicted time (full features)
                prev_state_full_old = prev_state_full
                prev_mask_old = prev_mask
                gt_state_full = ground_truth_states[:, T0 + step:T0 + step + 1, :, :]
                prev_state_full = gt_state_full
                # Infer mask (padding should be 0)
                gt_mask = (gt_state_full.abs().sum(dim=-1) > 0).float()
                current_latent = self.encoder(gt_state_full, gt_mask)
                prev_prev_state_full = prev_state_full_old
                prev_prev_mask = prev_mask_old
                prev_mask = gt_mask
            else:
                # Reconstruct full state for next iteration
                prev_state_full_old = prev_state_full
                prev_mask_old = prev_mask
                pred_state_full_masked = pred_state_full.clone()
                pred_state_full_masked[..., continuous_indices] = pred_state_cont * pred_mask.unsqueeze(-1)
                prev_state_full = pred_state_full_masked
                prev_prev_state_full = prev_state_full_old
                prev_prev_mask = prev_mask_old
                prev_mask = pred_mask

            # append current_latent to history, then predict next latent
            latent_hist = torch.cat([latent_hist, current_latent], dim=1)
            next_latent = self.dynamics.step(
                latent_hist,
                max_context=self.max_dynamics_context,
            ).view(B, 1, -1)
            current_latent = next_latent

        predicted_states = torch.cat(out_states, dim=1)  # [B,n_steps,K,F_cont] continuous only
        predicted_masks = torch.cat(out_masks, dim=1)    # [B,n_steps,K]
        return predicted_states, predicted_masks

    def rollout_train(
        self,
        initial_states: torch.Tensor,
        initial_masks: torch.Tensor,
        continuous_indices: List[int],
        discrete_indices: List[int],
        n_steps: int = 20,
        teacher_forcing_prob: float = 0.0,
        ground_truth_states: Optional[torch.Tensor] = None,
        use_soft_masks: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable rollout used for training-time auxiliary losses.

        Differences from `rollout()`:
        - Keeps gradients (no @torch.no_grad)
        - Optional scheduled sampling via `teacher_forcing_prob`
        - Uses soft existence masks by default to avoid hard thresholding non-differentiabilities

        Returns:
            predicted_states: [B, n_steps, K, F_cont]
            predicted_masks:  [B, n_steps, K] (probabilities if use_soft_masks else hard 0/1)
        """
        B, T0, K, F = initial_states.shape
        self._runtime_check_feature_dim(F)

        # Keep discrete features constant from the last context frame.
        discrete_template = initial_states[:, -1:, :, discrete_indices]  # [B,1,K,n_discrete]

        latent_ctx = self.encoder(initial_states, initial_masks)  # [B,T0,D]
        time_padding = (initial_masks.sum(dim=-1) == 0)  # [B,T0]
        pred_latent_ctx, _ = self.dynamics(latent_ctx, time_padding_mask=time_padding)  # [B,T0,D]
        current_latent = pred_latent_ctx[:, -1:, :]

        latent_hist = latent_ctx
        prev_state_full = initial_states[:, -1:, :, :]
        if T0 >= 2:
            prev_prev_state_full = initial_states[:, -2:-1, :, :]
            prev_prev_mask = initial_masks[:, -2:-1, :]
        else:
            prev_prev_state_full = prev_state_full
            prev_prev_mask = initial_masks[:, -1:, :]
        prev_mask = initial_masks[:, -1:, :]

        out_states = []
        out_masks = []

        tf_prob = float(max(0.0, min(1.0, teacher_forcing_prob)))

        for step in range(int(n_steps)):
            base_states_cont, exist_logits, residual_xy, _ = self.decoder(current_latent, return_residual_xy=True)
            pred_state_cont = base_states_cont.clone()

            # Reconstruct FULL state (continuous + discrete) for kinematic prior
            pred_state_full = torch.zeros(B, 1, K, F, device=base_states_cont.device, dtype=base_states_cont.dtype)
            pred_state_full[..., continuous_indices] = pred_state_cont
            pred_state_full[..., discrete_indices] = discrete_template

            # Apply kinematic prior + residual to (x,y)
            if self.rollout_prior_velocity_from_positions:
                prior_xy = self._kinematic_prior_xy_from_positions(
                    prev_prev_state_full, prev_state_full, prev_prev_masks=prev_prev_mask, prev_masks=prev_mask
                )
            else:
                prior_xy = self._kinematic_prior_xy(prev_state_full)  # [B,1,K,2]
            if residual_xy is not None:
                cont_idx_x = continuous_indices.index(self.idx_x)
                cont_idx_y = continuous_indices.index(self.idx_y)
                pred_state_cont[..., cont_idx_x] = prior_xy[..., 0] + residual_xy[..., 0]
                pred_state_cont[..., cont_idx_y] = prior_xy[..., 1] + residual_xy[..., 1]

            exist_prob = torch.sigmoid(exist_logits)  # [B,1,K]
            pred_mask = exist_prob if use_soft_masks else (exist_prob > 0.5).float()

            out_states.append(pred_state_cont)
            out_masks.append(pred_mask)

            # Scheduled sampling: with prob tf_prob, use GT as previous state; else use prediction.
            use_tf = (torch.rand(B, 1, 1, device=initial_states.device) < tf_prob).float() if tf_prob > 0 else 0.0
            if isinstance(use_tf, float):
                use_tf = torch.tensor(use_tf, device=initial_states.device, dtype=pred_state_cont.dtype)

            if ground_truth_states is not None:
                gt_state_full = ground_truth_states[:, T0 + step : T0 + step + 1, :, :]
            else:
                gt_state_full = None

            # Predicted full state for next iteration (soft-masked)
            pred_state_full_next = pred_state_full.clone()
            pred_state_full_next[..., continuous_indices] = pred_state_cont * pred_mask.unsqueeze(-1)

            prev_state_full_old = prev_state_full
            prev_mask_old = prev_mask

            if gt_state_full is None or tf_prob <= 0:
                prev_state_full = pred_state_full_next
                prev_mask = pred_mask
                # Keep current_latent as-is; next latent predicted from history
            else:
                prev_state_full = use_tf * gt_state_full + (1.0 - use_tf) * pred_state_full_next

                gt_mask = (gt_state_full.abs().sum(dim=-1) > 0).float()
                prev_mask = use_tf * gt_mask + (1.0 - use_tf) * pred_mask

                # When using GT as prev (teacher forcing), refresh latent with encoder(gt)
                tf_latent = self.encoder(gt_state_full, gt_mask)
                current_latent = use_tf.squeeze(-1) * tf_latent + (1.0 - use_tf.squeeze(-1)) * current_latent

            prev_prev_state_full = prev_state_full_old
            prev_prev_mask = prev_mask_old

            latent_hist = torch.cat([latent_hist, current_latent], dim=1)
            next_latent = self.dynamics.step(latent_hist, max_context=self.max_dynamics_context).view(B, 1, -1)
            current_latent = next_latent

        predicted_states = torch.cat(out_states, dim=1)
        predicted_masks = torch.cat(out_masks, dim=1)
        return predicted_states, predicted_masks
