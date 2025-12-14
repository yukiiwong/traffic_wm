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

        self.idx_x, self.idx_y = idx_x, idx_y
        self.idx_vx, self.idx_vy = idx_vx, idx_vy
        self.idx_ax, self.idx_ay = idx_ax, idx_ay
        self.use_acceleration = bool(use_acceleration)

        # Encoder: per-frame agent attention -> pooled scene latent [B,T,D]
        self.encoder = MultiAgentEncoder(
            input_dim=input_dim,
            hidden_dim=latent_dim,
            latent_dim=latent_dim,
            max_agents=max_agents,
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

        # Decoder: MLP decoder + residual_xy head (outputs only continuous features)
        self.decoder = StateDecoder(
            latent_dim=latent_dim,
            hidden_dim=latent_dim,
            output_dim=input_dim,
            continuous_dim=continuous_dim,
            max_agents=max_agents,
            enable_xy_residual=True,
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
        assert F == self.input_dim, f"Expected F={self.input_dim}, got {F}"

        latent = self.encoder(states, masks)  # [B,T,D]

        # time padding: True where all agents are absent at that timestep
        time_padding = (masks.sum(dim=-1) == 0)  # [B,T] bool

        predicted_latent, _ = self.dynamics(latent, time_padding_mask=time_padding)  # [B,T,D]

        recon_states, exist_logits, _ = self.decoder(latent, return_residual_xy=False)
        pred_states_base, pred_exist_logits, residual_xy = self.decoder(predicted_latent, return_residual_xy=True)

        # Apply kinematic prior + residual to (x,y) for prediction branch
        pred_states = pred_states_base.clone()
        if residual_xy is not None:
            prior_xy = self._kinematic_prior_xy(states)  # [B,T,K,2] prior for t+1 based on state_t
            # residual_xy is predicted in normalized space; also zero it out for padding agents
            residual_xy = residual_xy * masks.unsqueeze(-1)
            pred_states[..., self.idx_x] = prior_xy[..., 0] + residual_xy[..., 0]
            pred_states[..., self.idx_y] = prior_xy[..., 1] + residual_xy[..., 1]

        return {
            "latent": latent,
            "reconstructed_states": recon_states,
            "predicted_states": pred_states,
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

        # Extract constant discrete features (site_id, class_id, lane_id) from initial states
        # These will remain constant throughout rollout
        discrete_template = initial_states[:, -1:, :, discrete_indices]  # [B, 1, K, n_discrete]

        latent_ctx = self.encoder(initial_states, initial_masks)  # [B,T0,D]
        time_padding = (initial_masks.sum(dim=-1) == 0)  # [B,T0]

        pred_latent_ctx, _ = self.dynamics(latent_ctx, time_padding_mask=time_padding)  # [B,T0,D]
        current_latent = pred_latent_ctx[:, -1:, :]  # latent for time T0 (1-step after last context)

        latent_hist = latent_ctx  # history up to T0-1
        prev_state_full = initial_states[:, -1:, :, :]  # FULL state at time T0-1 [B,1,K,F]

        out_states = []  # Will contain continuous features only [F_cont]
        out_masks = []

        for step in range(n_steps):
            # Decode latent - outputs CONTINUOUS features only
            base_states_cont, exist_logits, residual_xy = self.decoder(current_latent, return_residual_xy=True)
            # base_states_cont: [B,1,K,F_cont=9], residual_xy: [B,1,K,2]
            pred_state_cont = base_states_cont.clone()

            # Reconstruct FULL state for kinematic prior (continuous + discrete)
            # Kinematic prior needs all features to compute physics properly
            pred_state_full = torch.zeros(B, 1, K, F, device=base_states_cont.device, dtype=base_states_cont.dtype)
            pred_state_full[..., continuous_indices] = pred_state_cont
            pred_state_full[..., discrete_indices] = discrete_template  # Keep discrete constant

            # Compute kinematic prior using FULL state (prev_state_full has all features)
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
                gt_state_full = ground_truth_states[:, T0 + step:T0 + step + 1, :, :]
                prev_state_full = gt_state_full
                # Infer mask (padding should be 0)
                gt_mask = (gt_state_full.abs().sum(dim=-1) > 0).float()
                current_latent = self.encoder(gt_state_full, gt_mask)
            else:
                # Reconstruct full state for next iteration
                pred_state_full_masked = pred_state_full.clone()
                pred_state_full_masked[..., continuous_indices] = pred_state_cont * pred_mask.unsqueeze(-1)
                prev_state_full = pred_state_full_masked

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
