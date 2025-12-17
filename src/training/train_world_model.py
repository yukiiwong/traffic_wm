"""
Train World Model (Transformer-only)

This script expects the processed NPZ created by your preprocess step, and the metadata.json
in the same directory (TrajectoryDataset loads it automatically).

Key fixes vs the old script:
- Uses ONE normalization stats file from TRAIN and reuses it for VAL/TEST.
- Passes continuous_indices into the loss (do NOT regress discrete IDs).
- Sets normalization stats into the model (needed for kinematic prior in normalized space).
- Transformer-only dynamics (no GRU/LSTM options).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.data.dataset import get_dataloader
from src.models.world_model import WorldModel
from src.training.losses import WorldModelLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", type=str, required=True, help="Path to train_episodes.npz")
    p.add_argument("--val_data", type=str, required=True, help="Path to val_episodes.npz")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/world_model")

    p.add_argument("--epochs", "--n_epochs", type=int, default=50, dest="epochs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", "--learning_rate", type=float, default=3e-4, dest="lr")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine", "step", "plateau"],
                   help="Learning rate scheduler: none, cosine, step, or plateau")
    p.add_argument("--lr_min", type=float, default=1e-6, help="Minimum LR for cosine scheduler")

    # loss weights
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--pred_weight", type=float, default=1.0)
    p.add_argument("--existence_weight", type=float, default=0.1)
    p.add_argument("--angle_weight", type=float, default=0.5, help="Weight for angle loss")
    p.add_argument("--velocity_direction_weight", type=float, default=1.0, 
                   help="Weight for velocity direction loss (supervises motion direction)")
    p.add_argument("--velocity_threshold", type=float, default=2,
                   help="Speed threshold (px/s in PHYSICAL space; vx/vy are computed as diff()/dt in preprocessing) for moving vehicle detection in direction loss")
    p.add_argument("--kinematic_weight", type=float, default=0.0,
                   help="Weight for kinematic consistency loss (v_pred ~ dx/dt). Start with 0.05-0.1, warmup recommended")

    # optional: stop supervising vx/vy outputs
    p.add_argument(
        "--disable_vxy_supervision",
        action="store_true",
        help=(
            "If set, excludes vx/vy from Huber recon/pred losses and disables velocity-direction loss. "
            "This keeps vx/vy as inputs/latents but stops direct supervision on them."
        ),
    )

    # open-loop short rollout training loss (xy-only)
    p.add_argument("--short_rollout_horizon", type=int, default=0,
                   help="If >0, add open-loop rollout loss over H steps (xy-only). 0 disables.")
    p.add_argument("--short_rollout_weight", type=float, default=0.0,
                   help="Weight for short open-loop rollout xy loss.")
    p.add_argument("--scheduled_sampling_start", type=float, default=0.0,
                   help="Scheduled sampling teacher-forcing prob at epoch 0 for short-rollout loss.")
    p.add_argument("--scheduled_sampling_end", type=float, default=0.0,
                   help="Scheduled sampling teacher-forcing prob at final epoch for short-rollout loss.")

    # soft boundary loss on denormalized pixel xy
    p.add_argument("--boundary_weight", type=float, default=0.0,
                   help="Weight for soft boundary penalty on predicted xy (in pixel space). 0 disables.")
    p.add_argument("--boundary_sigma", type=float, default=4.0,
                   help="Defines soft bounds as mean ± sigma*std (pixel space, via denorm).")
    p.add_argument("--boundary_margin_px", type=float, default=0.0,
                   help="Extra margin (px) added to the soft bounds.")

    # diagnostics
    p.add_argument(
        "--disp_stride",
        type=int,
        default=1,
        help="Stride (in frames) for displacement-direction diagnostics. Use 5 or 10 for less noisy direction estimates.",
    )

    # logging
    p.add_argument("--log_dir", type=str, default="logs/world_model")

    # model dims
    p.add_argument("--input_dim", type=int, default=12)
    p.add_argument("--max_agents", type=int, default=50)
    p.add_argument("--latent_dim", type=int, default=256)

    # transformer dynamics
    p.add_argument("--dynamics_layers", type=int, default=4)
    p.add_argument("--dynamics_heads", type=int, default=8)
    p.add_argument("--max_dynamics_len", type=int, default=512)
    p.add_argument("--max_dynamics_context", type=int, default=128)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging to both console and file."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "training.log"

    # Create logger
    logger = logging.getLogger("train_world_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_fmt)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def save_checkpoint(path: Path, model: WorldModel, optimizer: optim.Optimizer, epoch: int, config: dict = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    # Add model config if provided
    if config is not None:
        checkpoint_data["config"] = config
    
    torch.save(checkpoint_data, str(path))


def _safe_div(num, den, eps=1e-8):
    """Safe division avoiding division by zero."""
    return num / (den + eps)


@torch.no_grad()
def compute_val_diagnostics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    feature_names: Dict[int, str],
    continuous_indices: list,
    context_len: int = 65,
    pos_idx: tuple = (0, 1),
    vel_idx: tuple = (2, 3),
    velocity_threshold: float = 2.0,
    disp_stride: int = 1,
    rollout_pred_states: Optional[torch.Tensor] = None,
    normalization_mean: Optional[torch.Tensor] = None,
    normalization_std: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Compute detailed validation diagnostics including per-feature, per-timestep, and existence metrics.

    Args:
        outputs: Model outputs with keys:
            - "reconstructed_states": [B, T, K, F]
            - "predicted_states": [B, T, K, F]
            - "existence_logits": [B, T, K]
            - "predicted_existence_logits": [B, T, K]
        batch: Batch data with keys:
            - "states": [B, T, K, F]
            - "masks": [B, T, K]
        feature_names: Dict mapping feature index to name
        continuous_indices: List of indices for continuous features (e.g., [0,1,2,3,4,5,6,9,10])
        context_len: Length of context window (C)
        pos_idx: Indices of (x, y) features
        vel_idx: Indices of (vx, vy) features

    Returns:
        Dictionary with diagnostic metrics
    """
    states = batch["states"]
    masks = batch["masks"]

    recon = outputs["reconstructed_states"]
    pred = outputs["predicted_states"]

    B, T, K, F = states.shape
    C = context_len
    R = T - C  # Rollout horizon

    # Split into context and future
    # Filter ground truth to continuous features only (to match decoder output)
    gt_ctx_full = states[:, :C]                              # [B, C, K, F_full]
    gt_ctx = gt_ctx_full[..., continuous_indices]            # [B, C, K, F_cont]
    m_ctx = masks[:, :C]                                     # [B, C, K]

    gt_fut_full = states[:, C:C+R]                           # [B, R, K, F_full]
    gt_fut = gt_fut_full[..., continuous_indices]            # [B, R, K, F_cont]
    m_fut = masks[:, C:C+R]                                  # [B, R, K]

    # Decoder outputs are already continuous-only [B, T, K, F_cont]
    recon_ctx = recon[:, :C]                                 # [B, C, K, F_cont]

    # For prediction, we compare pred[:-1] with gt[1:] (one-step ahead)
    # But for rollout diagnostics, we want to compare future predictions
    # pred[:, C-1] predicts state at time C (first future state)
    pred_rollout = pred[:, C-1:C-1+R]                        # [B, R, K, F_cont]
    
    # 🔍 VERIFY: pred_rollout is from predicted_states (not reconstructed)
    if not hasattr(compute_val_diagnostics, '_rollout_verified'):
        print(f"\n🔍 [ROLLOUT-VERIFY] pred_rollout source check:")
        print(f"   pred_rollout.shape = {pred_rollout.shape} (should be [B, {R}, K, F_cont])")
        print(f"   gt_fut.shape = {gt_fut.shape}")
        print(f"   Slicing from predicted_states[:, {C-1}:{C-1+R}]")
        print(f"   Mean absolute value: {pred_rollout.abs().mean():.6f}")
        print(f"   ⚠️  If this is same as gt_fut, you're NOT using predictions!\n")
        compute_val_diagnostics._rollout_verified = True

    m_ctx_f = m_ctx.unsqueeze(-1)   # [B, C, K, 1]
    m_fut_f = m_fut.unsqueeze(-1)   # [B, R, K, 1]

    # Now both predictions and targets are continuous-only, same dimensions
    recon_abs_cont = (recon_ctx - gt_ctx).abs()              # [B, C, K, F_cont]
    pred_abs_cont = (pred_rollout - gt_fut).abs()            # [B, R, K, F_cont]

    diag = {}

    recon_mae_f = _safe_div(
        (recon_abs_cont * m_ctx_f).sum(dim=(0, 1, 2)),
        m_ctx_f.sum(dim=(0, 1, 2))
    )
    pred_mae_f = _safe_div(
        (pred_abs_cont * m_fut_f).sum(dim=(0, 1, 2)),
        m_fut_f.sum(dim=(0, 1, 2))
    )

    # Convert to named dict (only continuous features)
    diag["recon_mae_per_feature"] = {
        feature_names.get(continuous_indices[i], f"f{continuous_indices[i]}"): float(recon_mae_f[i])
        for i in range(len(continuous_indices))
    }
    diag["pred_mae_per_feature"] = {
        feature_names.get(continuous_indices[i], f"f{continuous_indices[i]}"): float(pred_mae_f[i])
        for i in range(len(continuous_indices))
    }

    # ========== (B) Per-timestep MAE for rollout (CONTINUOUS ONLY) ==========
    pred_mae_t = _safe_div(
        (pred_abs_cont * m_fut_f).sum(dim=(0, 2, 3)),
        (m_fut_f.sum(dim=(0, 2, 3)) * len(continuous_indices))
    )
    diag["pred_mae_per_step"] = pred_mae_t.detach().cpu().tolist()

    # ========== (C) XY ADE/FDE ==========
    px, py = pos_idx
    pred_xy = pred_rollout[..., [px, py]]  # [B, R, K, 2]
    gt_xy = gt_fut[..., [px, py]]
    l2 = torch.sqrt(((pred_xy - gt_xy) ** 2).sum(dim=-1) + 1e-8)  # [B, R, K]

    diag["pred_ADE_xy"] = _safe_div((l2 * m_fut).sum(), m_fut.sum().clamp_min(1.0)).item()

    # Final displacement error
    l2_final = l2[:, -1, :]  # [B, K]
    m_final = m_fut[:, -1, :]  # [B, K]
    diag["pred_FDE_xy"] = _safe_div((l2_final * m_final).sum(), m_final.sum().clamp_min(1.0)).item()

    # ========== (D) Velocity error ==========
    vx_idx, vy_idx = vel_idx
    pred_vel = pred_rollout[..., [vx_idx, vy_idx]]  # [B, R, K, 2]
    gt_vel = gt_fut[..., [vx_idx, vy_idx]]
    vel_l2 = torch.sqrt(((pred_vel - gt_vel) ** 2).sum(dim=-1) + 1e-8)  # [B, R, K]

    diag["pred_avg_vel_error"] = _safe_div((vel_l2 * m_fut).sum(), m_fut.sum().clamp_min(1.0)).item()

    # ========== (D2) Velocity direction error ==========
    # Compute angular error for moving vehicles (same as velocity_direction_loss but as diagnostic)
    # IMPORTANT: Denormalize to physical space to avoid anisotropic distortion
    vx_idx, vy_idx = vel_idx
    vx_norm = gt_fut[..., vx_idx]
    vy_norm = gt_fut[..., vy_idx]
    
    # Denormalize if stats available
    if normalization_mean is not None and normalization_std is not None:
        vx_continuous_idx = continuous_indices.index(2) if 2 in continuous_indices else None
        vy_continuous_idx = continuous_indices.index(3) if 3 in continuous_indices else None
        
        if vx_continuous_idx is not None and vy_continuous_idx is not None:
            vx_mean = normalization_mean[vx_continuous_idx]
            vx_std = normalization_std[vx_continuous_idx]
            vy_mean = normalization_mean[vy_continuous_idx]
            vy_std = normalization_std[vy_continuous_idx]
            
            vx_phys = vx_norm * vx_std + vx_mean
            vy_phys = vy_norm * vy_std + vy_mean
        else:
            vx_phys, vy_phys = vx_norm, vy_norm  # fallback
    else:
        vx_phys, vy_phys = vx_norm, vy_norm  # fallback
    
    gt_speed_phys = torch.sqrt(vx_phys**2 + vy_phys**2 + 1e-8)  # [B, R, K] in px/frame
    gt_speed_norm = torch.sqrt(vx_norm**2 + vy_norm**2 + 1e-8)  # [B, R, K] normalized
    
    # Diagnostic: print speed distribution (only once at first call)
    if not hasattr(compute_val_diagnostics, '_speed_stats_printed'):
        valid_speeds_phys = gt_speed_phys[m_fut > 0.5]
        valid_speeds_norm = gt_speed_norm[m_fut > 0.5]
        if len(valid_speeds_phys) > 0:
            percentiles = [0.5, 0.75, 0.9, 0.95, 0.99]
            speed_p_phys = torch.quantile(valid_speeds_phys, torch.tensor(percentiles, device=valid_speeds_phys.device))
            speed_p_norm = torch.quantile(valid_speeds_norm, torch.tensor(percentiles, device=valid_speeds_norm.device))
            
            print(f"\n🔍 GT Speed Distribution:")
            print(f"   PHYSICAL (px/s): p50={speed_p_phys[0]:.3f}, p75={speed_p_phys[1]:.3f}, p90={speed_p_phys[2]:.3f}, p95={speed_p_phys[3]:.3f}, p99={speed_p_phys[4]:.3f}")
            print(f"   NORMALIZED (std units): p50={speed_p_norm[0]:.3f}, p75={speed_p_norm[1]:.3f}, p90={speed_p_norm[2]:.3f}, p95={speed_p_norm[3]:.3f}, p99={speed_p_norm[4]:.3f}")
            print(f"   Threshold={velocity_threshold} px/s filters {(gt_speed_phys[m_fut > 0.5] > velocity_threshold).float().mean()*100:.1f}% of valid vehicles")
            
            # Bin coverage (align with sanity check bins)
            bins = [(0.5, 2), (2, 5), (5, 10), (10, 20)]
            print(f"   Bin coverage (sanity check alignment):")
            for low, high in bins:
                in_bin = ((gt_speed_phys > low) & (gt_speed_phys <= high) & (m_fut > 0.5)).float().sum()
                coverage = in_bin / (m_fut > 0.5).float().sum() * 100
                print(f"     {low}-{high} px/frame: {coverage:.1f}%")
            
            compute_val_diagnostics._speed_stats_printed = True
    
    moving_mask = m_fut * (gt_speed_phys > velocity_threshold).float()  # Use PHYSICAL threshold
    
    # Diagnostic: print moving_mask coverage and feature verification (only once)
    if not hasattr(compute_val_diagnostics, '_coverage_printed'):
        coverage_pct = (moving_mask.sum() / m_fut.sum()).item() * 100 if m_fut.sum() > 0 else 0
        print(f"\n🔍 Moving mask coverage: {coverage_pct:.1f}% of valid vehicles (threshold={velocity_threshold} px/s)")
        
        # Verify feature indices
        vx_continuous_idx = continuous_indices.index(2) if 2 in continuous_indices else None
        vy_continuous_idx = continuous_indices.index(3) if 3 in continuous_indices else None
        print(f"   Feature verification:")
        print(f"     vx: full_state[2] -> continuous[{vx_continuous_idx}]")
        print(f"     vy: full_state[3] -> continuous[{vy_continuous_idx}]")
        print(f"     Using for direction: gt_vel shape={gt_vel.shape}, pred_vel shape={pred_vel.shape}")
        compute_val_diagnostics._coverage_printed = True
    
    if moving_mask.sum() > 0:
        # Denormalize predicted velocity to physical space
        pred_vx_norm = pred_vel[..., 0]
        pred_vy_norm = pred_vel[..., 1]
        
        if normalization_mean is not None and normalization_std is not None:
            vx_continuous_idx = continuous_indices.index(2) if 2 in continuous_indices else None
            vy_continuous_idx = continuous_indices.index(3) if 3 in continuous_indices else None
            
            if vx_continuous_idx is not None and vy_continuous_idx is not None:
                vx_mean = normalization_mean[vx_continuous_idx]
                vx_std = normalization_std[vx_continuous_idx]
                vy_mean = normalization_mean[vy_continuous_idx]
                vy_std = normalization_std[vy_continuous_idx]
                
                pred_vx_phys = pred_vx_norm * vx_std + vx_mean
                pred_vy_phys = pred_vy_norm * vy_std + vy_mean
            else:
                pred_vx_phys, pred_vy_phys = pred_vx_norm, pred_vy_norm
        else:
            pred_vx_phys, pred_vy_phys = pred_vx_norm, pred_vy_norm
        
        # Compute directions in PHYSICAL space (no distortion)
        pred_dir = torch.atan2(pred_vy_phys, pred_vx_phys)  # [B, R, K]
        gt_dir = torch.atan2(vy_phys, vx_phys)               # [B, R, K]
        
        # Compute angular difference (periodic, in [0, pi])
        dir_diff = torch.abs(
            torch.atan2(
                torch.sin(pred_dir - gt_dir),
                torch.cos(pred_dir - gt_dir)
            )
        )  # [B, R, K] in radians
        
        # Average over moving vehicles, convert to degrees for interpretability
        diag["pred_velocity_direction_error_rad"] = _safe_div(
            (dir_diff * moving_mask).sum(), 
            moving_mask.sum().clamp_min(1.0)
        ).item()
        diag["pred_velocity_direction_error_deg"] = diag["pred_velocity_direction_error_rad"] * 180.0 / 3.14159265359
        
        # Store moving_mask coverage for logging
        diag["moving_mask_coverage"] = (moving_mask.sum() / m_fut.sum()).item() if m_fut.sum() > 0 else 0.0
    else:
        diag["pred_velocity_direction_error_rad"] = 0.0
        diag["pred_velocity_direction_error_deg"] = 0.0
        diag["moving_mask_coverage"] = 0.0
    
    # ========== (D3) Displacement direction error (Δx/Δy) as diagnostic contrast ==========
    # Compare PREDICTED trajectory direction vs GT trajectory direction
    # If VEL-DIR-ERROR is high but DISP-DIR-ERROR is low → velocity field issue
    # If both are high → trajectory geometry issue
    disp_stride = int(disp_stride)
    if disp_stride < 1:
        disp_stride = 1

    def _compute_disp_dir_metrics(
        *,
        pred_x_phys_src: torch.Tensor,
        pred_y_phys_src: torch.Tensor,
        gt_x_phys_src: torch.Tensor,
        gt_y_phys_src: torch.Tensor,
        moving_mask_src: torch.Tensor,
        stride: int,
        suffix: str,
    ) -> None:
        """Populate diag with displacement-direction metrics for a given prediction source."""
        nonlocal diag

        if stride < 1:
            stride = 1
        if pred_x_phys_src.size(1) <= stride:
            diag[f"disp_dir_err_{suffix}_gt_rad"] = 0.0
            diag[f"disp_dir_err_{suffix}_gt_deg"] = 0.0
            diag[f"gt_kin_baseline_{suffix}_gt_rad"] = 0.0
            diag[f"gt_kin_baseline_{suffix}_gt_deg"] = 0.0
            return

        # Compute displacements (t -> t+stride)
        gt_dx = gt_x_phys_src[:, stride:] - gt_x_phys_src[:, :-stride]  # [B, R-stride, K]
        gt_dy = gt_y_phys_src[:, stride:] - gt_y_phys_src[:, :-stride]
        pred_dx = pred_x_phys_src[:, stride:] - pred_x_phys_src[:, :-stride]
        pred_dy = pred_y_phys_src[:, stride:] - pred_y_phys_src[:, :-stride]

        gt_disp_dir = torch.atan2(gt_dy, gt_dx)  # [B, R-stride, K]
        pred_disp_dir = torch.atan2(pred_dy, pred_dx)

        # Use the moving mask aligned with the *velocity time index*.
        # NOTE: In preprocessing, v[t] = (p[t] - p[t-1]) / dt (backward difference).
        # For displacement from t -> t+stride, the naturally aligned velocity is at t+stride.
        moving_mask_disp = moving_mask_src[:, stride:]  # [B, R-stride, K]

        # GT velocity direction baseline: v[t+stride] vs displacement (t -> t+stride)
        gt_vel_dir_trunc = gt_vel_dir[:, stride:]  # [B, R-stride, K]

        # A. DISP-DIR-ERR: pred displacement vs GT displacement
        disp_dir_err_pred_gt = torch.abs(
            torch.atan2(
                torch.sin(pred_disp_dir - gt_disp_dir),
                torch.cos(pred_disp_dir - gt_disp_dir),
            )
        )

        # B. GT-KIN-BASELINE: GT velocity direction vs GT displacement direction
        gt_kin_baseline = torch.abs(
            torch.atan2(
                torch.sin(gt_vel_dir_trunc - gt_disp_dir),
                torch.cos(gt_vel_dir_trunc - gt_disp_dir),
            )
        )

        if moving_mask_disp.sum() > 0:
            diag[f"disp_dir_err_{suffix}_gt_rad"] = _safe_div(
                (disp_dir_err_pred_gt * moving_mask_disp).sum(),
                moving_mask_disp.sum().clamp_min(1.0),
            ).item()
            diag[f"disp_dir_err_{suffix}_gt_deg"] = diag[f"disp_dir_err_{suffix}_gt_rad"] * 180.0 / 3.14159265359

            diag[f"gt_kin_baseline_{suffix}_gt_rad"] = _safe_div(
                (gt_kin_baseline * moving_mask_disp).sum(),
                moving_mask_disp.sum().clamp_min(1.0),
            ).item()
            diag[f"gt_kin_baseline_{suffix}_gt_deg"] = (
                diag[f"gt_kin_baseline_{suffix}_gt_rad"] * 180.0 / 3.14159265359
            )
        else:
            diag[f"disp_dir_err_{suffix}_gt_rad"] = 0.0
            diag[f"disp_dir_err_{suffix}_gt_deg"] = 0.0
            diag[f"gt_kin_baseline_{suffix}_gt_rad"] = 0.0
            diag[f"gt_kin_baseline_{suffix}_gt_deg"] = 0.0

    if moving_mask.sum() > 0 and R > disp_stride:
        # Compute displacement from position differences (t -> t+1)
        # Need to denormalize positions to physical space
        px_idx, py_idx = pos_idx
        
        # GT positions (already in continuous features)
        gt_x_norm = gt_fut[..., px_idx]  # [B, R, K]
        gt_y_norm = gt_fut[..., py_idx]  # [B, R, K]
        
        # PRED positions (teacher-forced one-step)
        pred_x_norm = pred_rollout[..., px_idx]  # [B, R, K]
        pred_y_norm = pred_rollout[..., py_idx]  # [B, R, K]

        # Optional open-loop rollout prediction positions
        rollout_x_norm = None
        rollout_y_norm = None
        if rollout_pred_states is not None:
            if rollout_pred_states.dim() != 4:
                raise ValueError(
                    f"rollout_pred_states must be [B,R,K,F_cont], got {tuple(rollout_pred_states.shape)}"
                )
            if rollout_pred_states.shape[:3] != pred_rollout.shape[:3]:
                raise ValueError(
                    f"rollout_pred_states shape {tuple(rollout_pred_states.shape)} incompatible with pred_rollout {tuple(pred_rollout.shape)}"
                )
            rollout_x_norm = rollout_pred_states[..., px_idx]
            rollout_y_norm = rollout_pred_states[..., py_idx]
        
        # Denormalize positions if stats available
        if normalization_mean is not None and normalization_std is not None:
            x_continuous_idx = continuous_indices.index(0) if 0 in continuous_indices else None
            y_continuous_idx = continuous_indices.index(1) if 1 in continuous_indices else None
            
            if x_continuous_idx is not None and y_continuous_idx is not None:
                x_mean = normalization_mean[x_continuous_idx]
                x_std = normalization_std[x_continuous_idx]
                y_mean = normalization_mean[y_continuous_idx]
                y_std = normalization_std[y_continuous_idx]
                
                gt_x_phys = gt_x_norm * x_std + x_mean
                gt_y_phys = gt_y_norm * y_std + y_mean
                pred_x_phys = pred_x_norm * x_std + x_mean
                pred_y_phys = pred_y_norm * y_std + y_mean
            else:
                gt_x_phys, gt_y_phys = gt_x_norm, gt_y_norm
                pred_x_phys, pred_y_phys = pred_x_norm, pred_y_norm
        else:
            gt_x_phys, gt_y_phys = gt_x_norm, gt_y_norm
            pred_x_phys, pred_y_phys = pred_x_norm, pred_y_norm
        
        # 🔍 HARD DIAGNOSTIC 1: Verify pred != GT (one-time print)
        if not hasattr(compute_val_diagnostics, '_disp_debug_printed'):
            delta_x = (pred_x_phys - gt_x_phys).abs().mean().item()
            delta_y = (pred_y_phys - gt_y_phys).abs().mean().item()
            print(f"\n🔍 [DISP-DBG-1] Position difference (MUST be >0.01 if using predictions):")
            print(f"   mean|pred_x - gt_x| = {delta_x:.6f} px")
            print(f"   mean|pred_y - gt_y| = {delta_y:.6f} px")
            print(f"   ⚠️  If both ~0, pred_x/pred_y are ACTUALLY GT (variable mix-up!)")
            compute_val_diagnostics._disp_debug_printed = True
        
        # 🔍 HARD DIAGNOSTIC 1: Assert pred != GT (MUST pass or DISP-DIR will be constant)
        if not hasattr(compute_val_diagnostics, '_disp_assert_checked'):
            diff_x = (pred_x_phys - gt_x_phys).abs().mean().item()
            diff_y = (pred_y_phys - gt_y_phys).abs().mean().item()
            print(f"\n🔍 [HARDDBG-1] Position difference assertion:")
            print(f"   mean|pred_x - gt_x| = {diff_x:.4f} px")
            print(f"   mean|pred_y - gt_y| = {diff_y:.4f} px")
            if diff_x < 1e-4 and diff_y < 1e-4:
                print(f"   ❌ ASSERTION FAILED: pred == gt! DISP-DIR will be constant!")
                raise AssertionError("Predictions are identical to GT - check pred_rollout source!")
            else:
                print(f"   ✅ PASS: pred != gt")
            compute_val_diagnostics._disp_assert_checked = True
        
        # Also need GT velocity direction for baseline calculation
        # Extract GT velocities (already denormalized earlier for velocity direction error)
        # We need to get them again for this block
        gt_vx_for_baseline = gt_fut[..., vx_idx]  # [B, R, K]
        gt_vy_for_baseline = gt_fut[..., vy_idx]
        if normalization_mean is not None and normalization_std is not None:
            vx_continuous_idx = continuous_indices.index(2) if 2 in continuous_indices else None
            vy_continuous_idx = continuous_indices.index(3) if 3 in continuous_indices else None
            if vx_continuous_idx is not None and vy_continuous_idx is not None:
                vx_mean_bl = normalization_mean[vx_continuous_idx]
                vx_std_bl = normalization_std[vx_continuous_idx]
                vy_mean_bl = normalization_mean[vy_continuous_idx]
                vy_std_bl = normalization_std[vy_continuous_idx]
                gt_vx_phys_bl = gt_vx_for_baseline * vx_std_bl + vx_mean_bl
                gt_vy_phys_bl = gt_vy_for_baseline * vy_std_bl + vy_mean_bl
            else:
                gt_vx_phys_bl = gt_vx_for_baseline
                gt_vy_phys_bl = gt_vy_for_baseline
        else:
            gt_vx_phys_bl = gt_vx_for_baseline
            gt_vy_phys_bl = gt_vy_for_baseline
        
        gt_vel_dir = torch.atan2(gt_vy_phys_bl, gt_vx_phys_bl)  # [B, R, K]

        # Teacher-forced (one-step) displacement-direction metrics
        _compute_disp_dir_metrics(
            pred_x_phys_src=pred_x_phys,
            pred_y_phys_src=pred_y_phys,
            gt_x_phys_src=gt_x_phys,
            gt_y_phys_src=gt_y_phys,
            moving_mask_src=moving_mask,
            stride=disp_stride,
            suffix="pred",
        )

        # Backward-compatible aliases (teacher-forced baseline)
        diag["gt_kin_baseline_rad"] = diag.get("gt_kin_baseline_pred_gt_rad", 0.0)
        diag["gt_kin_baseline_deg"] = diag.get("gt_kin_baseline_pred_gt_deg", 0.0)

        # Open-loop rollout displacement-direction metrics (if provided)
        if rollout_x_norm is not None and rollout_y_norm is not None:
            if normalization_mean is not None and normalization_std is not None:
                # Reuse x/y stats already resolved above
                rollout_x_phys = rollout_x_norm * x_std + x_mean
                rollout_y_phys = rollout_y_norm * y_std + y_mean
            else:
                rollout_x_phys = rollout_x_norm
                rollout_y_phys = rollout_y_norm

            _compute_disp_dir_metrics(
                pred_x_phys_src=rollout_x_phys,
                pred_y_phys_src=rollout_y_phys,
                gt_x_phys_src=gt_x_phys,
                gt_y_phys_src=gt_y_phys,
                moving_mask_src=moving_mask,
                stride=disp_stride,
                suffix="rollout_pred",
            )

        # 🔍 FINAL ASSERTION (teacher-forced): A must NOT equal B (except in perfect model)
        if not hasattr(compute_val_diagnostics, '_final_assertion_checked'):
            a = diag.get("disp_dir_err_pred_gt_deg", 0.0)
            b = diag.get("gt_kin_baseline_pred_gt_deg", 0.0)
            print(f"\n🔍 [HARDDBG-FINAL] ASSERTION:")
            print(f"   diag['disp_dir_err_pred_gt_deg'] = {a:.2f}° [predΔ vs gtΔ]")
            print(f"   diag['gt_kin_baseline_pred_gt_deg'] = {b:.2f}° [gtV vs gtΔ]")
            print(f"   Difference: {abs(a-b):.2f}°")
            if abs(a - b) < 1e-3:
                print(f"   ❌ CRITICAL: Metrics are identical! DISP-DIR is using GT-only path!")
                raise AssertionError(f"DISP-DIR metric is computing GT-only! A={a:.4f}°, B={b:.4f}°")
            else:
                print(f"   ✅ PASS: Metrics differ correctly\n")
            compute_val_diagnostics._final_assertion_checked = True
    
    # ========== (D4) Kinematic direction error: pred velocity vs pred displacement direction ==========
    # This directly measures the kinematic consistency we're trying to enforce
    if moving_mask.sum() > 0 and R > 1:
        # Denormalize predicted velocities to physical space
        if normalization_mean is not None and normalization_std is not None:
            vx_continuous_idx = continuous_indices.index(2) if 2 in continuous_indices else None
            vy_continuous_idx = continuous_indices.index(3) if 3 in continuous_indices else None
            
            if vx_continuous_idx is not None and vy_continuous_idx is not None:
                vx_idx, vy_idx = vel_idx
                pred_vx_norm = pred_rollout[..., vx_idx]  # [B, R, K]
                pred_vy_norm = pred_rollout[..., vy_idx]
                
                vx_mean = normalization_mean[vx_continuous_idx]
                vx_std = normalization_std[vx_continuous_idx]
                vy_mean = normalization_mean[vy_continuous_idx]
                vy_std = normalization_std[vy_continuous_idx]
                
                pred_vx_phys = pred_vx_norm * vx_std + vx_mean
                pred_vy_phys = pred_vy_norm * vy_std + vy_mean
                
                # Velocity direction
                pred_vel_dir = torch.atan2(pred_vy_phys, pred_vx_phys)  # [B, R, K]
                
                # Displacement direction (already computed above if DISP-DIR was computed)
                # Need to recompute because we need full R frames not R-1
                # Use pred positions that were denormalized above
                pred_x_phys_full = pred_x_phys  # [B, R, K] from above
                pred_y_phys_full = pred_y_phys
                
                pred_dx_kin = pred_x_phys_full[:, 1:] - pred_x_phys_full[:, :-1]  # [B, R-1, K]
                pred_dy_kin = pred_y_phys_full[:, 1:] - pred_y_phys_full[:, :-1]
                pred_disp_dir_kin = torch.atan2(pred_dy_kin, pred_dx_kin)  # [B, R-1, K]
                
                # Time alignment: preprocessing defines v[t] as backward-diff (t-1 -> t).
                # Therefore compare v[t+1] with displacement (t -> t+1).
                kin_dir_diff = torch.abs(
                    torch.atan2(
                        torch.sin(pred_vel_dir[:, 1:] - pred_disp_dir_kin),
                        torch.cos(pred_vel_dir[:, 1:] - pred_disp_dir_kin)
                    )
                )  # [B, R-1, K]
                
                # Use same moving mask
                moving_mask_kin = moving_mask[:, 1:]  # [B, R-1, K]
                
                if moving_mask_kin.sum() > 0:
                    diag["kinematic_direction_error_rad"] = _safe_div(
                        (kin_dir_diff * moving_mask_kin).sum(),
                        moving_mask_kin.sum().clamp_min(1.0)
                    ).item()
                    diag["kinematic_direction_error_deg"] = diag["kinematic_direction_error_rad"] * 180.0 / 3.14159265359
                else:
                    diag["kinematic_direction_error_rad"] = 0.0
                    diag["kinematic_direction_error_deg"] = 0.0
            else:
                diag["kinematic_direction_error_rad"] = 0.0
                diag["kinematic_direction_error_deg"] = 0.0
        else:
            diag["kinematic_direction_error_rad"] = 0.0
            diag["kinematic_direction_error_deg"] = 0.0
    else:
        diag["kinematic_direction_error_rad"] = 0.0
        diag["kinematic_direction_error_deg"] = 0.0

    # ========== (E) Existence metrics (context reconstruction) ==========
    if "existence_logits" in outputs:
        logits = outputs["existence_logits"][:, :C]  # [B, C, K]
        prob = torch.sigmoid(logits)
        pred_e = (prob > 0.5).float()
        gt_e = m_ctx.float()

        tp = ((pred_e == 1) & (gt_e == 1)).sum().item()
        tn = ((pred_e == 0) & (gt_e == 0)).sum().item()
        fp = ((pred_e == 1) & (gt_e == 0)).sum().item()
        fn = ((pred_e == 0) & (gt_e == 1)).sum().item()
        total = tp + tn + fp + fn + 1e-8

        diag["exist_ctx_acc"] = (tp + tn) / total
        diag["exist_ctx_prec"] = tp / (tp + fp + 1e-8)
        diag["exist_ctx_rec"] = tp / (tp + fn + 1e-8)
        diag["exist_ctx_fp_rate"] = fp / (fp + tn + 1e-8)
        diag["exist_ctx_fn_rate"] = fn / (fn + tp + 1e-8)

    # ========== (F) Existence metrics (rollout prediction) ==========
    if "predicted_existence_logits" in outputs:
        pred_logits = outputs["predicted_existence_logits"][:, C-1:C-1+R]  # [B, R, K]
        pred_prob = torch.sigmoid(pred_logits)
        pred_m = (pred_prob > 0.5).float()
        gt_m = m_fut.float()

        tp = ((pred_m == 1) & (gt_m == 1)).sum().item()
        tn = ((pred_m == 0) & (gt_m == 0)).sum().item()
        fp = ((pred_m == 1) & (gt_m == 0)).sum().item()
        fn = ((pred_m == 0) & (gt_m == 1)).sum().item()
        total = tp + tn + fp + fn + 1e-8

        diag["exist_pred_acc"] = (tp + tn) / total
        diag["exist_pred_prec"] = tp / (tp + fp + 1e-8)
        diag["exist_pred_rec"] = tp / (tp + fn + 1e-8)
        diag["exist_pred_fp_rate"] = fp / (fp + tn + 1e-8)
        diag["exist_pred_fn_rate"] = fn / (fn + tp + 1e-8)

    return diag


@torch.no_grad()
def evaluate(
    model: WorldModel,
    loader,
    loss_fn: WorldModelLoss,
    device: str,
    context_len: int = 65,
    feature_names: Dict[int, str] = None,
    continuous_indices: list = None,
    discrete_indices: list = None,
    disp_stride: int = 1,
    compute_diagnostics: bool = True,
    short_rollout_horizon: int = 0,
    short_rollout_weight: float = 0.0,
    boundary_weight: float = 0.0,
    boundary_sigma: float = 4.0,
    boundary_margin_px: float = 0.0,
) -> Dict[str, float]:
    """
    Evaluate model on validation set with optional detailed diagnostics.

    Args:
        model: WorldModel instance
        loader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to use
        context_len: Context length (C)
        feature_names: Dict mapping feature index to name
        continuous_indices: List of indices for continuous features
        compute_diagnostics: Whether to compute detailed diagnostics

    Returns:
        Dictionary with loss metrics and optional diagnostics
    """
    model.eval()
    totals = {
        "total_loss": 0.0, 
        "recon_loss": 0.0, 
        "pred_loss": 0.0, 
        "exist_loss": 0.0, 
        "pred_exist_loss": 0.0,
        "recon_vel_dir_loss": 0.0,
        "pred_vel_dir_loss": 0.0,
        "kinematic_loss": 0.0,
        "short_rollout_loss": 0.0,
        "boundary_loss": 0.0,
        "total_loss_with_aux": 0.0,
    }
    n = 0

    # Diagnostic accumulators
    if compute_diagnostics:
        diag_sums = defaultdict(float)
        diag_counts = defaultdict(float)
        diag_lists = defaultdict(list)
        diag_dicts = defaultdict(lambda: defaultdict(float))
        diag_dict_counts = defaultdict(lambda: defaultdict(float))

    for batch in loader:
        states = batch["states"].to(device)
        masks = batch["masks"].to(device)
        preds = model(states, masks)
        losses = loss_fn(preds, {"states": states, "masks": masks})
        bs = states.size(0)

        # Optional: auxiliary losses (computed under no_grad)
        aux_roll = torch.tensor(0.0, device=device)
        aux_bnd = torch.tensor(0.0, device=device)
        if (
            short_rollout_weight > 0
            and short_rollout_horizon
            and discrete_indices is not None
            and loss_fn._stats_initialized
            and continuous_indices is not None
        ):
            C = int(context_len)
            T = int(states.shape[1])
            H = int(min(short_rollout_horizon, max(0, T - C)))
            if H > 0:
                ctx_states = states[:, :C]
                ctx_masks = masks[:, :C]
                pred_roll_cont, _ = model.rollout(
                    ctx_states,
                    ctx_masks,
                    continuous_indices=continuous_indices,
                    discrete_indices=discrete_indices,
                    n_steps=H,
                    teacher_forcing=False,
                )

                stable_mask = masks[:, C - 1 : C - 1 + H] * masks[:, C : C + H]  # [B,H,K]
                try:
                    x_ci = continuous_indices.index(0)
                    y_ci = continuous_indices.index(1)
                    x_mean, x_std = loss_fn.norm_mean[x_ci], loss_fn.norm_std[x_ci]
                    y_mean, y_std = loss_fn.norm_mean[y_ci], loss_fn.norm_std[y_ci]
                except ValueError:
                    x_ci = y_ci = None

                if x_ci is not None:
                    pred_x = pred_roll_cont[..., x_ci] * x_std + x_mean
                    pred_y = pred_roll_cont[..., y_ci] * y_std + y_mean
                    gt_x = states[:, C : C + H, :, 0] * x_std + x_mean
                    gt_y = states[:, C : C + H, :, 1] * y_std + y_mean

                    dx = pred_x - gt_x
                    dy = pred_y - gt_y
                    err = torch.stack([dx, dy], dim=-1)
                    abs_err = err.abs()
                    beta = 1.0
                    hub = torch.where(abs_err < beta, 0.5 * (err ** 2) / beta, abs_err - 0.5 * beta)
                    hub = hub * stable_mask.unsqueeze(-1)
                    denom = stable_mask.sum() * 2.0
                    aux_roll = hub.sum() / denom.clamp(min=1.0)

                    if boundary_weight > 0:
                        margin = float(boundary_margin_px)
                        sigma = float(boundary_sigma)
                        x_min = (x_mean - sigma * x_std - margin)
                        x_max = (x_mean + sigma * x_std + margin)
                        y_min = (y_mean - sigma * y_std - margin)
                        y_max = (y_mean + sigma * y_std + margin)
                        pen_x = F.relu(x_min - pred_x) + F.relu(pred_x - x_max)
                        pen_y = F.relu(y_min - pred_y) + F.relu(pred_y - y_max)
                        pen = (pen_x ** 2 + pen_y ** 2) * stable_mask
                        aux_bnd = pen.sum() / stable_mask.sum().clamp(min=1.0)

        # Accumulate base losses
        for k in [
            "total_loss",
            "recon_loss",
            "pred_loss",
            "exist_loss",
            "pred_exist_loss",
            "recon_vel_dir_loss",
            "pred_vel_dir_loss",
            "kinematic_loss",
        ]:
            totals[k] += float(losses[k].item()) * bs

        totals["short_rollout_loss"] += float(aux_roll.item()) * bs
        totals["boundary_loss"] += float(aux_bnd.item()) * bs
        totals["total_loss_with_aux"] += float(
            (losses["total_loss"] + float(short_rollout_weight) * aux_roll + float(boundary_weight) * aux_bnd).item()
        ) * bs
        n += bs

        # Compute diagnostics
        if compute_diagnostics and feature_names is not None and continuous_indices is not None:
            # ---------- Open-loop rollout (no teacher forcing) ----------
            rollout_pred_states = None
            if discrete_indices is not None:
                C = int(context_len)
                T = int(states.shape[1])
                R = max(0, T - C)
                if R > 0:
                    ctx_states = states[:, :C]
                    ctx_masks = masks[:, :C]
                    rollout_pred_states, _ = model.rollout(
                        ctx_states,
                        ctx_masks,
                        continuous_indices=continuous_indices,
                        discrete_indices=discrete_indices,
                        n_steps=R,
                        teacher_forcing=False,
                    )

            diag = compute_val_diagnostics(
                preds,
                {"states": states, "masks": masks},
                feature_names,
                continuous_indices,
                context_len=context_len,
                velocity_threshold=loss_fn.velocity_threshold,
                disp_stride=disp_stride,
                rollout_pred_states=rollout_pred_states,
                normalization_mean=loss_fn.norm_mean if loss_fn._stats_initialized else None,
                normalization_std=loss_fn.norm_std if loss_fn._stats_initialized else None,
            )

            # Accumulate scalar metrics
            for k, v in diag.items():
                if isinstance(v, (int, float)):
                    diag_sums[k] += float(v) * bs
                    diag_counts[k] += bs

            # Accumulate per-step list
            if "pred_mae_per_step" in diag:
                diag_lists["pred_mae_per_step"].append(diag["pred_mae_per_step"])

            # Accumulate per-feature dicts
            for dict_key in ["recon_mae_per_feature", "pred_mae_per_feature"]:
                if dict_key in diag:
                    for feat_name, val in diag[dict_key].items():
                        diag_dicts[dict_key][feat_name] += float(val) * bs
                        diag_dict_counts[dict_key][feat_name] += bs

    # Average losses
    for k in totals:
        totals[k] /= max(1, n)

    # Average diagnostics
    if compute_diagnostics:
        # Average scalars
        for k in diag_sums:
            totals[f"diag_{k}"] = diag_sums[k] / max(1, diag_counts[k])

        # Average per-step
        if "pred_mae_per_step" in diag_lists and len(diag_lists["pred_mae_per_step"]) > 0:
            import numpy as np
            # Average across batches
            step_arrays = [np.array(x) for x in diag_lists["pred_mae_per_step"]]
            totals["diag_pred_mae_per_step"] = np.mean(step_arrays, axis=0).tolist()

        # Average per-feature dicts
        for dict_key in diag_dicts:
            totals[f"diag_{dict_key}"] = {
                feat_name: diag_dicts[dict_key][feat_name] / max(1, diag_dict_counts[dict_key][feat_name])
                for feat_name in diag_dicts[dict_key]
            }

    return totals


def main() -> None:
    args = parse_args()

    if args.disable_vxy_supervision:
        # If user asks to stop supervising vx/vy outputs, we should also disable the velocity-direction loss.
        # (Otherwise vx/vy still get supervised indirectly through direction.)
        args.velocity_direction_weight = 0.0

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting training with args: {args}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ckpt_dir / "normalization_stats.npz"

    # TRAIN loader computes stats; save them once
    train_loader = get_dataloader(
        args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        normalize=True,
        stats_path=None,
    )
    if not stats_path.exists():
        train_loader.dataset.save_stats(str(stats_path))

    # VAL loader must reuse train stats
    val_loader = get_dataloader(
        args.val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        normalize=True,
        stats_path=str(stats_path),
    )

    # Pull metadata for dt and discrete vocab sizes
    meta = train_loader.dataset.metadata
    dt = float(meta.get("dt", 1.0 / 30.0))
    num_lanes = int(meta.get("num_lanes", 100))
    num_sites = int(meta.get("num_sites", 10))
    num_classes = int(meta.get("num_classes", 10))
    context_len = int(meta.get("context_length", 65))

    # Extract feature names and continuous indices for diagnostics
    feature_layout = meta.get("feature_layout", {})
    feature_names = {int(k): v for k, v in feature_layout.items()}
    continuous_indices = train_loader.dataset.continuous_indices
    discrete_indices = train_loader.dataset.discrete_indices

    # Resolve discrete feature indices from metadata (pass into model to avoid hard-coded defaults)
    from src.utils.common import parse_discrete_feature_indices_from_metadata

    lane_feature_idx, class_feature_idx, site_feature_idx = parse_discrete_feature_indices_from_metadata(
        meta, fallback=(8, 7, 11), strict=False
    )

    # Log feature classification
    logger.info("=" * 60)
    logger.info("Feature Classification:")
    logger.info(f"  Continuous features ({len(continuous_indices)}): {[feature_names.get(i, f'feature_{i}') for i in continuous_indices]}")
    logger.info(f"  Discrete features ({len(discrete_indices)}): {[feature_names.get(i, f'feature_{i}') for i in discrete_indices]}")
    logger.info(f"  NOTE: MAE metrics will ONLY be computed for continuous features")
    logger.info("=" * 60)

    device = args.device

    model = WorldModel(
        input_dim=args.input_dim,
        continuous_dim=len(continuous_indices),  # Only continuous features are decoded
        max_agents=args.max_agents,
        latent_dim=args.latent_dim,
        dynamics_layers=args.dynamics_layers,
        dynamics_heads=args.dynamics_heads,
        dt=dt,
        max_dynamics_len=args.max_dynamics_len,
        max_dynamics_context=args.max_dynamics_context,
        num_lanes=num_lanes,
        num_sites=num_sites,
        num_classes=num_classes,
        use_acceleration=bool(meta.get("use_acceleration", True)),
        lane_feature_idx=lane_feature_idx,
        class_feature_idx=class_feature_idx,
        site_feature_idx=site_feature_idx,
    ).to(device)

    # If vx/vy are not supervised, open-loop rollout should not depend on predicted vx/vy.
    # Derive rollout kinematic prior velocity from predicted position differences instead.
    model.rollout_prior_velocity_from_positions = bool(args.disable_vxy_supervision)

    # Set normalization stats into the model (needed for kinematic prior)
    model.set_normalization_stats(
        train_loader.dataset.mean,
        train_loader.dataset.std,
        train_loader.dataset.continuous_indices,
    )

    # Check if angle feature exists in continuous_indices
    # Angle (heading/yaw) is typically feature index 6 in full state
    # But if it's not in continuous_indices, we should NOT use angle loss
    angle_idx_full = 6  # Default full state index for angle
    angle_idx = None  # Will be set to continuous index if angle exists
    angle_weight_actual = args.angle_weight
    
    if angle_idx_full in continuous_indices:
        angle_idx = continuous_indices.index(angle_idx_full)
        logger.info(f"✅ Angle feature found at full_state[{angle_idx_full}] -> continuous[{angle_idx}], using angle_loss")
    else:
        angle_weight_actual = 0.0  # Force disable
        logger.warning(f"⚠️  Angle feature (idx {angle_idx_full}) NOT in continuous_indices {continuous_indices}")
        logger.warning(f"    Disabling angle_loss (angle_weight forced to 0.0)")
        logger.warning(f"    Current continuous features: {[feature_names.get(i, f'f{i}') for i in continuous_indices]}")
    
    # Diagnostic: print vx/vy normalization stats to check anisotropy
    vx_idx_in_continuous = continuous_indices.index(2) if 2 in continuous_indices else None
    vy_idx_in_continuous = continuous_indices.index(3) if 3 in continuous_indices else None
    if vx_idx_in_continuous is not None and vy_idx_in_continuous is not None:
        vx_std = train_loader.dataset.std[vx_idx_in_continuous].item()
        vy_std = train_loader.dataset.std[vy_idx_in_continuous].item()
        vx_mean = train_loader.dataset.mean[vx_idx_in_continuous].item()
        vy_mean = train_loader.dataset.mean[vy_idx_in_continuous].item()
        logger.info(f"🔍 Velocity normalization stats:")
        logger.info(f"   vx: mean={vx_mean:.4f}, std={vx_std:.4f}")
        logger.info(f"   vy: mean={vy_mean:.4f}, std={vy_std:.4f}")
        logger.info(f"   std_vx/std_vy ratio = {vx_std/vy_std:.4f} ({'isotropic' if abs(vx_std/vy_std - 1.0) < 0.1 else 'ANISOTROPIC - direction will be distorted!'})")

    loss_fn = WorldModelLoss(
        recon_weight=args.recon_weight,
        pred_weight=args.pred_weight,
        exist_weight=args.existence_weight,
        angle_weight=angle_weight_actual,  # Use actual (may be forced to 0)
        velocity_direction_weight=args.velocity_direction_weight,
        huber_beta=1.0,
        continuous_indices=train_loader.dataset.continuous_indices,
        angle_idx=angle_idx,  # May be None if angle not in continuous
        use_pred_existence_loss=True,
        velocity_threshold=args.velocity_threshold,
        normalization_mean=train_loader.dataset.mean,
        normalization_std=train_loader.dataset.std,
        kinematic_weight=args.kinematic_weight,
        dt=dt,
        disabled_full_state_indices=[2, 3] if args.disable_vxy_supervision else None,
    )
    
    logger.info(
        f"Loss weights: recon={args.recon_weight}, pred={args.pred_weight}, "
        f"exist={args.existence_weight}, angle={angle_weight_actual} (actual, may differ from arg), "
        f"vel_dir={args.velocity_direction_weight} (threshold={args.velocity_threshold} px/s in PHYSICAL space), "
        f"kinematic={args.kinematic_weight}, "
        f"short_rollout=(H={args.short_rollout_horizon}, w={args.short_rollout_weight}, ss={args.scheduled_sampling_start}->{args.scheduled_sampling_end}), "
        f"boundary=(sigma={args.boundary_sigma}, margin={args.boundary_margin_px}, w={args.boundary_weight}), "
        f"disable_vxy_supervision={args.disable_vxy_supervision}"
    )

    # Precompute soft bounds for boundary penalty in PHYSICAL (pixel) space from normalization stats.
    x_min = x_max = y_min = y_max = None
    if args.boundary_weight > 0 and loss_fn._stats_initialized and continuous_indices is not None:
        try:
            x_ci = continuous_indices.index(0)
            y_ci = continuous_indices.index(1)
            x_mean, x_std = loss_fn.norm_mean[x_ci], loss_fn.norm_std[x_ci]
            y_mean, y_std = loss_fn.norm_mean[y_ci], loss_fn.norm_std[y_ci]
            margin = float(args.boundary_margin_px)
            sigma = float(args.boundary_sigma)
            x_min = (x_mean - sigma * x_std - margin)
            x_max = (x_mean + sigma * x_std + margin)
            y_min = (y_mean - sigma * y_std - margin)
            y_max = (y_mean + sigma * y_std + margin)
        except ValueError:
            logger.warning("⚠️  Boundary loss enabled but x/y not in continuous_indices; disabling boundary penalty")
            args.boundary_weight = 0.0

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Setup learning rate scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )
        logger.info(f"Using CosineAnnealingLR scheduler (eta_min={args.lr_min})")
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.1
        )
        logger.info("Using StepLR scheduler (step_size=epochs/3, gamma=0.1)")
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        logger.info("Using ReduceLROnPlateau scheduler (patience=5, factor=0.5)")
    else:
        logger.info("No learning rate scheduler")

    best_val = float("inf")
    
    # Prepare model config for checkpoint saving
    model_config = {
        "input_dim": args.input_dim,
        "continuous_dim": len(continuous_indices),
        "max_agents": args.max_agents,
        "latent_dim": args.latent_dim,
        "dynamics_layers": args.dynamics_layers,
        "dynamics_heads": args.dynamics_heads,
        "dt": dt,
        "max_dynamics_len": args.max_dynamics_len,
        "max_dynamics_context": args.max_dynamics_context,
        "num_lanes": num_lanes,
        "num_sites": num_sites,
        "num_classes": num_classes,
        "use_acceleration": bool(meta.get("use_acceleration", True)),
        "lane_feature_idx": lane_feature_idx,
        "class_feature_idx": class_feature_idx,
        "site_feature_idx": site_feature_idx,
        "continuous_indices": continuous_indices.tolist() if hasattr(continuous_indices, 'tolist') else list(continuous_indices),
        "discrete_indices": discrete_indices.tolist() if hasattr(discrete_indices, 'tolist') else list(discrete_indices),
        "angle_idx": angle_idx,
        "rollout_prior_velocity_from_positions": bool(args.disable_vxy_supervision),
    }

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        running = 0.0
        n = 0

        for bi, batch in enumerate(pbar):
            states = batch["states"].to(device)
            masks  = batch["masks"].to(device)

            # ====== DEBUG CHECK (run once) ======
            if epoch == 0 and bi == 0:
                assert states.ndim == 4 and masks.ndim == 3
                assert states.shape[:3] == masks.shape
                assert states.shape[-1] == args.input_dim, (states.shape[-1], args.input_dim)
                print("mask ratio:", masks.float().mean().item(), "state dim:", states.shape[-1])
                print("states:", states.shape, "masks:", masks.shape)
                print("mask min/max:", masks.min().item(), masks.max().item())
                print("states finite ratio:", torch.isfinite(states).float().mean().item())
                # 看看 mask=0 的地方 states 是否被置零（很多实现会这样做）
                print("masked states abs mean:", states[masks==0].abs().mean().item())
            # ====================================

            optimizer.zero_grad(set_to_none=True)
            preds = model(states, masks)
            losses = loss_fn(preds, {"states": states, "masks": masks})

            loss = losses["total_loss"]

            # ---- Short open-loop rollout loss (xy-only, physical space) ----
            short_rollout_loss = torch.tensor(0.0, device=device)
            boundary_loss = torch.tensor(0.0, device=device)

            if args.short_rollout_weight > 0 and args.short_rollout_horizon and discrete_indices is not None:
                C = int(context_len)
                T = int(states.shape[1])
                H = int(min(args.short_rollout_horizon, max(0, T - C)))

                if H > 0 and loss_fn._stats_initialized:
                    # scheduled sampling probability (linear decay from start->end)
                    if args.epochs > 1:
                        frac = float(epoch) / float(max(1, args.epochs - 1))
                    else:
                        frac = 1.0
                    tf_prob = float(args.scheduled_sampling_start) * (1.0 - frac) + float(args.scheduled_sampling_end) * frac
                    tf_prob = max(0.0, min(1.0, tf_prob))

                    ctx_states = states[:, :C]
                    ctx_masks = masks[:, :C]

                    pred_roll_cont, _ = model.rollout_train(
                        initial_states=ctx_states,
                        initial_masks=ctx_masks,
                        continuous_indices=continuous_indices,
                        discrete_indices=discrete_indices,
                        n_steps=H,
                        teacher_forcing_prob=tf_prob,
                        ground_truth_states=states,
                        use_soft_masks=True,
                    )  # [B,H,K,F_cont]

                    # Stable existence mask: require gt exists at t-1 and t to avoid 0->1 jumps.
                    stable_mask = masks[:, C - 1 : C - 1 + H] * masks[:, C : C + H]  # [B,H,K]

                    # Denormalize xy (pixel space)
                    x_ci = continuous_indices.index(0)
                    y_ci = continuous_indices.index(1)
                    x_mean, x_std = loss_fn.norm_mean[x_ci], loss_fn.norm_std[x_ci]
                    y_mean, y_std = loss_fn.norm_mean[y_ci], loss_fn.norm_std[y_ci]

                    pred_x = pred_roll_cont[..., x_ci] * x_std + x_mean
                    pred_y = pred_roll_cont[..., y_ci] * y_std + y_mean

                    gt_x = states[:, C : C + H, :, 0] * x_std + x_mean
                    gt_y = states[:, C : C + H, :, 1] * y_std + y_mean

                    dx = pred_x - gt_x
                    dy = pred_y - gt_y
                    err = torch.stack([dx, dy], dim=-1)  # [B,H,K,2]
                    abs_err = err.abs()
                    beta = 1.0
                    hub = torch.where(abs_err < beta, 0.5 * (err ** 2) / beta, abs_err - 0.5 * beta)  # [B,H,K,2]

                    hub = hub * stable_mask.unsqueeze(-1)
                    denom = stable_mask.sum() * 2.0
                    short_rollout_loss = hub.sum() / denom.clamp(min=1.0)

                    # ---- Soft boundary penalty on rollout xy ----
                    if args.boundary_weight > 0 and x_min is not None:
                        pen_x = F.relu(x_min - pred_x) + F.relu(pred_x - x_max)
                        pen_y = F.relu(y_min - pred_y) + F.relu(pred_y - y_max)
                        pen = (pen_x ** 2 + pen_y ** 2) * stable_mask
                        boundary_loss = pen.sum() / stable_mask.sum().clamp(min=1.0)

                    loss = loss + float(args.short_rollout_weight) * short_rollout_loss + float(args.boundary_weight) * boundary_loss

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = states.size(0)
            running += float(loss.item()) * bs
            n += bs
            if args.short_rollout_weight > 0 or args.boundary_weight > 0:
                pbar.set_postfix(
                    loss=running / max(1, n),
                    roll=float(short_rollout_loss.detach().item()),
                    bnd=float(boundary_loss.detach().item()),
                )
            else:
                pbar.set_postfix(loss=running / max(1, n))

        val_metrics = evaluate(
            model, val_loader, loss_fn, device,
            context_len=context_len,
            feature_names=feature_names,
            continuous_indices=continuous_indices,
            discrete_indices=discrete_indices,
            disp_stride=args.disp_stride,
            compute_diagnostics=True,
            short_rollout_horizon=args.short_rollout_horizon,
            short_rollout_weight=args.short_rollout_weight,
            boundary_weight=args.boundary_weight,
            boundary_sigma=args.boundary_sigma,
            boundary_margin_px=args.boundary_margin_px,
        )
        val_loss = val_metrics["total_loss"]

        # Log basic losses (including angle losses)
        logger.info(f"[Epoch {epoch+1}/{args.epochs}] train_loss={running/max(1,n):.4f}  val_loss={val_loss:.4f}  "
                    f"recon={val_metrics['recon_loss']:.4f} pred={val_metrics['pred_loss']:.4f} "
                    f"exist={val_metrics['exist_loss']:.4f} pred_exist={val_metrics['pred_exist_loss']:.4f}")
        
        # Log angle losses separately
        if "recon_angle_loss" in val_metrics and "pred_angle_loss" in val_metrics:
            logger.info(f"  [ANGLE] recon_angle={val_metrics['recon_angle_loss']:.4f}  "
                        f"pred_angle={val_metrics['pred_angle_loss']:.4f}")
        
        # Log velocity direction losses
        if "recon_vel_dir_loss" in val_metrics and "pred_vel_dir_loss" in val_metrics:
            if args.disable_vxy_supervision:
                logger.info(
                    "  [VEL-DIR-LOSS] (disabled; vx/vy not supervised) "
                    f"recon={val_metrics['recon_vel_dir_loss']:.4f}  pred={val_metrics['pred_vel_dir_loss']:.4f}"
                )
            else:
                logger.info(
                    f"  [VEL-DIR-LOSS] recon={val_metrics['recon_vel_dir_loss']:.4f}  "
                    f"pred={val_metrics['pred_vel_dir_loss']:.4f}"
                )
        
        # Log kinematic consistency loss
        if "kinematic_loss" in val_metrics and val_metrics["kinematic_loss"] > 0:
            logger.info(f"  [KINEMATIC-LOSS] {val_metrics['kinematic_loss']:.4f} (v_pred ~ dx/dt consistency)")

        # Log diagnostics
        if "diag_pred_ADE_xy" in val_metrics:
            logger.info(f"  [DIAG] ADE={val_metrics['diag_pred_ADE_xy']:.4f}  FDE={val_metrics['diag_pred_FDE_xy']:.4f}  "
                        f"VelErr={val_metrics.get('diag_pred_avg_vel_error', 0):.4f}")
        
        # Log velocity direction error (the key metric we want to improve)
        if "diag_pred_velocity_direction_error_deg" in val_metrics:
            coverage_pct = val_metrics.get("diag_moving_mask_coverage", 0) * 100
            if args.disable_vxy_supervision:
                logger.info(
                    "  [VEL-DIR-ERROR (diag-only; vx/vy not supervised)] "
                    f"{val_metrics['diag_pred_velocity_direction_error_deg']:.2f}° "
                    f"({val_metrics['diag_pred_velocity_direction_error_rad']:.4f} rad) "
                    f"[mask_cov={coverage_pct:.1f}%]"
                )
            else:
                logger.info(
                    f"  [VEL-DIR-ERROR (vx/vy)] {val_metrics['diag_pred_velocity_direction_error_deg']:.2f}° "
                    f"({val_metrics['diag_pred_velocity_direction_error_rad']:.4f} rad) "
                    f"[mask_cov={coverage_pct:.1f}%]"
                )
        
        # Log displacement direction errors (THREE SEPARATE METRICS)
        # A. DISP-DIR-ERR: pred displacement vs GT displacement (THE KEY METRIC)
        if "diag_disp_dir_err_pred_gt_deg" in val_metrics:
            if args.disable_vxy_supervision:
                logger.info(
                    f"  [VEL-DIR (position-based) TF Δp̂ vs Δp] {val_metrics['diag_disp_dir_err_pred_gt_deg']:.5f}° "
                    f"(teacher-forced one-step, stride={args.disp_stride})"
                )
            else:
                logger.info(
                    f"  [DISP-DIR-ERR TF predΔ vs gtΔ] {val_metrics['diag_disp_dir_err_pred_gt_deg']:.5f}° "
                    f"(teacher-forced one-step, stride={args.disp_stride})"
                )

        if "diag_disp_dir_err_rollout_pred_gt_deg" in val_metrics:
            if args.disable_vxy_supervision:
                logger.info(
                    f"  [VEL-DIR (position-based) RL Δp̂ vs Δp] {val_metrics['diag_disp_dir_err_rollout_pred_gt_deg']:.5f}° "
                    f"(open-loop rollout, stride={args.disp_stride})"
                )
            else:
                logger.info(
                    f"  [DISP-DIR-ERR RL predΔ vs gtΔ] {val_metrics['diag_disp_dir_err_rollout_pred_gt_deg']:.5f}° "
                    f"(open-loop rollout, stride={args.disp_stride})"
                )
        
        # B. GT-KIN-BASELINE: GT velocity vs GT displacement (data baseline)
        if "diag_gt_kin_baseline_pred_gt_deg" in val_metrics:
            logger.info(
                f"  [GT-KIN-BASELINE gtV vs gtΔ] {val_metrics['diag_gt_kin_baseline_pred_gt_deg']:.5f}° "
                f"(data baseline, stride={args.disp_stride})"
            )
        
        # Legacy old key (if exists, for backward compatibility)
        if "diag_pred_displacement_direction_error_deg" in val_metrics:
            logger.info(f"  [LEGACY DISP-DIR] {val_metrics['diag_pred_displacement_direction_error_deg']:.2f}° "
                        f"(old key, may be incorrect)")
        
        # C. KIN-DIR-ERR: pred velocity vs pred displacement (should decrease with kinematic loss)
        if "diag_kinematic_direction_error_deg" in val_metrics and val_metrics['diag_kinematic_direction_error_deg'] > 0:
            logger.info(f"  [KIN-DIR-ERR predV vs predΔ] {val_metrics['diag_kinematic_direction_error_deg']:.2f}° "
                        f"(should decrease with kinematic_loss)")
        
        # Legacy compatibility (if old key exists)
        if "diag_pred_displacement_direction_error_deg" in val_metrics:
            logger.info(f"  [LEGACY DISP-DIR] {val_metrics['diag_pred_displacement_direction_error_deg']:.2f}° (old key, may be incorrect)")

        # Existence metrics
        if "diag_exist_ctx_acc" in val_metrics:
            logger.info(f"  [EXIST-CTX] Acc={val_metrics['diag_exist_ctx_acc']:.4f}  "
                        f"Prec={val_metrics['diag_exist_ctx_prec']:.4f}  Rec={val_metrics['diag_exist_ctx_rec']:.4f}  "
                        f"FP={val_metrics['diag_exist_ctx_fp_rate']:.4f}  FN={val_metrics['diag_exist_ctx_fn_rate']:.4f}")

        if "diag_exist_pred_acc" in val_metrics:
            logger.info(f"  [EXIST-PRED] Acc={val_metrics['diag_exist_pred_acc']:.4f}  "
                        f"Prec={val_metrics['diag_exist_pred_prec']:.4f}  Rec={val_metrics['diag_exist_pred_rec']:.4f}  "
                        f"FP={val_metrics['diag_exist_pred_fp_rate']:.4f}  FN={val_metrics['diag_exist_pred_fn_rate']:.4f}")

        # Per-step MAE
        if "diag_pred_mae_per_step" in val_metrics:
            steps = val_metrics["diag_pred_mae_per_step"]
            step_str = ", ".join([f"{x:.4f}" for x in steps[:min(10, len(steps))]])
            if len(steps) > 10:
                step_str += ", ..."
            logger.info(f"  [MAE-PER-STEP] {step_str}")

        # All features MAE (reconstruction and prediction)
        # Create reverse mapping: feature_name -> index for sorting
        name_to_idx = {v: k for k, v in feature_names.items()}

        if "diag_recon_mae_per_feature" in val_metrics:
            recon_feat = val_metrics["diag_recon_mae_per_feature"]
            # Sort by feature index (0-11) for consistent order
            sorted_recon = sorted(recon_feat.items(), key=lambda x: name_to_idx.get(x[0], 999))
            logger.info("  [RECON MAE PER FEATURE]")
            for name, val in sorted_recon:
                logger.info(f"    {name:15s}: {val:.4f}")

        if "diag_pred_mae_per_feature" in val_metrics:
            pred_feat = val_metrics["diag_pred_mae_per_feature"]
            # Sort by feature index for consistent order
            sorted_pred = sorted(pred_feat.items(), key=lambda x: name_to_idx.get(x[0], 999))
            logger.info("  [PRED MAE PER FEATURE]")
            for name, val in sorted_pred:
                logger.info(f"    {name:15s}: {val:.4f}")

        # Update learning rate scheduler
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"  [LR] Current learning rate: {current_lr:.6f}")

        # Save diagnostics to JSON
        diag_json_path = Path(args.log_dir) / f"val_diag_epoch_{epoch+1:04d}.json"
        with open(diag_json_path, "w") as f:
            json.dump(val_metrics, f, indent=2)

        save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch, model_config)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "checkpoint_best.pt", model, optimizer, epoch, model_config)
            logger.info(f"New best model saved! val_loss={best_val:.4f}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
