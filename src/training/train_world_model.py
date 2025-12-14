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
from typing import Dict
from collections import defaultdict

import torch
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

    # loss weights
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--pred_weight", type=float, default=1.0)
    p.add_argument("--existence_weight", type=float, default=0.1)

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


def save_checkpoint(path: Path, model: WorldModel, optimizer: optim.Optimizer, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        str(path),
    )


def _safe_div(num, den, eps=1e-8):
    """Safe division avoiding division by zero."""
    return num / (den + eps)


@torch.no_grad()
def compute_val_diagnostics(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    feature_names: Dict[int, str],
    context_len: int = 65,
    pos_idx: tuple = (0, 1),
    vel_idx: tuple = (2, 3),
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
    gt_ctx = states[:, :C]          # [B, C, K, F]
    m_ctx = masks[:, :C]            # [B, C, K]
    gt_fut = states[:, C:C+R]       # [B, R, K, F]
    m_fut = masks[:, C:C+R]         # [B, R, K]

    recon_ctx = recon[:, :C]        # [B, C, K, F]
    pred_fut = pred[:, :C]          # [B, C, K, F] - predicted states for times 1:C+1

    # For prediction, we compare pred[:-1] with gt[1:] (one-step ahead)
    # But for rollout diagnostics, we want to compare future predictions
    # pred[:, C-1] predicts state at time C (first future state)
    pred_rollout = pred[:, C-1:C-1+R]  # [B, R, K, F]

    m_ctx_f = m_ctx.unsqueeze(-1)   # [B, C, K, 1]
    m_fut_f = m_fut.unsqueeze(-1)   # [B, R, K, 1]

    recon_abs = (recon_ctx - gt_ctx).abs()
    pred_abs = (pred_rollout - gt_fut).abs()

    diag = {}

    # ========== (A) Per-feature MAE ==========
    recon_mae_f = _safe_div(
        (recon_abs * m_ctx_f).sum(dim=(0, 1, 2)),
        m_ctx_f.sum(dim=(0, 1, 2))
    )
    pred_mae_f = _safe_div(
        (pred_abs * m_fut_f).sum(dim=(0, 1, 2)),
        m_fut_f.sum(dim=(0, 1, 2))
    )

    # Convert to named dict
    diag["recon_mae_per_feature"] = {
        feature_names.get(i, f"f{i}"): float(recon_mae_f[i])
        for i in range(F)
    }
    diag["pred_mae_per_feature"] = {
        feature_names.get(i, f"f{i}"): float(pred_mae_f[i])
        for i in range(F)
    }

    # ========== (B) Per-timestep MAE for rollout ==========
    pred_mae_t = _safe_div(
        (pred_abs * m_fut_f).sum(dim=(0, 2, 3)),
        m_fut_f.sum(dim=(0, 2, 3))
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
    compute_diagnostics: bool = True,
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
        compute_diagnostics: Whether to compute detailed diagnostics

    Returns:
        Dictionary with loss metrics and optional diagnostics
    """
    model.eval()
    totals = {"total_loss": 0.0, "recon_loss": 0.0, "pred_loss": 0.0, "exist_loss": 0.0, "pred_exist_loss": 0.0}
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

        # Accumulate losses
        for k in totals:
            totals[k] += float(losses[k].item()) * bs
        n += bs

        # Compute diagnostics
        if compute_diagnostics and feature_names is not None:
            diag = compute_val_diagnostics(
                preds,
                {"states": states, "masks": masks},
                feature_names,
                context_len=context_len,
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

    # Extract feature names for diagnostics
    feature_layout = meta.get("feature_layout", {})
    feature_names = {int(k): v for k, v in feature_layout.items()}

    device = args.device

    model = WorldModel(
        input_dim=args.input_dim,
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
    ).to(device)

    # Set normalization stats into the model (needed for kinematic prior)
    model.set_normalization_stats(
        train_loader.dataset.mean,
        train_loader.dataset.std,
        train_loader.dataset.continuous_indices,
    )

    loss_fn = WorldModelLoss(
        recon_weight=args.recon_weight,
        pred_weight=args.pred_weight,
        exist_weight=args.existence_weight,
        huber_beta=1.0,
        continuous_indices=train_loader.dataset.continuous_indices,
        use_pred_existence_loss=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        running = 0.0
        n = 0

        for batch in pbar:
            states = batch["states"].to(device)
            masks = batch["masks"].to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(states, masks)
            losses = loss_fn(preds, {"states": states, "masks": masks})
            loss = losses["total_loss"]

            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = states.size(0)
            running += float(loss.item()) * bs
            n += bs
            pbar.set_postfix(loss=running / max(1, n))

        val_metrics = evaluate(
            model, val_loader, loss_fn, device,
            context_len=context_len,
            feature_names=feature_names,
            compute_diagnostics=True
        )
        val_loss = val_metrics["total_loss"]

        # Log basic losses
        logger.info(f"[Epoch {epoch+1}/{args.epochs}] train_loss={running/max(1,n):.4f}  val_loss={val_loss:.4f}  "
                    f"recon={val_metrics['recon_loss']:.4f} pred={val_metrics['pred_loss']:.4f} "
                    f"exist={val_metrics['exist_loss']:.4f} pred_exist={val_metrics['pred_exist_loss']:.4f}")

        # Log diagnostics
        if "diag_pred_ADE_xy" in val_metrics:
            logger.info(f"  [DIAG] ADE={val_metrics['diag_pred_ADE_xy']:.4f}  FDE={val_metrics['diag_pred_FDE_xy']:.4f}  "
                        f"VelErr={val_metrics.get('diag_pred_avg_vel_error', 0):.4f}")

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

        # Save diagnostics to JSON
        diag_json_path = Path(args.log_dir) / f"val_diag_epoch_{epoch+1:04d}.json"
        with open(diag_json_path, "w") as f:
            json.dump(val_metrics, f, indent=2)

        save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "checkpoint_best.pt", model, optimizer, epoch)
            logger.info(f"New best model saved! val_loss={best_val:.4f}")

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
