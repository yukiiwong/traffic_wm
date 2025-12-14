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
from pathlib import Path
from typing import Dict

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


@torch.no_grad()
def evaluate(model: WorldModel, loader, loss_fn: WorldModelLoss, device: str) -> Dict[str, float]:
    model.eval()
    totals = {"total_loss": 0.0, "recon_loss": 0.0, "pred_loss": 0.0, "exist_loss": 0.0, "pred_exist_loss": 0.0}
    n = 0
    for batch in loader:
        states = batch["states"].to(device)
        masks = batch["masks"].to(device)
        preds = model(states, masks)
        losses = loss_fn(preds, {"states": states, "masks": masks})
        bs = states.size(0)
        for k in totals:
            totals[k] += float(losses[k].item()) * bs
        n += bs
    for k in totals:
        totals[k] /= max(1, n)
    return totals


def main() -> None:
    args = parse_args()
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

        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["total_loss"]

        print(f"[Epoch {epoch+1}] train_loss={running/max(1,n):.4f}  val_loss={val_loss:.4f}  "
              f"recon={val_metrics['recon_loss']:.4f} pred={val_metrics['pred_loss']:.4f} "
              f"exist={val_metrics['exist_loss']:.4f} pred_exist={val_metrics['pred_exist_loss']:.4f}")

        save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, epoch)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "checkpoint_best.pt", model, optimizer, epoch)

    print("Training finished.")


if __name__ == "__main__":
    main()
