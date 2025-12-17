"""
Two-panel animation on a blank canvas:
  Left  : Ground Truth (green) + moving circles
  Right : Prediction  (red)   + moving circles + "growing" predicted trajectory

- Background defaults to BLANK canvas (to avoid confusing real-image vehicles).
- Optionally you can use real site images if you still want.

Usage example (GIF):
python -m src.evaluation.visualize_predictions_two_panel \
  --checkpoint checkpoints/world_model_siteA_v0.5/checkpoint_best.pt \
  --test_data data/processed_siteA_v0.5/val_episodes.npz \
  --metadata data/processed_siteA_v0.5/metadata.json \
  --output_dir results/two_panel_gifs \
  --context_length 65 \
  --rollout_horizon 60 \
  --num_samples 30 \
  --max_agents 10 \
  --select_agents presence \
  --fps 12 \
  --blank_bg \
  --roi_margin_px 80

MP4 (requires ffmpeg installed):
  --save_mp4

Notes:
- WorldModel.rollout(...) is open-loop when teacher_forcing=False.
- We normalize ONLY the context before feeding into rollout.
- Predicted xy are denormalized back to pixels for plotting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from tqdm import tqdm

from src.models.world_model import WorldModel
from src.data.dataset import TrajectoryDataset
from src.utils.common import parse_discrete_feature_indices_from_metadata


# If your dataset uses different IDs, adjust here.
SITE_NAMES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I"
}


# ----------------------------
# Utilities (checkpoint inference)
# ----------------------------
def infer_embedding_sizes_from_ckpt(state_dict: dict) -> Tuple[int, int, int]:
    """Infer num_lanes/num_sites/num_classes from checkpoint embedding weights."""
    def _get(name: str, default: int) -> int:
        w = state_dict.get(name, None)
        return int(w.shape[0]) if w is not None else int(default)

    num_lanes = _get("encoder.lane_embedding.weight", 100)
    num_sites = _get("encoder.site_embedding.weight", 10)
    num_classes = _get("encoder.class_embedding.weight", 10)
    return num_lanes, num_sites, num_classes


def infer_latent_dim_from_ckpt(state_dict: dict, default: int = 256) -> int:
    """Infer latent_dim from encoder.to_latent.0.bias if present."""
    b = state_dict.get("encoder.to_latent.0.bias", None)
    return int(b.shape[0]) if b is not None else int(default)


# ----------------------------
# Stats + normalization helpers
# ----------------------------
def compute_mean_std_from_episodes_npz(
    episodes_npz_path: str,
    continuous_indices: List[int],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean/std for continuous features from train_episodes.npz.
    Expected keys: 'states' [N,T,K,F], 'masks' [N,T,K]
    """
    z = np.load(episodes_npz_path, allow_pickle=True)
    if "states" not in z.files or "masks" not in z.files:
        raise KeyError(f"{episodes_npz_path} must contain 'states' and 'masks'. Found keys: {z.files}")

    states = z["states"]  # [N,T,K,F]
    masks = z["masks"]    # [N,T,K]
    m = masks > 0.5

    # Some datasets compute derived interaction features (20..23) in __getitem__
    # and thus the visualization may request stats for indices beyond raw F.
    f_raw = int(states.shape[-1])
    max_idx = max(continuous_indices) if len(continuous_indices) else -1
    if max_idx >= f_raw:
        derived_supported = {20, 21, 22, 23}
        missing = sorted({i for i in continuous_indices if i >= f_raw})
        unsupported = [i for i in missing if i not in derived_supported]
        if unsupported:
            raise ValueError(
                f"Cannot compute stats for feature indices {unsupported}: raw episodes have F={f_raw} and "
                f"only derived indices {sorted(derived_supported)} are supported here."
            )

        f_aug = max(f_raw, max_idx + 1)
        states_aug = np.zeros(states.shape[:-1] + (f_aug,), dtype=states.dtype)
        states_aug[..., :f_raw] = states

        # 20: velocity_direction from vx=2, vy=3
        if 20 in missing:
            vx = states[..., 2]
            vy = states[..., 3]
            states_aug[..., 20] = np.arctan2(vy, vx)

        # 21: headway from rel_x_preceding (12)
        if 21 in missing:
            states_aug[..., 21] = states[..., 12]

        # 22/23: ttc and preceding_distance derived from relative states
        if 22 in missing or 23 in missing:
            rel_x = states[..., 12]
            rel_y = states[..., 13]
            rel_vx = states[..., 14]
            distance = np.sqrt(rel_x ** 2 + rel_y ** 2)

            if 23 in missing:
                states_aug[..., 23] = distance

            if 22 in missing:
                approaching = rel_vx < -0.1
                ttc = np.full(distance.shape, 100.0, dtype=np.float64)
                ttc[approaching] = (-distance[approaching] / rel_vx[approaching]).astype(np.float64)
                ttc = np.clip(ttc / 30.0, 0.0, 100.0)
                states_aug[..., 22] = ttc.astype(states.dtype, copy=False)

        states = states_aug

    mean_list, std_list = [], []
    for feat_idx in continuous_indices:
        vals = states[..., feat_idx][m]
        if vals.size == 0:
            raise RuntimeError(f"No valid values for feature {feat_idx} in {episodes_npz_path}")
        vals = vals.astype(np.float64)
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        if sd < eps:
            sd = eps
        mean_list.append(mu)
        std_list.append(sd)

    mean = torch.tensor(mean_list, dtype=torch.float32)
    std = torch.tensor(std_list, dtype=torch.float32)
    return mean, std


def normalize_full_states(
    states_full: torch.Tensor,
    mean_cont: torch.Tensor,
    std_cont: torch.Tensor,
    continuous_indices: List[int],
) -> torch.Tensor:
    """
    Normalize continuous features in FULL state tensor.
    states_full: [B,T,K,F_full] raw (pixel/raw)
    mean_cont/std_cont: [n_cont] aligned with continuous_indices order
    """
    out = states_full.clone()
    for j, feat_idx in enumerate(continuous_indices):
        out[..., feat_idx] = (out[..., feat_idx] - mean_cont[j]) / std_cont[j]
    return out


def denorm_pred_cont_to_xy_pixels(
    pred_cont: torch.Tensor,  # [B,H,K,F_cont] (order == continuous_indices)
    mean_cont: torch.Tensor,
    std_cont: torch.Tensor,
    continuous_indices: List[int],
) -> torch.Tensor:
    """Return [B,H,K,2] in pixel coords."""
    ix = continuous_indices.index(0)
    iy = continuous_indices.index(1)
    x = pred_cont[..., ix] * std_cont[ix] + mean_cont[ix]
    y = pred_cont[..., iy] * std_cont[iy] + mean_cont[iy]
    return torch.stack([x, y], dim=-1)


def get_site_id_from_states(states_full: torch.Tensor, masks: torch.Tensor, site_id_feat: int) -> int:
    """
    states_full: [T,K,F], masks: [T,K]
    Pick first valid agent at early timesteps and read site_id.
    """
    T, K, _ = states_full.shape
    for t in range(min(T, 5)):
        valid = masks[t] > 0.5
        if valid.any():
            k = int(torch.nonzero(valid, as_tuple=False)[0].item())
            return int(states_full[t, k, site_id_feat].item())
    return int(states_full[0, 0, site_id_feat].item())


# ----------------------------
# Agent selection
# ----------------------------
def select_agents_by_error(
    gt_xy: torch.Tensor,           # [H,K,2]
    pred_xy: torch.Tensor,         # [H,K,2]
    gt_mask: torch.Tensor,         # [H,K]
    max_agents: int,
) -> torch.Tensor:
    """Select agents with largest future ADE (pixels)."""
    err = ((pred_xy - gt_xy) ** 2).sum(-1).sqrt()  # [H,K]
    err = err * gt_mask
    ade = err.sum(0) / (gt_mask.sum(0) + 1e-6)     # [K]
    ade = torch.where(gt_mask.sum(0) > 0, ade, torch.full_like(ade, -1e9))
    idx = torch.argsort(ade, descending=True)
    return idx[:max_agents]


def select_agents_by_presence(gt_mask: torch.Tensor, max_agents: int) -> torch.Tensor:
    """Select agents that appear for the longest duration in horizon."""
    counts = gt_mask.sum(0)
    idx = torch.argsort(counts, descending=True)
    return idx[:max_agents]


# ----------------------------
# ROI + background
# ----------------------------
def _clip_roi(xmin, ymin, xmax, ymax, W, H):
    xmin = max(0, min(W - 1, xmin))
    xmax = max(0, min(W - 1, xmax))
    ymin = max(0, min(H - 1, ymin))
    ymax = max(0, min(H - 1, ymax))
    if xmax <= xmin + 2:
        xmax = min(W - 1, xmin + 3)
    if ymax <= ymin + 2:
        ymax = min(H - 1, ymin + 3)
    return xmin, ymin, xmax, ymax


def compute_roi_from_all_agents(
    ctx_xy: torch.Tensor,   # [C,A,2]
    gt_xy: torch.Tensor,    # [H,A,2]
    pred_xy: torch.Tensor,  # [H,A,2]
    margin_px: int,
    canvas_W: int,
    canvas_H: int,
) -> Tuple[int, int, int, int]:
    pts = torch.cat([ctx_xy.reshape(-1, 2), gt_xy.reshape(-1, 2), pred_xy.reshape(-1, 2)], dim=0)
    xmin = int(torch.floor(pts[:, 0].min()).item()) - margin_px
    xmax = int(torch.ceil(pts[:, 0].max()).item()) + margin_px
    ymin = int(torch.floor(pts[:, 1].min()).item()) - margin_px
    ymax = int(torch.ceil(pts[:, 1].max()).item()) + margin_px
    return _clip_roi(xmin, ymin, xmax, ymax, canvas_W, canvas_H)


def make_blank_canvas_like(img: np.ndarray, white: bool = True) -> np.ndarray:
    H, W = img.shape[0], img.shape[1]
    if white:
        return np.full((H, W, 3), 255, dtype=np.uint8)
    return np.zeros((H, W, 3), dtype=np.uint8)


# ----------------------------
# Two-panel animation (Left GT, Right Pred)
# ----------------------------
def make_two_panel_animation(
    background_img: np.ndarray,     # [H,W,3]
    ctx_xy: np.ndarray,             # [C,A,2]
    ctx_mask: Optional[np.ndarray], # [C,A]
    gt_xy: np.ndarray,              # [Hh,A,2]
    pred_xy: np.ndarray,            # [Hh,A,2]
    gt_mask: np.ndarray,            # [Hh,A]
    pred_mask: Optional[np.ndarray],
    title: Optional[str],
    roi: Optional[Tuple[int, int, int, int]],
    fps: int,
    circle_radius: int,
    show_ctx: bool,
    show_full_pred_faint: bool,
) -> Tuple[plt.Figure, animation.FuncAnimation]:
    Hh, A, _ = gt_xy.shape
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=200, constrained_layout=True)
    ax_gt, ax_pr = axes

    for ax in axes:
        ax.imshow(background_img)
        ax.axis("off")
        ax.set_aspect("equal", adjustable="box")
        if roi is not None:
            xmin, ymin, xmax, ymax = roi
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)  # image coords

    if title:
        fig.suptitle(title, fontsize=12)
    ax_gt.set_title("GT (green) — moving circles", fontsize=11)
    ax_pr.set_title("Pred (red) — moving circles + growing traj", fontsize=11)

    # Optional: show context as static (blue-ish default)
    if show_ctx:
        for a in range(A):
            if ctx_mask is not None:
                traj = ctx_xy[:, a, :].astype(np.float32, copy=True)
                m = ctx_mask[:, a] > 0.5
                traj[~m, :] = np.nan
                ax_gt.plot(traj[:, 0], traj[:, 1], linewidth=2.0, alpha=0.25)
                ax_pr.plot(traj[:, 0], traj[:, 1], linewidth=2.0, alpha=0.25)
            else:
                ax_gt.plot(ctx_xy[:, a, 0], ctx_xy[:, a, 1], linewidth=2.0, alpha=0.25)
                ax_pr.plot(ctx_xy[:, a, 0], ctx_xy[:, a, 1], linewidth=2.0, alpha=0.25)

    # Optional: show full predicted future trajectory faintly on right (helps perception)
    if show_full_pred_faint:
        for a in range(A):
            if pred_mask is not None:
                traj = pred_xy[:, a, :].astype(np.float32, copy=True)
                m = pred_mask[:, a] > 0.5
                traj[~m, :] = np.nan
                ax_pr.plot(traj[:, 0], traj[:, 1], linewidth=2.0, alpha=0.15, color="tab:red")
            else:
                ax_pr.plot(pred_xy[:, a, 0], pred_xy[:, a, 1], linewidth=2.0, alpha=0.15, color="tab:red")

    # Dynamic artists
    gt_lines, pr_lines = [], []
    gt_circles, pr_circles = [], []

    for _ in range(A):
        (gt_ln,) = ax_gt.plot([], [], linewidth=2.8, alpha=0.95, color="tab:green")
        (pr_ln,) = ax_pr.plot([], [], linewidth=3.0, alpha=0.95, color="tab:red")
        gt_lines.append(gt_ln)
        pr_lines.append(pr_ln)

        gc = Circle((0, 0), radius=circle_radius, fill=False, linewidth=2.2, alpha=0.95, edgecolor="tab:green")
        pc = Circle((0, 0), radius=circle_radius, fill=False, linewidth=2.2, alpha=0.95, edgecolor="tab:red")
        ax_gt.add_patch(gc)
        ax_pr.add_patch(pc)
        gt_circles.append(gc)
        pr_circles.append(pc)

    def init():
        for a in range(A):
            gt_lines[a].set_data([], [])
            pr_lines[a].set_data([], [])
            gt_circles[a].center = (-9999, -9999)
            pr_circles[a].center = (-9999, -9999)
        return gt_lines + pr_lines + gt_circles + pr_circles

    def update(t: int):
        for a in range(A):
            # LEFT: GT grows + circle moves
            if gt_mask[t, a] > 0.5:
                # grow line, but break on gaps using NaNs
                seg = gt_xy[: t + 1, a, :].astype(np.float32, copy=True)
                m = gt_mask[: t + 1, a] > 0.5
                seg[~m, :] = np.nan
                gt_lines[a].set_data(seg[:, 0], seg[:, 1])
                gt_circles[a].center = (gt_xy[t, a, 0], gt_xy[t, a, 1])
            else:
                gt_circles[a].center = (-9999, -9999)

            # RIGHT: Pred grows + circle moves
            if pred_mask is None:
                seg = pred_xy[: t + 1, a, :]
                pr_lines[a].set_data(seg[:, 0], seg[:, 1])
                pr_circles[a].center = (pred_xy[t, a, 0], pred_xy[t, a, 1])
            else:
                if pred_mask[t, a] > 0.5:
                    seg = pred_xy[: t + 1, a, :].astype(np.float32, copy=True)
                    m = pred_mask[: t + 1, a] > 0.5
                    seg[~m, :] = np.nan
                    pr_lines[a].set_data(seg[:, 0], seg[:, 1])
                    pr_circles[a].center = (pred_xy[t, a, 0], pred_xy[t, a, 1])
                else:
                    pr_circles[a].center = (-9999, -9999)

        return gt_lines + pr_lines + gt_circles + pr_circles

    ani = animation.FuncAnimation(
        fig, update, frames=Hh, init_func=init,
        interval=int(1000 / max(1, fps)), blit=True
    )
    return fig, ani


# ----------------------------
# Main
# ----------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Two-panel (GT vs Pred) animations on blank canvas.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/two_panel_gifs")

    parser.add_argument("--context_length", type=int, default=65)
    parser.add_argument("--rollout_horizon", type=int, default=60)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--max_agents", type=int, default=10)
    parser.add_argument("--sampling", type=str, default="random", choices=["random", "uniform"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select_agents", type=str, default="presence", choices=["error", "presence"])

    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--circle_radius", type=int, default=10)
    parser.add_argument("--roi_margin_px", type=int, default=80)

    # Background control
    parser.add_argument("--blank_bg", action="store_true", help="Use blank background (recommended).")
    parser.add_argument("--bg_white", action="store_true", help="Blank background is white (default).")
    parser.add_argument("--site_images_dir", type=str, default=None,
                        help="If provided and --blank_bg is NOT set, use real images from this dir (SiteA.jpg...).")
    parser.add_argument("--site_id_feat", type=int, default=11)

    # Visual toggles
    parser.add_argument("--show_ctx", action="store_true", help="Draw context trajectories faintly on both panels.")
    parser.add_argument("--show_full_pred_faint", action="store_true", help="Draw full pred trajectory faintly on right.")

    # Export formats
    parser.add_argument("--save_mp4", action="store_true", help="Save MP4 instead of GIF (needs ffmpeg).")

    # Model architecture overrides (optional)
    parser.add_argument("--input_dim", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--dynamics_layers", type=int, default=None)
    parser.add_argument("--dynamics_heads", type=int, default=None)
    parser.add_argument("--continuous_dim", type=int, default=None)
    parser.add_argument("--max_agents_model", type=int, default=None)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    # Load dataset (raw pixels)
    test_dataset = TrajectoryDataset(data_path=args.test_data, normalize=False)
    N = len(test_dataset)
    if N == 0:
        raise RuntimeError("Empty dataset!")

    # Infer full feature dim from data (more reliable than metadata)
    sample0 = test_dataset[0]
    F_full = int(sample0["states"].shape[-1])

    # Discrete indices (use helper)
    lane_idx, class_idx, site_idx = parse_discrete_feature_indices_from_metadata(
        metadata, fallback=(8, 7, 11), strict=False
    )
    discrete_indices = [int(class_idx), int(lane_idx), int(site_idx)]

    # Continuous indices: all except discrete & angle_idx (if valid)
    vi = metadata.get("validation_info", {})
    angle_idx = int(vi.get("angle_idx", 6))
    all_indices = set(range(F_full))
    skip = set(discrete_indices)
    if 0 <= angle_idx < F_full:
        skip.add(angle_idx)
    continuous_indices = sorted(list(all_indices - skip))

    # Prefer using the exact train-time normalization stats/indices if present next to checkpoint.
    # This keeps (mean/std, continuous_indices, discrete_indices) consistent with the model.
    stats_npz = Path(args.checkpoint).parent / "normalization_stats.npz"
    mean_cont = None
    std_cont = None
    if stats_npz.exists():
        zstats = np.load(str(stats_npz))
        if {"mean", "std", "continuous_indices", "discrete_indices"}.issubset(set(zstats.files)):
            mean_cont = torch.from_numpy(zstats["mean"]).float()
            std_cont = torch.from_numpy(zstats["std"]).float().clamp(min=1e-6)
            continuous_indices = [int(x) for x in zstats["continuous_indices"].tolist()]
            discrete_indices = [int(x) for x in zstats["discrete_indices"].tolist()]
        else:
            print(f"Warning: {stats_npz} missing expected keys; falling back to on-the-fly stats.")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"]
    model_config = ckpt.get("config", {})

    # Resolve model dims
    input_dim = args.input_dim or model_config.get("input_dim", F_full)
    latent_dim = args.latent_dim or model_config.get("latent_dim", infer_latent_dim_from_ckpt(state_dict, default=256))
    dynamics_layers = args.dynamics_layers or model_config.get("dynamics_layers", 4)
    dynamics_heads = args.dynamics_heads or model_config.get("dynamics_heads", 8)

    model_max_agents = int(args.max_agents_model or model_config.get("max_agents", 50))

    if args.continuous_dim is not None:
        continuous_dim = int(args.continuous_dim)
    elif "continuous_dim" in model_config:
        continuous_dim = int(model_config["continuous_dim"])
    elif "decoder.state_head.bias" in state_dict:
        decoder_out = state_dict["decoder.state_head.bias"].shape[0]
        continuous_dim = int(decoder_out // model_max_agents)
    else:
        # fallback guess
        continuous_dim = len(continuous_indices)

    num_lanes, num_sites, num_classes = infer_embedding_sizes_from_ckpt(state_dict)

    model = WorldModel(
        input_dim=int(input_dim),
        continuous_dim=int(continuous_dim),
        max_agents=int(model_max_agents),
        latent_dim=int(latent_dim),
        dynamics_layers=int(dynamics_layers),
        dynamics_heads=int(dynamics_heads),
        num_lanes=int(num_lanes),
        num_sites=int(num_sites),
        num_classes=int(num_classes),
        lane_feature_idx=int(lane_idx),
        class_feature_idx=int(class_idx),
        site_feature_idx=int(site_idx),
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Keep rollout behavior consistent with train-time config (if present)
    if bool(model_config.get("rollout_prior_velocity_from_positions", False)):
        model.rollout_prior_velocity_from_positions = True

    # Mean/std (prefer stats file, else compute from train_episodes.npz)
    if mean_cont is None or std_cont is None:
        train_npz = str(Path(args.test_data).parent / "train_episodes.npz")
        mean_cont, std_cont = compute_mean_std_from_episodes_npz(train_npz, continuous_indices)

    # Final sanity: continuous_indices must match model continuous_dim.
    if len(continuous_indices) != int(continuous_dim):
        # Common default layout in this repo for continuous_dim=15.
        if int(continuous_dim) == 15 and F_full >= 24:
            continuous_indices = [0, 1, 2, 3, 4, 5, 9, 12, 13, 14, 15, 20, 21, 22, 23]
            if mean_cont.numel() != len(continuous_indices) or std_cont.numel() != len(continuous_indices):
                train_npz = str(Path(args.test_data).parent / "train_episodes.npz")
                mean_cont, std_cont = compute_mean_std_from_episodes_npz(train_npz, continuous_indices)
        else:
            raise ValueError(
                f"continuous_indices length ({len(continuous_indices)}) does not match model continuous_dim ({continuous_dim}). "
                f"Use a compatible stats file (normalization_stats.npz) or adjust indices."
            )
    mean_cont_d = mean_cont.to(device)
    std_cont_d = std_cont.to(device)

    # Set stats for kinematic prior
    model.set_normalization_stats(mean_cont.cpu().numpy(), std_cont.cpu().numpy(), continuous_indices)

    # Background images (optional)
    site_images: Dict[int, np.ndarray] = {}
    if (not args.blank_bg) and (args.site_images_dir is not None):
        from PIL import Image
        site_images_dir = Path(args.site_images_dir)
        for sid, name in SITE_NAMES.items():
            p = site_images_dir / f"Site{name}.jpg"
            if p.exists():
                site_images[sid] = np.array(Image.open(p).convert("RGB"))

    # Choose indices
    n_take = min(args.num_samples, N)
    if args.sampling == "random":
        idxs = np.random.choice(np.arange(N), size=n_take, replace=False).tolist()
    else:
        step = max(1, N // max(1, n_take))
        idxs = list(range(0, N, step))[:n_take]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for ds_idx in tqdm(idxs, desc="Export animations"):
        item = test_dataset[ds_idx]
        states_full = item["states"].to(device)  # [T,K,F] raw pixels
        masks = item["masks"].to(device)         # [T,K]
        T, K, _ = states_full.shape

        if T < args.context_length + args.rollout_horizon:
            continue

        # Determine background canvas
        if args.blank_bg or args.site_images_dir is None:
            # if we have no real image, make a blank canvas size based on metadata or infer from positions
            # simplest: choose a reasonable canvas size; or if site image exists, match it
            # Here: if any site image loaded, match it; else use fixed 1920x1080
            if len(site_images) > 0:
                # pick first loaded image
                any_img = next(iter(site_images.values()))
                bg = make_blank_canvas_like(any_img, white=True if args.bg_white or True else False)
            else:
                bg = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        else:
            site_id = get_site_id_from_states(states_full, masks, args.site_id_feat)
            if site_id not in site_images:
                continue
            bg = site_images[site_id]

        H_img, W_img = bg.shape[0], bg.shape[1]

        # Slice context + future
        C = args.context_length
        Hh = args.rollout_horizon

        ctx_full = states_full[:C].unsqueeze(0)       # [1,C,K,F]
        ctx_masks = masks[:C].unsqueeze(0)            # [1,C,K]
        gt_future_full = states_full[C:C+Hh]          # [H,K,F]
        gt_future_masks = masks[C:C+Hh]               # [H,K]

        # Normalize context for model
        ctx_norm_full = normalize_full_states(ctx_full, mean_cont_d, std_cont_d, continuous_indices)

        # Open-loop rollout
        with torch.no_grad():
            pred_cont, pred_masks = model.rollout(
                initial_states=ctx_norm_full,
                initial_masks=ctx_masks,
                continuous_indices=continuous_indices,
                discrete_indices=discrete_indices,
                n_steps=Hh,
                teacher_forcing=False,
            )

        # Denorm to xy pixels
        pred_xy = denorm_pred_cont_to_xy_pixels(pred_cont, mean_cont_d, std_cont_d, continuous_indices)[0]  # [H,K,2]
        gt_xy = gt_future_full[:, :, :2]  # [H,K,2]
        ctx_xy = states_full[:C, :, :2]   # [C,K,2]

        # Select agents
        if args.select_agents == "error":
            agent_idx = select_agents_by_error(gt_xy, pred_xy, gt_future_masks, args.max_agents)
        else:
            agent_idx = select_agents_by_presence(gt_future_masks, args.max_agents)

        # Gather A agents
        ctx_xy_a = ctx_xy[:, agent_idx].detach().cpu().numpy()               # [C,A,2]
        gt_xy_a = gt_xy[:, agent_idx].detach().cpu().numpy()                 # [H,A,2]
        pred_xy_a = pred_xy[:, agent_idx].detach().cpu().numpy()             # [H,A,2]
        gt_mask_a = gt_future_masks[:, agent_idx].detach().cpu().numpy()     # [H,A]
        pred_mask_a = pred_masks[0, :, agent_idx].detach().cpu().numpy()     # [H,A]

        # ROI (fixed for both panels)
        roi = compute_roi_from_all_agents(
            torch.from_numpy(ctx_xy_a),
            torch.from_numpy(gt_xy_a),
            torch.from_numpy(pred_xy_a),
            margin_px=args.roi_margin_px,
            canvas_W=W_img,
            canvas_H=H_img,
        )

        # Title
        title = f"sample {ds_idx} | agents={ctx_xy_a.shape[1]} | ctx={C} | H={Hh}"

        # Make animation
        fig, ani = make_two_panel_animation(
            background_img=bg,
            ctx_xy=ctx_xy_a,
            ctx_mask=ctx_masks[0, :, agent_idx].detach().cpu().numpy(),
            gt_xy=gt_xy_a,
            pred_xy=pred_xy_a,
            gt_mask=gt_mask_a,
            pred_mask=pred_mask_a,
            title=title,
            roi=roi,
            fps=args.fps,
            circle_radius=args.circle_radius,
            show_ctx=args.show_ctx,
            show_full_pred_faint=args.show_full_pred_faint,
        )

        # Save
        if args.save_mp4:
            out_path = out_dir / f"idx{ds_idx:06d}_A{ctx_xy_a.shape[1]:02d}.mp4"
            writer = animation.FFMpegWriter(fps=args.fps)
            ani.save(out_path, writer=writer)
        else:
            out_path = out_dir / f"idx{ds_idx:06d}_A{ctx_xy_a.shape[1]:02d}.gif"
            writer = animation.PillowWriter(fps=args.fps)
            ani.save(out_path, writer=writer)

        plt.close(fig)
        saved += 1

    print("=" * 60)
    print(f"Done. Saved {saved} animations to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
