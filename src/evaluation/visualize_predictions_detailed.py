
"""
Visualize trajectory predictions on real UAV site images (paper-style).

Outputs per-sample triple-panel figure + zoom-in inset:
  Panel A: Context + Ground Truth
  Panel B: Context + Prediction
  Panel C: Ground Truth vs Prediction (+ zoom-in)

Compatible with current codebase:
- WorldModel.__init__ has NO dynamics_type argument
- WorldModel.rollout(initial_states, initial_masks, continuous_indices, discrete_indices, ...) -> (pred_cont, pred_masks)
- WorldModel.set_normalization_stats(mean_cont, std_cont, continuous_indices) must be called so kinematic prior works

This script:
- Loads checkpoint, infers embedding sizes from checkpoint to avoid size mismatch
- Computes mean/std on-the-fly from train_episodes.npz (no separate stats file needed)
- Samples MANY episodes (default 50) and saves figures under output_dir

Usage:
python -m src.evaluation.visualize_predictions \
  --checkpoint checkpoints/world_model_siteA2/checkpoint_best.pt \
  --test_data data/processed_siteA/test_episodes.npz \
  --metadata data/processed_siteA/metadata.json \
  --site_images_dir src/evaluation/sites \
  --output_dir results/visualizations \
  --context_length 65 \
  --rollout_horizon 15 \
  --num_samples 50 \
  --max_agents 10 \
  --sampling random \
  --seed 42 \
  --select_agents error \
  --arrows_every 3

Expect site images named like:
  SiteA.jpg, SiteB.jpg, ...

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
from PIL import Image
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.models.world_model import WorldModel
from src.data.dataset import TrajectoryDataset


# If your dataset uses different IDs, adjust here.
SITE_NAMES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I"
}


# ----------------------------
# Utilities
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
    if b is None:
        return int(default)
    return int(b.shape[0])


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
    Pick first valid agent at t=0 (or later) and read site_id.
    """
    T, K, F = states_full.shape
    for t in range(min(T, 5)):
        valid = masks[t] > 0.5
        if valid.any():
            k = int(torch.nonzero(valid, as_tuple=False)[0].item())
            return int(states_full[t, k, site_id_feat].item())
    return int(states_full[0, 0, site_id_feat].item())


def select_agents_by_error(
    gt_xy: torch.Tensor,           # [H,K,2] pixel
    pred_xy: torch.Tensor,         # [H,K,2] pixel
    gt_mask: torch.Tensor,         # [H,K]
    max_agents: int,
) -> torch.Tensor:
    """
    Select agents with largest future ADE (pixels).
    Returns indices [A]
    """
    err = ((pred_xy - gt_xy) ** 2).sum(-1).sqrt()  # [H,K]
    err = err * gt_mask
    ade = err.sum(0) / (gt_mask.sum(0) + 1e-6)     # [K]
    ade = torch.where(gt_mask.sum(0) > 0, ade, torch.full_like(ade, -1e9))
    idx = torch.argsort(ade, descending=True)
    return idx[:max_agents]


def select_agents_by_presence(gt_mask: torch.Tensor, max_agents: int) -> torch.Tensor:
    counts = gt_mask.sum(0)  # [K]
    idx = torch.argsort(counts, descending=True)
    return idx[:max_agents]


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


def compute_zoom_roi_from_key_agent(
    gt_xy: torch.Tensor,     # [H,2] pixel
    pred_xy: torch.Tensor,   # [H,2] pixel
    margin_px: int,
    img_W: int,
    img_H: int,
) -> Tuple[int, int, int, int]:
    pts = torch.cat([gt_xy, pred_xy], dim=0)  # [2H,2]
    xmin = int(torch.floor(pts[:, 0].min()).item()) - margin_px
    xmax = int(torch.ceil(pts[:, 0].max()).item()) + margin_px
    ymin = int(torch.floor(pts[:, 1].min()).item()) - margin_px
    ymax = int(torch.ceil(pts[:, 1].max()).item()) + margin_px
    xmin, ymin, xmax, ymax = _clip_roi(xmin, ymin, xmax, ymax, img_W, img_H)
    return xmin, ymin, xmax, ymax


def draw_background(ax, img: np.ndarray):
    ax.imshow(img)
    ax.axis("off")


def draw_traj(ax, traj_xy: np.ndarray, color: str, lw: float = 2.2, alpha: float = 0.92, arrows_every: int = 0):
    if traj_xy.shape[0] < 2:
        return
    x = traj_xy[:, 0]
    y = traj_xy[:, 1]
    ax.plot(x, y, color=color, linewidth=lw, alpha=alpha)
    ax.scatter([x[0]], [y[0]], s=18, color=color, alpha=alpha, marker="o")
    ax.scatter([x[-1]], [y[-1]], s=22, color=color, alpha=alpha, marker="s")
    if arrows_every and traj_xy.shape[0] > 2:
        for t in range(0, traj_xy.shape[0] - 1, arrows_every):
            dx = x[t + 1] - x[t]
            dy = y[t + 1] - y[t]
            ax.arrow(
                x[t], y[t], dx, dy,
                length_includes_head=True,
                head_width=6, head_length=8,
                color=color, alpha=min(1.0, alpha),
                linewidth=0
            )


def draw_agents(
    ax,
    ctx_xy: np.ndarray,      # [C,A,2]
    gt_xy: np.ndarray,       # [H,A,2]
    pred_xy: np.ndarray,     # [H,A,2]
    gt_mask: np.ndarray,     # [H,A]
    pred_mask: Optional[np.ndarray],
    mode: str,
    arrows_every: int = 0,
):
    """
    mode:
      - "ctx_gt": ctx (blue) + gt (green)
      - "ctx_pred": ctx (blue) + pred (red)
      - "cmp": gt (green) + pred (red)
    """
    C, A, _ = ctx_xy.shape
    H, _, _ = gt_xy.shape

    for a in range(A):
        if mode in ("ctx_gt", "ctx_pred"):
            draw_traj(ax, ctx_xy[:, a, :], color="tab:blue", arrows_every=arrows_every)

        # GT masked
        m = gt_mask[:, a] > 0.5
        gt_traj = gt_xy[m, a, :] if m.any() else gt_xy[:, a, :]
        if mode in ("ctx_gt", "cmp"):
            draw_traj(ax, gt_traj, color="tab:green", arrows_every=arrows_every)

        # Pred masked (optional)
        if mode in ("ctx_pred", "cmp"):
            if pred_mask is not None:
                pm = pred_mask[:, a] > 0.5
                pred_traj = pred_xy[pm, a, :] if pm.any() else pred_xy[:, a, :]
            else:
                pred_traj = pred_xy[:, a, :]
            draw_traj(ax, pred_traj, color="tab:red", arrows_every=arrows_every)


def make_triple_panel_figure(
    background_img: np.ndarray,
    ctx_xy: np.ndarray,      # [C,A,2]
    gt_xy: np.ndarray,       # [H,A,2]
    pred_xy: np.ndarray,     # [H,A,2]
    gt_mask: np.ndarray,     # [H,A]
    pred_mask: Optional[np.ndarray],
    zoom_roi: Tuple[int, int, int, int],
    arrows_every: int = 0,
    title: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=220, constrained_layout=True)
    ax_gt, ax_pred, ax_cmp = axes

    for ax in axes:
        draw_background(ax, background_img)

    draw_agents(ax_gt, ctx_xy, gt_xy, pred_xy, gt_mask, pred_mask, mode="ctx_gt", arrows_every=arrows_every)
    ax_gt.set_title("Context + Ground Truth", fontsize=12)

    draw_agents(ax_pred, ctx_xy, gt_xy, pred_xy, gt_mask, pred_mask, mode="ctx_pred", arrows_every=arrows_every)
    ax_pred.set_title("Context + Prediction", fontsize=12)

    draw_agents(ax_cmp, ctx_xy, gt_xy, pred_xy, gt_mask, pred_mask, mode="cmp", arrows_every=arrows_every)
    ax_cmp.set_title("GT vs Prediction", fontsize=12)

    xmin, ymin, xmax, ymax = zoom_roi
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         edgecolor="black", facecolor="none", linewidth=2.0)
    ax_cmp.add_patch(rect)

    axins = inset_axes(ax_cmp, width="42%", height="42%", loc="upper right", borderpad=1.0)
    draw_background(axins, background_img)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymax, ymin)
    draw_agents(axins, ctx_xy, gt_xy, pred_xy, gt_mask, pred_mask, mode="cmp", arrows_every=arrows_every)
    axins.set_title("Zoom-in", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=13)

    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize predictions on real site images (paper-style).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--site_images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/visualizations")
    parser.add_argument("--context_length", type=int, default=65)
    parser.add_argument("--rollout_horizon", type=int, default=15)
    parser.add_argument("--num_samples", type=int, default=50, help="Total number of samples to export.")
    parser.add_argument("--max_agents", type=int, default=10, help="Max agents to draw per sample.")
    parser.add_argument("--sampling", type=str, default="random", choices=["random", "uniform"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select_agents", type=str, default="error", choices=["error", "presence"])
    parser.add_argument("--zoom_margin_px", type=int, default=60)
    parser.add_argument("--arrows_every", type=int, default=0, help="Draw direction arrows every N steps (0 disables).")
    parser.add_argument("--site_id_feat", type=int, default=11)
    
    # Model architecture parameters (optional, auto-detect from checkpoint)
    parser.add_argument("--input_dim", type=int, default=None, help="Input dimension (auto-detect if not provided)")
    parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension (auto-detect if not provided)")
    parser.add_argument("--dynamics_layers", type=int, default=None, help="Dynamics layers (auto-detect if not provided)")
    parser.add_argument("--dynamics_heads", type=int, default=None, help="Attention heads (auto-detect if not provided)")
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Site images: {args.site_images_dir}")
    print(f"Context: {args.context_length} frames, Rollout: {args.rollout_horizon} frames")

    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    continuous_indices = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    vi = metadata.get("validation_info", {})
    df = vi.get("discrete_features", {"class_id": 7, "lane_id": 8, "site_id": 11})
    discrete_indices = [int(df["class_id"]), int(df["lane_id"]), int(df["site_id"])]

    print(f"Continuous features: {continuous_indices}")
    print(f"Discrete features: {discrete_indices}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"]
    
    # Try to get model config from checkpoint (for newer checkpoints)
    model_config = ckpt.get("config", {})
    
    # Priority: CLI args > checkpoint config > inference from state_dict
    if args.input_dim is not None:
        input_dim = args.input_dim
    elif "input_dim" in model_config:
        input_dim = model_config["input_dim"]
    else:
        input_dim = int(metadata.get("n_features", 12))
    
    if args.latent_dim is not None:
        latent_dim = args.latent_dim
    elif "latent_dim" in model_config:
        latent_dim = model_config["latent_dim"]
    else:
        latent_dim = infer_latent_dim_from_ckpt(state_dict, default=256)
    
    if args.dynamics_layers is not None:
        dynamics_layers = args.dynamics_layers
    elif "dynamics_layers" in model_config:
        dynamics_layers = model_config["dynamics_layers"]
    else:
        dynamics_layers = 4  # default
    
    if args.dynamics_heads is not None:
        dynamics_heads = args.dynamics_heads
    elif "dynamics_heads" in model_config:
        dynamics_heads = model_config["dynamics_heads"]
    else:
        dynamics_heads = 8  # default
    
    # Get continuous_dim and max_agents from config or infer
    if "continuous_dim" in model_config:
        continuous_dim = int(model_config["continuous_dim"])
    elif "decoder.state_head.bias" in state_dict:
        # Infer from decoder output size
        decoder_out = state_dict["decoder.state_head.bias"].shape[0]
        model_max_agents = int(model_config.get("max_agents", 50))
        continuous_dim = decoder_out // model_max_agents
        print(f"  Inferred continuous_dim: {continuous_dim} (decoder output {decoder_out} / {model_max_agents} agents)")
    else:
        continuous_dim = 9  # default for 12-dim input
    
    model_max_agents = int(model_config.get("max_agents", 50))
    
    num_lanes, num_sites, num_classes = infer_embedding_sizes_from_ckpt(state_dict)

    from src.utils.common import parse_discrete_feature_indices_from_metadata
    lane_idx, class_idx, site_idx = parse_discrete_feature_indices_from_metadata(metadata, fallback=(8, 7, 11), strict=False)

    model = WorldModel(
        input_dim=input_dim,
        continuous_dim=continuous_dim,
        max_agents=model_max_agents,
        latent_dim=latent_dim,
        dynamics_layers=dynamics_layers,
        dynamics_heads=dynamics_heads,
        num_lanes=num_lanes,
        num_sites=num_sites,
        num_classes=num_classes,
        lane_feature_idx=lane_idx,
        class_feature_idx=class_idx,
        site_feature_idx=site_idx,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded: input_dim={input_dim}, continuous_dim={continuous_dim}, max_agents={model_max_agents}")
    print(f"  Architecture: latent_dim={latent_dim}, dynamics_layers={dynamics_layers}, dynamics_heads={dynamics_heads}")
    print(f"Embeddings: num_lanes={num_lanes}, num_sites={num_sites}, num_classes={num_classes}")


    # On-the-fly stats from train_episodes.npz (same folder)
    train_npz = str(Path(args.test_data).parent / "train_episodes.npz")
    print(f"Computing mean/std from: {train_npz}")
    mean_cont, std_cont = compute_mean_std_from_episodes_npz(train_npz, continuous_indices)
    mean_cont_d = mean_cont.to(device)
    std_cont_d = std_cont.to(device)

    model.set_normalization_stats(mean_cont.cpu().numpy(), std_cont.cpu().numpy(), continuous_indices)
    print("✅ set_normalization_stats() applied (cont_index_map initialized)")

    # Load site images
    site_images_dir = Path(args.site_images_dir)
    site_images: Dict[int, np.ndarray] = {}
    for sid, name in SITE_NAMES.items():
        p = site_images_dir / f"Site{name}.jpg"
        if p.exists():
            site_images[sid] = np.array(Image.open(p).convert("RGB"))
            print(f"Loaded image: {p.name} shape={site_images[sid].shape}")

    if not site_images:
        raise FileNotFoundError(f"No site images found in {site_images_dir}. Expect files like SiteA.jpg")

    # Dataset raw pixels
    test_dataset = TrajectoryDataset(data_path=args.test_data, normalize=False)
    N = len(test_dataset)

    # Choose indices (more images)
    n_take = min(args.num_samples, N)
    if args.sampling == "random":
        idxs = np.random.choice(np.arange(N), size=n_take, replace=False).tolist()
    else:
        step = max(1, N // max(1, n_take))
        idxs = list(range(0, N, step))[:n_take]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    print("\nGenerating figures...")
    for ds_idx in tqdm(idxs, desc="Samples"):
        item = test_dataset[ds_idx]
        states_full = item["states"].to(device)  # [T,K,F] pixel/raw
        masks = item["masks"].to(device)         # [T,K]

        T, K, F = states_full.shape
        if T < args.context_length + args.rollout_horizon:
            continue

        site_id = get_site_id_from_states(states_full, masks, args.site_id_feat)
        if site_id not in site_images:
            continue

        bg = site_images[site_id]
        H_img, W_img = bg.shape[0], bg.shape[1]

        # slices
        ctx_full = states_full[: args.context_length].unsqueeze(0)  # [1,C,K,F]
        ctx_masks = masks[: args.context_length].unsqueeze(0)       # [1,C,K]
        gt_future_full = states_full[args.context_length: args.context_length + args.rollout_horizon]  # [H,K,F]
        gt_future_masks = masks[args.context_length: args.context_length + args.rollout_horizon]       # [H,K]

        # normalize context for model input
        ctx_norm_full = normalize_full_states(ctx_full, mean_cont_d, std_cont_d, continuous_indices)

        # rollout
        pred_cont, pred_masks = model.rollout(
            initial_states=ctx_norm_full,
            initial_masks=ctx_masks,
            continuous_indices=continuous_indices,
            discrete_indices=discrete_indices,
            n_steps=args.rollout_horizon,
            teacher_forcing=False,
        )

        # denorm predicted xy to pixels
        pred_xy = denorm_pred_cont_to_xy_pixels(pred_cont, mean_cont_d, std_cont_d, continuous_indices)[0]  # [H,K,2]
        gt_xy = gt_future_full[:, :, :2]  # [H,K,2]
        ctx_xy = states_full[: args.context_length, :, :2]  # [C,K,2]

        # agent selection
        if args.select_agents == "error":
            agent_idx = select_agents_by_error(gt_xy, pred_xy, gt_future_masks, args.max_agents)
        else:
            agent_idx = select_agents_by_presence(gt_future_masks, args.max_agents)

        # gather
        ctx_xy_a = ctx_xy[:, agent_idx].detach().cpu().numpy()          # [C,A,2]
        gt_xy_a = gt_xy[:, agent_idx].detach().cpu().numpy()            # [H,A,2]
        pred_xy_a = pred_xy[:, agent_idx].detach().cpu().numpy()        # [H,A,2]
        gt_mask_a = gt_future_masks[:, agent_idx].detach().cpu().numpy()    # [H,A]
        pred_mask_a = pred_masks[0, :, agent_idx].detach().cpu().numpy()    # [H,A]

        # zoom ROI from key agent (first selected)
        key = 0
        gt_key = torch.from_numpy(gt_xy_a[:, key, :]).to(device)
        pr_key = torch.from_numpy(pred_xy_a[:, key, :]).to(device)
        zoom_roi = compute_zoom_roi_from_key_agent(gt_key, pr_key, args.zoom_margin_px, W_img, H_img)

        site_name = SITE_NAMES.get(site_id, f"id{site_id}")
        fig_title = f"Site {site_name} | sample {ds_idx} | agents={ctx_xy_a.shape[1]} | sel={args.select_agents}"

        fig = make_triple_panel_figure(
            background_img=bg,
            ctx_xy=ctx_xy_a,
            gt_xy=gt_xy_a,
            pred_xy=pred_xy_a,
            gt_mask=gt_mask_a,
            pred_mask=pred_mask_a,
            zoom_roi=zoom_roi,
            arrows_every=args.arrows_every,
            title=fig_title,
        )

        out_path = output_dir / f"site{site_name}_idx{ds_idx:06d}_A{ctx_xy_a.shape[1]:02d}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    print("\n" + "=" * 60)
    print(f"Done. Saved {saved} figures to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
