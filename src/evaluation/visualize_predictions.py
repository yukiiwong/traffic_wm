"""
Visualize Trajectory Predictions on Site Images

Draw ground truth and predicted trajectories on actual site aerial images.
Compatible with current WorldModel.rollout API:
  rollout(initial_states, initial_masks, continuous_indices, discrete_indices, ...) -> (pred_cont, pred_masks)
and current WorldModel __init__ (no dynamics_type arg).
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from PIL import Image

from src.models.world_model import WorldModel
from src.data.dataset import TrajectoryDataset


# Site ID to name mapping (adjust if your site ids differ)
SITE_NAMES = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I"
}


def _infer_embedding_sizes_from_ckpt(state_dict: dict) -> Tuple[int, int, int]:
    """Infer num_lanes/num_sites/num_classes from checkpoint weights to avoid size mismatch."""
    num_lanes = state_dict["encoder.lane_embedding.weight"].shape[0] if "encoder.lane_embedding.weight" in state_dict else 100
    num_sites = state_dict["encoder.site_embedding.weight"].shape[0] if "encoder.site_embedding.weight" in state_dict else 10
    num_classes = state_dict["encoder.class_embedding.weight"].shape[0] if "encoder.class_embedding.weight" in state_dict else 10
    return int(num_lanes), int(num_sites), int(num_classes)


def _compute_mean_std_from_episodes_npz(
    episodes_npz_path: str,
    continuous_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean/std for continuous features from train_episodes.npz.
    Expected keys: 'states' [N,T,K,F], 'masks' [N,T,K]
    Returns:
      mean_cont [n_cont], std_cont [n_cont]
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
        mu = vals.mean()
        sd = vals.std(ddof=0)
        if sd < 1e-6:
            sd = 1e-6
        mean_list.append(mu)
        std_list.append(sd)

    mean = torch.tensor(np.asarray(mean_list, dtype=np.float32))
    std = torch.tensor(np.asarray(std_list, dtype=np.float32))
    return mean, std


def normalize_states_full(
    states_full: torch.Tensor,
    mean_cont: torch.Tensor,
    std_cont: torch.Tensor,
    continuous_indices: List[int],
) -> torch.Tensor:
    """
    states_full: [B,T,K,F_full] in pixel/raw (NOT normalized)
    mean_cont/std_cont aligned with continuous_indices order
    """
    out = states_full.clone()
    for i, feat_idx in enumerate(continuous_indices):
        out[..., feat_idx] = (out[..., feat_idx] - mean_cont[i]) / std_cont[i]
    return out


def denormalize_pred_cont_to_xy_pixels(
    pred_cont: torch.Tensor,
    mean_cont: torch.Tensor,
    std_cont: torch.Tensor,
    continuous_indices: List[int],
) -> torch.Tensor:
    """
    pred_cont: [B,T,K,F_cont] where F_cont == len(continuous_indices) in the SAME ORDER as continuous_indices
    Return: pred_xy_pixels [B,T,K,2] in pixel coordinates
    """
    # find where x(0) and y(1) sit in the continuous vector
    ix = continuous_indices.index(0)
    iy = continuous_indices.index(1)
    x = pred_cont[..., ix] * std_cont[ix] + mean_cont[ix]
    y = pred_cont[..., iy] * std_cont[iy] + mean_cont[iy]
    return torch.stack([x, y], dim=-1)


def draw_polyline_on_image(img: np.ndarray, traj_xy: np.ndarray, color=(255, 0, 0), thickness=2):
    """
    img: HxWx3 uint8
    traj_xy: [T,2] (x,y) pixels
    """
    import cv2
    if traj_xy.shape[0] < 2:
        return img
    pts = traj_xy.astype(np.int32)
    for i in range(len(pts) - 1):
        x1, y1 = int(pts[i, 0]), int(pts[i, 1])
        x2, y2 = int(pts[i + 1, 0]), int(pts[i + 1, 1])
        cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
    # start/end markers
    cv2.circle(img, (int(pts[0, 0]), int(pts[0, 1])), 4, color, -1)
    cv2.rectangle(img, (int(pts[-1, 0]) - 3, int(pts[-1, 1]) - 3), (int(pts[-1, 0]) + 3, int(pts[-1, 1]) + 3), color, -1)
    return img


@torch.no_grad()
def visualize_batch_predictions(
    model: WorldModel,
    test_dataset: TrajectoryDataset,
    site_images_dir: Path,
    output_dir: Path,
    context_length: int,
    rollout_horizon: int,
    mean_cont: torch.Tensor,
    std_cont: torch.Tensor,
    continuous_indices: List[int],
    discrete_indices: List[int],
    device: torch.device,
    num_samples: int = 5,
    max_agents_per_sample: int = 10,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # load site images
    site_images: Dict[int, np.ndarray] = {}
    for sid, name in SITE_NAMES.items():
        img_path = site_images_dir / f"Site{name}.jpg"
        if img_path.exists():
            site_images[sid] = np.array(Image.open(img_path).convert("RGB"))
            print(f"Loaded {img_path.name}: {site_images[sid].shape}")

    if len(site_images) == 0:
        raise FileNotFoundError(f"No site images found in {site_images_dir}. Expect files like SiteA.jpg")

    # simple sampler: iterate dataset sequentially and collect per-site
    samples_by_site: Dict[int, List[dict]] = {sid: [] for sid in SITE_NAMES.keys()}

    print("\nCollecting samples...")
    for idx in tqdm(range(len(test_dataset)), desc="Scanning dataset"):
        item = test_dataset[idx]
        states_full = item["states"].unsqueeze(0).to(device)   # [1,T,K,F]
        masks = item["masks"].unsqueeze(0).to(device)          # [1,T,K]
        scene_ids = item.get("scene_ids", None)
        if scene_ids is None:
            # fallback: read from states_full[..., site_id]
            site_id_feat = 11  # from your metadata
            site_id = int(states_full[0, 0, 0, site_id_feat].item())
        else:
            site_id = int(scene_ids.item())

        if site_id not in site_images:
            continue

        B, T, K, F = states_full.shape
        if T < context_length + rollout_horizon:
            continue

        # normalize full states for model input
        mean_cont_d = mean_cont.to(device)
        std_cont_d = std_cont.to(device)
        states_norm_full = normalize_states_full(states_full, mean_cont_d, std_cont_d, continuous_indices)

        ctx_states = states_norm_full[:, :context_length]
        ctx_masks = masks[:, :context_length]

        gt_future_full = states_full[:, context_length:context_length + rollout_horizon]  # pixel
        gt_future_masks = masks[:, context_length:context_length + rollout_horizon]

        # rollout (returns continuous-only normalized)
        pred_cont, pred_masks = model.rollout(
            initial_states=ctx_states,
            initial_masks=ctx_masks,
            continuous_indices=continuous_indices,
            discrete_indices=discrete_indices,
            n_steps=rollout_horizon,
            teacher_forcing=False,
        )  # pred_cont [1,H,K,9]

        # denormalize predicted x,y to pixel
        pred_xy = denormalize_pred_cont_to_xy_pixels(pred_cont, mean_cont_d, std_cont_d, continuous_indices)  # [1,H,K,2]

        # context x,y in pixel
        ctx_xy = states_full[:, :context_length, :, :2]  # [1,C,K,2]

        # select agents: top by existence in future (gt mask)
        valid_counts = gt_future_masks[0].sum(dim=0)  # [K]
        topk = torch.argsort(valid_counts, descending=True)[:max_agents_per_sample].tolist()

        sample = dict(
            site_id=site_id,
            ctx_xy=ctx_xy[0, :, topk].detach().cpu().numpy(),      # [C,A,2]
            gt_xy=gt_future_full[0, :, topk, :2].detach().cpu().numpy(),  # [H,A,2]
            pred_xy=pred_xy[0, :, topk].detach().cpu().numpy(),    # [H,A,2]
            gt_mask=gt_future_masks[0, :, topk].detach().cpu().numpy(),   # [H,A]
            pred_mask=pred_masks[0, :, topk].detach().cpu().numpy(),      # [H,A]
        )
        samples_by_site[site_id].append(sample)

        # stop if enough
        if len(samples_by_site[site_id]) >= num_samples and all(
            (sid not in site_images) or (len(samples_by_site[sid]) >= num_samples)
            for sid in samples_by_site.keys()
        ):
            break

    print("\nCreating visualizations...")
    for site_id, samples in samples_by_site.items():
        if site_id not in site_images or len(samples) == 0:
            continue

        site_name = SITE_NAMES.get(site_id, str(site_id))
        base_img = site_images[site_id]

        for sidx, s in enumerate(samples[:num_samples]):
            img = base_img.copy()

            C, A, _ = s["ctx_xy"].shape
            H, _, _ = s["pred_xy"].shape

            # draw each agent
            for a in range(A):
                # masks
                gt_m = s["gt_mask"][:, a] > 0.5
                # context always drawn if appears
                ctx_traj = s["ctx_xy"][:, a, :]
                gt_traj = s["gt_xy"][gt_m, a, :] if gt_m.any() else s["gt_xy"][:, a, :]
                pred_traj = s["pred_xy"][:, a, :]

                # context (blue), gt future (green), pred (red)
                img = draw_polyline_on_image(img, ctx_traj, color=(0, 0, 255), thickness=2)
                img = draw_polyline_on_image(img, gt_traj, color=(0, 255, 0), thickness=2)
                img = draw_polyline_on_image(img, pred_traj, color=(255, 0, 0), thickness=2)

            out_path = output_dir / f"site{site_name}_sample{sidx:02d}.png"
            Image.fromarray(img).save(out_path)
            print(f"Saved: {out_path}")


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--site_images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/visualizations")
    parser.add_argument("--context_length", type=int, default=65)
    parser.add_argument("--rollout_horizon", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_agents", type=int, default=10)
    # Optional model architecture parameters (for old checkpoints without config)
    parser.add_argument("--input_dim", type=int, default=None, help="Input feature dimension (auto-detect from metadata if not specified)")
    parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension (auto-detect from checkpoint if not specified)")
    parser.add_argument("--dynamics_layers", type=int, default=None, help="Dynamics transformer layers (default: 4)")
    parser.add_argument("--dynamics_heads", type=int, default=None, help="Attention heads (default: 8)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Site images: {args.site_images_dir}")
    print(f"Context: {args.context_length} frames, Rollout: {args.rollout_horizon} frames")

    # metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    # indices
    continuous_indices = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    vi = metadata.get("validation_info", {})
    df = vi.get("discrete_features", {"class_id": 7, "lane_id": 8, "site_id": 11})
    discrete_indices = [int(df["class_id"]), int(df["lane_id"]), int(df["site_id"])]

    print(f"Continuous features: {continuous_indices}")
    print(f"Discrete features: {discrete_indices}")

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"]
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
    elif "encoder.to_latent.0.bias" in state_dict:
        latent_dim = int(state_dict["encoder.to_latent.0.bias"].shape[0])
    else:
        latent_dim = 256
    
    if args.dynamics_layers is not None:
        dynamics_layers = args.dynamics_layers
    elif "dynamics_layers" in model_config:
        dynamics_layers = model_config["dynamics_layers"]
    else:
        dynamics_layers = 4
    
    if args.dynamics_heads is not None:
        dynamics_heads = args.dynamics_heads
    elif "dynamics_heads" in model_config:
        dynamics_heads = model_config["dynamics_heads"]
    else:
        dynamics_heads = 8
    
    # Get continuous_dim and max_agents from config or infer
    if "continuous_dim" in model_config:
        continuous_dim = int(model_config["continuous_dim"])
    elif "decoder.state_head.bias" in state_dict:
        decoder_out = state_dict["decoder.state_head.bias"].shape[0]
        model_max_agents = int(model_config.get("max_agents", 50))
        continuous_dim = decoder_out // model_max_agents
    else:
        continuous_dim = 9
    
    model_max_agents = int(model_config.get("max_agents", 50))

    # infer embedding sizes to avoid mismatch
    num_lanes, num_sites, num_classes = _infer_embedding_sizes_from_ckpt(state_dict)
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
    print(f"  Embeddings: num_lanes={num_lanes}, num_sites={num_sites}, num_classes={num_classes}")

    # compute normalization stats from train_episodes.npz (same folder as test_data)
    train_npz = str(Path(args.test_data).parent / "train_episodes.npz")
    print(f"Computing mean/std from: {train_npz}")
    mean_cont, std_cont = _compute_mean_std_from_episodes_npz(train_npz, continuous_indices)
    print(f"mean_cont shape={mean_cont.shape}, std_cont shape={std_cont.shape}")

    # set stats into model for kinematic prior
    model.set_normalization_stats(mean_cont.cpu().numpy(), std_cont.cpu().numpy(), continuous_indices)
    print("✅ set_normalization_stats() applied (cont_index_map initialized)")

    # dataset (raw pixel, no normalization)
    test_dataset = TrajectoryDataset(data_path=args.test_data, normalize=False)

    visualize_batch_predictions(
        model=model,
        test_dataset=test_dataset,
        site_images_dir=Path(args.site_images_dir),
        output_dir=Path(args.output_dir),
        context_length=args.context_length,
        rollout_horizon=args.rollout_horizon,
        mean_cont=mean_cont,
        std_cont=std_cont,
        continuous_indices=continuous_indices,
        discrete_indices=discrete_indices,
        device=device,
        num_samples=args.num_samples,
        max_agents_per_sample=args.max_agents,
    )

    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
