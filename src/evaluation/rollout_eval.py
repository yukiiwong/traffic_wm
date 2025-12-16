"""
Rollout Evaluation Utilities

Evaluate multi-step prediction performance for WorldModel.rollout.

This file supports TWO types of --stats_path:
1) a real stats .npz that contains keys 'mean' and 'std'
2) a train_episodes .npz that does NOT contain 'mean/std'
   -> compute mean/std from it on the fly (continuous features only, masked)
   -> write a temporary stats file under output_dir
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from src.evaluation.prediction_metrics import compute_all_metrics, compute_extended_metrics
from src.utils.common import get_pixel_to_meter_conversion


# ---------------------------
# Helpers
# ---------------------------
def _average_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if len(all_metrics) == 0:
        return {}
    avg: Dict[str, float] = {}
    for k in all_metrics[0].keys():
        avg[k] = sum(m[k] for m in all_metrics) / len(all_metrics)
    return avg


def _infer_embedding_sizes_from_ckpt(state_dict: dict) -> Tuple[int, int, int]:
    """Infer num_lanes/num_sites/num_classes from checkpoint weights to avoid size mismatch."""
    num_lanes = (
        state_dict["encoder.lane_embedding.weight"].shape[0]
        if "encoder.lane_embedding.weight" in state_dict
        else 100
    )
    num_sites = (
        state_dict["encoder.site_embedding.weight"].shape[0]
        if "encoder.site_embedding.weight" in state_dict
        else 10
    )
    num_classes = (
        state_dict["encoder.class_embedding.weight"].shape[0]
        if "encoder.class_embedding.weight" in state_dict
        else 10
    )
    return int(num_lanes), int(num_sites), int(num_classes)


def _resolve_indices(metadata: dict, loader) -> Tuple[List[int], List[int]]:
    """
    Resolve continuous/discrete indices.

    Priority:
    1) dataset attributes (if present)
    2) metadata.validation_info discrete_features (if present)
    3) fallback defaults
    """
    cont = None
    disc = None

    ds = getattr(loader, "dataset", None)
    if ds is not None:
        cont = getattr(ds, "continuous_indices", None)
        disc = getattr(ds, "discrete_indices", None)

    vi = metadata.get("validation_info", {}) if metadata is not None else {}

    # Build discrete from metadata if present
    if disc is None and "discrete_features" in vi:
        df = vi["discrete_features"]
        disc = [int(df["class_id"]), int(df["lane_id"]), int(df["site_id"])]

    # continuous indices are not explicitly stored in your metadata snippet
    if cont is None:
        cont = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    if disc is None:
        disc = [7, 8, 11]

    return list(map(int, cont)), list(map(int, disc))


def _gt_to_continuous_order(gt_full: torch.Tensor, continuous_indices: List[int]) -> torch.Tensor:
    """FULL gt [B,T,K,F_full] -> continuous-only [B,T,K,F_cont] in order of continuous_indices."""
    return gt_full[..., continuous_indices]


def _load_stats_or_compute_from_episodes(
    stats_path: str,
    output_dir: Path,
    continuous_indices: List[int],
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """
    If stats_path contains mean/std, load them.
    Otherwise treat stats_path as train_episodes.npz and compute mean/std using masks.

    Returns:
      (resolved_stats_path, mean_cont_tensor, std_cont_tensor)
    """
    import numpy as np

    z = np.load(stats_path, allow_pickle=True)

    # Case 1: real stats file
    if "mean" in z.files and "std" in z.files:
        mean = z["mean"]
        std = z["std"]
        mean_t = torch.tensor(mean, dtype=torch.float32)
        std_t = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)
        return stats_path, mean_t, std_t

    # Case 2: episodes file -> compute stats
    if "states" not in z.files or "masks" not in z.files:
        raise KeyError(
            f"{stats_path} has no mean/std and also doesn't have ('states','masks'). "
            f"Found keys: {z.files}"
        )

    states = z["states"]  # [N,T,K,F]
    masks = z["masks"]    # [N,T,K]
    m = masks > 0.5

    mean_list = []
    std_list = []

    for feat_idx in continuous_indices:
        vals = states[..., feat_idx][m]
        if vals.size == 0:
            raise RuntimeError(f"No valid values for feature {feat_idx} when computing stats from {stats_path}.")
        vals = vals.astype(np.float64)
        mu = vals.mean()
        sd = vals.std(ddof=0)
        if sd < 1e-6:
            sd = 1e-6
        mean_list.append(mu)
        std_list.append(sd)

    mean = np.asarray(mean_list, dtype=np.float32)
    std = np.asarray(std_list, dtype=np.float32)

    computed_path = output_dir / "computed_train_stats.npz"
    np.savez(computed_path, mean=mean, std=std)
    print(f"✅ Computed mean/std from episodes and wrote stats to: {computed_path}")

    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)
    return str(computed_path), mean_t, std_t


# ---------------------------
# Evaluation functions (EXPORTED)
# ---------------------------
def evaluate_rollout(
    model: nn.Module,
    data_loader,
    continuous_indices: List[int],
    discrete_indices: List[int],
    context_length: int = 10,
    rollout_length: int = 20,
    device: torch.device = torch.device("cpu"),
    pixel_to_meter: Optional[float] = None,
    convert_to_meters: bool = True,
) -> Dict[str, float]:
    model.eval()
    all_metrics: List[Dict[str, float]] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Rollout"):
            states_full = batch["states"].to(device)  # [B,T,K,F_full]
            masks = batch["masks"].to(device)         # [B,T,K]

            B, T, K, F_full = states_full.shape
            if T < context_length + rollout_length:
                continue

            context_states = states_full[:, :context_length]
            context_masks = masks[:, :context_length]

            target_full = states_full[:, context_length:context_length + rollout_length]
            target_masks = masks[:, context_length:context_length + rollout_length]
            target_cont = _gt_to_continuous_order(target_full, continuous_indices)

            pred_cont, _pred_masks = model.rollout(
                initial_states=context_states,
                initial_masks=context_masks,
                continuous_indices=continuous_indices,
                discrete_indices=discrete_indices,
                n_steps=rollout_length,
                teacher_forcing=False,
            )

            # Determine correct heading_idx
            # angle is feature 6, check if it's in continuous_indices
            angle_feature_idx = 6
            heading_idx = None
            if angle_feature_idx in continuous_indices:
                heading_idx = continuous_indices.index(angle_feature_idx)
            
            metrics = compute_all_metrics(
                predicted=pred_cont,
                ground_truth=target_cont,
                masks=target_masks,
                pixel_to_meter=pixel_to_meter,
                convert_to_meters=convert_to_meters,
                heading_idx=heading_idx,
            )
            
            # Add extended metrics (moving agents, velocity direction, etc.)
            # velocity_threshold: 0.5 m/s for moving agents after conversion
            velocity_threshold = 0.5 if convert_to_meters else 15.0  # ~15 px/s = 0.5 m/s
            extended = compute_extended_metrics(
                predicted=pred_cont,
                ground_truth=target_cont,
                masks=target_masks,
                velocity_threshold=velocity_threshold,
            )
            metrics.update(extended)
            
            all_metrics.append(metrics)

    return _average_metrics(all_metrics)


def evaluate_multihorizon(
    model: nn.Module,
    data_loader,
    continuous_indices: List[int],
    discrete_indices: List[int],
    context_length: int = 10,
    horizons: List[int] = [1, 3, 5, 10, 20],
    device: torch.device = torch.device("cpu"),
    pixel_to_meter: Optional[float] = None,
    convert_to_meters: bool = True,
) -> Dict[int, Dict[str, float]]:
    model.eval()
    results: Dict[int, List[Dict[str, float]]] = {h: [] for h in horizons}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Multi-horizon Evaluation"):
            states_full = batch["states"].to(device)
            masks = batch["masks"].to(device)

            B, T, K, F_full = states_full.shape
            max_h = max(horizons)
            if T < context_length + max_h:
                continue

            context_states = states_full[:, :context_length]
            context_masks = masks[:, :context_length]

            pred_cont_all, _ = model.rollout(
                initial_states=context_states,
                initial_masks=context_masks,
                continuous_indices=continuous_indices,
                discrete_indices=discrete_indices,
                n_steps=max_h,
                teacher_forcing=False,
            )

            # Determine correct heading_idx once per batch
            angle_feature_idx = 6
            heading_idx = None
            if angle_feature_idx in continuous_indices:
                heading_idx = continuous_indices.index(angle_feature_idx)
            
            velocity_threshold = 0.5 if convert_to_meters else 15.0
            
            for h in horizons:
                pred_h = pred_cont_all[:, :h]
                target_full_h = states_full[:, context_length:context_length + h]
                target_cont_h = _gt_to_continuous_order(target_full_h, continuous_indices)
                mask_h = masks[:, context_length:context_length + h]

                metrics = compute_all_metrics(
                    predicted=pred_h,
                    ground_truth=target_cont_h,
                    masks=mask_h,
                    pixel_to_meter=pixel_to_meter,
                    convert_to_meters=convert_to_meters,
                    heading_idx=heading_idx,
                )
                
                extended = compute_extended_metrics(
                    predicted=pred_h,
                    ground_truth=target_cont_h,
                    masks=mask_h,
                    velocity_threshold=velocity_threshold,
                )
                metrics.update(extended)
                
                results[h].append(metrics)

    return {h: _average_metrics(results[h]) for h in horizons}


def evaluate_with_teacher_forcing(
    model: nn.Module,
    data_loader,
    continuous_indices: List[int],
    discrete_indices: List[int],
    context_length: int = 10,
    rollout_length: int = 20,
    device: torch.device = torch.device("cpu"),
    pixel_to_meter: Optional[float] = None,
    convert_to_meters: bool = True,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    open_loop_metrics: List[Dict[str, float]] = []
    closed_loop_metrics: List[Dict[str, float]] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Teacher Forcing Evaluation"):
            states_full = batch["states"].to(device)
            masks = batch["masks"].to(device)

            B, T, K, F_full = states_full.shape
            if T < context_length + rollout_length:
                continue

            context_states = states_full[:, :context_length]
            context_masks = masks[:, :context_length]

            target_full = states_full[:, context_length:context_length + rollout_length]
            target_masks = masks[:, context_length:context_length + rollout_length]
            target_cont = _gt_to_continuous_order(target_full, continuous_indices)

            full_states = states_full[:, :context_length + rollout_length]

            # Determine correct heading_idx
            angle_feature_idx = 6
            heading_idx = None
            if angle_feature_idx in continuous_indices:
                heading_idx = continuous_indices.index(angle_feature_idx)
            
            velocity_threshold = 0.5 if convert_to_meters else 15.0
            
            # open-loop
            open_pred_cont, _ = model.rollout(
                initial_states=context_states,
                initial_masks=context_masks,
                continuous_indices=continuous_indices,
                discrete_indices=discrete_indices,
                n_steps=rollout_length,
                teacher_forcing=False,
            )
            open_metrics = compute_all_metrics(
                predicted=open_pred_cont,
                ground_truth=target_cont,
                masks=target_masks,
                pixel_to_meter=pixel_to_meter,
                convert_to_meters=convert_to_meters,
                heading_idx=heading_idx,
            )
            open_extended = compute_extended_metrics(
                predicted=open_pred_cont,
                ground_truth=target_cont,
                masks=target_masks,
                velocity_threshold=velocity_threshold,
            )
            open_metrics.update(open_extended)
            open_loop_metrics.append(open_metrics)

            # teacher forcing
            closed_pred_cont, _ = model.rollout(
                initial_states=context_states,
                initial_masks=context_masks,
                continuous_indices=continuous_indices,
                discrete_indices=discrete_indices,
                n_steps=rollout_length,
                teacher_forcing=True,
                ground_truth_states=full_states,
            )
            closed_metrics = compute_all_metrics(
                predicted=closed_pred_cont,
                ground_truth=target_cont,
                masks=target_masks,
                pixel_to_meter=pixel_to_meter,
                convert_to_meters=convert_to_meters,
                heading_idx=heading_idx,
            )
            closed_extended = compute_extended_metrics(
                predicted=closed_pred_cont,
                ground_truth=target_cont,
                masks=target_masks,
                velocity_threshold=velocity_threshold,
            )
            closed_metrics.update(closed_extended)
            closed_loop_metrics.append(closed_metrics)

    return {"open_loop": _average_metrics(open_loop_metrics), "closed_loop": _average_metrics(closed_loop_metrics)}


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    import argparse
    import json
    from src.models.world_model import WorldModel
    from src.data.dataset import get_dataloader

    parser = argparse.ArgumentParser(description="Evaluate world model with rollout")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data NPZ file")
    parser.add_argument("--metadata", type=str, default="data/processed/metadata.json", help="Path to metadata.json")
    # can be stats.npz (mean/std) OR train_episodes.npz (compute stats)
    parser.add_argument(
        "--stats_path",
        type=str,
        default="data/processed/train_stats.npz",
        help="Path to stats .npz (mean/std) OR train_episodes .npz to compute stats",
    )
    parser.add_argument("--context_length", type=int, default=65, help="Number of context frames (default: 65)")
    parser.add_argument("--rollout_horizon", type=int, default=15, help="Number of frames to predict (default: 15)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="results/", help="Directory to save results")
    parser.add_argument("--convert_to_meters", action="store_true", default=True, help="Convert to meters (default: True)")
    
    # Model architecture parameters (optional, will be read from checkpoint if available)
    parser.add_argument("--input_dim", type=int, default=None, help="Input feature dimension (auto-detect if not provided)")
    parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension (auto-detect if not provided)")
    parser.add_argument("--dynamics_layers", type=int, default=None, help="Number of dynamics layers (auto-detect if not provided)")
    parser.add_argument("--dynamics_heads", type=int, default=None, help="Number of attention heads (auto-detect if not provided)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # metadata
    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    n_features = int(metadata.get("n_features", 12))

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Context length: {args.context_length}")
    print(f"Rollout horizon: {args.rollout_horizon}")
    print(f"Features: {n_features}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Priority: 1. command line args, 2. checkpoint config, 3. inference/defaults
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        input_dim = args.input_dim if args.input_dim is not None else int(cfg.get("input_dim", n_features))
        latent_dim = args.latent_dim if args.latent_dim is not None else int(cfg.get("latent_dim", 256))
        dynamics_layers = args.dynamics_layers if args.dynamics_layers is not None else int(cfg.get("dynamics_layers", 4))
        dynamics_heads = args.dynamics_heads if args.dynamics_heads is not None else int(cfg.get("dynamics_heads", 8))
        continuous_dim = int(cfg.get("continuous_dim", 9))
        print(f"✅ Loaded config from checkpoint")
    else:
        print("Warning: No config found in checkpoint, using args or inferring from state_dict")
        input_dim = args.input_dim if args.input_dim is not None else n_features
        dynamics_layers = args.dynamics_layers if args.dynamics_layers is not None else 4
        dynamics_heads = args.dynamics_heads if args.dynamics_heads is not None else 8
        
        if args.latent_dim is not None:
            latent_dim = args.latent_dim
        elif "encoder.to_latent.0.bias" in state_dict:
            latent_dim = int(state_dict["encoder.to_latent.0.bias"].shape[0])
            print(f"  Inferred latent_dim: {latent_dim}")
        else:
            latent_dim = 256
            print(f"  Could not infer latent_dim, using default: {latent_dim}")
        
        # Try to infer continuous_dim from decoder output
        if "decoder.state_head.bias" in state_dict:
            decoder_out = state_dict["decoder.state_head.bias"].shape[0]
            max_agents = 50  # assume default
            continuous_dim = decoder_out // max_agents
            print(f"  Inferred continuous_dim: {continuous_dim} (decoder output {decoder_out} / {max_agents} agents)")
        else:
            continuous_dim = 9
    
    # Get max_agents from config or use default
    max_agents = int(cfg.get("max_agents", 50)) if "config" in checkpoint else 50
    
    print(f"Model config: input_dim={input_dim}, latent_dim={latent_dim}, "
          f"dynamics_layers={dynamics_layers}, dynamics_heads={dynamics_heads}, "
          f"continuous_dim={continuous_dim}, max_agents={max_agents}")

    num_lanes_ckpt, num_sites_ckpt, num_classes_ckpt = _infer_embedding_sizes_from_ckpt(state_dict)
    print(f"✅ Using embedding sizes from checkpoint: num_lanes={num_lanes_ckpt}, num_sites={num_sites_ckpt}, num_classes={num_classes_ckpt}")

    from src.utils.common import parse_discrete_feature_indices_from_metadata
    lane_idx, class_idx, site_idx = parse_discrete_feature_indices_from_metadata(
        metadata, fallback=(8, 7, 11), strict=False
    )

    model = WorldModel(
        input_dim=input_dim,
        continuous_dim=continuous_dim,
        max_agents=max_agents,
        latent_dim=latent_dim,
        dynamics_layers=dynamics_layers,
        dynamics_heads=dynamics_heads,
        num_lanes=num_lanes_ckpt,
        num_sites=num_sites_ckpt,
        num_classes=num_classes_ckpt,
        lane_feature_idx=lane_idx,
        class_feature_idx=class_idx,
        site_feature_idx=site_idx,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # continuous/discrete indices (use metadata fallback here)
    vi = metadata.get("validation_info", {})
    if "discrete_features" in vi:
        df = vi["discrete_features"]
        discrete_indices_fallback = [int(df["class_id"]), int(df["lane_id"]), int(df["site_id"])]
    else:
        discrete_indices_fallback = [7, 8, 11]
    
    # Determine continuous indices based on n_features
    n_features = int(metadata.get("n_features", 12))
    do_not_norm = vi.get("do_not_normalize", ["lane_id", "class_id", "site_id", "angle"])
    angle_idx = int(vi.get("angle_idx", 6))
    
    # Build continuous_indices: all except discrete features and angle
    all_indices = set(range(n_features))
    skip_indices = set(discrete_indices_fallback + [angle_idx])
    continuous_indices_fallback = sorted(list(all_indices - skip_indices))

    # compute/load stats (fallback indices)
    resolved_stats_path, mean_cont_t, std_cont_t = _load_stats_or_compute_from_episodes(
        stats_path=args.stats_path,
        output_dir=output_dir,
        continuous_indices=continuous_indices_fallback,
    )

    # dataloader (dataset.py needs a real stats file with mean/std)
    test_loader = get_dataloader(
        data_path=args.test_data,
        batch_size=args.batch_size,
        shuffle=False,
        stats_path=resolved_stats_path,
    )

    # resolve indices from dataset/metadata
    continuous_indices, discrete_indices = _resolve_indices(metadata, test_loader)
    print(f"Using continuous_indices: {continuous_indices}")
    print(f"Using discrete_indices: {discrete_indices}")

    # if indices differ, recompute stats & rebuild loader
    if continuous_indices != continuous_indices_fallback:
        resolved_stats_path, mean_cont_t, std_cont_t = _load_stats_or_compute_from_episodes(
            stats_path=args.stats_path,
            output_dir=output_dir,
            continuous_indices=continuous_indices,
        )
        test_loader = get_dataloader(
            data_path=args.test_data,
            batch_size=args.batch_size,
            shuffle=False,
            stats_path=resolved_stats_path,
        )

    # set stats into model for kinematic prior
    model.set_normalization_stats(mean_cont_t.cpu().numpy(), std_cont_t.cpu().numpy(), continuous_indices)
    print("✅ set_normalization_stats() applied (cont_index_map initialized)")

    # pixel-to-meter
    pixel_to_meter = None
    if args.convert_to_meters:
        try:
            pixel_to_meter = get_pixel_to_meter_conversion()
            print(f"Pixel to meter conversion: {pixel_to_meter:.6f}")
        except Exception as e:
            print(f"Warning: Could not load pixel_to_meter conversion: {e}")
            print("Metrics will be in pixels")
            args.convert_to_meters = False

    # evaluate
    print("\n" + "=" * 60)
    print("Starting Rollout Evaluation")
    print("=" * 60)

    metrics = evaluate_rollout(
        model=model,
        data_loader=test_loader,
        continuous_indices=continuous_indices,
        discrete_indices=discrete_indices,
        context_length=args.context_length,
        rollout_length=args.rollout_horizon,
        device=device,
        pixel_to_meter=pixel_to_meter,
        convert_to_meters=args.convert_to_meters,
    )

    units = "meters" if args.convert_to_meters else "pixels"
    print("\n" + "=" * 60)
    print(f"Rollout Evaluation Results ({units})")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # save
    results_file = output_dir / "rollout_metrics.json"
    results = {
        "metrics": metrics,
        "config": {
            "checkpoint": args.checkpoint,
            "test_data": args.test_data,
            "metadata": args.metadata,
            "stats_path_arg": args.stats_path,
            "resolved_stats_path_used": resolved_stats_path,
            "context_length": args.context_length,
            "rollout_horizon": args.rollout_horizon,
            "convert_to_meters": args.convert_to_meters,
            "units": units,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "num_lanes_ckpt": num_lanes_ckpt,
            "num_sites_ckpt": num_sites_ckpt,
            "num_classes_ckpt": num_classes_ckpt,
            "continuous_indices": continuous_indices,
            "discrete_indices": discrete_indices,
        },
    }

    import json as _json
    with open(results_file, "w") as f:
        _json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("=" * 60)
