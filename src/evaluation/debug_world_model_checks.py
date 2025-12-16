"""
Debug checks: verify this is truly a latent world model (imagination rollout),
not just trajectory prediction / one-step regression.

Run:
python -m src.evaluation.debug_world_model_checks \
  --checkpoint checkpoints/world_model_siteA2/checkpoint_best.pt \
  --test_data data/processed_siteA/test_episodes.npz \
  --metadata data/processed_siteA/metadata.json \
  --num_samples 20 \
  --context_length 65 \
  --rollout_horizon 15
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.models.world_model import WorldModel
from src.data.dataset import TrajectoryDataset
from src.evaluation.prediction_metrics import compute_all_metrics


# -----------------------------
# ckpt inference helpers
# -----------------------------
def infer_embedding_sizes_from_ckpt(state_dict: dict) -> Tuple[int, int, int]:
    def _get(name: str, default: int) -> int:
        w = state_dict.get(name, None)
        return int(w.shape[0]) if w is not None else int(default)

    return (
        _get("encoder.lane_embedding.weight", 100),
        _get("encoder.site_embedding.weight", 10),
        _get("encoder.class_embedding.weight", 10),
    )


def infer_latent_dim_from_ckpt(state_dict: dict, default: int = 256) -> int:
    b = state_dict.get("encoder.to_latent.0.bias", None)
    return int(b.shape[0]) if b is not None else int(default)


# -----------------------------
# normalization from train npz
# -----------------------------
def compute_mean_std_from_episodes_npz(npz_path: str, continuous_indices: List[int], eps: float = 1e-6):
    z = np.load(npz_path, allow_pickle=True)
    if "states" not in z.files or "masks" not in z.files:
        raise KeyError(f"{npz_path} must contain 'states' and 'masks'. Found {z.files}")

    states = z["states"]  # [N,T,K,F]
    masks = z["masks"]    # [N,T,K]
    m = masks > 0.5

    means, stds = [], []
    for feat in continuous_indices:
        vals = states[..., feat][m]
        vals = vals.astype(np.float64)
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        if sd < eps:
            sd = eps
        means.append(mu)
        stds.append(sd)
    return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)


def normalize_full_states(states_full: torch.Tensor, mean_cont: torch.Tensor, std_cont: torch.Tensor, continuous_indices: List[int]):
    """
    states_full: [B,T,K,F] raw
    """
    out = states_full.clone()
    for j, feat in enumerate(continuous_indices):
        out[..., feat] = (out[..., feat] - mean_cont[j]) / std_cont[j]
    return out


def denorm_xy_from_cont(cont_pred: torch.Tensor, mean_cont: torch.Tensor, std_cont: torch.Tensor, continuous_indices: List[int]):
    """
    cont_pred: [B,T,K,F_cont]
    return xy raw pixels: [B,T,K,2]
    """
    ix = continuous_indices.index(0)  # x feature idx in full space is 0
    iy = continuous_indices.index(1)  # y feature idx in full space is 1
    x = cont_pred[..., ix] * std_cont[ix] + mean_cont[ix]
    y = cont_pred[..., iy] * std_cont[iy] + mean_cont[iy]
    return torch.stack([x, y], dim=-1)


def cont_to_full_state_denorm(
    pred_cont_norm: torch.Tensor,          # [B,H,K,F_cont]  (normalized)
    mean_cont: torch.Tensor,               # [F_cont]
    std_cont: torch.Tensor,                # [F_cont]
    continuous_indices: List[int],         # full-space feature indices, length=F_cont
    full_F: int,
    discrete_template: torch.Tensor = None # optional [B,1,K,n_disc] or [B,K,n_disc]
):
    B,H,K,Fc = pred_cont_norm.shape
    out = torch.zeros((B,H,K,full_F), device=pred_cont_norm.device, dtype=pred_cont_norm.dtype)

    # denorm each continuous feature and write back
    for j, feat_idx in enumerate(continuous_indices):
        out[..., feat_idx] = pred_cont_norm[..., j] * std_cont[j] + mean_cont[j]

    # optional: fill discrete columns if you want
    if discrete_template is not None:
        pass

    return out


# -------------------------------------------------
# rollout variants
# -------------------------------------------------
@torch.no_grad()
def rollout_open_loop(
    model: WorldModel,
    ctx_norm_full: torch.Tensor,
    ctx_masks: torch.Tensor,
    continuous_indices: List[int],
    discrete_indices: List[int],
    n_steps: int,
    threshold: float,
):
    pred_cont, pred_masks = model.rollout(
        initial_states=ctx_norm_full,
        initial_masks=ctx_masks,
        continuous_indices=continuous_indices,
        discrete_indices=discrete_indices,
        n_steps=n_steps,
        threshold=threshold,
        teacher_forcing=False,
        ground_truth_states=None,
    )
    return pred_cont, pred_masks


@torch.no_grad()
def rollout_teacher_forcing(
    model: WorldModel,
    ctx_norm_full: torch.Tensor,
    ctx_masks: torch.Tensor,
    full_norm_states: torch.Tensor,  # [B, C+H, K, F]
    continuous_indices: List[int],
    discrete_indices: List[int],
    n_steps: int,
    threshold: float,
):
    pred_cont, pred_masks = model.rollout(
        initial_states=ctx_norm_full,
        initial_masks=ctx_masks,
        continuous_indices=continuous_indices,
        discrete_indices=discrete_indices,
        n_steps=n_steps,
        threshold=threshold,
        teacher_forcing=True,
        ground_truth_states=full_norm_states,
    )
    return pred_cont, pred_masks


@torch.no_grad()
def rollout_latent_frozen(
    model: WorldModel,
    ctx_norm_full: torch.Tensor,
    ctx_masks: torch.Tensor,
    continuous_indices: List[int],
    discrete_indices: List[int],
    n_steps: int,
    threshold: float,
):
    orig_step = model.dynamics.step

    def step_identity(latent_hist, max_context: int):
        return latent_hist[:, -1, :]  # do not advance latent

    model.dynamics.step = step_identity
    try:
        return rollout_open_loop(model, ctx_norm_full, ctx_masks, continuous_indices, discrete_indices, n_steps, threshold)
    finally:
        model.dynamics.step = orig_step


@torch.no_grad()
def rollout_latent_zeroed(
    model: WorldModel,
    ctx_norm_full: torch.Tensor,
    ctx_masks: torch.Tensor,
    continuous_indices: List[int],
    discrete_indices: List[int],
    n_steps: int,
    threshold: float,
):
    orig_forward = model.decoder.forward

    def forward_zero_latent(latent, return_residual_xy=False):
        return orig_forward(torch.zeros_like(latent), return_residual_xy=return_residual_xy)

    model.decoder.forward = forward_zero_latent
    try:
        return rollout_open_loop(model, ctx_norm_full, ctx_masks, continuous_indices, discrete_indices, n_steps, threshold)
    finally:
        model.decoder.forward = orig_forward


@torch.no_grad()
def rollout_context_shuffled(
    model: WorldModel,
    ctx_norm_full: torch.Tensor,
    ctx_masks: torch.Tensor,
    continuous_indices: List[int],
    discrete_indices: List[int],
    n_steps: int,
    threshold: float,
):
    perm = torch.randperm(ctx_norm_full.shape[1], device=ctx_norm_full.device)
    ctx_s = ctx_norm_full[:, perm]
    m_s = ctx_masks[:, perm]
    return rollout_open_loop(model, ctx_s, m_s, continuous_indices, discrete_indices, n_steps, threshold)


def avg_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ms[0].keys()
    return {k: float(np.mean([m[k] for m in ms])) for k in keys}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_data", type=str, required=True)
    p.add_argument("--metadata", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--context_length", type=int, default=65)
    p.add_argument("--rollout_horizon", type=int, default=15)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    # Optional model architecture parameters (for old checkpoints without config)
    p.add_argument("--input_dim", type=int, default=None, help="Input feature dimension (auto-detect from metadata if not specified)")
    p.add_argument("--latent_dim", type=int, default=None, help="Latent dimension (auto-detect from checkpoint if not specified)")
    p.add_argument("--dynamics_layers", type=int, default=None, help="Dynamics transformer layers (default: 4)")
    p.add_argument("--dynamics_heads", type=int, default=None, help="Attention heads (default: 8)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    # indices must match your training pipeline
    continuous_indices = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    from src.utils.common import parse_discrete_feature_indices_from_metadata
    lane_idx, class_idx, site_idx = parse_discrete_feature_indices_from_metadata(metadata, fallback=(8, 7, 11), strict=False)
    discrete_indices = [class_idx, lane_idx, site_idx]

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
    else:
        latent_dim = infer_latent_dim_from_ckpt(state_dict, 256)
    
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
    
    num_lanes, num_sites, num_classes = infer_embedding_sizes_from_ckpt(state_dict)

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
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Compute mean/std from train_episodes.npz next to test_data
    train_npz = str(Path(args.test_data).parent / "train_episodes.npz")
    mean_cont, std_cont = compute_mean_std_from_episodes_npz(train_npz, continuous_indices)
    mean_cont = mean_cont.to(device)
    std_cont = std_cont.to(device)
    model.set_normalization_stats(mean_cont.cpu().numpy(), std_cont.cpu().numpy(), continuous_indices)

    # raw dataset (pixels), no normalization
    ds = TrajectoryDataset(data_path=args.test_data, normalize=False)
    idxs = np.random.choice(np.arange(len(ds)), size=min(args.num_samples, len(ds)), replace=False).tolist()

    agg = {k: [] for k in ["open_loop", "teacher_forcing", "latent_frozen", "latent_zeroed", "context_shuffled"]}

    C = args.context_length
    H = args.rollout_horizon

    for i in tqdm(idxs, desc="Checking"):
        item = ds[i]
        states = item["states"].to(device)  # [T,K,F] raw
        masks = item["masks"].to(device)    # [T,K]
        T = states.shape[0]
        if T < C + H:
            continue

        # Build batched tensors
        ctx_full = states[:C].unsqueeze(0)             # [1,C,K,F] raw
        ctx_masks = masks[:C].unsqueeze(0)             # [1,C,K]
        fut_full = states[C:C+H].unsqueeze(0)          # [1,H,K,F] raw
        fut_masks = masks[C:C+H].unsqueeze(0)          # [1,H,K]

        full_raw = torch.cat([ctx_full, fut_full], dim=1)  # [1,C+H,K,F]
        full_norm = normalize_full_states(full_raw, mean_cont, std_cont, continuous_indices)
        ctx_norm = full_norm[:, :C]  # [1,C,K,F] normalized

        # Rollout variants
        pred_open_cont, _ = rollout_open_loop(model, ctx_norm, ctx_masks, continuous_indices, discrete_indices, H, args.threshold)
        pred_tf_cont, _   = rollout_teacher_forcing(model, ctx_norm, ctx_masks, full_norm, continuous_indices, discrete_indices, H, args.threshold)
        pred_froz_cont, _ = rollout_latent_frozen(model, ctx_norm, ctx_masks, continuous_indices, discrete_indices, H, args.threshold)
        pred_zero_cont, _ = rollout_latent_zeroed(model, ctx_norm, ctx_masks, continuous_indices, discrete_indices, H, args.threshold)
        pred_shuf_cont, _ = rollout_context_shuffled(model, ctx_norm, ctx_masks, continuous_indices, discrete_indices, H, args.threshold)

        # Convert predicted continuous -> full state (xy only filled), keep batch dim!
        pred_open_full = cont_to_full_state_denorm(pred_open_cont, mean_cont, std_cont, continuous_indices, input_dim)
        pred_tf_full   = cont_to_full_state_denorm(pred_tf_cont,   mean_cont, std_cont, continuous_indices, input_dim)
        pred_froz_full = cont_to_full_state_denorm(pred_froz_cont, mean_cont, std_cont, continuous_indices, input_dim)
        pred_zero_full = cont_to_full_state_denorm(pred_zero_cont, mean_cont, std_cont, continuous_indices, input_dim)
        pred_shuf_full = cont_to_full_state_denorm(pred_shuf_cont, mean_cont, std_cont, continuous_indices, input_dim)

        # compute_all_metrics expects [B,T,K,F] + masks [B,T,K]
        m_open = compute_all_metrics(predicted=pred_open_full, ground_truth=fut_full, masks=fut_masks,
                                     pixel_to_meter=None, convert_to_meters=False)
        m_tf   = compute_all_metrics(predicted=pred_tf_full,   ground_truth=fut_full, masks=fut_masks,
                                     pixel_to_meter=None, convert_to_meters=False)
        m_froz = compute_all_metrics(predicted=pred_froz_full, ground_truth=fut_full, masks=fut_masks,
                                     pixel_to_meter=None, convert_to_meters=False)
        m_zero = compute_all_metrics(predicted=pred_zero_full, ground_truth=fut_full, masks=fut_masks,
                                     pixel_to_meter=None, convert_to_meters=False)
        m_shuf = compute_all_metrics(predicted=pred_shuf_full, ground_truth=fut_full, masks=fut_masks,
                                     pixel_to_meter=None, convert_to_meters=False)

        agg["open_loop"].append(m_open)
        agg["teacher_forcing"].append(m_tf)
        agg["latent_frozen"].append(m_froz)
        agg["latent_zeroed"].append(m_zero)
        agg["context_shuffled"].append(m_shuf)

    print("\n==================== RESULTS (pixels) ====================")
    for k, ms in agg.items():
        if len(ms) == 0:
            print(f"{k}: (no samples)")
            continue
        a = avg_metrics(ms)
        print(f"\n[{k}]")
        for mk, mv in a.items():
            print(f"  {mk:16s}: {mv:.6f}")

    print("\n==================== HOW TO INTERPRET ====================")
    print("- teacher_forcing << open_loop  => rollout is truly autoregressive (imagination matters).")
    print("- latent_frozen/latent_zeroed worse than open_loop => latent dynamics/latent decoding contributes.")
    print("- context_shuffled similar to open_loop => weak use of temporal order (risk of one-step-like behavior).")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
