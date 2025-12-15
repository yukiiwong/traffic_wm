"""
World-model visualization: shows Kinematic Prior vs WorldModel (prior+residual) and latent rollout traces.

Run:
python -m src.evaluation.visualize_predictions \
  --checkpoint checkpoints/world_model_siteA2/checkpoint_best.pt \
  --test_data data/processed_siteA/test_episodes.npz \
  --metadata data/processed_siteA/metadata.json \
  --site_images_dir src/evaluation/sites \
  --output_dir results/visualizations \
  --context_length 65 --rollout_horizon 15 --num_samples 80 \
  --max_agents 5 --select_agents presence

Extra alignment (if needed):
  --flip_y
  --rotate_deg 90

Agent selection:
  --select_agents presence|random|best_error|median_error|error

Notes:
- Expects site images named SiteA.jpg, SiteB.jpg... in --site_images_dir
- Computes mean/std from train_episodes.npz located next to --test_data
"""

from __future__ import annotations
import sys, json, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.world_model import WorldModel
from src.data.dataset import TrajectoryDataset

SITE_NAMES = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G",7:"H",8:"I",9:"J"}


def infer_embedding_sizes_from_ckpt(state_dict: dict) -> Tuple[int,int,int]:
    def _get(name: str, default: int) -> int:
        w = state_dict.get(name, None)
        return int(w.shape[0]) if w is not None else int(default)
    return (_get("encoder.lane_embedding.weight",100),
            _get("encoder.site_embedding.weight",10),
            _get("encoder.class_embedding.weight",10))

def infer_latent_dim_from_ckpt(state_dict: dict, default: int = 256) -> int:
    b = state_dict.get("encoder.to_latent.0.bias", None)
    return int(b.shape[0]) if b is not None else int(default)

def compute_mean_std_from_episodes_npz(npz_path: str, continuous_indices: List[int], eps: float=1e-6):
    z = np.load(npz_path, allow_pickle=True)
    if "states" not in z.files or "masks" not in z.files:
        raise KeyError(f"{npz_path} must contain 'states' and 'masks'. Found {z.files}")
    states = z["states"]  # [N,T,K,F]
    masks  = z["masks"]   # [N,T,K]
    m = masks > 0.5
    means, stds = [], []
    for feat in continuous_indices:
        vals = states[..., feat][m]
        if vals.size == 0:
            raise RuntimeError(f"No valid values for feature {feat} in {npz_path}")
        vals = vals.astype(np.float64)
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        if sd < eps: sd = eps
        means.append(mu); stds.append(sd)
    return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)

def normalize_full_states(states_full: torch.Tensor, mean_cont: torch.Tensor, std_cont: torch.Tensor, continuous_indices: List[int]):
    out = states_full.clone()
    for j, feat in enumerate(continuous_indices):
        out[..., feat] = (out[..., feat] - mean_cont[j]) / std_cont[j]
    return out

def denorm_xy_from_cont(cont_pred: torch.Tensor, mean_cont: torch.Tensor, std_cont: torch.Tensor, continuous_indices: List[int]):
    ix = continuous_indices.index(0)
    iy = continuous_indices.index(1)
    x = cont_pred[..., ix] * std_cont[ix] + mean_cont[ix]
    y = cont_pred[..., iy] * std_cont[iy] + mean_cont[iy]
    return torch.stack([x,y], dim=-1)

def apply_flip_y(xy: torch.Tensor, H_img: int) -> torch.Tensor:
    out = xy.clone()
    out[...,1] = (H_img-1) - out[...,1]
    return out

def rotate_xy(xy: torch.Tensor, deg: float, cx: float, cy: float) -> torch.Tensor:
    if abs(deg) < 1e-6: return xy
    theta = torch.tensor(deg*np.pi/180.0, dtype=xy.dtype, device=xy.device)
    c, s = torch.cos(theta), torch.sin(theta)
    x = xy[...,0] - cx
    y = xy[...,1] - cy
    xr = x*c - y*s + cx
    yr = x*s + y*c + cy
    return torch.stack([xr,yr], dim=-1)

@torch.no_grad()
def rollout_with_traces(model: WorldModel,
                        ctx_norm_full: torch.Tensor, ctx_masks: torch.Tensor,
                        continuous_indices: List[int], discrete_indices: List[int],
                        n_steps: int, threshold: float=0.5):
    device = ctx_norm_full.device
    B, C, K, F = ctx_norm_full.shape
    discrete_template = ctx_norm_full[:, -1:, :, discrete_indices]  # [B,1,K,n_disc]

    latent_ctx = model.encoder(ctx_norm_full, ctx_masks)  # [B,C,D]
    time_padding = (ctx_masks.sum(dim=-1) == 0)
    pred_latent_ctx, _ = model.dynamics(latent_ctx, time_padding_mask=time_padding)
    current_latent = pred_latent_ctx[:, -1:, :]  # [B,1,D]
    latent_hist = latent_ctx
    prev_state_full = ctx_norm_full[:, -1:, :, :]

    out_cont, out_prior_cont, out_masks = [], [], []
    latent_norm, residual_norm = [], []

    cont_ix = continuous_indices.index(model.idx_x)
    cont_iy = continuous_indices.index(model.idx_y)

    for step in range(n_steps):
        base_cont, exist_logits, residual_xy = model.decoder(current_latent, return_residual_xy=True)
        pred_cont = base_cont.clone()

        exist_prob = torch.sigmoid(exist_logits)
        pred_mask = (exist_prob > threshold).float()

        prior_xy = model._kinematic_prior_xy(prev_state_full)  # [B,1,K,2]

        # final = prior + residual
        if residual_xy is not None:
            residual_xy = residual_xy * pred_mask.unsqueeze(-1)
            pred_cont[..., cont_ix] = prior_xy[...,0] + residual_xy[...,0]
            pred_cont[..., cont_iy] = prior_xy[...,1] + residual_xy[...,1]

        # prior-only curve
        prior_cont = base_cont.clone()
        prior_cont[..., cont_ix] = prior_xy[...,0]
        prior_cont[..., cont_iy] = prior_xy[...,1]

        latent_norm.append(current_latent.norm(dim=-1).mean().detach().cpu())
        residual_norm.append((residual_xy.norm(dim=-1).mean().detach().cpu() if residual_xy is not None else torch.tensor(0.0)))

        out_cont.append(pred_cont)
        out_prior_cont.append(prior_cont)
        out_masks.append(pred_mask)

        # build next prev_state_full
        pred_full = torch.zeros(B,1,K,F, device=device, dtype=ctx_norm_full.dtype)
        pred_full[..., continuous_indices] = pred_cont
        pred_full[..., discrete_indices] = discrete_template
        pred_full[..., continuous_indices] = pred_cont * pred_mask.unsqueeze(-1)
        prev_state_full = pred_full

        # next latent
        latent_hist = torch.cat([latent_hist, current_latent], dim=1)
        next_latent = model.dynamics.step(latent_hist, max_context=model.max_dynamics_context).view(B,1,-1)
        current_latent = next_latent

    return {
        "pred_cont": torch.cat(out_cont, dim=1),
        "prior_cont": torch.cat(out_prior_cont, dim=1),
        "pred_masks": torch.cat(out_masks, dim=1),
        "latent_norm": torch.stack(latent_norm),
        "residual_norm": torch.stack(residual_norm),
    }

def compute_ade_per_agent(pred_xy: torch.Tensor, gt_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dist = torch.norm(pred_xy-gt_xy, dim=-1) * mask
    denom = mask.sum(dim=0).clamp(min=1.0)
    ade = dist.sum(dim=0)/denom
    ade = torch.where(mask.sum(dim=0)>0, ade, torch.full_like(ade, 1e9))
    return ade

def select_agents(mode: str,
                  ctx_mask: torch.Tensor, fut_mask: torch.Tensor,
                  pred_xy: torch.Tensor, gt_xy: torch.Tensor,
                  max_agents: int, seed: int=0) -> torch.Tensor:
    valid = (fut_mask>0.5).all(dim=0) & (ctx_mask>0.5).any(dim=0)
    valid_ids = torch.where(valid)[0]
    if len(valid_ids)==0:
        valid_ids = torch.where((fut_mask>0.5).any(dim=0))[0]
    if len(valid_ids)==0:
        return torch.arange(min(max_agents, pred_xy.shape[1]))

    if mode=="presence":
        return valid_ids[:max_agents]
    if mode=="random":
        g = torch.Generator(device=pred_xy.device); g.manual_seed(seed)
        perm = torch.randperm(len(valid_ids), generator=g, device=pred_xy.device)
        return valid_ids[perm[:max_agents]]

    ade = compute_ade_per_agent(pred_xy, gt_xy, (fut_mask>0.5).float())
    ade_valid = ade[valid_ids]

    if mode=="error":
        order = torch.argsort(ade_valid, descending=True)
        return valid_ids[order[:max_agents]]
    if mode=="best_error":
        order = torch.argsort(ade_valid, descending=False)
        return valid_ids[order[:max_agents]]
    if mode=="median_error":
        order = torch.argsort(ade_valid, descending=False)
        mid = len(order)//2
        chosen = order[mid:mid+max_agents]
        return valid_ids[chosen]
    return valid_ids[:max_agents]

def draw_background(ax, img: np.ndarray):
    ax.imshow(img); ax.axis("off")

def draw_traj(ax, traj_xy: np.ndarray, color: str, lw: float=2.0, alpha: float=0.9,
              linestyle: str="-", arrows_every: int=0):
    if traj_xy.shape[0] < 2: return
    x,y = traj_xy[:,0], traj_xy[:,1]
    ax.plot(x,y, color=color, linewidth=lw, alpha=alpha, linestyle=linestyle)
    ax.scatter([x[0]],[y[0]], s=18, color=color, alpha=alpha, marker="o")
    ax.scatter([x[-1]],[y[-1]], s=22, color=color, alpha=alpha, marker="s")
    if arrows_every and traj_xy.shape[0] > 2:
        for t in range(0, traj_xy.shape[0]-1, arrows_every):
            dx, dy = x[t+1]-x[t], y[t+1]-y[t]
            ax.arrow(x[t], y[t], dx, dy, length_includes_head=True,
                     head_width=6, head_length=8, color=color, alpha=min(1.0,alpha), linewidth=0)

def masked_traj(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask > 0.5
    if m.sum() < 2:
        return xy[m] if m.sum() > 0 else xy[:0]
    return xy[m]

def compute_zoom_roi(gt_key: np.ndarray, pr_key: np.ndarray, margin_px: int, W: int, H: int):
    pts = np.concatenate([gt_key, pr_key], axis=0)
    xmin = int(np.floor(pts[:,0].min()))-margin_px
    xmax = int(np.ceil (pts[:,0].max()))+margin_px
    ymin = int(np.floor(pts[:,1].min()))-margin_px
    ymax = int(np.ceil (pts[:,1].max()))+margin_px
    xmin = max(0, min(W-1, xmin)); xmax = max(0, min(W-1, xmax))
    ymin = max(0, min(H-1, ymin)); ymax = max(0, min(H-1, ymax))
    if xmax <= xmin+2: xmax = min(W-1, xmin+3)
    if ymax <= ymin+2: ymax = min(H-1, ymin+3)
    return xmin, ymin, xmax, ymax

def make_fig(bg: np.ndarray,
             ctx_xy: np.ndarray, ctx_mask: np.ndarray,
             gt_xy: np.ndarray, gt_mask: np.ndarray,
             pred_xy: np.ndarray, pred_mask: np.ndarray,
             prior_xy: np.ndarray,
             latent_norm: np.ndarray, residual_norm: np.ndarray,
             zoom_roi, arrows_every: int=0, title: Optional[str]=None):
    fig = plt.figure(figsize=(18,7), dpi=220, constrained_layout=True)
    gs = fig.add_gridspec(2,3, height_ratios=[10,2])
    ax_gt   = fig.add_subplot(gs[0,0])
    ax_pred = fig.add_subplot(gs[0,1])
    ax_cmp  = fig.add_subplot(gs[0,2])
    ax_tl   = fig.add_subplot(gs[1,:])

    for ax in (ax_gt, ax_pred, ax_cmp):
        draw_background(ax, bg)

    C,A,_ = ctx_xy.shape
    H = gt_xy.shape[0]

    for a in range(A):
        ct = masked_traj(ctx_xy[:,a,:], ctx_mask[:,a])
        gt = masked_traj(gt_xy[:,a,:], gt_mask[:,a])
        if ct.shape[0] >= 2: draw_traj(ax_gt, ct, "tab:blue", arrows_every=arrows_every)
        if gt.shape[0] >= 2: draw_traj(ax_gt, gt, "tab:green", arrows_every=arrows_every)
    ax_gt.set_title("Context + Ground Truth", fontsize=12)

    for a in range(A):
        ct = masked_traj(ctx_xy[:,a,:], ctx_mask[:,a])
        pr = masked_traj(pred_xy[:,a,:], pred_mask[:,a])
        kp = masked_traj(prior_xy[:,a,:], pred_mask[:,a])
        if ct.shape[0] >= 2: draw_traj(ax_pred, ct, "tab:blue", arrows_every=arrows_every)
        if kp.shape[0] >= 2: draw_traj(ax_pred, kp, "gray", linestyle="--", lw=1.8, alpha=0.85)
        if pr.shape[0] >= 2: draw_traj(ax_pred, pr, "tab:red", arrows_every=arrows_every)
    ax_pred.set_title("Context + WorldModel (Prior + Residual)", fontsize=12)

    for a in range(A):
        gt = masked_traj(gt_xy[:,a,:], gt_mask[:,a])
        pr = masked_traj(pred_xy[:,a,:], pred_mask[:,a])
        kp = masked_traj(prior_xy[:,a,:], pred_mask[:,a])
        if kp.shape[0] >= 2: draw_traj(ax_cmp, kp, "gray", linestyle="--", lw=1.8, alpha=0.85)
        if gt.shape[0] >= 2: draw_traj(ax_cmp, gt, "tab:green", arrows_every=arrows_every)
        if pr.shape[0] >= 2: draw_traj(ax_cmp, pr, "tab:red", arrows_every=arrows_every)
    ax_cmp.set_title("GT vs WorldModel (+ Kinematic Prior)", fontsize=12)

    xmin,ymin,xmax,ymax = zoom_roi
    rect = plt.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, edgecolor="black", facecolor="none", linewidth=2.0)
    ax_cmp.add_patch(rect)

    axins = inset_axes(ax_cmp, width="42%", height="42%", loc="upper right", borderpad=1.0)
    draw_background(axins, bg)
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymax, ymin)
    for a in range(A):
        gt = masked_traj(gt_xy[:,a,:], gt_mask[:,a])
        pr = masked_traj(pred_xy[:,a,:], pred_mask[:,a])
        kp = masked_traj(prior_xy[:,a,:], pred_mask[:,a])
        if kp.shape[0] >= 2: draw_traj(axins, kp, "gray", linestyle="--", lw=1.6, alpha=0.85)
        if gt.shape[0] >= 2: draw_traj(axins, gt, "tab:green", lw=2.0, alpha=0.9)
        if pr.shape[0] >= 2: draw_traj(axins, pr, "tab:red", lw=2.0, alpha=0.9)
    axins.set_title("Zoom-in", fontsize=10)

    steps = np.arange(1, H+1)
    ax_tl.plot(steps, latent_norm, linewidth=2.0, label="||z_t|| (latent norm)")
    ax_tl.plot(steps, residual_norm, linewidth=2.0, label="||r_xy|| (decoded residual)")
    ax_tl.set_xlabel("Rollout step", fontsize=10)
    ax_tl.set_ylabel("Magnitude", fontsize=10)
    ax_tl.grid(True, alpha=0.25)
    ax_tl.legend(loc="upper right", fontsize=9, ncol=2)
    ax_tl.set_title("Imagined Latent Rollout Traces (World Model)", fontsize=11)

    if title: fig.suptitle(title, fontsize=13)
    return fig

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_data", type=str, required=True)
    p.add_argument("--metadata", type=str, required=True)
    p.add_argument("--site_images_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="results/visualizations")
    p.add_argument("--context_length", type=int, default=65)
    p.add_argument("--rollout_horizon", type=int, default=15)
    p.add_argument("--num_samples", type=int, default=60)
    p.add_argument("--max_agents", type=int, default=5)
    p.add_argument("--sampling", type=str, default="random", choices=["random","uniform"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--select_agents", type=str, default="presence",
                   choices=["presence","random","best_error","median_error","error"])
    p.add_argument("--zoom_margin_px", type=int, default=70)
    p.add_argument("--arrows_every", type=int, default=0)
    p.add_argument("--site_id_feat", type=int, default=11)
    p.add_argument("--flip_y", action="store_true")
    p.add_argument("--rotate_deg", type=float, default=0.0)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.metadata,"r") as f:
        metadata = json.load(f)

    # feature indices (match your pipeline)
    continuous_indices = [0,1,2,3,4,5,6,9,10]
    vi = metadata.get("validation_info", {})
    df = vi.get("discrete_features", {"class_id":7, "lane_id":8, "site_id":11})
    discrete_indices = [int(df["class_id"]), int(df["lane_id"]), int(df["site_id"])]

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"]
    latent_dim = infer_latent_dim_from_ckpt(state_dict, 256)
    num_lanes, num_sites, num_classes = infer_embedding_sizes_from_ckpt(state_dict)
    input_dim = int(metadata.get("n_features", 12))

    from src.utils.common import parse_discrete_feature_indices_from_metadata
    lane_idx, class_idx, site_idx = parse_discrete_feature_indices_from_metadata(metadata, fallback=(8, 7, 11), strict=False)

    model = WorldModel(input_dim=input_dim, latent_dim=latent_dim,
                       num_lanes=num_lanes, num_sites=num_sites, num_classes=num_classes,
                       lane_feature_idx=lane_idx,
                       class_feature_idx=class_idx,
                       site_feature_idx=site_idx)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # stats from train_episodes.npz next to test_data
    train_npz = str(Path(args.test_data).parent / "train_episodes.npz")
    mean_cont, std_cont = compute_mean_std_from_episodes_npz(train_npz, continuous_indices)
    mean_cont = mean_cont.to(device); std_cont = std_cont.to(device)
    model.set_normalization_stats(mean_cont.cpu().numpy(), std_cont.cpu().numpy(), continuous_indices)

    # site images
    site_dir = Path(args.site_images_dir)
    site_images: Dict[int, np.ndarray] = {}
    for sid, name in SITE_NAMES.items():
        fp = site_dir / f"Site{name}.jpg"
        if fp.exists():
            site_images[sid] = np.array(Image.open(fp).convert("RGB"))
    if not site_images:
        raise FileNotFoundError(f"No SiteX.jpg found in {site_dir}")

    # dataset (raw pixels)
    ds = TrajectoryDataset(data_path=args.test_data, normalize=False)
    N = len(ds)
    take = min(args.num_samples, N)
    if args.sampling=="random":
        idxs = np.random.choice(np.arange(N), size=take, replace=False).tolist()
    else:
        step = max(1, N // max(1,take))
        idxs = list(range(0,N,step))[:take]

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for ds_idx in tqdm(idxs, desc="Samples"):
        item = ds[ds_idx]
        states = item["states"].to(device)  # [T,K,F] raw pixels
        masks  = item["masks"].to(device)   # [T,K]
        T,K,F = states.shape
        C,Hh = args.context_length, args.rollout_horizon
        if T < C+Hh: continue

        # infer site id from first valid
        site_id = None
        for t in range(min(T,5)):
            valid = masks[t] > 0.5
            if valid.any():
                k0 = int(torch.nonzero(valid, as_tuple=False)[0].item())
                site_id = int(states[t,k0,args.site_id_feat].item())
                break
        if site_id is None or site_id not in site_images:
            continue

        bg = site_images[site_id]
        H_img, W_img = bg.shape[0], bg.shape[1]
        cx, cy = (W_img-1)/2.0, (H_img-1)/2.0

        ctx_full = states[:C].unsqueeze(0)
        ctx_masks = masks[:C].unsqueeze(0)
        gt_full = states[C:C+Hh]
        gt_masks = masks[C:C+Hh]

        ctx_norm = normalize_full_states(ctx_full, mean_cont, std_cont, continuous_indices)

        out = rollout_with_traces(model, ctx_norm, ctx_masks, continuous_indices, discrete_indices, n_steps=Hh)

        pred_xy = denorm_xy_from_cont(out["pred_cont"], mean_cont, std_cont, continuous_indices)[0]
        prior_xy = denorm_xy_from_cont(out["prior_cont"], mean_cont, std_cont, continuous_indices)[0]
        pred_masks = out["pred_masks"][0]

        gt_xy = gt_full[:,:, :2]
        ctx_xy = states[:C, :, :2]

        if args.flip_y:
            pred_xy = apply_flip_y(pred_xy, H_img)
            prior_xy = apply_flip_y(prior_xy, H_img)
            gt_xy = apply_flip_y(gt_xy, H_img)
            ctx_xy = apply_flip_y(ctx_xy, H_img)
        if abs(args.rotate_deg) > 1e-6:
            pred_xy = rotate_xy(pred_xy, args.rotate_deg, cx, cy)
            prior_xy = rotate_xy(prior_xy, args.rotate_deg, cx, cy)
            gt_xy = rotate_xy(gt_xy, args.rotate_deg, cx, cy)
            ctx_xy = rotate_xy(ctx_xy, args.rotate_deg, cx, cy)

        agent_idx = select_agents(args.select_agents, ctx_masks[0], gt_masks, pred_xy, gt_xy,
                                  max_agents=args.max_agents, seed=args.seed+ds_idx)

        ctx_xy_a = ctx_xy[:, agent_idx].detach().cpu().numpy()
        ctx_mask_a = ctx_masks[0,:,agent_idx].detach().cpu().numpy()
        gt_xy_a = gt_xy[:, agent_idx].detach().cpu().numpy()
        gt_mask_a = gt_masks[:, agent_idx].detach().cpu().numpy()
        pred_xy_a = pred_xy[:, agent_idx].detach().cpu().numpy()
        pred_mask_a = pred_masks[:, agent_idx].detach().cpu().numpy()
        prior_xy_a = prior_xy[:, agent_idx].detach().cpu().numpy()

        latent_norm = out["latent_norm"].detach().cpu().numpy()
        residual_norm = out["residual_norm"].detach().cpu().numpy()

        zoom_roi = compute_zoom_roi(gt_xy_a[:,0,:], pred_xy_a[:,0,:], args.zoom_margin_px, W_img, H_img)
        site_name = SITE_NAMES.get(site_id, f"id{site_id}")
        title = f"Site {site_name} | sample {ds_idx} | agents={len(agent_idx)} | sel={args.select_agents}"

        fig = make_fig(bg, ctx_xy_a, ctx_mask_a, gt_xy_a, gt_mask_a, pred_xy_a, pred_mask_a, prior_xy_a,
                       latent_norm, residual_norm, zoom_roi, arrows_every=args.arrows_every, title=title)

        out_path = out_dir / f"site{site_name}_idx{ds_idx:06d}_sel-{args.select_agents}_A{len(agent_idx):02d}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    print(f"\nSaved {saved} figures to: {out_dir}")

if __name__ == "__main__":
    main()
