"""
Prediction Metrics for Multi-Agent Trajectory Evaluation.

This module is used by src.evaluation.rollout_eval and provides:
- ADE / FDE (position errors)
- velocity error
- heading error (wrap-around safe, radians -> degrees)
- collision metrics (robust to padding leakage; supports pair/frame/episode)

Assumptions (match your metadata / feature layout):
- position: indices (0,1) -> (x,y)
- velocity: indices (2,3) -> (vx,vy) if present
- acceleration: indices (4,5) -> (ax,ay) if present
- heading/angle: index is configurable (for your SiteA layout it's feature 6: "angle")
"""

from __future__ import annotations

from typing import Dict, Optional, Literal
import math

import torch

# IMPORTANT: Use absolute imports so `python -m src.evaluation.rollout_eval` works reliably.
from src.utils.common import convert_pixels_to_meters, get_pixel_to_meter_conversion


def compute_ade(predicted: torch.Tensor, ground_truth: torch.Tensor, masks: torch.Tensor) -> float:
    """
    Average Displacement Error (ADE): mean L2 distance over all valid (t,k).

    Args:
        predicted: [B, T, K, F] (F >= 2)
        ground_truth: [B, T, K, F]
        masks: [B, T, K] (0/1)

    Returns:
        ADE (same unit as x,y; meters if you converted before calling)
    """
    pred_pos = predicted[..., :2]
    gt_pos = ground_truth[..., :2]
    displacement = torch.norm(pred_pos - gt_pos, dim=-1)  # [B,T,K]
    masked = displacement * masks
    return (masked.sum() / masks.sum().clamp(min=1)).item()


def compute_fde(predicted: torch.Tensor, ground_truth: torch.Tensor, masks: torch.Tensor) -> float:
    """
    Final Displacement Error (FDE): L2 distance at final time step.

    Returns:
        FDE (same unit as x,y)
    """
    pred_final = predicted[:, -1, :, :2]
    gt_final = ground_truth[:, -1, :, :2]
    mask_final = masks[:, -1, :]
    displacement = torch.norm(pred_final - gt_final, dim=-1)
    masked = displacement * mask_final
    return (masked.sum() / mask_final.sum().clamp(min=1)).item()


def compute_velocity_error(predicted: torch.Tensor, ground_truth: torch.Tensor, masks: torch.Tensor) -> float:
    """
    Mean L2 velocity error over all valid (t,k).

    Returns:
        velocity error (same unit as vx,vy; m/s if converted before calling)
    """
    if predicted.shape[-1] < 4 or ground_truth.shape[-1] < 4:
        return 0.0
    pred_vel = predicted[..., 2:4]
    gt_vel = ground_truth[..., 2:4]
    vel_error = torch.norm(pred_vel - gt_vel, dim=-1)  # [B,T,K]
    masked = vel_error * masks
    return (masked.sum() / masks.sum().clamp(min=1)).item()


def _angular_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Smallest angular difference |a-b| on a circle, assuming radians.

    Output in [0, pi].
    """
    # Using atan2(sin, cos) is numerically stable.
    return torch.abs(torch.atan2(torch.sin(a - b), torch.cos(a - b)))


def compute_heading_error(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor,
    heading_idx: int,
    assume_radians: bool = True,
) -> float:
    """
    Average heading error.

    Args:
        heading_idx: which feature dimension corresponds to heading/angle.
                    (For your metadata: angle is feature 6.)
        assume_radians: if True, converts result to degrees for readability.

    Returns:
        mean heading error in degrees (default) or radians (if assume_radians=False)
    """
    if predicted.shape[-1] <= heading_idx or ground_truth.shape[-1] <= heading_idx:
        return 0.0

    pred_h = predicted[..., heading_idx]
    gt_h = ground_truth[..., heading_idx]

    diff = _angular_diff_rad(pred_h, gt_h)  # [B,T,K] in radians
    masked = diff * masks
    avg = (masked.sum() / masks.sum().clamp(min=1))

    if assume_radians:
        return (avg * (180.0 / math.pi)).item()
    return avg.item()


CollisionMode = Literal["pair", "frame", "episode"]


def compute_collision_rate(
    predicted: torch.Tensor,
    masks: torch.Tensor,
    safety_margin: float = 2.0,
    mode: CollisionMode = "frame",
    zero_pos_eps: float = 1e-3,
) -> float:
    """
    Robust collision metric based on inter-agent distances on each frame.

    Why "robust":
    - filters near-zero positions (often padding leakage where x=y=0) even if mask==1
    - supports different reporting modes:
        * "pair": percentage of colliding pairs among all valid pairs (can be overly sensitive when K is large)
        * "frame": percentage of evaluated frames (b,t) where ANY collision occurs (recommended)
        * "episode": percentage of episodes b where ANY collision occurs over the horizon

    Args:
        predicted: [B,T,K,F] (expects x,y in [:2])
        masks: [B,T,K]
        safety_margin: collision distance threshold in the same unit as x,y (meters if converted)
        mode: "pair" | "frame" | "episode"
        zero_pos_eps: filter points with ||(x,y)|| <= eps (padding leakage)

    Returns:
        collision rate in percentage (%)
    """
    B, T, K, _ = predicted.shape
    pos = predicted[..., :2]  # [B,T,K,2]

    if mode == "pair":
        total_pairs = 0
        total_collisions = 0

        for b in range(B):
            for t in range(T):
                valid = masks[b, t].bool()
                p = pos[b, t, valid]
                # filter near-zero positions
                keep = (p.norm(dim=-1) > zero_pos_eps)
                p = p[keep]
                n = p.shape[0]
                if n < 2:
                    continue
                d = torch.cdist(p, p)  # [n,n]
                tri = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
                rel = d[tri]
                total_collisions += (rel < safety_margin).sum().item()
                total_pairs += rel.numel()

        return 0.0 if total_pairs == 0 else (total_collisions / total_pairs) * 100.0

    # frame / episode
    frames_evaluated = 0
    frames_with_collision = 0
    episodes_with_collision = 0

    for b in range(B):
        episode_has_collision = False
        for t in range(T):
            valid = masks[b, t].bool()
            p = pos[b, t, valid]
            keep = (p.norm(dim=-1) > zero_pos_eps)
            p = p[keep]
            n = p.shape[0]
            if n < 2:
                continue

            d = torch.cdist(p, p)
            tri = torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)
            rel = d[tri]

            any_col = bool((rel < safety_margin).any().item())
            frames_evaluated += 1
            if any_col:
                frames_with_collision += 1
                episode_has_collision = True

        if episode_has_collision:
            episodes_with_collision += 1

    if mode == "frame":
        return 0.0 if frames_evaluated == 0 else (frames_with_collision / frames_evaluated) * 100.0
    if mode == "episode":
        return (episodes_with_collision / B) * 100.0
    raise ValueError(f"Unknown collision mode: {mode}")


def compute_existence_metrics(
    predicted_existence: torch.Tensor,
    ground_truth_masks: torch.Tensor,
) -> Dict[str, float]:
    """
    Precision / Recall / F1 for existence prediction.

    Args:
        predicted_existence: [B,T,K] logits or probs
        ground_truth_masks: [B,T,K] (0/1)

    Returns:
        dict: precision, recall, f1
    """
    if predicted_existence.min().item() < 0:  # logits
        pred_bin = (torch.sigmoid(predicted_existence) > 0.5).float()
    else:
        pred_bin = (predicted_existence > 0.5).float()

    tp = (pred_bin * ground_truth_masks).sum().item()
    fp = (pred_bin * (1 - ground_truth_masks)).sum().item()
    fn = ((1 - pred_bin) * ground_truth_masks).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_all_metrics(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor,
    predicted_existence: Optional[torch.Tensor] = None,
    pixel_to_meter: Optional[float] = None,
    convert_to_meters: bool = False,
    heading_idx: Optional[int] = None,
    # collision configs
    collision_mode: CollisionMode = "frame",
    safety_margin: float = 2.0,
    zero_pos_eps: float = 1e-3,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Notes for your project:
    - your world_model.rollout returns CONTINUOUS features only, with layout matching continuous_indices.
    - for SiteA layout: angle is feature 6 -> if predicted/gt include angle at index 6, pass heading_idx=6.
      If predicted/gt are continuous-only, pass heading_idx = continuous_indices.index(6).

    Args:
        convert_to_meters:
            If True, converts x,y (and vx,vy / ax,ay if present) from pixels to meters.
            If your predicted/gt are already in meters, set False.
    """
    if convert_to_meters:
        if pixel_to_meter is None:
            pixel_to_meter = get_pixel_to_meter_conversion()

        predicted = convert_pixels_to_meters(
            predicted,
            pixel_to_meter,
            position_indices=(0, 1),
            velocity_indices=(2, 3) if predicted.shape[-1] > 3 else None,
            acceleration_indices=(4, 5) if predicted.shape[-1] > 5 else None,
        )
        ground_truth = convert_pixels_to_meters(
            ground_truth,
            pixel_to_meter,
            position_indices=(0, 1),
            velocity_indices=(2, 3) if ground_truth.shape[-1] > 3 else None,
            acceleration_indices=(4, 5) if ground_truth.shape[-1] > 5 else None,
        )

    # sensible default: if not provided and we have enough dims, assume angle is at index 6
    if heading_idx is None:
        heading_idx = 6 if predicted.shape[-1] > 6 else 4

    metrics: Dict[str, float] = {
        "ade": compute_ade(predicted, ground_truth, masks),
        "fde": compute_fde(predicted, ground_truth, masks),
        "velocity_error": compute_velocity_error(predicted, ground_truth, masks),
        "heading_error": compute_heading_error(predicted, ground_truth, masks, heading_idx=heading_idx),
        "collision_rate": compute_collision_rate(
            predicted, masks, safety_margin=safety_margin, mode=collision_mode, zero_pos_eps=zero_pos_eps
        ),
    }

    if predicted_existence is not None:
        ex = compute_existence_metrics(predicted_existence, masks)
        metrics.update(
            {
                "existence_precision": ex["precision"],
                "existence_recall": ex["recall"],
                "existence_f1": ex["f1"],
            }
        )

    return metrics


if __name__ == "__main__":
    # quick sanity check
    B, T, K, F = 2, 5, 6, 12
    predicted = torch.randn(B, T, K, F)
    gt = torch.randn(B, T, K, F)
    masks = (torch.rand(B, T, K) > 0.3).float()

    out = compute_all_metrics(
        predicted, gt, masks,
        heading_idx=6,
        collision_mode="frame",
        convert_to_meters=False,
    )
    for k, v in out.items():
        print(k, v)
