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
    heading_idx: Optional[int],
    assume_radians: bool = True,
) -> float:
    """
    Average heading error.

    Args:
        heading_idx: which feature dimension corresponds to heading/angle.
                    (For your metadata: angle is feature 6.)
                    If None, returns float('nan') to indicate unavailable.
        assume_radians: if True, converts result to degrees for readability.

    Returns:
        mean heading error in degrees (default) or radians (if assume_radians=False)
        Returns NaN if heading_idx is None or out of range.
    """
    if heading_idx is None:
        return float('nan')
    if predicted.shape[-1] <= heading_idx or ground_truth.shape[-1] <= heading_idx:
        return float('nan')

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


def compute_moving_agent_metrics(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor,
    velocity_threshold: float = 0.5,  # m/s or px/s depending on conversion
) -> Dict[str, float]:
    """
    Compute metrics only for moving agents (velocity > threshold).
    
    Useful for traffic scenarios with many stationary vehicles (at traffic lights, etc.)
    
    Args:
        velocity_threshold: minimum velocity magnitude to consider agent as "moving"
                           Should be in same units as velocity features (m/s if converted)
    
    Returns:
        Dict with moving_ade, moving_fde, moving_velocity_error, moving_agent_ratio
    """
    if predicted.shape[-1] < 4 or ground_truth.shape[-1] < 4:
        return {
            "moving_ade": float('nan'),
            "moving_fde": float('nan'), 
            "moving_velocity_error": float('nan'),
            "moving_agent_ratio": 0.0,
        }
    
    # Compute velocity magnitude from ground truth
    gt_vel = ground_truth[..., 2:4]  # [B,T,K,2]
    vel_mag = torch.norm(gt_vel, dim=-1)  # [B,T,K]
    
    # Moving mask: valid AND velocity > threshold
    moving_mask = masks * (vel_mag > velocity_threshold).float()
    
    moving_count = moving_mask.sum()
    total_count = masks.sum()
    
    if moving_count < 1:
        return {
            "moving_ade": float('nan'),
            "moving_fde": float('nan'),
            "moving_velocity_error": float('nan'),
            "moving_agent_ratio": 0.0,
        }
    
    # ADE for moving agents
    pred_pos = predicted[..., :2]
    gt_pos = ground_truth[..., :2]
    displacement = torch.norm(pred_pos - gt_pos, dim=-1)
    moving_ade = (displacement * moving_mask).sum() / moving_count
    
    # FDE for moving agents (last timestep)
    moving_mask_final = moving_mask[:, -1, :]
    if moving_mask_final.sum() > 0:
        displacement_final = torch.norm(predicted[:, -1, :, :2] - ground_truth[:, -1, :, :2], dim=-1)
        moving_fde = (displacement_final * moving_mask_final).sum() / moving_mask_final.sum()
    else:
        moving_fde = torch.tensor(float('nan'))
    
    # Velocity error for moving agents
    pred_vel = predicted[..., 2:4]
    vel_error = torch.norm(pred_vel - gt_vel, dim=-1)
    moving_vel_error = (vel_error * moving_mask).sum() / moving_count
    
    return {
        "moving_ade": moving_ade.item(),
        "moving_fde": moving_fde.item(),
        "moving_velocity_error": moving_vel_error.item(),
        "moving_agent_ratio": (moving_count / total_count).item() if total_count > 0 else 0.0,
    }


def compute_velocity_direction_error(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor,
    velocity_threshold: float = 0.5,
) -> float:
    """
    Compute angular difference between predicted and GT velocity directions.
    
    This measures if the predicted motion direction is correct, which is often
    more important than exact heading angle for trajectory prediction.
    
    Args:
        velocity_threshold: only compute for agents with velocity > threshold
    
    Returns:
        Mean angular error in degrees
    """
    if predicted.shape[-1] < 4 or ground_truth.shape[-1] < 4:
        return float('nan')
    
    pred_vel = predicted[..., 2:4]  # [B,T,K,2]
    gt_vel = ground_truth[..., 2:4]
    
    # Filter by velocity threshold
    gt_vel_mag = torch.norm(gt_vel, dim=-1)
    moving_mask = masks * (gt_vel_mag > velocity_threshold).float()
    
    if moving_mask.sum() < 1:
        return float('nan')
    
    # Compute angles from velocities
    pred_angle = torch.atan2(pred_vel[..., 1], pred_vel[..., 0])  # [B,T,K]
    gt_angle = torch.atan2(gt_vel[..., 1], gt_vel[..., 0])
    
    # Angular difference
    angle_diff = _angular_diff_rad(pred_angle, gt_angle)  # [B,T,K] in radians
    
    # Average over moving agents
    masked_diff = angle_diff * moving_mask
    mean_error_rad = masked_diff.sum() / moving_mask.sum()
    
    return (mean_error_rad * 180.0 / math.pi).item()


def compute_acceleration_error(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor,
) -> float:
    """
    Mean L2 acceleration error over all valid agents.
    
    Returns:
        acceleration error (m/s^2 if converted)
    """
    if predicted.shape[-1] < 6 or ground_truth.shape[-1] < 6:
        return float('nan')
    
    pred_acc = predicted[..., 4:6]
    gt_acc = ground_truth[..., 4:6]
    acc_error = torch.norm(pred_acc - gt_acc, dim=-1)
    masked = acc_error * masks
    return (masked.sum() / masks.sum().clamp(min=1)).item()


def compute_position_variance(
    predicted: torch.Tensor,
    masks: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute variance of predicted positions to detect mode collapse.
    
    Low variance suggests the model is not capturing trajectory diversity.
    
    Returns:
        Dict with pos_variance_x, pos_variance_y, pos_variance_total
    """
    pred_pos = predicted[..., :2]  # [B,T,K,2]
    
    # Only compute over valid agents
    valid_count = masks.sum()
    if valid_count < 2:
        return {
            "pos_variance_x": float('nan'),
            "pos_variance_y": float('nan'),
            "pos_variance_total": float('nan'),
        }
    
    # Flatten to [N, 2] where N = valid agents
    valid_pos = pred_pos[masks > 0.5]  # [N, 2]
    
    var_x = valid_pos[:, 0].var().item()
    var_y = valid_pos[:, 1].var().item()
    var_total = var_x + var_y
    
    return {
        "pos_variance_x": var_x,
        "pos_variance_y": var_y,
        "pos_variance_total": var_total,
    }


def compute_extended_metrics(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor,
    velocity_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute extended metrics including moving-agent-specific and direction metrics.
    
    Args:
        velocity_threshold: m/s threshold for considering an agent as "moving"
    
    Returns:
        Dict with all extended metrics
    """
    metrics = {}
    
    # Moving agent metrics
    moving_metrics = compute_moving_agent_metrics(
        predicted, ground_truth, masks, velocity_threshold
    )
    metrics.update(moving_metrics)
    
    # Velocity direction error
    metrics["velocity_direction_error"] = compute_velocity_direction_error(
        predicted, ground_truth, masks, velocity_threshold
    )
    
    # Acceleration error
    metrics["acceleration_error"] = compute_acceleration_error(
        predicted, ground_truth, masks
    )
    
    # Position variance (mode collapse detection)
    variance_metrics = compute_position_variance(predicted, masks)
    metrics.update(variance_metrics)
    
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
