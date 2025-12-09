"""
Prediction Metrics for Multi-Agent Trajectory Evaluation

Includes ADE, FDE, collision rate, and other metrics.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def compute_ade(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor
) -> float:
    """
    Average Displacement Error (ADE).

    Average L2 distance between predicted and ground truth positions
    across all time steps and agents.

    Args:
        predicted: [B, T, K, F] predicted states (F >= 2 for x, y)
        ground_truth: [B, T, K, F] ground truth states
        masks: [B, T, K] binary mask for valid agents

    Returns:
        ADE value (meters)
    """
    # Extract positions (x, y)
    pred_pos = predicted[..., :2]  # [B, T, K, 2]
    gt_pos = ground_truth[..., :2]  # [B, T, K, 2]

    # Compute L2 distance
    displacement = torch.norm(pred_pos - gt_pos, dim=-1)  # [B, T, K]

    # Apply mask
    masked_displacement = displacement * masks

    # Average over valid elements
    ade = masked_displacement.sum() / masks.sum().clamp(min=1)

    return ade.item()


def compute_fde(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor
) -> float:
    """
    Final Displacement Error (FDE).

    L2 distance between predicted and ground truth positions
    at the final time step.

    Args:
        predicted: [B, T, K, F] predicted states
        ground_truth: [B, T, K, F] ground truth states
        masks: [B, T, K] binary mask

    Returns:
        FDE value (meters)
    """
    # Extract final positions
    pred_final = predicted[:, -1, :, :2]  # [B, K, 2]
    gt_final = ground_truth[:, -1, :, :2]  # [B, K, 2]
    mask_final = masks[:, -1, :]  # [B, K]

    # Compute L2 distance
    displacement = torch.norm(pred_final - gt_final, dim=-1)  # [B, K]

    # Apply mask
    masked_displacement = displacement * mask_final

    # Average
    fde = masked_displacement.sum() / mask_final.sum().clamp(min=1)

    return fde.item()


def compute_velocity_error(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor
) -> float:
    """
    Average velocity error.

    Args:
        predicted: [B, T, K, F] predicted states (F >= 4 for vx, vy)
        ground_truth: [B, T, K, F] ground truth states
        masks: [B, T, K] binary mask

    Returns:
        Average velocity error (m/s)
    """
    if predicted.shape[-1] < 4:
        return 0.0

    # Extract velocities (vx, vy)
    pred_vel = predicted[..., 2:4]  # [B, T, K, 2]
    gt_vel = ground_truth[..., 2:4]  # [B, T, K, 2]

    # Compute L2 error
    vel_error = torch.norm(pred_vel - gt_vel, dim=-1)  # [B, T, K]

    # Apply mask
    masked_error = vel_error * masks

    # Average
    avg_error = masked_error.sum() / masks.sum().clamp(min=1)

    return avg_error.item()


def compute_heading_error(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor,
    heading_idx: int = 4
) -> float:
    """
    Average heading error (in degrees).

    Args:
        predicted: [B, T, K, F] predicted states
        ground_truth: [B, T, K, F] ground truth states
        masks: [B, T, K] binary mask
        heading_idx: Index of heading in feature dimension

    Returns:
        Average heading error (degrees)
    """
    if predicted.shape[-1] <= heading_idx:
        return 0.0

    # Extract headings
    pred_heading = predicted[..., heading_idx]  # [B, T, K]
    gt_heading = ground_truth[..., heading_idx]  # [B, T, K]

    # Compute angular difference (handling wrap-around)
    diff = torch.abs(pred_heading - gt_heading)
    diff = torch.min(diff, 2 * np.pi - diff)  # Wrap to [0, pi]

    # Apply mask
    masked_diff = diff * masks

    # Average and convert to degrees
    avg_error = masked_diff.sum() / masks.sum().clamp(min=1)
    avg_error_deg = avg_error * (180.0 / np.pi)

    return avg_error_deg.item()


def compute_collision_rate(
    predicted: torch.Tensor,
    masks: torch.Tensor,
    safety_margin: float = 2.0
) -> float:
    """
    Estimate collision rate based on inter-vehicle distances.

    Args:
        predicted: [B, T, K, F] predicted states
        masks: [B, T, K] binary mask
        safety_margin: Minimum safe distance (meters)

    Returns:
        Collision rate (percentage)
    """
    B, T, K, F = predicted.shape

    # Extract positions
    positions = predicted[..., :2]  # [B, T, K, 2]

    total_checks = 0
    total_collisions = 0

    for b in range(B):
        for t in range(T):
            # Get valid agents at this timestep
            valid_mask = masks[b, t].bool()  # [K]
            valid_positions = positions[b, t, valid_mask]  # [n_valid, 2]

            n_valid = valid_positions.shape[0]
            if n_valid < 2:
                continue

            # Compute pairwise distances
            pos_expanded_1 = valid_positions.unsqueeze(1)  # [n_valid, 1, 2]
            pos_expanded_2 = valid_positions.unsqueeze(0)  # [1, n_valid, 2]

            distances = torch.norm(pos_expanded_1 - pos_expanded_2, dim=-1)  # [n_valid, n_valid]

            # Count collisions (distance < safety_margin)
            # Exclude self-comparisons (diagonal)
            mask_upper_tri = torch.triu(torch.ones_like(distances), diagonal=1).bool()
            relevant_distances = distances[mask_upper_tri]

            collisions = (relevant_distances < safety_margin).sum().item()

            total_collisions += collisions
            total_checks += len(relevant_distances)

    if total_checks == 0:
        return 0.0

    collision_rate = (total_collisions / total_checks) * 100.0
    return collision_rate


def compute_existence_metrics(
    predicted_existence: torch.Tensor,
    ground_truth_masks: torch.Tensor
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for agent existence prediction.

    Args:
        predicted_existence: [B, T, K] existence logits or probabilities
        ground_truth_masks: [B, T, K] binary masks

    Returns:
        Dictionary with precision, recall, F1
    """
    # Convert to binary predictions
    if predicted_existence.min() < 0:  # Logits
        pred_binary = (torch.sigmoid(predicted_existence) > 0.5).float()
    else:  # Probabilities
        pred_binary = (predicted_existence > 0.5).float()

    # True positives, false positives, false negatives
    tp = (pred_binary * ground_truth_masks).sum().item()
    fp = (pred_binary * (1 - ground_truth_masks)).sum().item()
    fn = ((1 - pred_binary) * ground_truth_masks).sum().item()

    # Precision, recall, F1
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_all_metrics(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    masks: torch.Tensor,
    predicted_existence: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        predicted: [B, T, K, F] predicted states
        ground_truth: [B, T, K, F] ground truth states
        masks: [B, T, K] binary masks
        predicted_existence: [B, T, K] optional existence predictions

    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'ade': compute_ade(predicted, ground_truth, masks),
        'fde': compute_fde(predicted, ground_truth, masks),
        'velocity_error': compute_velocity_error(predicted, ground_truth, masks),
        'heading_error': compute_heading_error(predicted, ground_truth, masks),
        'collision_rate': compute_collision_rate(predicted, masks)
    }

    if predicted_existence is not None:
        existence_metrics = compute_existence_metrics(predicted_existence, masks)
        metrics.update({
            'existence_precision': existence_metrics['precision'],
            'existence_recall': existence_metrics['recall'],
            'existence_f1': existence_metrics['f1']
        })

    return metrics


if __name__ == '__main__':
    # Test metrics
    B, T, K, F = 4, 10, 20, 6

    predicted = torch.randn(B, T, K, F)
    ground_truth = torch.randn(B, T, K, F)
    masks = torch.randint(0, 2, (B, T, K)).float()

    metrics = compute_all_metrics(predicted, ground_truth, masks)

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
