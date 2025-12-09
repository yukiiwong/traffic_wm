"""
Rollout Evaluation Utilities

Evaluate multi-step prediction performance.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from tqdm import tqdm

from .prediction_metrics import compute_all_metrics


def evaluate_rollout(
    model: nn.Module,
    data_loader,
    context_length: int = 10,
    rollout_length: int = 20,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Evaluate model on multi-step rollout.

    Args:
        model: Trained world model
        data_loader: DataLoader with test data
        context_length: Number of context frames
        rollout_length: Number of frames to predict
        device: Device to run on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Rollout"):
            states = batch['states'].to(device)  # [B, T, K, F]
            masks = batch['masks'].to(device)    # [B, T, K]

            B, T, K, F = states.shape

            # Ensure we have enough frames
            if T < context_length + rollout_length:
                continue

            # Split into context and target
            context_states = states[:, :context_length]
            context_masks = masks[:, :context_length]

            target_states = states[:, context_length:context_length + rollout_length]
            target_masks = masks[:, context_length:context_length + rollout_length]

            # Perform rollout
            rollout_output = model.rollout(
                initial_states=context_states,
                initial_masks=context_masks,
                n_steps=rollout_length,
                teacher_forcing=False
            )

            predicted_states = rollout_output['predicted_states']

            # Compute metrics
            metrics = compute_all_metrics(
                predicted=predicted_states,
                ground_truth=target_states,
                masks=target_masks
            )

            all_metrics.append(metrics)

    # Average metrics across batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    return avg_metrics


def evaluate_multihorizon(
    model: nn.Module,
    data_loader,
    context_length: int = 10,
    horizons: List[int] = [1, 3, 5, 10, 20],
    device: torch.device = torch.device('cpu')
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate at multiple prediction horizons.

    Args:
        model: Trained world model
        data_loader: DataLoader with test data
        context_length: Number of context frames
        horizons: List of prediction horizons to evaluate
        device: Device to run on

    Returns:
        Dictionary mapping horizon to metrics
    """
    model.eval()

    results = {h: [] for h in horizons}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Multi-horizon Evaluation"):
            states = batch['states'].to(device)
            masks = batch['masks'].to(device)

            B, T, K, F = states.shape

            max_horizon = max(horizons)
            if T < context_length + max_horizon:
                continue

            # Context
            context_states = states[:, :context_length]
            context_masks = masks[:, :context_length]

            # Rollout for max horizon
            rollout_output = model.rollout(
                initial_states=context_states,
                initial_masks=context_masks,
                n_steps=max_horizon,
                teacher_forcing=False
            )

            predicted_states = rollout_output['predicted_states']

            # Evaluate at each horizon
            for h in horizons:
                pred_h = predicted_states[:, :h]
                target_h = states[:, context_length:context_length + h]
                mask_h = masks[:, context_length:context_length + h]

                metrics = compute_all_metrics(
                    predicted=pred_h,
                    ground_truth=target_h,
                    masks=mask_h
                )

                results[h].append(metrics)

    # Average across batches
    avg_results = {}
    for h in horizons:
        if len(results[h]) > 0:
            avg_results[h] = {}
            for key in results[h][0].keys():
                avg_results[h][key] = sum(m[key] for m in results[h]) / len(results[h])

    return avg_results


def evaluate_with_teacher_forcing(
    model: nn.Module,
    data_loader,
    context_length: int = 10,
    rollout_length: int = 20,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Dict[str, float]]:
    """
    Compare open-loop (no teacher forcing) vs closed-loop (teacher forcing).

    Args:
        model: Trained world model
        data_loader: DataLoader with test data
        context_length: Number of context frames
        rollout_length: Number of frames to predict
        device: Device to run on

    Returns:
        Dictionary with 'open_loop' and 'closed_loop' metrics
    """
    model.eval()

    open_loop_metrics = []
    closed_loop_metrics = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Teacher Forcing Evaluation"):
            states = batch['states'].to(device)
            masks = batch['masks'].to(device)

            B, T, K, F = states.shape

            if T < context_length + rollout_length:
                continue

            context_states = states[:, :context_length]
            context_masks = masks[:, :context_length]

            target_states = states[:, context_length:context_length + rollout_length]
            target_masks = masks[:, context_length:context_length + rollout_length]

            full_states = states[:, :context_length + rollout_length]

            # Open-loop (autoregressive)
            open_loop_output = model.rollout(
                initial_states=context_states,
                initial_masks=context_masks,
                n_steps=rollout_length,
                teacher_forcing=False
            )

            open_loop_pred = open_loop_output['predicted_states']
            open_metrics = compute_all_metrics(
                predicted=open_loop_pred,
                ground_truth=target_states,
                masks=target_masks
            )
            open_loop_metrics.append(open_metrics)

            # Closed-loop (teacher forcing)
            closed_loop_output = model.rollout(
                initial_states=context_states,
                initial_masks=context_masks,
                n_steps=rollout_length,
                teacher_forcing=True,
                ground_truth_states=full_states
            )

            closed_loop_pred = closed_loop_output['predicted_states']
            closed_metrics = compute_all_metrics(
                predicted=closed_loop_pred,
                ground_truth=target_states,
                masks=target_masks
            )
            closed_loop_metrics.append(closed_metrics)

    # Average
    def average_metrics(metric_list):
        if len(metric_list) == 0:
            return {}
        avg = {}
        for key in metric_list[0].keys():
            avg[key] = sum(m[key] for m in metric_list) / len(metric_list)
        return avg

    return {
        'open_loop': average_metrics(open_loop_metrics),
        'closed_loop': average_metrics(closed_loop_metrics)
    }


if __name__ == '__main__':
    # Example usage
    from src.models.world_model import WorldModel
    from src.data.dataset import get_dataloader

    # Load model
    model = WorldModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load data
    test_loader = get_dataloader(
        data_path='../../data/processed/episodes.npz',
        batch_size=8,
        shuffle=False
    )

    # Evaluate
    metrics = evaluate_rollout(
        model=model,
        data_loader=test_loader,
        context_length=10,
        rollout_length=20,
        device=device
    )

    print("Rollout Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
