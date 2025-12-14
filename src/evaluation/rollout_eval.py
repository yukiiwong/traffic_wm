"""
Rollout Evaluation Utilities

Evaluate multi-step prediction performance.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from tqdm import tqdm

from src.evaluation.prediction_metrics import compute_all_metrics
from src.utils.common import get_pixel_to_meter_conversion


def evaluate_rollout(
    model: nn.Module,
    data_loader,
    context_length: int = 10,
    rollout_length: int = 20,
    device: torch.device = torch.device('cpu'),
    pixel_to_meter: Optional[float] = None,
    convert_to_meters: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on multi-step rollout.

    Args:
        model: Trained world model
        data_loader: DataLoader with test data
        context_length: Number of context frames
        rollout_length: Number of frames to predict
        device: Device to run on
        pixel_to_meter: Conversion factor from pixels to meters.
                       If None and convert_to_meters=True, will auto-load.
        convert_to_meters: If True, convert coordinates from pixels to meters
                          before computing metrics (default: True)

    Returns:
        Dictionary of evaluation metrics (in meters if convert_to_meters=True)
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
                masks=target_masks,
                pixel_to_meter=pixel_to_meter,
                convert_to_meters=convert_to_meters
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
    device: torch.device = torch.device('cpu'),
    pixel_to_meter: Optional[float] = None,
    convert_to_meters: bool = True
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate at multiple prediction horizons.

    Args:
        model: Trained world model
        data_loader: DataLoader with test data
        context_length: Number of context frames
        horizons: List of prediction horizons to evaluate
        device: Device to run on
        pixel_to_meter: Conversion factor from pixels to meters.
                       If None and convert_to_meters=True, will auto-load.
        convert_to_meters: If True, convert coordinates from pixels to meters
                          before computing metrics (default: True)

    Returns:
        Dictionary mapping horizon to metrics (in meters if convert_to_meters=True)
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
                    masks=mask_h,
                    pixel_to_meter=pixel_to_meter,
                    convert_to_meters=convert_to_meters
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
    device: torch.device = torch.device('cpu'),
    pixel_to_meter: Optional[float] = None,
    convert_to_meters: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare open-loop (no teacher forcing) vs closed-loop (teacher forcing).

    Args:
        model: Trained world model
        data_loader: DataLoader with test data
        context_length: Number of context frames
        rollout_length: Number of frames to predict
        device: Device to run on
        pixel_to_meter: Conversion factor from pixels to meters.
                       If None and convert_to_meters=True, will auto-load.
        convert_to_meters: If True, convert coordinates from pixels to meters
                          before computing metrics (default: True)

    Returns:
        Dictionary with 'open_loop' and 'closed_loop' metrics (in meters if convert_to_meters=True)
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
                masks=target_masks,
                pixel_to_meter=pixel_to_meter,
                convert_to_meters=convert_to_meters
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
                masks=target_masks,
                pixel_to_meter=pixel_to_meter,
                convert_to_meters=convert_to_meters
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
    import argparse
    import json
    import numpy as np
    from src.models.world_model import WorldModel
    from src.data.dataset import get_dataloader

    parser = argparse.ArgumentParser(description='Evaluate world model with rollout')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data NPZ file')
    parser.add_argument('--metadata', type=str, default='data/processed/metadata.json', help='Path to metadata.json')
    parser.add_argument('--stats_path', type=str, default='data/processed/train_stats.npz', help='Path to training statistics')
    parser.add_argument('--context_length', type=int, default=65, help='Number of context frames (default: 65)')
    parser.add_argument('--rollout_horizon', type=int, default=15, help='Number of frames to predict (default: 15)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results/', help='Directory to save results')
    parser.add_argument('--convert_to_meters', action='store_true', default=True, help='Convert to meters (default: True)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)

    n_features = metadata['n_features']
    n_lanes = metadata['validation_info']['num_lanes']

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Context length: {args.context_length}")
    print(f"Rollout horizon: {args.rollout_horizon}")
    print(f"Features: {n_features}, Lanes: {n_lanes}")

    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Extract model configuration from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        input_dim = config.get('input_dim', n_features)
        latent_dim = config.get('latent_dim', 256)
        dynamics_type = config.get('dynamics_type', 'gru')
    else:
        # Try to infer from checkpoint state_dict
        print("Warning: No config found in checkpoint, inferring from state_dict")
        state_dict = checkpoint['model_state_dict']

        # Infer latent_dim from encoder.to_latent.0.bias shape
        if 'encoder.to_latent.0.bias' in state_dict:
            latent_dim = state_dict['encoder.to_latent.0.bias'].shape[0]
            print(f"  Inferred latent_dim: {latent_dim}")
        else:
            latent_dim = 256
            print(f"  Could not infer latent_dim, using default: {latent_dim}")

        # Infer dynamics_type and hidden_dim from checkpoint keys and weight shapes
        dynamics_keys = [k for k in state_dict.keys() if k.startswith('dynamics.')]
        dynamics_hidden = 512  # default

        if any('transformer' in k for k in dynamics_keys):
            dynamics_type = 'transformer'
        elif any('rnn.weight_ih_l0' in k for k in dynamics_keys):
            # Check weight shape to distinguish LSTM (4x) from GRU (3x)
            weight_key = [k for k in dynamics_keys if 'rnn.weight_ih_l0' in k][0]
            weight_shape = state_dict[weight_key].shape
            # weight_shape[0] = hidden_dim * num_gates
            # weight_shape[1] = latent_dim (input size)

            # Try to infer: ratio = hidden_dim * num_gates / latent_dim
            total_hidden = weight_shape[0]
            input_size = weight_shape[1]

            # Assume input_size == latent_dim
            if abs(total_hidden / input_size - 4.0) < 0.1:  # LSTM has 4 gates
                dynamics_type = 'lstm'
                dynamics_hidden = total_hidden // 4
            elif abs(total_hidden / input_size - 3.0) < 0.1:  # GRU has 3 gates
                dynamics_type = 'gru'
                dynamics_hidden = total_hidden // 3
            else:
                print(f"  Warning: Unexpected ratio {total_hidden / input_size:.1f}, defaulting to GRU")
                dynamics_type = 'gru'
                dynamics_hidden = total_hidden // 3

            print(f"  Inferred hidden_dim: {dynamics_hidden}")
        else:
            dynamics_type = 'gru'
        print(f"  Inferred dynamics_type: {dynamics_type}")

        input_dim = n_features

    # Create model with inferred or config parameters
    if 'config' in checkpoint:
        model = WorldModel(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dynamics_type=dynamics_type,
            dynamics_hidden=config.get('dynamics_hidden', 512)
        )
    else:
        model = WorldModel(
            input_dim=input_dim,
            latent_dim=latent_dim,
            dynamics_type=dynamics_type,
            dynamics_hidden=dynamics_hidden
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {dynamics_type}, latent_dim={latent_dim}")

    # Load data
    test_loader = get_dataloader(
        data_path=args.test_data,
        batch_size=args.batch_size,
        shuffle=False,
        stats_path=args.stats_path
    )

    # Get pixel to meter conversion
    pixel_to_meter = None
    if args.convert_to_meters:
        try:
            pixel_to_meter = get_pixel_to_meter_conversion()
            print(f"Pixel to meter conversion: {pixel_to_meter:.6f}")
        except Exception as e:
            print(f"Warning: Could not load pixel_to_meter conversion: {e}")
            print("Metrics will be in pixels")
            args.convert_to_meters = False

    # Evaluate
    print("\n" + "="*60)
    print("Starting Rollout Evaluation")
    print("="*60)

    metrics = evaluate_rollout(
        model=model,
        data_loader=test_loader,
        context_length=args.context_length,
        rollout_length=args.rollout_horizon,
        device=device,
        pixel_to_meter=pixel_to_meter,
        convert_to_meters=args.convert_to_meters
    )

    # Print results
    units = "meters" if args.convert_to_meters else "pixels"
    print("\n" + "="*60)
    print(f"Rollout Evaluation Results ({units})")
    print("="*60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save results
    results_file = output_dir / 'rollout_metrics.json'
    results = {
        'metrics': metrics,
        'config': {
            'checkpoint': args.checkpoint,
            'test_data': args.test_data,
            'context_length': args.context_length,
            'rollout_horizon': args.rollout_horizon,
            'convert_to_meters': args.convert_to_meters,
            'units': units,
            'model_type': dynamics_type,
            'latent_dim': latent_dim
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("="*60)
