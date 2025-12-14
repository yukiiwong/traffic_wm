"""
Visualize Trajectory Predictions on Site Images

Draw ground truth and predicted trajectories on actual site aerial images.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
import json
from tqdm import tqdm
from typing import Dict, List, Tuple

from src.models.world_model import WorldModel
from src.data.dataset import get_dataloader


# Site ID to name mapping
SITE_NAMES = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I'
}


def denormalize_states(states: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, continuous_indices: List[int]) -> torch.Tensor:
    """
    Denormalize states using training statistics.

    Args:
        states: [B, T, K, F] normalized states
        mean: Mean values for continuous features
        std: Std values for continuous features
        continuous_indices: Indices of continuous features

    Returns:
        Denormalized states
    """
    # Create output tensor (copy to avoid modifying input)
    denorm_states = states.clone()

    # Denormalize only continuous features
    for i, feat_idx in enumerate(continuous_indices):
        denorm_states[..., feat_idx] = states[..., feat_idx] * std[i] + mean[i]

    return denorm_states


def draw_trajectory_on_image(
    img: np.ndarray,
    trajectory: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 2,
    draw_points: bool = True
) -> np.ndarray:
    """
    Draw a single trajectory on the image.

    Args:
        img: Image array [H, W, 3]
        trajectory: [T, 2] trajectory coordinates (x, y)
        color: RGB color tuple
        thickness: Line thickness
        draw_points: Whether to draw points at each timestep

    Returns:
        Image with trajectory drawn
    """
    import cv2

    # Filter out invalid points (negative or zero coordinates)
    valid_mask = (trajectory[:, 0] > 0) & (trajectory[:, 1] > 0)
    trajectory = trajectory[valid_mask]

    if len(trajectory) < 2:
        return img

    # Draw lines connecting consecutive points
    for i in range(len(trajectory) - 1):
        pt1 = (int(trajectory[i, 0]), int(trajectory[i, 1]))
        pt2 = (int(trajectory[i+1, 0]), int(trajectory[i+1, 1]))
        cv2.line(img, pt1, pt2, color, thickness)

    # Draw points
    if draw_points:
        for pt in trajectory:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)

    # Draw start and end markers
    if len(trajectory) > 0:
        # Start: larger circle
        cv2.circle(img, (int(trajectory[0, 0]), int(trajectory[0, 1])), 6, color, 2)
        # End: square
        end_pt = (int(trajectory[-1, 0]), int(trajectory[-1, 1]))
        cv2.rectangle(img, (end_pt[0]-4, end_pt[1]-4), (end_pt[0]+4, end_pt[1]+4), color, -1)

    return img


def normalize_states(states: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, continuous_indices: List[int]) -> torch.Tensor:
    """Normalize states for model input."""
    norm_states = states.clone()
    for i, feat_idx in enumerate(continuous_indices):
        norm_states[..., feat_idx] = (states[..., feat_idx] - mean[i]) / std[i]
    return norm_states


def visualize_batch_predictions(
    model: torch.nn.Module,
    test_loader,
    site_images_dir: Path,
    output_dir: Path,
    context_length: int,
    rollout_horizon: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    continuous_indices: List[int],
    device: torch.device,
    num_samples: int = 5,
    max_agents_per_sample: int = 10
):
    """
    Visualize predictions for multiple samples across different sites.

    Args:
        model: Trained world model
        test_loader: Test data loader (unnormalized pixel coordinates)
        site_images_dir: Directory containing site images
        output_dir: Directory to save visualizations
        context_length: Number of context frames
        rollout_horizon: Number of prediction frames
        mean: Mean values for normalization/denormalization
        std: Std values for normalization/denormalization
        continuous_indices: Indices of continuous features
        device: Device to run on
        num_samples: Number of samples to visualize per site
        max_agents_per_sample: Maximum agents to visualize per sample
    """
    model.eval()

    # Load site images
    site_images = {}
    for site_id, site_name in SITE_NAMES.items():
        img_path = site_images_dir / f'Site{site_name}.jpg'
        if img_path.exists():
            site_images[site_id] = np.array(Image.open(img_path))
            print(f"Loaded {img_path.name}: {site_images[site_id].shape}")

    # Organize samples by site
    samples_by_site = {site_id: [] for site_id in SITE_NAMES.keys()}

    print("\nCollecting samples from test data...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing batches"):
            states = batch['states'].to(device)  # Unnormalized pixel coordinates
            masks = batch['masks'].to(device)
            scene_ids = batch['scene_id']  # Note: singular 'scene_id'

            B, T, K, F = states.shape

            # Ensure enough frames
            if T < context_length + rollout_horizon:
                continue

            # Normalize states for model
            states_norm = normalize_states(states, mean.to(device), std.to(device), continuous_indices)

            # Split into context and target
            context_states_norm = states_norm[:, :context_length]
            context_masks = masks[:, :context_length]
            target_states = states[:, context_length:context_length + rollout_horizon]  # Keep unnormalized for GT
            target_masks = masks[:, context_length:context_length + rollout_horizon]

            # Run rollout on normalized data
            rollout_output = model.rollout(
                initial_states=context_states_norm,
                initial_masks=context_masks,
                n_steps=rollout_horizon,
                teacher_forcing=False
            )

            predicted_states_norm = rollout_output['predicted_states']

            # Denormalize predictions
            predicted_states = denormalize_states(
                predicted_states_norm,
                mean.to(device), std.to(device),
                continuous_indices
            )

            # Collect samples for each site (all in pixel coordinates now)
            for b in range(B):
                site_id = scene_ids[b].item()

                if site_id not in site_images:
                    continue

                if len(samples_by_site[site_id]) >= num_samples:
                    continue

                sample = {
                    'context': states[b, :context_length].cpu(),  # Unnormalized context
                    'context_masks': context_masks[b].cpu(),
                    'ground_truth': target_states[b].cpu(),  # Unnormalized GT
                    'ground_truth_masks': target_masks[b].cpu(),
                    'predicted': predicted_states[b].cpu(),  # Denormalized predictions
                }

                samples_by_site[site_id].append(sample)

            # Check if we have enough samples for all sites
            if all(len(samples_by_site[sid]) >= num_samples for sid in site_images.keys()):
                break

    # Create visualizations for each site
    print("\nCreating visualizations...")
    for site_id, samples in samples_by_site.items():
        if len(samples) == 0 or site_id not in site_images:
            continue

        site_name = SITE_NAMES[site_id]
        print(f"\nSite {site_name}: {len(samples)} samples")

        for sample_idx, sample in enumerate(samples):
            # Load fresh image for each sample
            img = site_images[site_id].copy()

            # Data is already in pixel coordinates
            context = sample['context']  # [T_context, K, F]
            gt = sample['ground_truth']  # [T_rollout, K, F]
            pred = sample['predicted']  # [T_rollout, K, F]

            context_masks = sample['context_masks']  # [T_context, K]
            gt_masks = sample['ground_truth_masks']  # [T_rollout, K]

            # Find valid agents (exist in both context and ground truth)
            # An agent is valid if it appears in the last context frame
            last_context_valid = context_masks[-1] > 0.5
            agent_indices = torch.where(last_context_valid)[0][:max_agents_per_sample]

            # Draw trajectories for each agent
            for agent_idx in agent_indices:
                agent_idx = agent_idx.item()

                # Extract agent trajectories
                context_traj = context[:, agent_idx, :2].numpy()  # [T_context, 2]
                gt_traj = gt[:, agent_idx, :2].numpy()  # [T_rollout, 2]
                pred_traj = pred[:, agent_idx, :2].numpy()  # [T_rollout, 2]

                # Filter by mask
                context_valid = context_masks[:, agent_idx] > 0.5
                gt_valid = gt_masks[:, agent_idx] > 0.5

                context_traj = context_traj[context_valid.numpy()]
                gt_traj = gt_traj[gt_valid.numpy()]

                # Draw context (blue)
                img = draw_trajectory_on_image(
                    img, context_traj,
                    color=(0, 0, 255),  # Blue
                    thickness=2,
                    draw_points=False
                )

                # Draw ground truth (green)
                img = draw_trajectory_on_image(
                    img, gt_traj,
                    color=(0, 255, 0),  # Green
                    thickness=3,
                    draw_points=True
                )

                # Draw prediction (red)
                img = draw_trajectory_on_image(
                    img, pred_traj,
                    color=(255, 0, 0),  # Red
                    thickness=3,
                    draw_points=True
                )

            # Add legend
            import cv2
            legend_y = 30
            cv2.putText(img, "Blue: Context", (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "Green: Ground Truth", (10, legend_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, "Red: Prediction", (10, legend_y + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, f"Site {site_name} - Sample {sample_idx+1}", (10, legend_y + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Save visualization
            output_path = output_dir / f'site_{site_name}_sample_{sample_idx+1}.jpg'
            Image.fromarray(img).save(output_path)
            print(f"  Saved: {output_path.name}")

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize trajectory predictions on site images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data NPZ file')
    parser.add_argument('--metadata', type=str, default='data/processed/metadata.json', help='Path to metadata.json')
    parser.add_argument('--stats_path', type=str, default='data/processed/train_stats.npz', help='Path to training statistics')
    parser.add_argument('--site_images_dir', type=str, default='src/evaluation/sites', help='Directory containing site images')
    parser.add_argument('--context_length', type=int, default=65, help='Number of context frames')
    parser.add_argument('--rollout_horizon', type=int, default=15, help='Number of frames to predict')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='results/visualizations', help='Directory to save visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize per site')
    parser.add_argument('--max_agents', type=int, default=10, help='Maximum agents to visualize per sample')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    site_images_dir = Path(args.site_images_dir)

    # Load metadata
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)

    n_features = metadata['n_features']

    # Get continuous feature indices
    discrete_features = metadata['validation_info']['discrete_features']
    discrete_indices = list(discrete_features.values())
    continuous_indices = [i for i in range(n_features) if i not in discrete_indices]

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")
    print(f"Site images: {site_images_dir}")
    print(f"Context: {args.context_length} frames, Rollout: {args.rollout_horizon} frames")
    print(f"Continuous features: {continuous_indices}")

    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Infer model configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        input_dim = config.get('input_dim', n_features)
        latent_dim = config.get('latent_dim', 256)
        dynamics_type = config.get('dynamics_type', 'gru')
    else:
        print("Warning: No config found in checkpoint, inferring from state_dict")
        state_dict = checkpoint['model_state_dict']

        if 'encoder.to_latent.0.bias' in state_dict:
            latent_dim = state_dict['encoder.to_latent.0.bias'].shape[0]
        else:
            latent_dim = 256

        dynamics_keys = [k for k in state_dict.keys() if k.startswith('dynamics.')]
        if any('transformer' in k for k in dynamics_keys):
            dynamics_type = 'transformer'
        elif any('lstm' in k.lower() for k in dynamics_keys):
            dynamics_type = 'lstm'
        elif any('rnn' in k.lower() for k in dynamics_keys):
            dynamics_type = 'gru'
        else:
            dynamics_type = 'gru'

        input_dim = n_features

    print(f"Model: {dynamics_type}, latent_dim={latent_dim}")

    # Create model
    model = WorldModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        dynamics_type=dynamics_type
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load data and extract normalization stats
    from src.data.dataset import TrajectoryDataset

    # First load train dataset to get normalization stats
    train_data_path = str(Path(args.test_data).parent / 'train_episodes.npz')
    print(f"Loading train dataset to get normalization stats from: {train_data_path}")

    train_dataset = TrajectoryDataset(
        data_path=train_data_path,
        normalize=True,
        stats_path=None  # Train dataset computes its own stats
    )

    # Extract mean and std from train dataset
    mean = train_dataset.mean
    std = train_dataset.std

    print(f"Loaded normalization stats: mean shape {mean.shape}, std shape {std.shape}")

    # Now load test dataset with train stats
    test_dataset = TrajectoryDataset(
        data_path=args.test_data,
        normalize=False  # We'll use train stats manually, no need to normalize again
    )

    # Create dataloader from dataset
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Trajectory Visualizations")
    print("="*60)

    visualize_batch_predictions(
        model=model,
        test_loader=test_loader,
        site_images_dir=site_images_dir,
        output_dir=output_dir,
        context_length=args.context_length,
        rollout_horizon=args.rollout_horizon,
        mean=mean,
        std=std,
        continuous_indices=continuous_indices,
        device=device,
        num_samples=args.num_samples,
        max_agents_per_sample=args.max_agents
    )

    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
