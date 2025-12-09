"""
Attention Visualization and Model Interpretability Tools

Provides tools to visualize and understand what the model learns.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def visualize_attention_heatmap(
    attention_weights: np.ndarray,
    save_path: Optional[str] = None,
    vehicle_ids: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Visualize attention weights as a heatmap.

    Args:
        attention_weights: [K, K] attention matrix
        save_path: Path to save figure
        vehicle_ids: Optional list of vehicle IDs for labeling
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        attention_weights,
        annot=False,
        cmap='YlOrRd',
        square=True,
        cbar_kws={'label': 'Attention Weight'},
        ax=ax
    )

    if vehicle_ids:
        ax.set_xticklabels(vehicle_ids, rotation=45)
        ax.set_yticklabels(vehicle_ids, rotation=0)

    ax.set_xlabel('Attending To (Key)')
    ax.set_ylabel('Attending From (Query)')
    ax.set_title('Vehicle-to-Vehicle Attention Weights')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_spatial_attention(
    attention_weights: np.ndarray,
    positions: np.ndarray,
    query_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10)
):
    """
    Visualize attention weights in spatial context.

    Shows which vehicles a specific vehicle attends to, overlaid on position map.

    Args:
        attention_weights: [K, K] attention matrix
        positions: [K, 2] vehicle positions (x, y)
        query_idx: Index of the query vehicle to visualize
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    K = len(positions)
    query_attention = attention_weights[query_idx]  # [K]

    # Plot all vehicles
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=200,
        c=query_attention,
        cmap='YlOrRd',
        alpha=0.7,
        edgecolors='black',
        linewidths=2
    )

    # Highlight query vehicle
    ax.scatter(
        positions[query_idx, 0],
        positions[query_idx, 1],
        s=400,
        c='blue',
        marker='*',
        edgecolors='black',
        linewidths=3,
        label=f'Query Vehicle {query_idx}',
        zorder=10
    )

    # Draw attention arrows (only for top-k attended vehicles)
    top_k = min(5, K - 1)
    top_indices = np.argsort(query_attention)[-top_k-1:-1][::-1]

    for idx in top_indices:
        if idx != query_idx:
            ax.annotate(
                '',
                xy=positions[idx],
                xytext=positions[query_idx],
                arrowprops=dict(
                    arrowstyle='->',
                    lw=query_attention[idx] * 5,
                    color='blue',
                    alpha=0.5
                )
            )

            # Add attention weight label
            mid_x = (positions[query_idx, 0] + positions[idx, 0]) / 2
            mid_y = (positions[query_idx, 1] + positions[idx, 1]) / 2
            ax.text(
                mid_x, mid_y,
                f'{query_attention[idx]:.2f}',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Spatial Attention from Vehicle {query_idx}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved spatial attention to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_attention_patterns(
    attention_weights: torch.Tensor,
    masks: torch.Tensor,
    positions: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Analyze attention patterns to understand what the model learns.

    Args:
        attention_weights: [B, H, K, K] multi-head attention weights
        masks: [B, K] validity masks
        positions: [B, K, 2] optional positions

    Returns:
        Dictionary with analysis results
    """
    B, H, K, _ = attention_weights.shape

    results = {}

    # Average attention per head
    results['avg_attention_per_head'] = attention_weights.mean(dim=(0, 2, 3)).cpu().numpy()

    # Attention entropy (how focused is attention?)
    # Higher entropy = more uniform attention, lower entropy = more focused
    attention_probs = attention_weights + 1e-8
    entropy = -(attention_probs * torch.log(attention_probs)).sum(dim=-1).mean()
    results['attention_entropy'] = entropy.item()

    # Self-attention ratio (how much do vehicles attend to themselves?)
    diag_mask = torch.eye(K, device=attention_weights.device).bool()
    self_attention = attention_weights[:, :, diag_mask].mean()
    results['self_attention_ratio'] = self_attention.item()

    # Sparsity (how many vehicles are effectively attended to?)
    # Count vehicles with attention > threshold
    threshold = 0.1
    active_attention = (attention_weights > threshold).float().sum(dim=-1).mean()
    results['avg_attended_vehicles'] = active_attention.item()

    # If positions available, analyze spatial patterns
    if positions is not None:
        # Compute distances
        pos_i = positions.unsqueeze(2)  # [B, K, 1, 2]
        pos_j = positions.unsqueeze(1)  # [B, 1, K, 2]
        distances = torch.norm(pos_i - pos_j, dim=-1)  # [B, K, K]

        # Correlation between attention and distance
        # (Do vehicles attend more to nearby vehicles?)
        valid_mask = masks.unsqueeze(1) * masks.unsqueeze(2)  # [B, K, K]
        valid_mask = valid_mask.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, K, K]

        # Flatten valid entries
        attn_flat = attention_weights[valid_mask].cpu().numpy()
        dist_flat = distances.unsqueeze(1).expand(-1, H, -1, -1)[valid_mask].cpu().numpy()

        # Compute correlation
        if len(attn_flat) > 0:
            correlation = np.corrcoef(attn_flat, dist_flat)[0, 1]
            results['attention_distance_correlation'] = correlation

            # Average attention by distance bins
            dist_bins = [0, 10, 20, 30, 50, 100, np.inf]
            attention_by_distance = []

            for i in range(len(dist_bins) - 1):
                mask = (dist_flat >= dist_bins[i]) & (dist_flat < dist_bins[i+1])
                if mask.sum() > 0:
                    avg_attn = attn_flat[mask].mean()
                    attention_by_distance.append(avg_attn)
                else:
                    attention_by_distance.append(0.0)

            results['attention_by_distance'] = attention_by_distance
            results['distance_bins'] = [f'{dist_bins[i]}-{dist_bins[i+1]}' for i in range(len(dist_bins)-1)]

    return results


def plot_attention_statistics(
    attention_analysis: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot attention analysis statistics.

    Args:
        attention_analysis: Results from analyze_attention_patterns
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot 1: Attention per head
    if 'avg_attention_per_head' in attention_analysis:
        axes[0].bar(
            range(len(attention_analysis['avg_attention_per_head'])),
            attention_analysis['avg_attention_per_head']
        )
        axes[0].set_xlabel('Head Index')
        axes[0].set_ylabel('Average Attention')
        axes[0].set_title('Average Attention per Head')
        axes[0].grid(True, alpha=0.3)

    # Plot 2: Key metrics
    metrics = {
        'Entropy': attention_analysis.get('attention_entropy', 0),
        'Self-Attention': attention_analysis.get('self_attention_ratio', 0),
        'Avg Attended': attention_analysis.get('avg_attended_vehicles', 0)
    }

    axes[1].bar(range(len(metrics)), list(metrics.values()))
    axes[1].set_xticks(range(len(metrics)))
    axes[1].set_xticklabels(list(metrics.keys()), rotation=45)
    axes[1].set_ylabel('Value')
    axes[1].set_title('Attention Metrics')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Attention by distance
    if 'attention_by_distance' in attention_analysis:
        axes[2].plot(
            attention_analysis['attention_by_distance'],
            marker='o',
            linewidth=2
        )
        axes[2].set_xticks(range(len(attention_analysis['distance_bins'])))
        axes[2].set_xticklabels(attention_analysis['distance_bins'], rotation=45)
        axes[2].set_xlabel('Distance Range (m)')
        axes[2].set_ylabel('Average Attention')
        axes[2].set_title('Attention vs Distance')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention statistics to {save_path}")
    else:
        plt.show()

    plt.close()


def extract_attention_from_model(
    model: nn.Module,
    states: torch.Tensor,
    masks: torch.Tensor,
    layer_idx: int = 0
) -> torch.Tensor:
    """
    Extract attention weights from a trained model.

    Args:
        model: Trained world model
        states: [B, T, K, F] input states
        masks: [B, T, K] masks
        layer_idx: Which transformer layer to extract from

    Returns:
        attention_weights: [B*T, H, K, K]
    """
    model.eval()

    # Hook to capture attention
    attention_weights = []

    def attention_hook(module, input, output):
        # Capture attention weights from multi-head attention
        if hasattr(module, 'attn_weights'):
            attention_weights.append(module.attn_weights.detach())

    # Register hook
    if hasattr(model, 'encoder'):
        encoder = model.encoder
        if hasattr(encoder, 'transformer'):
            layer = encoder.transformer.layers[layer_idx]
            handle = layer.self_attn.register_forward_hook(attention_hook)

            # Forward pass
            with torch.no_grad():
                _ = model(states, masks)

            handle.remove()

            if len(attention_weights) > 0:
                return attention_weights[0]

    return None


def create_attention_report(
    model: nn.Module,
    dataloader,
    save_dir: str = './attention_analysis',
    n_samples: int = 5,
    device: torch.device = torch.device('cpu')
):
    """
    Create comprehensive attention analysis report.

    Args:
        model: Trained model
        dataloader: DataLoader with data
        save_dir: Directory to save analysis
        n_samples: Number of samples to analyze
        device: Device to run on
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= n_samples:
                break

            states = batch['states'].to(device)
            masks = batch['masks'].to(device)

            # Extract attention (if supported)
            attention = extract_attention_from_model(model, states, masks)

            if attention is not None:
                # Analyze patterns
                positions = states[..., :2]  # Assuming first 2 features are position
                analysis = analyze_attention_patterns(attention, masks[0], positions[0])

                # Save statistics plot
                plot_attention_statistics(
                    analysis,
                    save_path=save_path / f'attention_stats_sample_{sample_count}.png'
                )

                # Save spatial attention for first vehicle
                attn_np = attention[0, 0].cpu().numpy()  # First sample, first head
                pos_np = positions[0, 0].cpu().numpy()  # First sample, first timestep
                mask_np = masks[0, 0].cpu().numpy()

                # Only plot valid vehicles
                valid_idx = mask_np > 0.5
                if valid_idx.sum() > 1:
                    visualize_spatial_attention(
                        attn_np[valid_idx][:, valid_idx],
                        pos_np[valid_idx],
                        query_idx=0,
                        save_path=save_path / f'spatial_attention_sample_{sample_count}.png'
                    )

            sample_count += 1

    print(f"Attention analysis report saved to {save_dir}")


if __name__ == '__main__':
    # Test visualization
    K = 10

    # Generate random attention weights
    attention = np.random.rand(K, K)
    attention = attention / attention.sum(axis=1, keepdims=True)

    # Generate random positions
    positions = np.random.rand(K, 2) * 100

    # Visualize
    visualize_attention_heatmap(attention)
    visualize_spatial_attention(attention, positions, query_idx=0)

    print("Attention visualization test complete!")
