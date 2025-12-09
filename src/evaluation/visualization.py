"""
Visualization Utilities

Visualize trajectory predictions and rollouts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.animation import FuncAnimation
import torch
from typing import Optional, List


def visualize_trajectories(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    masks: np.ndarray,
    save_path: Optional[str] = None,
    time_step: int = 0,
    max_agents: int = 20,
    figsize: tuple = (12, 8)
):
    """
    Visualize predicted vs ground truth trajectories at a specific timestep.

    Args:
        predicted: [T, K, F] predicted states
        ground_truth: [T, K, F] ground truth states
        masks: [T, K] binary masks
        save_path: Path to save figure
        time_step: Which timestep to visualize
        max_agents: Maximum agents to display
        figsize: Figure size
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Extract positions
    pred_pos = predicted[time_step, :, :2]  # [K, 2]
    gt_pos = ground_truth[time_step, :, :2]  # [K, 2]
    mask = masks[time_step]  # [K]

    # Plot valid agents
    for k in range(min(max_agents, len(mask))):
        if mask[k] > 0.5:
            # Ground truth (green)
            ax.scatter(gt_pos[k, 0], gt_pos[k, 1],
                      c='green', s=100, alpha=0.7, marker='o',
                      label='Ground Truth' if k == 0 else '')

            # Prediction (red)
            ax.scatter(pred_pos[k, 0], pred_pos[k, 1],
                      c='red', s=100, alpha=0.7, marker='x',
                      label='Predicted' if k == 0 else '')

            # Connection line
            ax.plot([gt_pos[k, 0], pred_pos[k, 0]],
                   [gt_pos[k, 1], pred_pos[k, 1]],
                   'k--', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Trajectory Comparison at Timestep {time_step}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()



def visualize_rollout(predicted, ground_truth, masks,
                      save_path='rollout_comparison.png',
                      max_agents=10, figsize=(10, 4)):
    """
    predicted:  [B, T, K, 2] 或 [T, K, 2]
    ground_truth: 同上
    masks:      [B, T, K]   或 [T, K]
    """

    # ---- 转成 numpy ----
    predicted = np.asarray(predicted)
    ground_truth = np.asarray(ground_truth)
    masks = np.asarray(masks)

    # ---- 去掉 batch 维度（如果有）----
    if predicted.ndim == 4:
        predicted = predicted[0]      # [T, K, 2]
        ground_truth = ground_truth[0]
    if masks.ndim == 3:
        masks = masks[0]              # [T, K] 或 [K, T]

    # 现在 predicted: [T, K, 2]
    T, K, _ = predicted.shape

    # ---- 确保 masks 的形状是 [T, K] ----
    if masks.shape == (K, T):
        masks = masks.T
    assert masks.shape == (T, K), f"masks shape {masks.shape} incompatible with predicted {predicted.shape}"

    # ---- 开始画图 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左图：GT 轨迹
    for k in range(min(K, max_agents)):
        if masks[:, k].sum() > 0:
            valid_idx = masks[:, k] > 0.5
            traj_gt = ground_truth[valid_idx, k, :2]
            if len(traj_gt) > 0:
                ax1.plot(traj_gt[:, 0], traj_gt[:, 1],
                         alpha=0.6, linewidth=2)

    ax1.set_title("Ground Truth")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_aspect("equal", adjustable="box")

    # 右图：Pred 轨迹
    for k in range(min(K, max_agents)):
        if masks[:, k].sum() > 0:
            valid_idx = masks[:, k] > 0.5
            traj_pred = predicted[valid_idx, k, :2]
            if len(traj_pred) > 0:
                ax2.plot(traj_pred[:, 0], traj_pred[:, 1],
                         alpha=0.6, linewidth=2)

    ax2.set_title("Rollout Prediction")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"✅ rollout_comparison saved to {save_path}")


def visualize_error_heatmap(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    masks: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Visualize position error as a heatmap over time and agents.

    Args:
        predicted: [T, K, F] predictions
        ground_truth: [T, K, F] ground truth
        masks: [T, K] masks
        save_path: Save path
        figsize: Figure size
    """
    # Compute position errors
    pred_pos = predicted[:, :, :2]
    gt_pos = ground_truth[:, :, :2]

    errors = np.linalg.norm(pred_pos - gt_pos, axis=-1)  # [T, K]
    errors = errors * masks  # Apply mask

    # Replace zeros (masked) with NaN for better visualization
    errors[masks < 0.5] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    im = ax.imshow(errors.T, aspect='auto', cmap='hot', interpolation='nearest')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Agent ID')
    ax.set_title('Position Error Heatmap')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('L2 Error (m)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_metrics_over_time(
    metrics_dict: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Plot evaluation metrics over time/horizons.

    Args:
        metrics_dict: Dictionary mapping horizon to metrics
        save_path: Save path
        figsize: Figure size
    """
    horizons = sorted(metrics_dict.keys())

    # Extract metrics
    metric_names = list(metrics_dict[horizons[0]].keys())

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, metric_name in enumerate(metric_names[:4]):
        values = [metrics_dict[h][metric_name] for h in horizons]

        axes[idx].plot(horizons, values, marker='o', linewidth=2)
        axes[idx].set_xlabel('Prediction Horizon')
        axes[idx].set_ylabel(metric_name.replace('_', ' ').title())
        axes[idx].set_title(metric_name.replace('_', ' ').title())
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def create_animation(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    masks: np.ndarray,
    save_path: str,
    fps: int = 10,
    max_agents: int = 20
):
    """
    Create animated visualization of rollout.

    Args:
        predicted: [T, K, F] predictions
        ground_truth: [T, K, F] ground truth
        masks: [T, K] masks
        save_path: Path to save animation (e.g., 'rollout.gif' or 'rollout.mp4')
        fps: Frames per second
        max_agents: Max agents to visualize
    """
    T = predicted.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Set axis limits
    all_pos = np.concatenate([predicted[:, :, :2], ground_truth[:, :, :2]], axis=0)
    x_min, x_max = all_pos[:, :, 0].min(), all_pos[:, :, 0].max()
    y_min, y_max = all_pos[:, :, 1].min(), all_pos[:, :, 1].max()

    margin = 5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    def update(frame):
        ax.clear()

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_aspect('equal')

        # Plot trajectories up to current frame
        for k in range(min(max_agents, masks.shape[1])):
            if masks[frame, k] > 0.5:
                # Ground truth (green circle)
                ax.scatter(ground_truth[frame, k, 0], ground_truth[frame, k, 1],
                          c='green', s=150, alpha=0.7, marker='o', edgecolors='black')

                # Prediction (red x)
                ax.scatter(predicted[frame, k, 0], predicted[frame, k, 1],
                          c='red', s=150, alpha=0.7, marker='x')

                # History trails
                if frame > 0:
                    history_idx = masks[:frame, k] > 0.5
                    gt_trail = ground_truth[:frame, k, :2][history_idx]
                    pred_trail = predicted[:frame, k, :2][history_idx]

                    if len(gt_trail) > 0:
                        ax.plot(gt_trail[:, 0], gt_trail[:, 1],
                               'g-', alpha=0.3, linewidth=1)
                    if len(pred_trail) > 0:
                        ax.plot(pred_trail[:, 0], pred_trail[:, 1],
                               'r--', alpha=0.3, linewidth=1)

        ax.set_title(f'Frame {frame}/{T-1}')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)

    anim = FuncAnimation(fig, update, frames=T, interval=1000//fps)

    # Save animation
    if save_path.endswith('.gif'):
        anim.save(save_path, writer='pillow', fps=fps)
    else:
        anim.save(save_path, writer='ffmpeg', fps=fps)

    print(f"Saved animation to {save_path}")
    plt.close()


if __name__ == '__main__':
    # Test visualization
    T, K, F = 30, 10, 6

    predicted = np.random.randn(T, K, F) * 10
    ground_truth = np.random.randn(T, K, F) * 10
    masks = np.random.randint(0, 2, (T, K)).astype(float)

    # Visualize single frame
    visualize_trajectories(predicted, ground_truth, masks, time_step=15)

    # Visualize rollout
    visualize_rollout(predicted, ground_truth, masks)

    # Error heatmap
    visualize_error_heatmap(predicted, ground_truth, masks)
