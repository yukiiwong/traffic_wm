"""
Evaluation metrics and visualization tools.
"""

from .prediction_metrics import (
    compute_ade,
    compute_fde,
    compute_velocity_error,
    compute_heading_error,
    compute_collision_rate,
    compute_existence_metrics,
    compute_all_metrics
)
from .rollout_eval import (
    evaluate_rollout,
    evaluate_multihorizon,
    evaluate_with_teacher_forcing
)
from .visualization import (
    visualize_trajectories,
    visualize_rollout,
    visualize_error_heatmap,
    plot_metrics_over_time,
    create_animation
)
from .attention_visualization import (
    visualize_attention_heatmap,
    visualize_spatial_attention,
    analyze_attention_patterns,
    plot_attention_statistics,
    create_attention_report
)

__all__ = [
    # Prediction metrics
    "compute_ade",
    "compute_fde",
    "compute_velocity_error",
    "compute_heading_error",
    "compute_collision_rate",
    "compute_existence_metrics",
    "compute_all_metrics",
    # Rollout evaluation
    "evaluate_rollout",
    "evaluate_multihorizon",
    "evaluate_with_teacher_forcing",
    # Visualization
    "visualize_trajectories",
    "visualize_rollout",
    "visualize_error_heatmap",
    "plot_metrics_over_time",
    "create_animation",
    # Attention visualization
    "visualize_attention_heatmap",
    "visualize_spatial_attention",
    "analyze_attention_patterns",
    "plot_attention_statistics",
    "create_attention_report",
]
