"""GT direction consistency diagnostic.

Compares two direction definitions on *raw preprocessed* episodes:
- Displacement direction: θ_Δ = atan2(Δy, Δx)
- Velocity direction:     θ_v = atan2(vy, vx)

This is primarily useful to quantify how noisy vx/vy supervision is,
especially when agents disappear/re-appear (mask gaps) or when the
preprocessing estimates velocities from sparse frames.

Usage:
  python -m src.evaluation.debug.gt_direction_consistency \
    --data data/processed_siteA_20/train_episodes.npz \
    --out_dir results/direction_sanity_check

Notes:
- This script does not use model checkpoints or normalization stats.
- It works on the stored vx/vy in the NPZ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def angular_distance(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """Angular distance in [0, π]."""
    diff = wrap_angle(angle1 - angle2)
    return np.abs(diff)


def analyze_direction_consistency(
    data_path: str,
    out_dir: str,
    vel_threshold: float = 0.5,
    disp_threshold: float = 0.1,
    max_vel_plot: float = 20.0,
) -> dict:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    data = np.load(data_path)
    states = data["states"]  # [N, T, K, F]
    masks = data["masks"]  # [N, T, K]

    n_episodes, t_steps, n_agents, feat_dim = states.shape
    print(f"Data shape: N={n_episodes}, T={t_steps}, K={n_agents}, F={feat_dim}")

    x = states[:, :, :, 0]
    y = states[:, :, :, 1]
    vx = states[:, :, :, 2]
    vy = states[:, :, :, 3]

    dx = x[:, 1:, :] - x[:, :-1, :]
    dy = y[:, 1:, :] - y[:, :-1, :]
    theta_disp = np.arctan2(dy, dx)  # [N, T-1, K]

    theta_vel = np.arctan2(vy[:, :-1, :], vx[:, :-1, :])  # [N, T-1, K]

    mask_valid = masks[:, :-1, :] > 0
    vel_mag = np.sqrt(vx[:, :-1, :] ** 2 + vy[:, :-1, :] ** 2)
    disp_mag = np.sqrt(dx**2 + dy**2)

    valid_mask = mask_valid & (vel_mag > vel_threshold) & (disp_mag > disp_threshold)
    n_valid = int(valid_mask.sum())
    n_present = int(mask_valid.sum())
    print(f"Valid samples: {n_valid:,} / {n_present:,} ({100.0*n_valid/max(1,n_present):.1f}%)")

    angle_diff = angular_distance(theta_vel, theta_disp)
    angle_diff_deg = np.rad2deg(angle_diff)[valid_mask]
    vel_mag_valid = vel_mag[valid_mask]

    # Stats on the valid subset
    valid_diffs = np.rad2deg(angle_diff[valid_mask])

    summary = {
        "data_path": str(data_path),
        "thresholds": {
            "vel_threshold": float(vel_threshold),
            "disp_threshold": float(disp_threshold),
        },
        "overall": {
            "n_present": n_present,
            "n_valid": n_valid,
            "mean_deg": float(valid_diffs.mean()) if n_valid else None,
            "median_deg": float(np.median(valid_diffs)) if n_valid else None,
            "std_deg": float(valid_diffs.std()) if n_valid else None,
            "p95_deg": float(np.percentile(valid_diffs, 95)) if n_valid else None,
            "p99_deg": float(np.percentile(valid_diffs, 99)) if n_valid else None,
        },
    }

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(valid_diffs, bins=100, alpha=0.75, edgecolor="black")
    if n_valid:
        ax.axvline(valid_diffs.mean(), color="red", linestyle="--", label=f"Mean {valid_diffs.mean():.1f}°")
        ax.axvline(np.median(valid_diffs), color="orange", linestyle="--", label=f"Median {np.median(valid_diffs):.1f}°")
    ax.set_xlabel("|θ_v - θ_Δ| (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Direction inconsistency distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if n_valid:
        sorted_diff = np.sort(valid_diffs)
        cdf = np.arange(1, len(sorted_diff) + 1) / len(sorted_diff)
        ax.plot(sorted_diff, cdf, linewidth=2)
        ax.axvline(40, color="red", linestyle="--", label="40° reference")
    ax.set_xlabel("|θ_v - θ_Δ| (deg)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if n_valid:
        # Subsample to avoid giant scatter
        v = vel_mag_valid[valid_mask]
        d = valid_diffs
        if len(d) > 20000:
            idx = np.random.choice(len(d), 20000, replace=False)
            v = v[idx]
            d = d[idx]
        sc = ax.scatter(v, d, alpha=0.12, s=4, c=d, cmap="viridis", vmin=0, vmax=90)
        plt.colorbar(sc, ax=ax, label="|θ_v - θ_Δ| (deg)")
    ax.set_xlabel("|v| (px/frame)")
    ax.set_ylabel("|θ_v - θ_Δ| (deg)")
    ax.set_title("Angular diff vs speed")
    ax.set_xlim(0, max_vel_plot)
    ax.set_ylim(0, 90)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if n_valid:
        bins = [0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
        box_data = []
        labels = []
        flat_vel = vel_mag[valid_mask]
        flat_diff = valid_diffs
        for vmin, vmax in zip(bins[:-1], bins[1:]):
            m = (flat_vel >= vmin) & (flat_vel < vmax)
            if m.any():
                box_data.append(flat_diff[m])
                labels.append(f"{vmin:.1f}-{vmax:.1f}")
        if box_data:
            ax.boxplot(box_data, labels=labels)
            ax.set_xlabel("Velocity bin (px/frame)")
            ax.set_ylabel("|θ_v - θ_Δ| (deg)")
            ax.set_title("Angular diff by speed")
            ax.grid(alpha=0.3, axis="y")
            plt.setp(ax.get_xticklabels(), rotation=45)

    fig.tight_layout()

    plot_path = out_dir_path / "direction_consistency_analysis.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")

    summary_path = out_dir_path / "direction_consistency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to train_episodes.npz / val_episodes.npz / test_episodes.npz")
    parser.add_argument("--out_dir", default="results/direction_sanity_check")
    parser.add_argument("--vel_threshold", type=float, default=0.5)
    parser.add_argument("--disp_threshold", type=float, default=0.1)
    parser.add_argument("--max_vel_plot", type=float, default=20.0)
    args = parser.parse_args()

    analyze_direction_consistency(
        data_path=args.data,
        out_dir=args.out_dir,
        vel_threshold=args.vel_threshold,
        disp_threshold=args.disp_threshold,
        max_vel_plot=args.max_vel_plot,
    )


if __name__ == "__main__":
    main()
