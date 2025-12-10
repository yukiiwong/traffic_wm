"""
Data processing and loading utilities.
"""

from .preprocess import preprocess_trajectories

# Lazy import for torch-dependent modules
def __getattr__(name):
    if name == "TrajectoryDataset":
        from .dataset import TrajectoryDataset
        return TrajectoryDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["preprocess_trajectories", "TrajectoryDataset"]
