"""Data processing and loading utilities.

Supported preprocessing entry point:
- src.data.preprocess_multisite.preprocess_all_sites

Legacy/low-level preprocessing helpers are kept in src.data.preprocess.
"""

from .preprocess_multisite import preprocess_all_sites


# Lazy import for torch-dependent modules
def __getattr__(name):
    if name == "TrajectoryDataset":
        from .dataset import TrajectoryDataset

        return TrajectoryDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["preprocess_all_sites", "TrajectoryDataset"]
