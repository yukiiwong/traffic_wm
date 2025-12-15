"""
Common Utility Functions
"""

import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path
import json
from typing import Dict, Tuple


def parse_discrete_feature_indices_from_metadata(
        metadata: Dict,
        fallback: Tuple[int, int, int] = (8, 7, 11),  # (lane, class, site)
        strict: bool = False,
) -> Tuple[int, int, int]:
        """
        Parse discrete feature indices from metadata.json.

        Expected structure:
            metadata["validation_info"]["discrete_features"] = {
                 "lane_id": <int>, "class_id": <int>, "site_id": <int>
            }

        Returns:
            (lane_feature_idx, class_feature_idx, site_feature_idx)

        strict=True: if keys missing -> raise KeyError
        """
        vi = (metadata or {}).get("validation_info", {})
        df = (vi or {}).get("discrete_features", {})

        if strict and (not isinstance(df, dict) or any(k not in df for k in ("lane_id", "class_id", "site_id"))):
                raise KeyError("metadata.validation_info.discrete_features missing lane_id/class_id/site_id")

        lane_idx = int(df.get("lane_id", fallback[0]))
        class_idx = int(df.get("class_id", fallback[1]))
        site_idx = int(df.get("site_id", fallback[2]))

        return lane_idx, class_idx, site_idx


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get PyTorch device.

    Args:
        device_id: Specific GPU ID to use (None = auto)

    Returns:
        torch.device
    """
    if device_id is not None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
        else:
            print(f"CUDA not available, using CPU instead")
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    return device


def save_config(config: dict, save_path: str):
    """
    Save configuration dictionary to JSON file.

    Args:
        config: Configuration dictionary
        save_path: Path to save JSON file
    """
    import json
    from pathlib import Path

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Configuration saved to {save_path}")


def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Configuration dictionary
    """
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Configuration loaded from {config_path}")
    return config


def format_time(seconds: float) -> str:
    """
    Format time in seconds to readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the L2 norm of gradients.

    Args:
        model: PyTorch model

    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    return total_norm


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (minimize or maximize metric)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.

        Args:
            metric_value: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = metric_value
            return False

        if self.mode == 'min':
            improved = metric_value < (self.best_value - self.min_delta)
        else:
            improved = metric_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def get_pixel_to_meter_conversion(
    lane_geometry_path: Optional[str] = None,
    default_value: float = 0.07696103842104474
) -> float:
    """
    Get pixel to meter conversion factor.

    Args:
        lane_geometry_path: Path to lane_geometry_summary.json file.
                          If None, searches in default locations.
        default_value: Default conversion factor if file not found

    Returns:
        Pixel to meter conversion factor
    """
    if lane_geometry_path is None:
        # Try default locations
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "lane_extraction" / "01_data" / "lane_geometry_summary.json",
            Path("../lane_extraction/01_data/lane_geometry_summary.json"),
            Path("../../lane_extraction/01_data/lane_geometry_summary.json"),
        ]

        for path in possible_paths:
            if path.exists():
                lane_geometry_path = str(path)
                break

    if lane_geometry_path and Path(lane_geometry_path).exists():
        try:
            with open(lane_geometry_path, 'r') as f:
                data = json.load(f)
                pixel_to_meter = data.get('pixel_to_meter', default_value)
                print(f"Loaded pixel_to_meter = {pixel_to_meter:.6f} from {lane_geometry_path}")
                return float(pixel_to_meter)
        except Exception as e:
            print(f"Warning: Could not load conversion factor from {lane_geometry_path}: {e}")

    print(f"Using default pixel_to_meter = {default_value:.6f}")
    return default_value


def convert_pixels_to_meters(
    states: torch.Tensor,
    pixel_to_meter: float,
    position_indices: tuple = (0, 1),
    velocity_indices: tuple = (2, 3),
    acceleration_indices: tuple = (4, 5)
) -> torch.Tensor:
    """
    Convert pixel-based coordinates to meters.

    Args:
        states: Tensor of shape [..., F] where F is feature dimension
        pixel_to_meter: Conversion factor (meters per pixel)
        position_indices: Indices for position features (x, y)
        velocity_indices: Indices for velocity features (vx, vy)
        acceleration_indices: Indices for acceleration features (ax, ay)

    Returns:
        Converted states tensor (same shape as input)
    """
    states = states.clone()

    # Convert positions: pixels -> meters
    if position_indices is not None:
        for idx in position_indices:
            states[..., idx] *= pixel_to_meter

    # Convert velocities: pixels/frame -> meters/second
    # Note: velocities are already computed with dt, so just scale by pixel_to_meter
    if velocity_indices is not None:
        for idx in velocity_indices:
            states[..., idx] *= pixel_to_meter

    # Convert accelerations: pixels/frame^2 -> meters/second^2
    if acceleration_indices is not None:
        for idx in acceleration_indices:
            states[..., idx] *= pixel_to_meter

    return states


if __name__ == '__main__':
    # Test utilities
    set_seed(42)

    # Test parameter counting
    model = nn.Linear(10, 5)
    print(f"Model parameters: {count_parameters(model)}")

    # Test device
    device = get_device()

    # Test time formatting
    print(format_time(3661))  # 1h 1m 1s

    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode='min')

    for loss in [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84]:
        should_stop = early_stop(loss)
        print(f"Loss: {loss:.2f}, Counter: {early_stop.counter}, Stop: {should_stop}")
