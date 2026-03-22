from .logger import Logger, MetricsTracker
from .replay_buffer import EpisodeReplayBuffer

# Device selection utility
try:
    import torch
except ImportError:
    torch = None


def get_device(device=None):
    """
    Get the best available device (CUDA if available, otherwise CPU).
    Forces CPU over MPS on Mac for RL due to synchronization overhead.

    Args:
        device: Optional device string ("cuda", "cpu", "mps", or None for auto-detect)

    Returns:
        Device string
    """
    if device is not None:
        if device.lower() == "cuda" and torch is not None and torch.cuda.is_available():
            return "cuda"
        elif device.lower() == "cpu":
            return "cpu"
        elif device.lower() == "mps" and torch is not None and torch.backends.mps.is_available():
            return "mps"
        elif device.lower() == "cuda" and (torch is None or not torch.cuda.is_available()):
            print(
                "Warning: CUDA requested but not available. Falling back to CPU (MPS avoided for RL speed)."
            )
            return "cpu"
        else:
            return device.lower()

    # Auto-detect (prefer CUDA, fallback to CPU. Avoid MPS due to sync overhead)
    if torch is not None and torch.cuda.is_available():
        return "cuda"

    return "cpu"


def setup_experiment(base_dir: str = "results", exp_name: str = None):
    """Setup experiment directory."""
    import os
    from datetime import datetime

    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    return exp_dir


__all__ = ["EpisodeReplayBuffer", "Logger", "MetricsTracker", "get_device", "setup_experiment"]
