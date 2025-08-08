import numpy as np
import mlx.core as mx
import time
from typing import Optional, Any
import random

# Make PyTorch import optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: Optional[int]) -> None:
    """
    Set the seed for all random number generators used in mlx-engine to ensure reproducible results.
    This function synchronizes the random states across multiple libraries including MLX, NumPy,
    and Python's built-in random module. PyTorch is also supported if available.

    Args:
        seed: The seed value to initialize random number generators. If None, a seed will be
            automatically generated using the current nanosecond timestamp. The final seed
            value will be truncated to 32 bits for compatibility across all random number
            generators.

    Raises:
        ValueError: If the provided seed is negative.

    Returns:
        None

    Note:
        This function affects the following random number generators:
        - MLX (mx.random)
        - NumPy (np.random)
        - PyTorch (torch.manual_seed)
        - Python's built-in random module
    """
    if seed is None:
        # Get nanosecond timestamp and use it as seed
        seed = int(time.time_ns())

    if seed < 0:
        raise ValueError("Seed must be a non-negative integer.")
    seed_int = seed & (2**32 - 1)  # Ensure seed fits in 32 bits

    # For MLX and MLX_LM
    mx.random.seed(seed_int)

    # Set PyTorch seed if available
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_int)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            torch.manual_seed(seed_int)
        except Exception as e:
            # Silently continue if there's an issue with PyTorch
            pass

    # MLX_VLM depends on numpy 
    np.random.seed(seed_int)

    # Just in case
    random.seed(seed_int)
