import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python random, NumPy legacy global RNG, and PyTorch (CPU + MPS).

    Note: np.random.default_rng() instances are NOT affected - callers that
    use new-style NumPy generators must seed them explicitly, e.g.,
    np.random.default_rng(seed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
