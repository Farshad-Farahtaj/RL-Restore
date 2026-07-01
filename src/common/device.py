import os

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Resolution order: RLR_DEVICE env var > explicit arg > mps if available > cpu."""
    name = os.environ.get("RLR_DEVICE") or preferred
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
