import numpy as np

_EPS = 1e-10  # matches the original implementation's epsilon


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two float RGB images in [0,1] (peak = 1.0).

    Computed over all channels, matching the original RL-Restore code.
    Inputs are not clipped: tool outputs may exceed [0,1] by design.
    """
    diff = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(diff * diff))
    return float(10.0 * np.log10(1.0 / (mse + _EPS)))
