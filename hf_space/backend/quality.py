"""Image-quality metrics for the demo backend.

Two regimes (webapp_spec.md "PSNR only on the simulator path"):

- Reference present (simulator path): full-reference PSNR via the SAME
  ``rlrestore.common.metrics.psnr`` the models were evaluated with (peak 1.0,
  unclamped). This is the single source of truth — we do NOT reimplement it.
- No reference (uploads): a NumPy/OpenCV no-reference *proxy*. It is NOT NIQE
  and is not calibrated to any external scale; it is a monotone "looks sharper /
  less noisy" score so the climbing-meter UX has something honest to move.
  Labeled ``sharpness_proxy`` so the frontend can mark it "(estimated)".

The proxy combines the two signals the spec names, on the luma channel:

    sharpness = gradient energy (Tenengrad: mean |∇|², a focus measure) computed
                on a lightly pre-smoothed luma so additive noise does NOT inflate
                it. Reported in dB (10·log10). Higher = crisper structure.
    noise     = robust sigma via the median-absolute-deviation of the raw
                Laplacian high-pass response (higher = noisier).

    score = sharpness_db − k · noise_sigma

Deblur raises gradient energy; denoise lowers the noise sigma — so a good chain
climbs. The score is an open-ended dB-like number (NOT sigmoid-squashed, which
saturates and erases the blur ordering). It is monotone for the two degradation
axes the spec targets (blur, noise); JPEG *blocking* synthesises spurious edges
that can fool any gradient-based sharpness measure, so the proxy is least
reliable for compression artefacts — hence the "(estimated)" label and the
always-shown visual before/after.
"""

from __future__ import annotations

import cv2
import numpy as np

from rlrestore.common.metrics import psnr as _psnr

__all__ = ["psnr", "no_ref_quality", "NO_REF_LABEL", "REF_LABEL"]

REF_LABEL = "psnr"
NO_REF_LABEL = "sharpness_proxy"

# Weight on the noise sigma (on the [0,1] intensity scale) relative to the dB
# sharpness term. Tuned so a moderate noise sigma moves the score by a few dB,
# comparable to the swing blur produces.
_NOISE_WEIGHT = 120.0
# MAD -> Gaussian-sigma consistency constant (1 / Phi^-1(0.75)).
_MAD_TO_SIGMA = 1.4826
# Pre-smoothing for the sharpness measure (suppresses noise before gradients).
_SHARP_PRESMOOTH_SIGMA = 1.0
# Floor for degenerate (flat) inputs so the score stays finite and bounded.
_FLOOR = -120.0
_EPS = 1e-12


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Full-reference PSNR in dB (peak 1.0, unclamped) — repo metric passthrough."""
    return _psnr(a, b)


def _to_luma(img: np.ndarray) -> np.ndarray:
    """HWC float32 [0,1] RGB -> HxW float32 luma (Rec. 601)."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return (
        0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    ).astype(np.float32)


def _laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour discrete Laplacian with replicate padding."""
    padded = np.pad(gray, 1, mode="edge")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * gray
    ).astype(np.float32)


def no_ref_quality(img: np.ndarray) -> float:
    """Monotone no-reference quality proxy for an HWC float image in [0,1].

    Returns a single finite float. Higher = sharper and/or less noisy. The
    absolute value is meaningless; only its ordering along a restoration chain
    is. Monotone for blur and noise; robust to flat inputs (finite floor).
    """
    gray = _to_luma(img)
    if gray.size == 0:
        return 0.0

    # Sharpness (noise-robust): Tenengrad gradient energy on a pre-smoothed luma.
    smooth = cv2.GaussianBlur(gray, (0, 0), _SHARP_PRESMOOTH_SIGMA)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad_energy = float(np.mean(gx * gx + gy * gy))
    sharpness_db = 10.0 * float(np.log10(grad_energy + _EPS))

    # Noise: robust sigma from the MAD of the raw high-pass response.
    lap = _laplacian(gray)
    med = float(np.median(lap))
    mad = float(np.median(np.abs(lap - med)))
    noise_sigma = _MAD_TO_SIGMA * mad

    score = sharpness_db - _NOISE_WEIGHT * noise_sigma
    if not np.isfinite(score):
        return _FLOOR
    return float(max(_FLOOR, score))
