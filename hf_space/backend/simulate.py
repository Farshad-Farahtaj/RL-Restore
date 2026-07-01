"""Damage simulator: degrade a FULL clean image with a sampled recipe.

webapp_spec "/api/simulate": use ``sample_recipe(severity, rng)`` + ``apply_recipe``
on the FULL resized image (NOT ``make_training_pair``'s 63x63 crop). This gives
the demo a clean reference, so the restoration that follows can report true PSNR.

The recipe RNG is seeded so a (severity, seed) pair is reproducible — the noise
realisation and the band-within-level draws are deterministic.
"""

from __future__ import annotations

import numpy as np

from rlrestore.data.pipeline import apply_recipe
from rlrestore.data.recipes import (
    BLUR_GRID,
    JPEG_GRID,
    NOISE_GRID,
    RecipeParams,
    SEVERITY_LEVELS,
)

from .imaging import encode_dataurl
from .quality import psnr as ref_psnr

__all__ = ["simulate", "demo_recipe", "SEVERITIES"]

SEVERITIES = ["mild", "moderate", "severe"]

# Demo damage emphasises what the toolchain actually reverses well: heavy sensor
# noise and JPEG blocking (the realistic "low-light phone photo"), with only
# slight blur. High-level Gaussian blur is near-irreversible, so a blur-dominant
# draw would denoise to a still-blurry result and read as "nothing happened" —
# this keeps the simulator's before/after legible. Each triple stays inside its
# severity band (sum(levels) - 2 in the SEVERITY_LEVELS range); the scientific
# evaluation in reports/ uses the full sample_recipe distribution, not this.
_DEMO_LEVELS = {
    "mild": [(1, 6, 4), (1, 7, 3)],
    "moderate": [(1, 9, 4), (2, 8, 5), (1, 8, 6)],
    "severe": [(2, 10, 8), (1, 10, 9), (2, 10, 9)],
}


def demo_recipe(severity: str, rng) -> RecipeParams:
    """A noise/JPEG-dominant, low-blur recipe for the demo simulator."""
    choices = _DEMO_LEVELS[severity]
    b, n, j = choices[int(rng.integers(len(choices)))]
    return RecipeParams(
        (b, n, j),
        float(rng.uniform(BLUR_GRID[b - 1], BLUR_GRID[b])),
        float(rng.uniform(NOISE_GRID[n - 1], NOISE_GRID[n])),
        float(rng.uniform(JPEG_GRID[j], JPEG_GRID[j - 1])),
    )


def simulate(
    clean_img: np.ndarray,
    severity: str,
    seed: int,
    fmt: str = "png",
) -> dict:
    """Degrade ``clean_img`` (HWC float32 [0,1], already resized) at ``severity``.

    Returns a dict with clean/degraded data URLs, the recipe params, and the
    reference PSNR of degraded-vs-clean. ``degraded_img`` is included (not in the
    JSON) so the caller can chain a restoration with the clean reference.
    """
    if severity not in SEVERITY_LEVELS:
        raise ValueError(
            f"severity must be one of {sorted(SEVERITY_LEVELS)}, got {severity!r}"
        )

    rng = np.random.default_rng(seed)
    clean = np.ascontiguousarray(clean_img, dtype=np.float32)

    params = demo_recipe(severity, rng)
    degraded = apply_recipe(clean, params, rng)

    psnr_deg = ref_psnr(degraded, clean)

    return {
        "severity": severity,
        "seed": int(seed),
        "recipe": {
            "blur_sigma": float(params.blur_sigma),
            "noise_sigma": float(params.noise_sigma),
            "jpeg_quality": float(params.jpeg_quality),
            "levels": list(params.levels),
        },
        "clean": encode_dataurl(clean, clamp=True, fmt=fmt),
        "degraded": encode_dataurl(degraded, clamp=True, fmt=fmt),
        "psnr_degraded_vs_clean": float(psnr_deg),
        # Not serialised directly — used by the route to optionally restore.
        "_clean_img": clean,
        "_degraded_img": degraded,
    }
