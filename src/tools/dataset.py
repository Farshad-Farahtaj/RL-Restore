"""(degraded, clean) pair generation for one tool's degradation band.

robust_extras implements the paper's Sec 3.1 trick ("add slight Gaussian
noises and JPEG compression to all the training data", worth +0.2 dB).
Magnitudes are NOT stated in the paper - our documented assumption:
extra noise sigma ~ U[0,5], then JPEG Q ~ U[80,100], applied after the
band degradation. Recorded in docs/paper_config.md; ablatable via config.

Images may be float32 RGB [0,1] or uint8 RGB (preferred: 750 DIV2K images
are ~4-6 GB as uint8 vs 17+ GB as float32 on a 16 GB machine). uint8 crops
convert to float32/255 after cropping - identical to the original's
uint8-PNG -> im2double data path.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from rlrestore.data.degradations import gaussian_blur, gaussian_noise, jpeg_compress
from rlrestore.data.pipeline import SCALES, random_patch, rescale
from rlrestore.tools.specs import ToolSpec

ROBUST_NOISE_RANGE = (0.0, 5.0)
ROBUST_JPEG_RANGE = (80.0, 100.0)


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))


def build_pyramid(images) -> list[list[np.ndarray]]:
    """Pre-rescale every image to all SCALES once. Build BEFORE forking workers
    so Linux shares it copy-on-write; rebuild-per-epoch would be ruinous."""
    return [[rescale(im, s) for s in SCALES] for im in images]


def degrade_for_tool(
    img: np.ndarray, spec: ToolSpec, rng: np.random.Generator, robust_extras: bool
) -> np.ndarray:
    level = float(rng.uniform(spec.lo, spec.hi))
    if spec.kind == "blur":
        out = gaussian_blur(img, level)
    elif spec.kind == "noise":
        out = gaussian_noise(img, level, rng)
    else:
        out = jpeg_compress(img, level)
    if robust_extras:
        out = gaussian_noise(out, float(rng.uniform(*ROBUST_NOISE_RANGE)), rng)
        out = jpeg_compress(out, float(rng.uniform(*ROBUST_JPEG_RANGE)))
    return out


class ToolPairDataset(Dataset):
    """Index-deterministic: item i is fully determined by (seed, i).

    cache_pyramid (or env RLR_CACHE_PYRAMID=1): pre-rescale every image to the
    three pyramid scales once at construction. Costs ~1.55x the image RAM
    (~9 GB for 750 uint8 DIV2K images) but removes the per-item full-image
    rescale that starves fast GPUs. Identical outputs either way (rescale is
    deterministic); keep OFF on 16 GB machines.
    """

    def __init__(self, images, spec: ToolSpec, seed: int, length: int, robust_extras: bool,
                 cache_pyramid: bool | None = None, pyramid: list | None = None):
        self.images = images
        self.spec = spec
        self.seed = seed
        self.length = length
        self.robust_extras = robust_extras
        if pyramid is not None:
            self._pyramid = pyramid  # prebuilt and shared across epochs/datasets
        else:
            if cache_pyramid is None:
                cache_pyramid = os.environ.get("RLR_CACHE_PYRAMID", "0") == "1"
            self._pyramid = build_pyramid(images) if cache_pyramid else None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.spec.index, i))
        img_idx = int(rng.integers(len(self.images)))
        scale_idx = int(rng.integers(len(SCALES)))
        if self._pyramid is not None:
            scaled = self._pyramid[img_idx][scale_idx]
        else:
            scaled = rescale(self.images[img_idx], SCALES[scale_idx])
        patch = random_patch(scaled, rng)
        if patch.dtype == np.uint8:
            clean = patch.astype(np.float32) / 255.0
        else:
            clean = np.clip(patch, 0.0, 1.0).astype(np.float32)
        degraded = degrade_for_tool(clean, self.spec, rng, self.robust_extras)
        return _to_tensor(degraded), _to_tensor(clean)
