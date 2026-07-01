import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clean_img(rng):
    """Synthetic 96x96 RGB float32 [0,1] image with structure (not pure noise)."""
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    img = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    img += 0.1 * rng.standard_normal((96, 96, 3)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)
