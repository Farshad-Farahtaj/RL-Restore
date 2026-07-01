import numpy as np
import pytest

from rlrestore.common.metrics import psnr


def test_psnr_identical_images_is_huge():
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    assert psnr(img, img) > 90.0  # eps-bounded, not inf


def test_psnr_known_value():
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.full((4, 4, 3), 0.1, dtype=np.float32)
    # float32(0.1) -> MSE ~ 0.01 -> PSNR ~ 20 dB
    assert psnr(a, b) == pytest.approx(20.0, abs=1e-3)


def test_psnr_symmetric(clean_img, rng):
    noisy = np.clip(clean_img + 0.05 * rng.standard_normal(clean_img.shape), 0, 1).astype(
        np.float32
    )
    assert psnr(clean_img, noisy) == pytest.approx(psnr(noisy, clean_img))


def test_psnr_accepts_out_of_range_inputs():
    a = np.full((4, 4, 3), 0.0, dtype=np.float32)
    b = np.full((4, 4, 3), 1.1, dtype=np.float32)  # exceeds [0,1] like raw tool outputs
    result = psnr(a, b)
    assert isinstance(result, float)
    assert result < 20.0  # MSE=1.21 -> PSNR ~ -0.83 dB
