import numpy as np
import pytest

from rlrestore.common.metrics import psnr
from rlrestore.data.degradations import gaussian_blur, gaussian_noise, jpeg_compress


def _check_contract(out, ref):
    assert out.shape == ref.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_blur_zero_sigma_is_identity(clean_img):
    out = gaussian_blur(clean_img, 0.0)
    assert np.allclose(out, clean_img)


def test_blur_reduces_psnr_more_with_higher_sigma(clean_img):
    p1 = psnr(clean_img, gaussian_blur(clean_img, 1.0))
    p4 = psnr(clean_img, gaussian_blur(clean_img, 4.0))
    _check_contract(gaussian_blur(clean_img, 4.0), clean_img)
    assert p4 < p1 < 100


def test_noise_statistics(rng):
    flat = np.full((256, 256, 3), 0.5, dtype=np.float32)
    out = gaussian_noise(flat, 25.0, rng)
    _check_contract(out, flat)
    residual = out - flat
    assert abs(float(residual.mean())) < 1e-3
    assert float(residual.std()) == pytest.approx(25.0 / 255.0, rel=0.05)


def test_noise_zero_sigma_is_identity(clean_img, rng):
    assert np.allclose(gaussian_noise(clean_img, 0.0, rng), clean_img)


def test_jpeg_lower_quality_hurts_more(clean_img):
    p90 = psnr(clean_img, jpeg_compress(clean_img, 90))
    p15 = psnr(clean_img, jpeg_compress(clean_img, 15))
    _check_contract(jpeg_compress(clean_img, 15), clean_img)
    assert p15 < p90


def test_jpeg_quality_100_is_near_lossless_on_smooth_content():
    # On smooth content Q=100 is near-lossless; the noisy fixture isn't a fair
    # probe because JPEG chroma subsampling legitimately destroys per-pixel
    # chroma noise even at Q=100.
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    smooth = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    assert psnr(smooth, jpeg_compress(smooth, 100)) > 35.0
