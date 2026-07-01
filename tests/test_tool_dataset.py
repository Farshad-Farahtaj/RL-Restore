import numpy as np
import torch

from rlrestore.common.metrics import psnr
from rlrestore.tools.dataset import ToolPairDataset, degrade_for_tool
from rlrestore.tools.specs import TOOL_SPECS


def test_degrade_for_tool_blur_band(clean_img):
    rng = np.random.default_rng(0)
    spec = TOOL_SPECS[1]  # blur [1.25, 2.5]
    out = degrade_for_tool(clean_img, spec, rng, robust_extras=False)
    assert out.shape == clean_img.shape
    assert psnr(clean_img, out) < 45.0


def test_robust_extras_add_more_damage(clean_img):
    spec = TOOL_SPECS[0]  # blur [0, 1.25] (mild band)
    plain = degrade_for_tool(clean_img, spec, np.random.default_rng(1), robust_extras=False)
    robust = degrade_for_tool(clean_img, spec, np.random.default_rng(1), robust_extras=True)
    assert psnr(clean_img, robust) <= psnr(clean_img, plain) + 1e-6


def test_dataset_yields_tensor_pairs(clean_img):
    ds = ToolPairDataset(
        images=[clean_img], spec=TOOL_SPECS[4], seed=3, length=4, robust_extras=True
    )
    assert len(ds) == 4
    degraded, clean = ds[0]
    assert degraded.shape == clean.shape == (3, 63, 63)
    assert degraded.dtype == clean.dtype == torch.float32
    d2, c2 = ds[0]
    assert torch.equal(degraded, d2) and torch.equal(clean, c2)  # index-deterministic


def test_dataset_accepts_uint8_images(clean_img):
    u8 = (clean_img * 255 + 0.5).astype(np.uint8)
    as_float = u8.astype(np.float32) / 255.0
    ds_f = ToolPairDataset(images=[as_float], spec=TOOL_SPECS[4], seed=3, length=2,
                           robust_extras=False)
    ds_u = ToolPairDataset(images=[u8], spec=TOOL_SPECS[4], seed=3, length=2,
                           robust_extras=False)
    d_f, c_f = ds_f[0]
    d_u, c_u = ds_u[0]
    assert c_u.dtype == torch.float32
    assert 0.0 <= float(c_u.min()) and float(c_u.max()) <= 1.0
    assert torch.allclose(c_f, c_u, atol=5e-3)   # uint8 rescale rounds to integers
    assert torch.allclose(d_f, d_u, atol=2e-2)


def test_pyramid_cache_is_exactly_equivalent(clean_img):
    plain = ToolPairDataset(images=[clean_img], spec=TOOL_SPECS[4], seed=3, length=3,
                            robust_extras=True, cache_pyramid=False)
    cached = ToolPairDataset(images=[clean_img], spec=TOOL_SPECS[4], seed=3, length=3,
                             robust_extras=True, cache_pyramid=True)
    for i in range(3):
        dp, cp = plain[i]
        dc, cc = cached[i]
        assert torch.equal(dp, dc) and torch.equal(cp, cc)
