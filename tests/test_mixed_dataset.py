import numpy as np
import torch

from rlrestore.data import mixed_dataset as md
from rlrestore.data.mixed_dataset import MixedPairDataset


def _images(n=2, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    base = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    return [
        np.clip(base + 0.1 * rng.standard_normal(base.shape), 0, 1).astype(np.float32)
        for _ in range(n)
    ]


def test_getitem_shapes_dtype_and_determinism(clean_img):
    ds = MixedPairDataset(images=[clean_img], seed=3, length=4)
    assert len(ds) == 4
    degraded, clean = ds[0]
    assert degraded.shape == clean.shape == (3, 63, 63)
    assert degraded.dtype == clean.dtype == torch.float32
    d2, c2 = ds[0]
    assert torch.equal(degraded, d2) and torch.equal(clean, c2)  # index-deterministic


def test_severity_varies_across_indices(monkeypatch):
    """Sampling many items must hit more than one severity class."""
    seen: list[str] = []
    real = md.make_training_pair

    def spy(src, severity, rng, *a, **k):
        seen.append(severity)
        return real(src, severity, rng, *a, **k)

    monkeypatch.setattr(md, "make_training_pair", spy)
    ds = MixedPairDataset(images=_images(), seed=11, length=60)
    for i in range(len(ds)):
        ds[i]
    assert set(seen) == {"mild", "moderate", "severe"}  # all three appear


def test_single_severity_tuple_is_honoured(monkeypatch):
    seen: list[str] = []
    real = md.make_training_pair

    def spy(src, severity, rng, *a, **k):
        seen.append(severity)
        return real(src, severity, rng, *a, **k)

    monkeypatch.setattr(md, "make_training_pair", spy)
    ds = MixedPairDataset(images=_images(), seed=1, length=10, severities=("severe",))
    for i in range(len(ds)):
        ds[i]
    assert set(seen) == {"severe"}


def test_uint8_input_clean_patch_not_saturated(clean_img):
    """Guards the 2026-06-12 white-patch regression: uint8 input must be /255'd,
    not clipped, so the clean patch is NOT saturated to ~white."""
    u8 = (clean_img * 255 + 0.5).astype(np.uint8)
    ds = MixedPairDataset(images=[u8], seed=3, length=4)
    for i in range(4):
        _, clean = ds[i]
        assert float(clean.mean()) < 0.98  # not a white patch
        assert 0.0 <= float(clean.min()) and float(clean.max()) <= 1.0


def test_uint8_matches_float_equivalent(clean_img):
    u8 = (clean_img * 255 + 0.5).astype(np.uint8)
    as_float = u8.astype(np.float32) / 255.0
    ds_f = MixedPairDataset(images=[as_float], seed=5, length=2)
    ds_u = MixedPairDataset(images=[u8], seed=5, length=2)
    d_f, c_f = ds_f[0]
    d_u, c_u = ds_u[0]
    assert c_u.dtype == torch.float32
    # Clean is the regression-critical comparison: uint8 /255 must match the
    # pre-divided float to within integer-rounding (NOT clipped to white).
    assert torch.allclose(c_f, c_u, atol=5e-3)  # uint8 rescale rounds to integers
    # Degraded: the full blur->noise->jpeg recipe is identical, but JPEG
    # quantization is a step function, so a clean pixel within rounding distance
    # of a quantization boundary can flip a whole level (max diff ~0.07). The
    # meaningful equivalence is therefore in aggregate, not pixel-exact.
    assert float((d_f - d_u).abs().mean()) < 5e-3


def test_pyramid_cache_is_equivalent(clean_img):
    plain = MixedPairDataset(images=[clean_img], seed=3, length=3, cache_pyramid=False)
    cached = MixedPairDataset(images=[clean_img], seed=3, length=3, cache_pyramid=True)
    for i in range(3):
        dp, cp = plain[i]
        dc, cc = cached[i]
        assert torch.equal(dp, dc) and torch.equal(cp, cc)
