import numpy as np
import pytest

from rlrestore.common.metrics import psnr
from rlrestore.data.pipeline import (
    PATCH_SIZE,
    SCALES,
    apply_recipe,
    make_training_pair,
    random_patch,
)
from rlrestore.data.recipes import RecipeParams


def test_apply_recipe_order_and_effect(clean_img, rng):
    params = RecipeParams((5, 5, 5), 2.0, 20.0, 45.0)
    out = apply_recipe(clean_img, params, rng)
    assert out.shape == clean_img.shape and out.dtype == np.float32
    assert psnr(clean_img, out) < 35.0  # visibly damaged


def test_apply_recipe_mild_levels_are_near_identity(rng):
    # Smooth content: the noisy clean_img fixture is an unfair probe because
    # JPEG chroma subsampling destroys per-pixel chroma noise even at Q=100
    # (see test_degradations.py for the same reasoning).
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    smooth = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    params = RecipeParams((1, 1, 1), 0.0, 0.0, 100.0)
    out = apply_recipe(smooth, params, rng)
    assert psnr(smooth, out) > 35.0


def test_random_patch_size_and_determinism(clean_img):
    p1 = random_patch(clean_img, np.random.default_rng(5))
    p2 = random_patch(clean_img, np.random.default_rng(5))
    assert p1.shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert np.array_equal(p1, p2)


def test_scales_match_original():
    assert SCALES == pytest.approx([1.0, 2 / 3, 1 / 3])


def test_make_training_pair_skips_too_clean(clean_img):
    rng = np.random.default_rng(11)
    clean, degraded = make_training_pair(clean_img, "moderate", rng)
    assert clean.shape == degraded.shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert psnr(degraded, clean) <= 50.0


def test_make_training_pair_rejects_zero_tries(clean_img):
    with pytest.raises(ValueError):
        make_training_pair(clean_img, "moderate", np.random.default_rng(0), max_tries=0)


def test_make_training_pair_resamples_when_too_clean(clean_img, monkeypatch):
    import rlrestore.data.pipeline as pipe

    calls = {"n": 0}
    real_apply = pipe.apply_recipe

    def near_identity_twice(img, params, rng):
        calls["n"] += 1
        if calls["n"] < 3:
            return img.copy()  # PSNR ~inf -> must be skipped
        return real_apply(img, params, rng)

    monkeypatch.setattr(pipe, "apply_recipe", near_identity_twice)
    clean, degraded = pipe.make_training_pair(clean_img, "moderate", np.random.default_rng(0))
    assert calls["n"] >= 3, "skip-and-resample loop should have retried"


def test_make_training_pair_uint8_not_saturated():
    """REGRESSION (2026-06-12): uint8 images must be normalized, not clipped.

    The original float-only assumption clipped 0-255 uint8 values to [0,1],
    saturating every clean patch to ~pure white — the Phase 2 agent's first
    2M-step training run learned on that degenerate world. A natural-content
    uint8 image must yield patches matching its float/255 twin, never white.
    """
    rng_img = np.random.default_rng(7)
    # Textured synthetic image: smooth gradient + noise, mid-range values
    h, w = 400, 400
    gradient = np.linspace(40, 200, w, dtype=np.float32)[None, :, None]
    texture = rng_img.normal(0, 25, size=(h, w, 3)).astype(np.float32)
    img_u8 = np.clip(gradient + texture + 30, 0, 255).astype(np.uint8)

    clean_u8, degraded_u8 = make_training_pair(img_u8, "moderate", np.random.default_rng(3))
    assert clean_u8.dtype == np.float32
    assert clean_u8.mean() < 0.9, (
        f"uint8 patch saturated to white (mean={clean_u8.mean():.4f}) — "
        "dtype normalization regressed"
    )
    assert clean_u8.std() > 0.02, "uint8 patch lost all texture"

    # Same seed, float/255 twin -> near-identical pair (interp rounding only)
    img_f = img_u8.astype(np.float32) / 255.0
    clean_f, degraded_f = make_training_pair(img_f, "moderate", np.random.default_rng(3))
    assert np.allclose(clean_u8, clean_f, atol=2 / 255), "uint8 and float paths diverged"
    assert abs(float(degraded_u8.mean()) - float(degraded_f.mean())) < 0.02
