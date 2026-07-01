import itertools

import numpy as np
import pytest

from rlrestore.data.recipes import (
    BLUR_GRID,
    JPEG_GRID,
    NOISE_GRID,
    SEVERITY_LEVELS,
    RecipeParams,
    enumerate_level_rows,
    sample_recipe,
)


def test_grids_match_original():
    assert list(BLUR_GRID) == pytest.approx([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    assert list(NOISE_GRID) == pytest.approx([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    assert list(JPEG_GRID) == [100, 80, 60, 50, 40, 35, 30, 25, 20, 15, 10]


@pytest.mark.parametrize("severity", ["mild", "moderate", "severe"])
def test_enumerate_matches_bruteforce(severity):
    lo, hi = SEVERITY_LEVELS[severity]
    rows = enumerate_level_rows(lo, hi)
    brute = {
        (k, m, n)
        for k, m, n in itertools.product(range(1, 11), repeat=3)
        if lo <= k + m + n - 2 <= hi
    }
    assert set(rows) == brute


def test_severity_classes_are_disjoint():
    sets = {
        name: set(enumerate_level_rows(lo, hi)) for name, (lo, hi) in SEVERITY_LEVELS.items()
    }
    assert not (sets["mild"] & sets["moderate"])
    assert not (sets["moderate"] & sets["severe"])


def test_sample_recipe_params_lie_in_band():
    rng = np.random.default_rng(7)
    for _ in range(200):
        p = sample_recipe("moderate", rng)
        assert isinstance(p, RecipeParams)
        b, n, j = p.levels
        assert BLUR_GRID[b - 1] <= p.blur_sigma <= BLUR_GRID[b]
        assert NOISE_GRID[n - 1] <= p.noise_sigma <= NOISE_GRID[n]
        assert JPEG_GRID[j] <= p.jpeg_quality <= JPEG_GRID[j - 1]
        assert 12 <= sum(p.levels) - 2 <= 17


def test_sample_recipe_deterministic_with_seed():
    a = sample_recipe("severe", np.random.default_rng(3))
    b = sample_recipe("severe", np.random.default_rng(3))
    assert a == b
