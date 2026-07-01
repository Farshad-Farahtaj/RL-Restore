"""Tests for severity threading (P3.E1) and the TEST evaluation harness.

Test index
----------
(a) RestoreEnv severity parameter: mild/severe accepted, invalid rejected,
    and severity actually changes the degradation for a fixed seed.
(b) compute_baselines severity passthrough: runs for severity="mild" with
    fake tools and returns all B-keys.
(c) evaluate_on_class smoke: tiny inputs (2 episodes, fake tools, untrained
    AgentNet, 1 strip) produce a JSON-able dict with agent/baselines keys
    and the strip PNG on disk.

All tools are fakes — no real checkpoints required; everything runs on CPU
in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.evaluate_agent import evaluate_on_class
from torch import nn

from rlrestore.agent.env import RestoreEnv
from rlrestore.agent.net import AgentNet
from rlrestore.agent.train import compute_baselines

# ──────────────────────────── fakes and helpers ──────────────────────────────

_DEVICE = torch.device("cpu")
_N_TOOLS = 12


class _Identity(nn.Module):
    """Leaves the image unchanged — neutral PSNR effect."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class _Smooth(nn.Module):
    """3x3 box smoothing — same fake-tool pattern as tests/test_agent_runner.py."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 9.0)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _fake_tools() -> list[nn.Module]:
    return [_Smooth() for _ in range(_N_TOOLS)]


def _make_images(n: int = 4, seed: int = 0) -> list[np.ndarray]:
    """n synthetic 96x96 RGB float32 [0, 1] images with structure."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    base = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    return [
        np.clip(base + 0.1 * rng.standard_normal(base.shape), 0, 1).astype(np.float32)
        for _ in range(n)
    ]


def _make_env(severity: str, seed: int = 7) -> RestoreEnv:
    return RestoreEnv(
        images=_make_images(),
        tools=[_Identity() for _ in range(_N_TOOLS)],
        device=_DEVICE,
        seed=seed,
        training_mode=False,
        severity=severity,
    )


# ══════════════════════════ (a) RestoreEnv severity ═══════════════════════════


@pytest.mark.parametrize("severity", ["mild", "moderate", "severe"])
def test_a_env_accepts_valid_severities(severity: str) -> None:
    env = _make_env(severity)
    assert env.severity == severity
    img, oh = env.reset()
    assert img.shape == (63, 63, 3) and oh.shape == (12,)


def test_a_env_rejects_invalid_severity() -> None:
    with pytest.raises(AssertionError, match="severity"):
        _make_env("extreme")


def test_a_env_default_severity_is_moderate() -> None:
    env = RestoreEnv(
        images=_make_images(),
        tools=[_Identity() for _ in range(_N_TOOLS)],
        device=_DEVICE,
        seed=7,
    )
    assert env.severity == "moderate"


def test_a_severity_changes_degradation_for_same_seed() -> None:
    """Same seed, different severity -> same clean patch, different degraded."""
    env_mild = _make_env("mild", seed=123)
    env_severe = _make_env("severe", seed=123)
    img_mild, _ = env_mild.reset()
    img_severe, _ = env_severe.reset()

    # Scale/crop draws precede recipe sampling, so the clean patch matches...
    np.testing.assert_array_equal(env_mild.clean_image, env_severe.clean_image)
    # ...but the recipe (and hence the degraded image) must differ.
    assert not np.array_equal(img_mild, img_severe)


# ══════════════════════════ (b) baselines passthrough ═════════════════════════


def test_b_compute_baselines_severity_mild() -> None:
    result = compute_baselines(
        tools=_fake_tools(),
        images=_make_images(),
        device=_DEVICE,
        seeds=[0, 1, 2, 3],
        n=4,
        n_random_chains=3,
        severity="mild",
    )
    expected_keys = {
        "B0_do_nothing",
        "B1_best_single_tool",
        "B1_tool_gains",
        "B2_random_chain_mean",
        "B2_random_chain_std",
        "B3_greedy_oracle",
        "B4_oracle_with_stop",
        "n_episodes",
        "n_random_chains",
    }
    assert expected_keys <= set(result.keys())
    assert result["n_episodes"] == 4
    assert len(result["B1_tool_gains"]) == _N_TOOLS


# ══════════════════════════ (c) evaluate_on_class smoke ═══════════════════════


def test_c_evaluate_on_class_smoke(tmp_path: Path) -> None:
    """Tiny end-to-end run: JSON-able result with agent/baselines + strip PNG."""
    torch.manual_seed(0)
    net = AgentNet().to(_DEVICE).eval()  # untrained weights are fine for a smoke
    seeds = [10_000, 10_001]

    result = evaluate_on_class(
        net=net,
        tools=_fake_tools(),
        images=_make_images(),
        device=_DEVICE,
        severity="mild",
        seeds=seeds,
        n_random_chains=2,
        n_strips=1,
        strips_dir=tmp_path,
    )

    assert set(result.keys()) == {"agent", "baselines"}
    agent = result["agent"]
    for key in (
        "psnr_gain_db",
        "action_hist",
        "entropy",
        "stop_rate_t1",
        "stop_rate_t2",
        "stop_rate_t3",
        "mean_chain_length",
        "top_tools",
    ):
        assert key in agent, f"agent missing key: {key}"
    assert agent["n_episodes"] == 2
    assert 0.0 <= agent["mean_chain_length"] <= 3.0
    assert len(agent["top_tools"]) == 3
    assert "B3_greedy_oracle" in result["baselines"]

    # Must be JSON-serialisable as-is (the script dumps it verbatim)
    json.dumps(result)

    # Strip PNG for the first seed exists
    strip = tmp_path / f"mild_{seeds[0]}.png"
    assert strip.exists() and strip.stat().st_size > 0
