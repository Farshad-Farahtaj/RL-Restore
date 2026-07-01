"""Tests for the tool marginal-value analysis (P3.E2).

Test index
----------
(a) greedy_oracle_gain on a fake set where exactly one tool helps: removing the
    helpful tool strictly lowers the gain (drop > 0); removing a no-op tool
    leaves it unchanged (drop ~ 0). (Greedy is path-dependent and NOT monotone
    in the toolset in general — so we do NOT assert universal monotonicity.)
(b) leave-one-out shape: 12 entries, all required keys, every drop finite and
    signed; full ceiling equals compute_baselines' B3 on the same pairs.
(c) usage parsing: a synthetic action_hist normalises so the 12 tool fractions
    plus the STOP fraction sum to 1, and STOP is excluded from the tool share.
(d) specialist-diagonal parsing on the real matrix returns 12 floats.
(e) smoke: the importable core on 2 fake tools-as-12 + 2 episodes/class writes
    tool_value.json with per_tool + protocol keys into tmp_path.

All tools are fakes — no real checkpoints required; everything runs on CPU in
seconds. Fake-tool pattern mirrors tests/test_agent_runner.py / test_evaluate_agent.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scripts.analyze_tool_value import (
    analyze_tool_value,
    build_pairs,
    greedy_oracle_gain,
    leave_one_out_drops,
    load_agent_usage,
    parse_specialist_diagonal,
    usage_from_hist,
    verdict_for,
)
from torch import nn

from rlrestore.agent.train import compute_baselines

_DEVICE = torch.device("cpu")
_N_TOOLS = 12


# ──────────────────────────── fakes and helpers ──────────────────────────────


class _Identity(nn.Module):
    """Leaves the image unchanged — neutral PSNR effect (a no-op tool)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class _Smooth(nn.Module):
    """3x3 box smoothing — genuinely denoises noisy patches (a helpful tool)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 9.0)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _make_images(n: int = 4, seed: int = 0) -> list[np.ndarray]:
    """n synthetic 96x96 RGB float32 [0, 1] images with structure."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    base = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    return [
        np.clip(base + 0.1 * rng.standard_normal(base.shape), 0, 1).astype(np.float32)
        for _ in range(n)
    ]


def _one_helpful_tools() -> list[nn.Module]:
    """12 tools where exactly slot 0 helps (smooth) and the rest are no-ops."""
    tools: list[nn.Module] = [_Identity() for _ in range(_N_TOOLS)]
    tools[0] = _Smooth()
    return tools


def _all_smooth_tools() -> list[nn.Module]:
    return [_Smooth() for _ in range(_N_TOOLS)]


def _pairs(severity: str = "moderate", n: int = 3, base: int = 20_000):
    seeds = list(range(base, base + n))
    return build_pairs(_make_images(), _one_helpful_tools(), _DEVICE, severity, seeds)


# ══════════════════════════ (a) oracle monotonicity ══════════════════════════


def test_a_greedy_oracle_helpful_vs_noop_removal() -> None:
    """Removing the only helpful tool strictly hurts; removing a no-op is a wash.

    We deliberately do NOT assert universal monotonicity: the greedy oracle is
    path-dependent, so on real tools removing a myopically-over-used tool can
    even raise the 3-step gain (a small negative drop). This fake set isolates
    the two clean cases.
    """
    tools = _one_helpful_tools()  # slot 0 smooths, slots 1-11 are no-ops
    pairs = _pairs(n=3)
    full = greedy_oracle_gain(tools, pairs, _DEVICE)

    # Removing the one helpful tool (slot 0) strictly reduces the gain.
    without0 = greedy_oracle_gain(tools[1:], pairs, _DEVICE)
    assert full - without0 > 0.0

    # Removing a pure no-op (slot 5) leaves the gain unchanged.
    without5 = greedy_oracle_gain(
        [tool for i, tool in enumerate(tools) if i != 5], pairs, _DEVICE
    )
    assert full - without5 == pytest.approx(0.0, abs=1e-9)


def test_a_oracle_matches_b3_baseline() -> None:
    """greedy_oracle_gain on all 12 == compute_baselines' B3 for the same seeds."""
    images = _make_images()
    tools = _all_smooth_tools()
    seeds = list(range(20_000, 20_004))
    pairs = build_pairs(images, tools, _DEVICE, "moderate", seeds)
    ours = greedy_oracle_gain(tools, pairs, _DEVICE)
    b3 = compute_baselines(
        tools=tools,
        images=images,
        device=_DEVICE,
        seeds=seeds,
        n=len(seeds),
        n_random_chains=1,
        severity="moderate",
    )["B3_greedy_oracle"]
    assert ours == pytest.approx(b3, abs=1e-6)


def test_a_forward_selection_structure() -> None:
    """Forward selection: sizes 1..12, every tool added exactly once, the full
    set hits 100% of the full ceiling, and tools_to_reach is a sane 1..12.

    We do NOT assert which tool is added first: the no-STOP oracle forces 3x
    application, so at k=1 a powerful-but-over-applied tool can score below a
    gentle one — a real property, conservative for the toolbox-size readout.
    """
    import math

    from scripts.analyze_tool_value import greedy_forward_selection, tools_to_reach

    tools = _one_helpful_tools()
    pairs = _pairs(n=3)
    curve = greedy_forward_selection(tools, pairs, _DEVICE)

    assert [e["size"] for e in curve] == list(range(1, _N_TOOLS + 1))
    assert sorted(e["added_index"] for e in curve) == list(range(_N_TOOLS))
    assert all(math.isfinite(e["ceiling"]) for e in curve)
    assert curve[-1]["frac_of_full"] == pytest.approx(1.0, abs=1e-9)
    assert 1 <= tools_to_reach(curve, 0.95) <= _N_TOOLS


# ══════════════════════════ (b) leave-one-out shape ═══════════════════════════


def test_b_leave_one_out_shape_and_signed() -> None:
    import math

    tools = _one_helpful_tools()
    pairs = _pairs(n=3)
    ceiling, drops, n_negative = leave_one_out_drops(tools, pairs, _DEVICE)
    assert isinstance(ceiling, float)
    assert len(drops) == _N_TOOLS
    assert n_negative >= 0
    for d in drops:
        assert math.isfinite(d), f"drop {d} must be finite"
    # On this fake set only slot 0 helps, so its drop is the largest and > 0.
    assert drops[0] > 0.0
    assert drops[0] == max(drops)


def test_b_full_result_per_tool_keys() -> None:
    """analyze_tool_value per_tool entries carry every documented key."""
    result = analyze_tool_value(
        images=_make_images(),
        tools=_one_helpful_tools(),
        device=_DEVICE,
        per_class=2,
        agent_usage=None,
        specialist_diag=None,
    )
    assert len(result["per_tool"]) == _N_TOOLS
    required = {
        "index",
        "name",
        "band",
        "agent_usage_overall",
        "agent_usage_by_class",
        "leaveoneout_drop_overall",
        "leaveoneout_drop_by_class",
        "specialist_own_band_gain",
        "verdict",
    }
    for entry in result["per_tool"]:
        assert required <= set(entry.keys())
        assert entry["leaveoneout_drop_overall"] >= 0.0
        for cls in ("mild", "moderate", "severe"):
            assert cls in entry["leaveoneout_drop_by_class"]


def test_b_verdict_thresholds() -> None:
    assert verdict_for(0.5) == "keystone"
    assert verdict_for(0.30) == "keystone"
    assert verdict_for(0.20) == "useful"
    assert verdict_for(0.05) == "marginal"
    assert verdict_for(0.0) == "redundant"


# ══════════════════════════ (c) usage parsing ════════════════════════════════


def test_c_usage_fractions_sum_to_one_and_exclude_stop() -> None:
    # 12 tool counts + 1 STOP count = 13 bins.
    hist = [10, 0, 5, 0, 20, 50, 0, 0, 0, 0, 0, 0, 15]  # STOP=15
    out = usage_from_hist(hist)
    total = sum(hist)  # 100
    assert out["total_actions"] == total
    # 12 tool fractions + STOP fraction sum to 1.
    assert sum(out["tool_fractions"]) + out["stop_fraction"] == pytest.approx(1.0)
    # STOP excluded from the per-tool share.
    assert out["stop_fraction"] == pytest.approx(15 / total)
    assert out["tool_fractions"][0] == pytest.approx(10 / total)
    assert len(out["tool_fractions"]) == _N_TOOLS


def test_c_usage_empty_histogram_safe() -> None:
    out = usage_from_hist([0] * 13)
    assert out["stop_fraction"] == 0.0
    assert out["tool_fractions"] == [0.0] * _N_TOOLS
    assert out["total_actions"] == 0


def test_c_load_agent_usage_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_agent_usage(tmp_path / "nope.json") is None


def test_c_load_agent_usage_pools_classes(tmp_path: Path) -> None:
    """Overall histogram is the per-class sum, then normalised."""
    fake = {
        "per_class": {
            "mild": {"agent": {"action_hist": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}},
            "moderate": {
                "agent": {"action_hist": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]}
            },
            "severe": {
                "agent": {"action_hist": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]}
            },
        }
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(fake))
    usage = load_agent_usage(p)
    assert usage is not None
    # Pooled counts: tool0=1, tool4=1, STOP=3 -> total 5.
    assert usage["overall"]["tool_fractions"][0] == pytest.approx(1 / 5)
    assert usage["overall"]["tool_fractions"][4] == pytest.approx(1 / 5)
    assert usage["overall"]["stop_fraction"] == pytest.approx(3 / 5)
    assert set(usage["by_class"].keys()) == {"mild", "moderate", "severe"}


# ══════════════════════════ (d) specialist diagonal ══════════════════════════


def test_d_parse_specialist_diagonal_real_matrix() -> None:
    path = Path("reports/tools_specialist_matrix.md")
    diag = parse_specialist_diagonal(path)
    if not path.exists():
        pytest.skip("specialist matrix not present locally")
    assert diag is not None
    assert len(diag) == _N_TOOLS
    assert all(isinstance(v, float) for v in diag)


def test_d_parse_specialist_diagonal_missing_returns_none(tmp_path: Path) -> None:
    assert parse_specialist_diagonal(tmp_path / "nope.md") is None


# ══════════════════════════ (e) core smoke into tmp_path ═════════════════════


def test_e_core_smoke_writes_json(tmp_path: Path) -> None:
    """Importable core on fakes -> JSON-able dict with per_tool + protocol."""
    result = analyze_tool_value(
        images=_make_images(),
        tools=_all_smooth_tools(),
        device=_DEVICE,
        per_class=2,
        agent_usage=None,
        specialist_diag=None,
    )
    result["protocol"] = {"per_class": 2, "n_tools": _N_TOOLS}

    out = tmp_path / "tool_value.json"
    out.write_text(json.dumps(result, indent=2))
    assert out.exists() and out.stat().st_size > 0

    reloaded = json.loads(out.read_text())
    assert "per_tool" in reloaded and "protocol" in reloaded
    assert len(reloaded["per_tool"]) == _N_TOOLS
    assert "full_oracle_ceiling_by_class" in reloaded
    # Agent usage was None -> per-tool usage fields are null, STOP block null.
    assert reloaded["per_tool"][0]["agent_usage_overall"] is None
    assert reloaded["stop_fraction_overall"] is None
