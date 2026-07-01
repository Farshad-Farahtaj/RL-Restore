"""Tests for the chunked resumable runner and deployment health check.

Spec: docs/plan2/infra_spec.md §2-§5; docs/plan2/stability_spec.md §4.

Test index
----------
(a) Fresh run: writes agent_net.pt, target_net.pt, optimizer.pt,
    train_state.json, status.json, train_log.jsonl with >=1 line;
    status.json has the required schema keys.
(b) Resume round-trip: run to env_step X, call run_training again ->
    continues to total without redoing; env_step monotonic;
    train_state env_step == total at end.
(c) Atomicity: no *.tmp files left after run completes.
(d) NaN guard: monkeypatch dqn_train_step to return nan loss once ->
    run continues, nan_loss_count == 1.
(e) Health check — fake tools:
    (e1) all-improving fake set passes (exit 0, gate_pass True).
    (e2) one non-tool04 degrading tool fails (exit 1, gate_pass False).
    (e3) tool04-slot degrading only -> disabled_tools contains 4, exit 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from scripts.check_tool_health import exit_code_for, run_health_check

from rlrestore.agent.runner import run_training
from rlrestore.common.config import AgentCfg, Config, DataCfg, ToolsCfg

# ──────────────────────────── constants ──────────────────────────────────────

_DEVICE = torch.device("cpu")
_N_TOOLS = 12

# ──────────────────────────── fake tools ─────────────────────────────────────


class _Identity(nn.Module):
    """Leaves the image unchanged — neutral PSNR effect."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class _Worsen(nn.Module):
    """Makes images worse by adding a large bias (degrades PSNR)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.6


def _neutral_tools() -> list[nn.Module]:
    return [_Identity() for _ in range(_N_TOOLS)]


class _Smooth(nn.Module):
    """3x3 box smoothing — genuinely denoises the noisy test states, so it
    'helps somewhere' under the revised gate (frac_positive/p90 criteria)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / 9.0)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _passing_tools() -> list[nn.Module]:
    """Smoothing tools — positive delta on most noisy states (revised gate)."""
    return [_Smooth() for _ in range(_N_TOOLS)]


# ──────────────────────────── tiny images ────────────────────────────────────


def _make_images(n: int = 4, seed: int = 0) -> list[np.ndarray]:
    """Create n synthetic 96x96 RGB float32 [0, 0.4] images."""
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(0.0, 0.4, (96, 96, 3)).astype(np.float32) for _ in range(n)
    ]


# ──────────────────────────── tiny AgentCfg ──────────────────────────────────

_TINY_AGENT = AgentCfg(
    env_steps=200,
    warmup_steps=20,
    replay_capacity=2000,
    batch_episodes=4,
    train_every=2,
    target_sync_every=50,
    chunk_size=100,
    eval_pairs=4,
    epsilon_end=0.5,
    epsilon_anneal_steps=150,
    lr=1e-3,
    gamma=0.99,
    use_double_dqn=False,
)


def _make_cfg(agent: AgentCfg = _TINY_AGENT, name: str = "test") -> Config:
    return Config(
        name=name,
        seed=42,
        data=DataCfg(
            root="data/DIV2K",
            severity="moderate",
            val_pairs=4,
            max_train_images=4,
        ),
        tools=ToolsCfg(
            epochs=1,
            steps_per_epoch=10,
            batch_size=8,
            lr=0.1,
            lr_drop_every=5,
            lr_drop_factor=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            grad_clip_norm=0.4,
            robust_extras=False,
            optimizer="sgd",
        ),
        agent=agent,
    )


# ──────────────────────────── test helpers ────────────────────────────────────


def _run(
    tmp_path: Path,
    tools: list[nn.Module] | None = None,
    agent_cfg: AgentCfg = _TINY_AGENT,
    resume: bool = True,
) -> dict[str, Any]:
    if tools is None:
        tools = _neutral_tools()
    images = _make_images(n=4)
    val_images = _make_images(n=4, seed=99)
    cfg = _make_cfg(agent=agent_cfg)
    return run_training(
        cfg=cfg,
        tools=tools,
        images=images,
        val_images=val_images,
        device=_DEVICE,
        out_dir=tmp_path,
        resume=resume,
    )


# ══════════════════════════ (a) Fresh run artifacts ══════════════════════════


def test_a_fresh_run_artifacts(tmp_path: Path) -> None:
    """Fresh run creates all expected artifacts with valid content."""
    metrics = _run(tmp_path)

    # All checkpoint files must exist
    for fname in ["agent_net.pt", "target_net.pt", "optimizer.pt", "train_state.json"]:
        assert (tmp_path / fname).exists(), f"Missing artifact: {fname}"

    # status.json must exist with required keys
    status_path = tmp_path / "status.json"
    assert status_path.exists(), "status.json missing"
    status = json.loads(status_path.read_text())
    required_keys = {
        "env_step", "epsilon", "lr", "last_val_psnr",
        "buffer_fill", "updates_done", "wall_sec_total",
        "last_chunk_sec", "session_id", "updated_utc",
    }
    missing = required_keys - set(status.keys())
    assert not missing, f"status.json missing keys: {missing}"

    # train_log.jsonl must have >= 1 line
    jsonl_path = tmp_path / "train_log.jsonl"
    assert jsonl_path.exists(), "train_log.jsonl missing"
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) >= 1, "train_log.jsonl has no lines"

    # Each line must be valid JSON with env_step
    for line in lines:
        row = json.loads(line)
        assert "env_step" in row, "jsonl line missing env_step"

    # Final metrics must be a non-empty dict
    assert isinstance(metrics, dict) and metrics
    assert "env_step" in metrics

    # train_state must reach total env_steps
    state = json.loads((tmp_path / "train_state.json").read_text())
    assert state["env_step"] >= _TINY_AGENT.env_steps


# ══════════════════════════ (b) Resume round-trip ════════════════════════════


def test_b_resume_round_trip(tmp_path: Path) -> None:
    """Partial run + resume completes to total; env_step is monotonic; refill runs on resume."""
    import rlrestore.agent.runner as runner_mod

    # Phase 1: run only the first chunk (env_steps = chunk_size)
    partial_cfg = AgentCfg(
        **{
            **_TINY_AGENT.__dict__,
            "env_steps": _TINY_AGENT.chunk_size,  # stop after 1 chunk
        }
    )
    metrics1 = _run(tmp_path, agent_cfg=partial_cfg, resume=False)
    step_after_phase1 = metrics1.get("env_step", 0)
    assert step_after_phase1 > 0, "Phase 1 should make progress"

    # Read train_state to confirm partial progress
    state1 = json.loads((tmp_path / "train_state.json").read_text())
    assert state1["env_step"] == step_after_phase1

    # Phase 2: resume with the full env_steps budget.
    # Wrap _replay_refill to count calls and verify it fires at least once.
    refill_call_count: list[int] = [0]
    _original_refill = runner_mod._replay_refill

    def _counting_refill(*args: Any, **kwargs: Any) -> None:
        refill_call_count[0] += 1
        return _original_refill(*args, **kwargs)

    with patch.object(runner_mod, "_replay_refill", _counting_refill):
        metrics2 = _run(tmp_path, agent_cfg=_TINY_AGENT, resume=True)

    assert refill_call_count[0] >= 1, (
        "Expected _replay_refill to be called at least once on resume, "
        f"but it was called {refill_call_count[0]} times"
    )
    step_after_phase2 = metrics2.get("env_step", 0)

    # Must reach or exceed total (at least as far as phase1+chunk_size)
    assert step_after_phase2 >= step_after_phase1, (
        f"env_step went backwards: {step_after_phase1} -> {step_after_phase2}"
    )
    assert step_after_phase2 >= _TINY_AGENT.env_steps, (
        f"Did not reach env_steps={_TINY_AGENT.env_steps}, got {step_after_phase2}"
    )

    # train_state must reflect final env_step
    state2 = json.loads((tmp_path / "train_state.json").read_text())
    assert state2["env_step"] == step_after_phase2

    # Idempotent: calling again returns immediately without error
    metrics3 = _run(tmp_path, agent_cfg=_TINY_AGENT, resume=True)
    assert metrics3.get("env_step") == step_after_phase2


# ══════════════════════════ (c) No tmp leftovers ═════════════════════════════


def test_c_no_tmp_leftovers(tmp_path: Path) -> None:
    """After a successful run, no *.tmp files remain."""
    _run(tmp_path)
    tmp_files = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob("**/*.tmp"))
    assert not tmp_files, f"Found leftover .tmp files: {tmp_files}"


# ══════════════════════════ (d) NaN guard ════════════════════════════════════


def test_d_nan_guard(tmp_path: Path) -> None:
    """NaN loss halts the run WITHOUT checkpointing (F5 revision).

    The bad opt.step has already poisoned in-memory weights; continuing would
    overwrite the only clean rolling checkpoint at the next chunk boundary.
    Contract: RuntimeError raised, no agent_net.pt written for that chunk,
    status.json carries nan_halt=true.
    """
    import json

    import pytest

    import rlrestore.agent.runner as runner_mod

    def _nan_dqn(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {"loss": float("nan"), "grad_norm": 0.0}

    with patch.object(runner_mod, "dqn_train_step", _nan_dqn):
        with pytest.raises(RuntimeError, match="NaN/inf loss"):
            _run(tmp_path)

    # No weights checkpoint may exist (halt fired before end-of-chunk save)
    assert not (tmp_path / "agent_net.pt").exists(), (
        "agent_net.pt was written despite NaN halt — poisoned checkpoint risk"
    )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["nan_halt"] is True
    assert status["nan_loss_count"] == 1


# ══════════════════════════ (e) Health check ═════════════════════════════════


def test_e1_health_check_all_pass(tmp_path: Path) -> None:
    """Smoothing tools genuinely denoise, passing the revised helps-somewhere gate.

    Exercises run_health_check with pre-loaded fake tools and exit_code_for().
    """
    tools = _passing_tools()
    images = _make_images(n=8, seed=7)
    result = run_health_check(tools=tools, images=images, n_pairs=20, seed=42,
                              oracle_min_gain=0.2)
    assert result["all_pass"] is True
    assert result["gate_pass"] is True
    assert result["disabled_tools"] == []
    assert result["defect_tools"] == []
    assert exit_code_for(result) == 0, f"Expected exit 0; result={result}"


def test_e2_health_check_non_tool04_fails(tmp_path: Path) -> None:
    """If tool 0 strongly degrades quality, gate fails with exit 1."""
    tools = _passing_tools()
    tools[0] = _Worsen()
    images = _make_images(n=8, seed=7)
    result = run_health_check(tools=tools, images=images, n_pairs=30, seed=42,
                              oracle_min_gain=0.2)
    assert result["gate_pass"] is False
    assert result["all_pass"] is False
    assert 0 in result["defect_tools"]
    assert exit_code_for(result) == 1, f"Expected exit 1; result={result}"


def test_e3_health_check_tool04_disabled(tmp_path: Path) -> None:
    """Tool04-only failure -> disabled_tools contains 4, exit 0."""
    tools = _passing_tools()
    tools[4] = _Worsen()
    images = _make_images(n=8, seed=7)
    result = run_health_check(tools=tools, images=images, n_pairs=30, seed=42,
                              oracle_min_gain=0.2)
    assert result["gate_pass"] is True
    assert result["all_pass"] is True
    assert 4 in result["disabled_tools"]
    assert result["defect_tools"] == []
    assert exit_code_for(result) == 0, f"Expected exit 0 (tool04 contingency); result={result}"


# ══════════════════════════ (f) Config round-trip ════════════════════════════


def test_f_agent_cfg_float_coercion(tmp_path: Path) -> None:
    """AgentCfg fields lr/gamma/epsilon_end are float, not strings."""
    cfg = _make_cfg()
    assert isinstance(cfg.agent.lr, float)
    assert isinstance(cfg.agent.gamma, float)
    assert isinstance(cfg.agent.epsilon_end, float)


def test_g_config_without_agent_stays_valid(tmp_path: Path) -> None:
    """Config without agent: section has agent==None (Plan-1 compat)."""
    # smoke.yaml without agent key should still load (we test that None is ok)
    cfg_no_agent = Config(
        name="noagent",
        seed=0,
        data=DataCfg(root=".", severity="moderate", val_pairs=4, max_train_images=0),
        tools=ToolsCfg(
            epochs=1,
            steps_per_epoch=10,
            batch_size=8,
            lr=0.1,
            lr_drop_every=5,
            lr_drop_factor=0.1,
            momentum=0.9,
            weight_decay=1e-4,
            grad_clip_norm=0.4,
            robust_extras=False,
            optimizer="sgd",
        ),
    )
    assert cfg_no_agent.agent is None
