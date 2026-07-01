"""Correctness + smoke tests for joint fine-tuning (Algorithm 1, P3.E5).

Spec: docs/plan3/joint_finetune_spec.md §"Correctness tests".

T1 frozen agent unchanged after a step (params equal; requires_grad False).
T2 grad flows ONLY to tools applied in the batch (stub agent picks tool 3 then
   STOP; tool 3 has grad, an unused tool's grad is None).
T3 use-count averaging: a tool applied twice gets its grad scaled by 1/2 vs a
   one-application reference (tiny numeric check).
T4 differentiable chain: loss.backward() populates the applied tools' grads
   (finite, nonzero).
T5 originals preserved: a tiny FT into a tmp out-tools-name leaves
   checkpoints/tools/full/ byte-identical and writes the new tools dir.
T6 resume idempotence + NaN-halt (mirrors test_train_monolith.py).

Stub agents return a fixed action sequence so T2/T3 control exactly which tools
run; the tools themselves are real (tiny) modules so gradients are genuine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from rlrestore.agent.joint_finetune import (
    JointFTState,
    _to_nchw,
    freeze_agent,
    joint_ft_step,
    load_trainable_tools,
    run_joint_finetune,
    select_and_apply_chain,
)
from rlrestore.agent.net import AgentNet

_DEVICE = torch.device("cpu")
_N_TOOLS = 12
_STOP = 12

# Where the production originals live; T5 asserts they are never touched.
_ORIG_TOOLS_DIR = Path("checkpoints/tools/full")


# ──────────────────────────── stub agent ─────────────────────────────────────


class _ScriptedAgent(AgentNet):
    """AgentNet whose greedy choice follows a fixed action script.

    ``forward_step`` returns a one-hot-ish Q vector whose argmax is the next
    scripted action, advancing an internal pointer. The script is consumed per
    chain; reset() restarts it. Inherits the real AgentNet so freeze_agent and
    the no-grad selection path behave exactly as in production.
    """

    def __init__(self, script: list[int]) -> None:
        super().__init__()
        self._script = script
        self._ptr = 0

    def reset_script(self) -> None:
        self._ptr = 0

    def forward_step(self, img, prev_oh, hx):  # type: ignore[override]
        a = self._script[min(self._ptr, len(self._script) - 1)]
        self._ptr += 1
        q = torch.full((1, 13), -1.0, device=img.device)
        q[0, a] = 1.0
        return q, hx


class _AlwaysAction(AgentNet):
    """Greedy choice is always the same single action (no STOP, runs full 3)."""

    def __init__(self, action: int) -> None:
        super().__init__()
        self._action = action

    def forward_step(self, img, prev_oh, hx):  # type: ignore[override]
        q = torch.full((1, 13), -1.0, device=img.device)
        q[0, self._action] = 1.0
        return q, hx


# ──────────────────────────── tiny tools ─────────────────────────────────────


class _TinyTool(nn.Module):
    """1x1 conv with a global residual — same output=input+correction shape as
    the real tools, trainable, with genuine gradients."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


def _tiny_tools() -> list[nn.Module]:
    tools = [_TinyTool(seed=i) for i in range(_N_TOOLS)]
    for t in tools:
        t.train().requires_grad_(True)
    return tools


# ──────────────────────────── synthetic data ─────────────────────────────────


def _pair(seed: int = 0):
    """One (clean, degraded) NCHW pair of 63x63 synthetic patches on CPU."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:63, 0:63].astype(np.float32) / 63.0
    clean = np.clip(
        np.stack([yy, xx, (yy + xx) / 2], -1)
        + 0.05 * rng.standard_normal((63, 63, 3)).astype(np.float32),
        0,
        1,
    ).astype(np.float32)
    degraded = np.clip(
        clean + 0.2 * rng.standard_normal((63, 63, 3)).astype(np.float32), 0, 1
    ).astype(np.float32)
    return _to_nchw(clean, _DEVICE), _to_nchw(degraded, _DEVICE)


def _images(n: int = 3, seed: int = 0) -> list[np.ndarray]:
    """n synthetic 96x96 RGB float32 [0,1] images (croppable to 63x63 patches)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    base = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    return [
        np.clip(base + 0.1 * rng.standard_normal(base.shape), 0, 1).astype(np.float32)
        for _ in range(n)
    ]


def _hash_dir(d: Path) -> dict[str, str]:
    """sha256 of every file under d (relative path -> hex), for tamper checks."""
    out: dict[str, str] = {}
    for p in sorted(d.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ══════════════════════════════ T1 ═══════════════════════════════════════════


def test_t1_agent_frozen_after_step():
    """Agent params identical before/after a step and have requires_grad False."""
    agent = _AlwaysAction(action=3)
    freeze_agent(agent)
    assert not any(p.requires_grad for p in agent.parameters())

    before = [p.detach().clone() for p in agent.parameters()]

    tools = _tiny_tools()
    opt = torch.optim.Adam([p for t in tools for p in t.parameters()], lr=1e-3)
    joint_ft_step(agent, tools, opt, [_pair(0), _pair(1)], grad_clip_norm=0.5)

    after = list(agent.parameters())
    for b, a in zip(before, after):
        assert torch.equal(b, a), "frozen agent parameter changed after a FT step"
    # No grad ever attached to the agent.
    assert all(p.grad is None for p in agent.parameters())


# ══════════════════════════════ T2 ═══════════════════════════════════════════


def test_t2_grad_only_to_used_tools():
    """Stub agent picks tool 3 then STOP: tool 3 grads, unused tools' grad None."""
    agent = _ScriptedAgent(script=[3, _STOP])
    freeze_agent(agent)

    tools = _tiny_tools()
    opt = torch.optim.Adam([p for t in tools for p in t.parameters()], lr=1e-3)

    # Reset the script before each chain so every image picks the same path.
    def _reset_each(*a, **k):
        agent.reset_script()
        return _orig(*a, **k)

    _orig = select_and_apply_chain
    with patch(
        "rlrestore.agent.joint_finetune.select_and_apply_chain", _reset_each
    ):
        out = joint_ft_step(agent, tools, opt, [_pair(0), _pair(1)], grad_clip_norm=None)

    # Only tool 3 ran (once per image -> twice across the batch).
    assert out["uses"][3] == 2
    assert sum(out["uses"]) == 2

    used_grads = [p.grad for p in tools[3].parameters()]
    assert all(g is not None for g in used_grads), "used tool 3 has no grad"
    assert any(float(g.abs().sum()) > 0 for g in used_grads), "tool 3 grad all zero"

    # An unused tool must have grad None (never entered the graph).
    for t in (0, 5, 11):
        assert all(
            p.grad is None for p in tools[t].parameters()
        ), f"unused tool {t} unexpectedly received a gradient"


# ══════════════════════════════ T3 ═══════════════════════════════════════════


def test_t3_use_count_averaging():
    """Tool applied twice in one chain gets grad scaled by 1/2 vs one application.

    Reference: a single image whose chain applies tool 5 exactly ONCE, no
    averaging (uses==1, the 1/uses branch is a no-op). Subject: a single image
    whose chain applies tool 5 exactly TWICE; with use-count averaging the
    accumulated grad is divided by 2. We compare the parameter-update direction
    magnitude. The two-application grad (pre-averaging) is generally >= the
    one-application grad in norm, so after the 1/2 scaling the subject's grad is
    close to half the *summed* two-application grad — we assert the scaling was
    applied by comparing against the same step run WITHOUT averaging.
    """
    # ── Build two identical toolboxes so the only difference is averaging ──
    tools_avg = _tiny_tools()
    tools_raw = _tiny_tools()
    for ta, tr in zip(tools_avg, tools_raw):
        tr.load_state_dict(ta.state_dict())

    agent = _AlwaysAction(action=5)  # always tool 5 -> chain [5,5,5] (3 steps)
    freeze_agent(agent)

    pair = _pair(7)

    # Subject: real joint_ft_step (applies use-count averaging).
    opt_avg = torch.optim.Adam(
        [p for t in tools_avg for p in t.parameters()], lr=1e-3
    )
    opt_avg.zero_grad()
    out = joint_ft_step(agent, tools_avg, opt_avg, [pair], grad_clip_norm=None)
    # Tool 5 applied 3 times on the single image.
    assert out["uses"][5] == 3
    grad_avg = [
        p.grad.detach().clone()
        for p in tools_avg[5].parameters()
        if p.grad is not None
    ]

    # Reference: identical forward/backward but NO averaging (manual).
    import torch.nn.functional as F

    from rlrestore.agent.joint_finetune import select_and_apply_chain as _sac

    for p in tools_raw[5].parameters():
        p.grad = None
    restored, uses, actions = _sac(agent, tools_raw, pair[1])
    assert uses[5] == 3 and actions == [5, 5, 5]
    loss = F.mse_loss(restored, pair[0])  # n_active==1 -> same /1 as subject
    loss.backward()
    grad_raw = [p.grad.detach().clone() for p in tools_raw[5].parameters()]

    # averaged grad == raw grad * (1/3) elementwise (use-count == 3).
    assert len(grad_avg) == len(grad_raw)
    for ga, gr in zip(grad_avg, grad_raw):
        torch.testing.assert_close(ga, gr / 3.0, rtol=1e-5, atol=1e-7)


# ══════════════════════════════ T4 ═══════════════════════════════════════════


def test_t4_differentiable_chain_populates_grads():
    """loss.backward() through the applied tools yields finite, nonzero grads."""
    import torch.nn.functional as F

    agent = _AlwaysAction(action=2)
    freeze_agent(agent)
    tools = _tiny_tools()
    for p in (p for t in tools for p in t.parameters()):
        p.grad = None

    clean, degraded = _pair(3)
    restored, uses, actions = select_and_apply_chain(agent, tools, degraded)
    assert actions == [2, 2, 2] and uses[2] == 3
    assert restored.requires_grad, "chain output is not connected to the graph"

    loss = F.mse_loss(restored, clean)
    loss.backward()

    grads = [p.grad for p in tools[2].parameters()]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads), "non-finite tool grad"
    assert any(float(g.abs().sum()) > 0 for g in grads), "all-zero tool grad"


# ══════════════════════════════ T5 ═══════════════════════════════════════════


@pytest.mark.skipif(
    not _ORIG_TOOLS_DIR.exists(), reason="checkpoints/tools/full not present"
)
def test_t5_originals_preserved(tmp_path):
    """A tiny FT writes the new out-tools dir and leaves full/ byte-identical."""
    before = _hash_dir(_ORIG_TOOLS_DIR)
    assert before, "expected real tool checkpoints under checkpoints/tools/full"

    agent = _AlwaysAction(action=4)
    freeze_agent(agent)
    out_tools_name = "full_ft_test_t5"
    out_tools_dir = tmp_path / "tools" / out_tools_name

    state = run_joint_finetune(
        agent=agent,
        images=_images(3),
        val_images=_images(3, seed=9),
        out_dir=tmp_path / "run",
        out_tools_name=out_tools_name,
        seed=0,
        device=_DEVICE,
        iters=2,
        batch_size=2,
        chunk_size=2,
        lr=1e-3,
        severities=("moderate",),
        in_tools_name="full",  # initialise from the REAL originals
        tools_ckpt_root=tmp_path / "tools",  # but write under tmp
        val_pairs=3,
        own_band_pairs=4,
        patience_chunks=0,
        resume=False,
        tools=_tiny_tools(),  # tiny tools keep it seconds-fast
    )
    assert isinstance(state, JointFTState)

    # New fine-tuned toolbox written.
    assert (out_tools_dir / "tool00" / "best.pt").exists()

    # Originals untouched (hashes identical).
    after = _hash_dir(_ORIG_TOOLS_DIR)
    assert after == before, "checkpoints/tools/full was modified during fine-tuning"


def test_t5b_in_tools_dir_not_written_when_root_shared(tmp_path):
    """Even sharing the ckpt root, the in-tools dir is never written.

    Uses a fake 'orig' toolbox under tmp so we can assert byte-stability without
    depending on the real checkpoints being present.
    """
    # Materialise a fake originals dir 'orig' under a shared tmp root.
    root = tmp_path / "tools"
    orig_dir = root / "orig"
    seed_tools = _tiny_tools()
    for i, t in enumerate(seed_tools):
        d = orig_dir / f"tool{i:02d}"
        d.mkdir(parents=True)
        torch.save({"model": t.state_dict(), "spec_index": i}, d / "best.pt")
    before = _hash_dir(orig_dir)

    agent = _AlwaysAction(action=1)
    freeze_agent(agent)
    run_joint_finetune(
        agent=agent,
        images=_images(3),
        val_images=_images(3, seed=9),
        out_dir=tmp_path / "run",
        out_tools_name="orig_ft",
        seed=0,
        device=_DEVICE,
        iters=2,
        batch_size=2,
        chunk_size=2,
        lr=1e-3,
        severities=("moderate",),
        in_tools_name="orig",
        tools_ckpt_root=root,
        val_pairs=3,
        own_band_pairs=4,
        patience_chunks=0,
        resume=False,
        tools=_tiny_tools(),
    )
    assert (root / "orig_ft" / "tool00" / "best.pt").exists()
    assert _hash_dir(orig_dir) == before, "in-tools dir was modified"


# ══════════════════════════════ T6 ═══════════════════════════════════════════


def _run_tiny(tmp_path, *, iters, resume, out_name="ft", tools=None):
    agent = _AlwaysAction(action=4)
    freeze_agent(agent)
    return run_joint_finetune(
        agent=agent,
        images=_images(2),
        val_images=_images(2, seed=5),
        out_dir=tmp_path,
        out_tools_name=out_name,
        seed=0,
        device=_DEVICE,
        iters=iters,
        batch_size=2,
        chunk_size=2,
        lr=1e-3,
        severities=("moderate",),
        in_tools_name="full",
        tools_ckpt_root=tmp_path / "tools",
        val_pairs=2,
        own_band_pairs=4,
        patience_chunks=0,  # disable early stop so iters is the only budget
        resume=resume,
        tools=tools if tools is not None else _tiny_tools(),
    )


def test_t6_writes_artifacts_and_resumes(tmp_path):
    """Fresh run writes status/train_log/tools; a second call resumes & extends."""
    s1 = _run_tiny(tmp_path, iters=4, resume=False)
    assert isinstance(s1, JointFTState)
    assert s1.iters == 4 and s1.resumed_from == 0

    # Artifact set.
    assert (tmp_path / "optimizer.pt").exists()
    assert (tmp_path / "status.json").exists()
    assert (tmp_path / "train_log.jsonl").exists()
    assert (tmp_path / "tools" / "ft" / "tool00" / "best.pt").exists()

    status = json.loads((tmp_path / "status.json").read_text())
    for key in (
        "iter",
        "system_val_gain",
        "per_tool_ownband",
        "lr",
        "wall_sec",
        "updated_utc",
    ):
        assert key in status, f"status.json missing {key}"
    assert len(status["per_tool_ownband"]) == 12

    # iters=4 / chunk=2 -> exactly 2 chunk lines.
    lines = (tmp_path / "train_log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[-1])["iter"] == 4

    # Resume to 8 iters: picks up from iter 4 (resumed_from == 4), extends log.
    s2 = _run_tiny(tmp_path, iters=8, resume=True)
    assert s2.iters == 8 and s2.resumed_from == 4
    lines2 = (tmp_path / "train_log.jsonl").read_text().strip().splitlines()
    assert len(lines2) == 4

    # Re-running at the same budget is a no-op (already done).
    s3 = _run_tiny(tmp_path, iters=8, resume=True)
    assert s3.iters == 8
    lines3 = (tmp_path / "train_log.jsonl").read_text().strip().splitlines()
    assert len(lines3) == 4


def test_t6_nan_halt_preserves_best(tmp_path):
    """NaN loss halts WITHOUT overwriting the best tools (agent discipline).

    First do one clean chunk so best.pt tools exist on disk; then monkeypatch
    joint_ft_step to return a NaN loss and assert the run raises, status.json
    carries nan_halt, and the previously-saved best tools are unchanged.
    """
    # Clean run to lay down best tools + an optimizer checkpoint.
    _run_tiny(tmp_path, iters=2, resume=False)
    best_pt = tmp_path / "tools" / "ft" / "tool00" / "best.pt"
    assert best_pt.exists()
    best_hash = hashlib.sha256(best_pt.read_bytes()).hexdigest()

    import rlrestore.agent.joint_finetune as jf

    def _nan_step(*a, **k):
        return {"loss": float("nan"), "grad_norm": 0.0, "uses": [0] * 12, "n_active": 0}

    with patch.object(jf, "joint_ft_step", _nan_step):
        with pytest.raises(RuntimeError, match="NaN/inf loss"):
            _run_tiny(tmp_path, iters=6, resume=True)

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["nan_halt"] is True
    assert status["nan_loss_count"] == 1
    # Best tools on disk untouched by the poisoned run.
    assert hashlib.sha256(best_pt.read_bytes()).hexdigest() == best_hash


# ══════════════════════════ extra: load_trainable_tools ══════════════════════


@pytest.mark.skipif(
    not _ORIG_TOOLS_DIR.exists(), reason="checkpoints/tools/full not present"
)
def test_load_trainable_tools_are_trainable():
    """load_trainable_tools returns 12 train()-mode, requires_grad=True modules."""
    tools = load_trainable_tools("full", _DEVICE)
    assert len(tools) == _N_TOOLS
    assert all(t.training for t in tools)
    assert all(all(p.requires_grad for p in t.parameters()) for t in tools)


def test_no_tmp_leftovers(tmp_path):
    """After a successful run no *.tmp files remain (atomic I/O)."""
    _run_tiny(tmp_path, iters=2, resume=False)
    tmp_files = list(tmp_path.rglob("*.tmp"))
    assert not tmp_files, f"leftover .tmp files: {tmp_files}"
