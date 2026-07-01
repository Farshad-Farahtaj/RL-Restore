"""Joint fine-tuning of the toolbox (RL-Restore Algorithm 1, P3.E5).

Spec: docs/plan3/joint_finetune_spec.md.

The headline "beyond the paper" contribution: the original authors describe
Algorithm 1 but never released its code (the public repo trains tools in
isolation and freezes them). Here we close the loop — a FROZEN, already-trained
agent selects each image's <=3-tool chain, the chain is re-applied
differentiably, and the final MSE against the clean target backpropagates into
the TOOLS (the agent stays frozen). Each tool's accumulated gradient is averaged
by its use-count across the batch before the optimiser step, so a tool applied
many times does not take a larger step.

Core algorithm (fused single pass per image)
--------------------------------------------
::

    state = degraded                      # (1,3,63,63)
    hx = None; prev_oh = zeros(1,12)
    for step in range(3):
        with torch.no_grad():             # selection sees DETACHED state
            q, hx = agent.forward_step(state.detach(), prev_oh, hx)
            a = int(q.argmax())
        if a == STOP: break
        state = tools[a](state)           # WITH grad (train mode) -> builds graph
        uses[a] += 1
        prev_oh = one_hot(a, 12)
    loss_img = mse(state, clean)          # skip if no tool was applied

The batch loss is the mean over the per-image losses (images where the agent
STOPs at step 0 contribute zero and are excluded from the mean). After
``loss.backward()`` and BEFORE ``opt.step()``, each tool ``t`` with
``uses[t] > 0`` has its accumulated ``.grad`` scaled by ``1 / uses[t]``.

Why selection runs in ``no_grad`` on a detached state
-----------------------------------------------------
The agent picks the chain from the CURRENT (drifting) tool outputs — that is
what keeps its routing valid as the tools change — but its choice is a discrete
argmax that carries no gradient. Detaching the state for selection means the
agent's LSTM never enters the autograd graph (so the MPS-LSTM crash never blocks
a local CPU smoke), and the gradient path is purely degraded -> tools -> loss.

Infrastructure mirrors agent/runner.py and scripts/train_monolith.py: atomic
tmp+rename checkpoints, status.json heartbeat, train_log.jsonl per chunk, a
fixed system-validation set scored each chunk (greedy agent + current tools,
PSNR gain), early stopping on the system val gain, NaN-halt WITHOUT overwriting
the best tools, snapshots, and resume.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rlrestore.agent.net import AgentNet
from rlrestore.common.metrics import psnr
from rlrestore.data.pipeline import make_training_pair
from rlrestore.data.recipes import SEVERITY_LEVELS
from rlrestore.tools.models import build_tool
from rlrestore.tools.specs import TOOL_SPECS

# ─────────────────────────────── constants ───────────────────────────────────

_N_ACTIONS = 12  # tool count (0-11)
_STOP = 12       # stop action index
_MAX_STEPS = 3   # maximum tool steps per chain
MIN_IMPROVEMENT_DB = 0.02  # below this, a system-val "improvement" is noise
_SNAPSHOT_EVERY = 5  # chunks between immutable checkpoint snapshots


# ──────────────────────────── atomic I/O helpers ─────────────────────────────


def _atomic_save_torch(obj: Any, path: Path) -> None:
    """Save a torch object atomically via tmp+rename (tools/train.py pattern)."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def _atomic_save_json(obj: Any, path: Path) -> None:
    """Save a JSON-serialisable object atomically via tmp+os.replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _append_jsonl(line: dict[str, Any], path: Path) -> None:
    """Append one JSON line to a .jsonl file (open-append)."""
    with path.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ──────────────────────────── trainable tools ────────────────────────────────


def load_trainable_tools(
    name: str,
    device: torch.device,
    ckpt_root: Path | str = Path("checkpoints/tools"),
) -> list[nn.Module]:
    """Load the 12 tool CNNs as TRAINABLE modules for joint fine-tuning.

    Unlike :func:`rlrestore.agent.env.load_tools` (which freezes the tools and
    sets ``eval()`` for the DQN agent), this returns fresh tool modules from the
    same ``checkpoints/tools/<name>/toolNN/best.pt`` state dicts in ``train()``
    mode with ``requires_grad=True`` on every parameter, ready to be handed to an
    optimiser.

    Parameters
    ----------
    name : str
        Config-name sub-directory (e.g. ``"full"``).
    device : torch.device
        Device to map the weights to.
    ckpt_root : Path or str, optional
        Root directory; default ``"checkpoints/tools"``.

    Returns
    -------
    list of nn.Module
        12 tool modules in ``train()`` mode with ``requires_grad=True``.
    """
    assert len(TOOL_SPECS) == 12, "toolbox size drifted"
    ckpt_root = Path(ckpt_root)
    tools: list[nn.Module] = []
    for i, spec in enumerate(TOOL_SPECS):
        path = ckpt_root / name / f"tool{i:02d}" / "best.pt"
        model = build_tool(spec)
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.to(device).train().requires_grad_(True)
        tools.append(model)
    return tools


def freeze_agent(agent: AgentNet) -> AgentNet:
    """Put the agent in eval mode with all gradients disabled (Algorithm 1).

    The published algorithm updates the tools ONLY; the agent's parameters must
    never receive a gradient. Returns the same object for call chaining.
    """
    agent.eval()
    agent.requires_grad_(False)
    return agent


# ───────────────────────── chain selection + replay ──────────────────────────


def select_and_apply_chain(
    agent: AgentNet,
    tools: list[nn.Module],
    degraded: Tensor,
    max_steps: int = _MAX_STEPS,
) -> tuple[Tensor, np.ndarray, list[int]]:
    """Select a <=``max_steps`` tool chain with the frozen agent and apply it.

    Fused single pass: at every step the frozen agent chooses the next action
    greedily from the CURRENT (detached) state in ``no_grad``; if the action is a
    tool it is applied in ``train()`` mode, building the autograd graph on the
    image path. Selection never sees gradients (its argmax is discrete and its
    LSTM stays out of the graph), so the only differentiable path is
    ``degraded -> tools -> restored``.

    Parameters
    ----------
    agent : AgentNet
        Frozen agent (``requires_grad=False``, ``eval()``). Caller's
        responsibility; see :func:`freeze_agent`.
    tools : list of nn.Module
        Exactly 12 trainable tool modules (``train()`` mode).
    degraded : Tensor
        Degraded image, shape ``(1, 3, 63, 63)``.
    max_steps : int, optional
        Maximum number of tool applications (default 3).

    Returns
    -------
    restored : Tensor
        Chain output, shape ``(1, 3, 63, 63)``. If the agent STOPs at step 0
        this is ``degraded`` unchanged (and ``actions`` is empty), so callers
        can detect the no-tool case and skip the image's loss.
    uses : np.ndarray
        Length-12 int array; ``uses[t]`` = times tool ``t`` was applied.
    actions : list of int
        The tool indices applied, in order (excludes STOP).
    """
    device = degraded.device
    state = degraded
    hx: tuple[Tensor, Tensor] | None = None
    prev_oh = torch.zeros(1, _N_ACTIONS, device=device)
    uses = np.zeros(_N_ACTIONS, dtype=np.int64)
    actions: list[int] = []

    for _ in range(max_steps):
        with torch.no_grad():
            # Selection sees the DETACHED state: discrete argmax, no graph.
            q, hx = agent.forward_step(state.detach(), prev_oh, hx)
            a = int(q.argmax(dim=-1).item())
        if a == _STOP:
            break
        state = tools[a](state)  # WITH grad — extends the autograd graph
        uses[a] += 1
        actions.append(a)
        prev_oh = torch.zeros(1, _N_ACTIONS, device=device)
        prev_oh[0, a] = 1.0

    return state, uses, actions


def joint_ft_step(
    agent: AgentNet,
    tools: list[nn.Module],
    opt: torch.optim.Optimizer,
    batch: list[tuple[Tensor, Tensor]],
    grad_clip_norm: float | None = 0.5,
) -> dict[str, Any]:
    """One joint fine-tuning step over a batch of (clean, degraded) pairs.

    For each pair the frozen agent selects and differentiably applies its chain;
    the summed MSE is averaged over the images that actually used >=1 tool and a
    single ``backward()`` accumulates per-tool gradients. Before stepping, every
    used tool's gradient is divided by its use-count across the whole batch
    (paper's use-count averaging) and the global grad-norm is optionally clipped.

    Parameters
    ----------
    agent : AgentNet
        Frozen agent (eval, ``requires_grad=False``).
    tools : list of nn.Module
        12 trainable tool modules.
    opt : torch.optim.Optimizer
        Optimiser over the tools' parameters only.
    batch : list of (Tensor, Tensor)
        Each item ``(clean, degraded)``, both ``(1, 3, 63, 63)`` on the device.
    grad_clip_norm : float or None, optional
        Global grad-norm clip applied AFTER use-count averaging (default 0.5).
        ``None`` disables clipping.

    Returns
    -------
    dict
        ``loss`` (float, the averaged batch MSE; 0.0 if no tool ran),
        ``grad_norm`` (float, post-averaging pre-clip L2 norm over tool params),
        ``uses`` (length-12 int list, total tool use-counts this batch),
        ``n_active`` (int, images that applied >=1 tool).
    """
    opt.zero_grad()
    uses_total = np.zeros(_N_ACTIONS, dtype=np.int64)

    loss_sum: Tensor | None = None
    n_active = 0
    for clean, degraded in batch:
        restored, uses, actions = select_and_apply_chain(agent, tools, degraded)
        uses_total += uses
        if not actions:
            # Agent STOPped at step 0: restored is the un-graphed degraded image,
            # so there is nothing to learn from. Skip it (zero-grad that image).
            continue
        loss_img = F.mse_loss(restored, clean)
        loss_sum = loss_img if loss_sum is None else loss_sum + loss_img
        n_active += 1

    if loss_sum is None:
        # No image used a tool this batch — nothing to update.
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "uses": uses_total.tolist(),
            "n_active": 0,
        }

    loss = loss_sum / n_active
    loss.backward()

    # ── Per-tool use-count averaging (paper detail) ─────────────────────────
    # A tool applied uses_t times has accumulated uses_t superimposed gradient
    # contributions; scale by 1/uses_t so frequently-used tools do not step
    # larger. Tools with uses_t == 0 received no grad and are left untouched.
    for t, tool in enumerate(tools):
        ut = int(uses_total[t])
        if ut <= 1:
            continue
        scale = 1.0 / ut
        for p in tool.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    # ── Post-averaging grad norm (for logging) + optional clip ──────────────
    all_params = [p for tool in tools for p in tool.parameters()]
    if grad_clip_norm is not None:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(all_params, grad_clip_norm))
    else:
        sq = 0.0
        for p in all_params:
            if p.grad is not None:
                sq += float(p.grad.detach().norm(2).item() ** 2)
        grad_norm = float(sq**0.5)

    opt.step()

    return {
        "loss": float(loss.detach().item()),
        "grad_norm": grad_norm,
        "uses": uses_total.tolist(),
        "n_active": n_active,
    }


# ──────────────────────────── batch sampling ─────────────────────────────────


def _sample_batch(
    images: list[np.ndarray],
    severities: tuple[str, ...],
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> list[tuple[Tensor, Tensor]]:
    """Sample a batch of (clean, degraded) NCHW pairs, severities drawn uniform.

    Each pair: a random training image, a random severity from ``severities``,
    one ``make_training_pair`` draw. Returned tensors are ``(1, 3, 63, 63)`` on
    ``device``.
    """
    batch: list[tuple[Tensor, Tensor]] = []
    for _ in range(batch_size):
        img = images[int(rng.integers(len(images)))]
        sev = severities[int(rng.integers(len(severities)))]
        clean_np, degraded_np = make_training_pair(img, sev, rng)
        clean = _to_nchw(clean_np, device)
        degraded = _to_nchw(degraded_np, device)
        batch.append((clean, degraded))
    return batch


def _to_nchw(img_hwc: np.ndarray, device: torch.device) -> Tensor:
    """HWC float32 numpy -> (1, 3, H, W) float32 tensor on ``device``."""
    return torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0).to(device).float()


# ──────────────────────── system validation (whole pipeline) ─────────────────


def _build_val_set(
    val_images: list[np.ndarray],
    severities: tuple[str, ...],
    val_pairs: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Fixed (clean, degraded, severity) val triples, built ONCE.

    Round-robins severities across ``val_pairs`` so every class is evenly
    represented; each triple is fully determined by ``(seed, i)`` so the set is
    byte-identical across chunks and resumes (mirrors train_monolith).
    """
    triples: list[tuple[np.ndarray, np.ndarray, str]] = []
    for i in range(val_pairs):
        rng = np.random.default_rng((seed, i))
        img_idx = int(rng.integers(len(val_images)))
        severity = severities[i % len(severities)]
        clean, degraded = make_training_pair(val_images[img_idx], severity, rng)
        triples.append((clean, degraded, severity))
    return triples


@torch.no_grad()
def evaluate_system(
    agent: AgentNet,
    tools: list[nn.Module],
    val_set: list[tuple[np.ndarray, np.ndarray, str]],
    severities: tuple[str, ...],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Whole-system PSNR gain: greedy agent routes the CURRENT tools per image.

    For each fixed val triple the greedy agent selects and applies its chain on
    the current (fine-tuning) tools; the metric is mean
    ``psnr(restored, clean) - psnr(degraded, clean)`` overall and per severity.
    This is the same end-to-end quantity ``evaluate_agent.py`` reports, used here
    as the in-training early-stopping signal.

    Tools are flipped to ``eval()`` for the measurement and restored to
    ``train()`` afterwards.
    """
    was_training = [t.training for t in tools]
    for t in tools:
        t.eval()
    agent.eval()
    per_class: dict[str, list[float]] = {s: [] for s in severities}
    try:
        for clean, degraded, sev in val_set:
            x = _to_nchw(degraded, device)
            restored, _uses, _actions = select_and_apply_chain(agent, tools, x)
            rest_hwc = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gain = psnr(rest_hwc, clean) - psnr(degraded, clean)
            per_class.setdefault(sev, []).append(gain)
    finally:
        for t, was in zip(tools, was_training):
            t.train(was)
    class_means = {s: float(np.mean(v)) if v else 0.0 for s, v in per_class.items()}
    all_gains = [g for v in per_class.values() for g in v]
    overall = float(np.mean(all_gains)) if all_gains else 0.0
    return overall, class_means


# ─────────────────────────── forgetting monitor ──────────────────────────────


@torch.no_grad()
def own_band_gains(
    tools: list[nn.Module],
    images: list[np.ndarray],
    pairs_per_band: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    """Each tool's own-band specialist gain (the Phase-1 matrix diagonal).

    For tool ``t``, degrade patches with tool ``t``'s own band and report the
    mean ``ΔPSNR`` of ``t`` applied to them — a cheap proxy for catastrophic
    forgetting: joint fine-tuning for chains may erode single-band
    specialisation, and that tradeoff is a finding to REPORT, not hide.

    Imported lazily so the heavy ``tools.dataset`` / cv2 machinery is only pulled
    in when the monitor actually runs.
    """
    from rlrestore.tools.dataset import ToolPairDataset
    from rlrestore.tools.validate import select_eval_indices

    was_training = [t.training for t in tools]
    for t in tools:
        t.eval()
    diag: list[float] = []
    try:
        for j, spec in enumerate(TOOL_SPECS):
            ds = ToolPairDataset(
                images,
                spec,
                seed=seed + 20_000 + j,
                length=pairs_per_band * 4,
                robust_extras=False,
            )
            pairs = [ds[k] for k in range(len(ds))]
            clean_np = torch.stack([c for _, c in pairs]).numpy().transpose(0, 2, 3, 1)
            degraded_np = (
                torch.stack([d for d, _ in pairs]).numpy().transpose(0, 2, 3, 1)
            )
            base_all = np.array(
                [psnr(degraded_np[k], clean_np[k]) for k in range(len(pairs))]
            )
            keep = select_eval_indices(base_all, pairs_per_band)
            clean_keep = clean_np[keep]
            base = base_all[keep]
            deg_dev = torch.stack([pairs[k][0] for k in keep]).to(device)
            restored = tools[j](deg_dev).cpu().numpy().transpose(0, 2, 3, 1)
            gains = [
                psnr(restored[k], clean_keep[k]) - base[k] for k in range(len(keep))
            ]
            diag.append(float(np.mean(gains)) if gains else 0.0)
    finally:
        for t, was in zip(tools, was_training):
            t.train(was)
    return diag


# ──────────────────────────── tool checkpoint I/O ────────────────────────────


def _save_tools(tools: list[nn.Module], tools_dir: Path) -> None:
    """Atomically save each tool to ``tools_dir/toolNN/best.pt`` ({"model": sd}).

    Mirrors tools/train.py's checkpoint schema so the fine-tuned toolbox is a
    drop-in for ``evaluate_agent.py --tools-name <out>`` and
    :func:`load_trainable_tools`.
    """
    for i, tool in enumerate(tools):
        d = tools_dir / f"tool{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        _atomic_save_torch(
            {"model": tool.state_dict(), "spec_index": i}, d / "best.pt"
        )


def _load_tools_into(
    tools: list[nn.Module], tools_dir: Path, device: torch.device
) -> None:
    """Load ``tools_dir/toolNN/best.pt`` state dicts INTO the given modules.

    Restores weights without rebuilding architectures, so a caller that injected
    custom (e.g. tiny test) tool modules resumes into the SAME objects the
    optimiser already references. The optimiser state is parameter-tensor keyed,
    so we must mutate these tensors in place rather than swap the modules.
    """
    for i, tool in enumerate(tools):
        ck = torch.load(
            tools_dir / f"tool{i:02d}" / "best.pt",
            map_location=device,
            weights_only=False,
        )
        tool.load_state_dict(ck["model"])


# ──────────────────────────── train state ────────────────────────────────────


@dataclass
class JointFTState:
    """Outcome of a :func:`run_joint_finetune` call.

    Attributes
    ----------
    iters : int
        Total fine-tuning iterations completed.
    best_system_gain : float
        Best whole-system val PSNR gain reached.
    best_chunk : int
        Chunk index (1-based) at which the best gain was achieved.
    resumed_from : int
        Iteration count at the start of this call (0 = fresh).
    early_stopped : bool
        Whether early stopping fired on the system val gain.
    """

    iters: int
    best_system_gain: float
    best_chunk: int
    resumed_from: int
    early_stopped: bool


# ─────────────────────────── run_joint_finetune ──────────────────────────────


def run_joint_finetune(
    *,
    agent: AgentNet,
    images: list[np.ndarray],
    val_images: list[np.ndarray],
    out_dir: Path | str,
    out_tools_name: str,
    seed: int,
    device: torch.device,
    iters: int = 200_000,
    batch_size: int = 32,
    chunk_size: int = 5_000,
    lr: float = 1e-4,
    grad_clip_norm: float | None = 0.5,
    severities: tuple[str, ...] = ("mild", "moderate", "severe"),
    in_tools_name: str = "full",
    tools_ckpt_root: Path | str = Path("checkpoints/tools"),
    val_pairs: int = 128,
    own_band_pairs: int = 32,
    patience_chunks: int = 6,
    resume: bool = True,
    tools: list[nn.Module] | None = None,
) -> JointFTState:
    """Resumable, early-stopping joint fine-tuning loop (Algorithm 1).

    The frozen ``agent`` routes the trainable tools; the chain MSE updates the
    tools with per-tool use-count gradient averaging. Each chunk validates the
    whole system on a FIXED val set (greedy agent + current tools, PSNR gain),
    early-stops on that gain (``MIN_IMPROVEMENT_DB`` + ``patience_chunks``), saves
    the tools to ``checkpoints/tools/<out_tools_name>/`` atomically (originals in
    ``in_tools_name`` are NEVER overwritten), and logs the per-tool own-band gain
    (forgetting monitor). NaN loss halts WITHOUT overwriting the saved tools.

    Parameters
    ----------
    agent : AgentNet
        Trained agent; frozen in place at entry (eval, ``requires_grad=False``).
    images, val_images : list of np.ndarray
        Train / validation image pools (uint8 or float HWC).
    out_dir : Path or str
        Directory for the optimiser checkpoint, logs, status and snapshots.
    out_tools_name : str
        Sub-directory under ``tools_ckpt_root`` for the fine-tuned tools.
    seed : int
        Base RNG seed (training stream + fixed val/own-band sets are derived).
    device : torch.device
        Torch device.
    iters : int, optional
        Iteration cap / safety rail (default 200_000); early stopping is the
        real judge.
    batch_size : int, optional
        Images (chains) per iteration (paper "M=64"; default 32).
    chunk_size : int, optional
        Iterations per validation+checkpoint chunk (default 5_000).
    lr : float, optional
        Adam learning rate (paper's alpha; default 1e-4).
    grad_clip_norm : float or None, optional
        Global grad-norm clip after use-count averaging (default 0.5).
    severities : tuple of str, optional
        Severity classes sampled uniformly (default all three).
    in_tools_name : str, optional
        Source toolbox to initialise from (default ``"full"``; never written).
    tools_ckpt_root : Path or str, optional
        Root of the tool checkpoints (default ``"checkpoints/tools"``).
    val_pairs : int, optional
        Fixed system-val episodes across severities (default 128).
    own_band_pairs : int, optional
        Pairs per band for the forgetting monitor (default 32).
    patience_chunks : int, optional
        Early-stop patience in chunks (default 6).
    resume : bool, optional
        If True and an optimiser checkpoint exists, resume from it.
    tools : list of nn.Module, optional
        Pre-built trainable tools (tests inject tiny/fake tools). When ``None``
        they are loaded from ``in_tools_name`` via :func:`load_trainable_tools`.

    Returns
    -------
    JointFTState
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tools_ckpt_root = Path(tools_ckpt_root)
    out_tools_dir = tools_ckpt_root / out_tools_name

    severities = tuple(severities)
    for s in severities:
        assert s in SEVERITY_LEVELS, f"unknown severity {s!r}"

    session_id = str(uuid.uuid4())[:8]
    wall_start = time.monotonic()

    # ── Frozen agent + trainable tools + optimiser ──────────────────────────
    freeze_agent(agent.to(device))
    if tools is None:
        tools = load_trainable_tools(in_tools_name, device, ckpt_root=tools_ckpt_root)
    tool_params = [p for tool in tools for p in tool.parameters()]
    opt = torch.optim.Adam(tool_params, lr=lr)

    rng = np.random.default_rng(seed)

    # ── Fixed sets (built ONCE, byte-identical across chunks/resumes) ───────
    val_set = _build_val_set(val_images, severities, val_pairs, seed + 10_000)

    # ── Resume / idempotent-exit ────────────────────────────────────────────
    last_path = out_dir / "optimizer.pt"
    done_iters = 0
    best = -float("inf")
    best_chunk = 0
    significant = -float("inf")
    wall_prev = 0.0
    chunk_idx = 0
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        # Restore the EXACT in-memory tool weights from the rolling _last dir
        # (best.pt only tracks improvements). Load in place so the optimiser's
        # parameter references — and thus its loaded state — stay valid.
        _load_tools_into(tools, out_tools_dir / "_last", device)
        opt.load_state_dict(ck["opt"])
        done_iters = int(ck["iters"])
        best = float(ck["best_system_gain"])
        best_chunk = int(ck.get("best_chunk", 0))
        significant = float(ck.get("significant_best", best))
        chunk_idx = int(ck.get("chunk_idx", 0))
        wall_prev = float(ck.get("wall_sec_total", 0.0))
        if bool(ck.get("early_stopped", False)) or done_iters >= iters:
            return JointFTState(
                iters=done_iters,
                best_system_gain=best,
                best_chunk=best_chunk,
                resumed_from=done_iters,
                early_stopped=bool(ck.get("early_stopped", False)),
            )
    resumed_from = done_iters

    early_stopped = False
    nan_count = 0

    # ══════════════════════════ CHUNK LOOP ═════════════════════════════════
    while done_iters < iters:
        chunk_idx += 1
        chunk_start_iter = done_iters
        chunk_wall_start = time.monotonic()
        loss_running: list[float] = []
        grad_running: list[float] = []
        uses_chunk = np.zeros(_N_ACTIONS, dtype=np.int64)

        # ── Inner iteration loop ────────────────────────────────────────────
        while (done_iters - chunk_start_iter) < chunk_size and done_iters < iters:
            for t in tools:
                t.train()
            batch = _sample_batch(images, severities, batch_size, rng, device)
            step = joint_ft_step(
                agent, tools, opt, batch, grad_clip_norm=grad_clip_norm
            )
            done_iters += 1

            loss_val = float(step["loss"])
            if not np.isfinite(loss_val):
                # NaN-halt: the bad opt.step already poisoned in-memory tools.
                # Halt BEFORE saving so the best tools on disk stay clean
                # (agent/runner.py discipline).
                nan_count += 1
                total_wall = (time.monotonic() - wall_start) + wall_prev
                _atomic_save_json(
                    {
                        "iter": done_iters,
                        "nan_halt": True,
                        "nan_loss_count": nan_count,
                        "session_id": session_id,
                        "wall_sec": total_wall,
                        "updated_utc": _utc_iso(),
                    },
                    out_dir / "status.json",
                )
                raise RuntimeError(
                    f"NaN/inf loss at iter={done_iters} — halting WITHOUT saving "
                    "(last good tools preserved; resume restores pre-NaN state)."
                )

            if step["n_active"] > 0:
                loss_running.append(loss_val)
                grad_running.append(float(step["grad_norm"]))
            uses_chunk += np.asarray(step["uses"], dtype=np.int64)

        # ── End-of-chunk: validate, monitor, log, checkpoint ────────────────
        system_gain, per_class = evaluate_system(
            agent, tools, val_set, severities, device
        )
        ownband = own_band_gains(tools, val_images, own_band_pairs, seed, device)
        chunk_sec = time.monotonic() - chunk_wall_start
        total_wall = (time.monotonic() - wall_start) + wall_prev
        mean_loss = float(np.mean(loss_running)) if loss_running else 0.0
        mean_grad = float(np.mean(grad_running)) if grad_running else 0.0

        # best.pt tools track the TRUE max system gain (every improvement).
        if system_gain > best:
            best = system_gain
            _save_tools(tools, out_tools_dir)
        # Patience resets only on a MEANINGFUL improvement (>= MIN_IMPROVEMENT_DB);
        # measurement-noise creep must not keep fine-tuning alive.
        if system_gain > significant + MIN_IMPROVEMENT_DB:
            significant = system_gain
            best_chunk = chunk_idx
        early_stopped = (
            patience_chunks > 0 and chunk_idx - best_chunk >= patience_chunks
        )

        # status.json heartbeat (spec schema)
        _atomic_save_json(
            {
                "iter": done_iters,
                "system_val_gain": system_gain,
                "per_class_gain": per_class,
                "per_tool_ownband": ownband,
                "lr": lr,
                "wall_sec": total_wall,
                "session_id": session_id,
                "updated_utc": _utc_iso(),
            },
            out_dir / "status.json",
        )
        # train_log.jsonl — one line per chunk
        _append_jsonl(
            {
                "chunk": chunk_idx,
                "iter": done_iters,
                "system_val_gain": system_gain,
                "per_class_gain": per_class,
                "per_tool_ownband": ownband,
                "mean_loss": mean_loss,
                "mean_grad_norm": mean_grad,
                "uses_chunk": uses_chunk.tolist(),
                "best_system_gain": best,
                "best_chunk": best_chunk,
                "lr": lr,
                "wall_sec": total_wall,
                "chunk_sec": chunk_sec,
            },
            out_dir / "train_log.jsonl",
        )
        # optimizer.pt — rolling resume point (tool weights live in out_tools_dir).
        _atomic_save_torch(
            {
                "opt": opt.state_dict(),
                "iters": done_iters,
                "chunk_idx": chunk_idx,
                "best_system_gain": best,
                "best_chunk": best_chunk,
                "significant_best": significant,
                "early_stopped": early_stopped,
                "wall_sec_total": total_wall,
            },
            last_path,
        )
        # Also persist a rolling last.pt of the tools so resume restores the
        # exact in-memory weights (best.pt only updates on improvement).
        _save_tools(tools, out_tools_dir / "_last")

        # Immutable snapshot every _SNAPSHOT_EVERY chunks (divergence insurance).
        if chunk_idx % _SNAPSHOT_EVERY == 0:
            snap = out_dir / "snapshots" / f"chunk_{chunk_idx}"
            if not snap.exists():
                snap.mkdir(parents=True)
                if last_path.exists():
                    shutil.copy2(last_path, snap / "optimizer.pt")
                _save_tools(tools, snap / "tools")

        if early_stopped:
            break

    return JointFTState(
        iters=done_iters,
        best_system_gain=best,
        best_chunk=best_chunk,
        resumed_from=resumed_from,
        early_stopped=early_stopped,
    )


# Re-export for callers/tests that want the type without importing the dataclass
# module path directly.
__all__ = [
    "JointFTState",
    "evaluate_system",
    "freeze_agent",
    "joint_ft_step",
    "load_trainable_tools",
    "own_band_gains",
    "run_joint_finetune",
    "select_and_apply_chain",
]
