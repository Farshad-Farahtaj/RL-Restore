"""Chunked, resumable DQN training runner for RL-Restore.

Spec: docs/plan2/infra_spec.md §2-§5; docs/plan2/stability_spec.md §1.

Artifact layout under out_dir
------------------------------
agent_net.pt         — online network weights (atomic)
target_net.pt        — target network weights (atomic)
optimizer.pt         — optimizer state dict (atomic)
train_state.json     — {env_step, rng_state_numpy, rng_state_random,
                        wall_seconds, updates_done}
status.json          — heartbeat; infra_spec §5 fields (atomic)
train_log.jsonl      — one JSON line per chunk; stability_spec §1 fields
agent_baselines.json — B0-B4 baselines (computed once, then cached)

Resume semantics
----------------
If train_state.json exists: load net/opt/env_step, then REPLAY REFILL:
  collect min(50_000, env_steps//4) steps at current epsilon, optimizer
  frozen.  This re-warms the replay without counting those steps toward
  the total (env_step is NOT incremented during refill).

NaN guard (REVISED 2026-06-12, F5 review)
-----------------------------------------
dqn_train_step returns {"loss": float, "grad_norm": float}.  If the
returned loss is nan/inf the in-memory weights are already poisoned
(opt.step ran inside dqn_train_step).  The original guard zeroed grads
and CONTINUED — which meant the next end-of-chunk checkpoint overwrote
the only clean rolling checkpoint on disk with NaN weights.  The run
now HALTS: write a nan_halt status.json (no weights touched), raise
RuntimeError.  The last good checkpoint stays on disk and a re-run
resumes from it.  Test contract: "monkeypatch dqn_train_step to return
nan loss once -> RuntimeError, no checkpoint for that chunk, status.json
has nan_halt=true."

Snapshots: every _SNAPSHOT_EVERY env steps the four checkpoint files are
copied to snapshots/step_<N>k/ as immutable restore points against soft
(non-NaN) divergence contaminating the rolling checkpoint.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlrestore.agent.env import RestoreEnv
from rlrestore.agent.net import AgentNet
from rlrestore.agent.replay import EpisodeReplay
from rlrestore.agent.train import (
    check_smoke_thresholds,  # re-exported so importers can get it here  # noqa: F401
    collect_episode,
    compute_baselines,
    dqn_train_step,
    epsilon_at,
    evaluate,
    lr_at,
)
from rlrestore.common.config import Config

# ──────────────────────────── atomic I/O helpers ─────────────────────────────


def _atomic_save_torch(obj: Any, path: Path) -> None:
    """Save a torch object atomically via tmp+rename (same pattern as tools/train.py)."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def _atomic_save_json(obj: Any, path: Path) -> None:
    """Save a JSON-serialisable object atomically via tmp+os.replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _append_jsonl(line: dict[str, Any], path: Path) -> None:
    """Append one JSON line to a .jsonl file (open-append; not atomic per line)."""
    with path.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


_SNAPSHOT_EVERY = 250_000  # env steps between immutable checkpoint snapshots


# ──────────────────────────── helpers ────────────────────────────────────────


def _build_val_envs(
    images: list[np.ndarray],
    tools: list,
    device: torch.device,
    n: int,
    base_seed: int,
    severity: str = "moderate",
) -> list[RestoreEnv]:
    """Build n fixed validation envs with seeds base_seed .. base_seed+n-1."""
    return [
        RestoreEnv(
            images=images,
            tools=tools,
            device=device,
            seed=base_seed + i,
            training_mode=False,
            severity=severity,
        )
        for i in range(n)
    ]


def _replay_refill(
    env: RestoreEnv,
    online: AgentNet,
    replay: EpisodeReplay,
    rng_py: random.Random,
    device: torch.device,
    refill_steps: int,
    eps: float,
) -> None:
    """Collect refill_steps env steps at eps WITHOUT optimizer updates."""
    collected = 0
    while collected < refill_steps:
        ep = collect_episode(
            env=env,
            net=online,
            eps=eps,
            rng=rng_py,
            device=device,
            replay=replay,
        )
        collected += ep["length"]


# ─────────────────────────── run_training ────────────────────────────────────


def run_training(
    cfg: Config,
    tools: list,
    images: list[np.ndarray],
    val_images: list[np.ndarray],
    device: torch.device,
    out_dir: Path | str,
    resume: bool = True,
) -> dict[str, Any]:
    """Chunked, resumable DQN training loop.

    Parameters
    ----------
    cfg        : Config with a non-None cfg.agent section.
    tools      : List of 12 frozen tool nn.Module objects.
    images     : Training images (float32 HWC numpy arrays).
    val_images : Validation images for evaluate() and baselines.
    device     : Torch device.
    out_dir    : Directory for checkpoints and logs.
    resume     : If True and train_state.json exists, resume from checkpoint.

    Returns
    -------
    dict — final metrics from the last chunk (stability_spec §1 fields
           plus runner-owned extras).
    """
    assert cfg.agent is not None, "cfg.agent must be set for run_training"
    acfg = cfg.agent

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = str(uuid.uuid4())[:8]
    wall_start = time.monotonic()

    # ── Build network objects ──────────────────────────────────────────────
    online = AgentNet().to(device)
    target = AgentNet().to(device)
    target.requires_grad_(False)
    target.load_state_dict(online.state_dict())

    opt = torch.optim.Adam(online.parameters(), lr=acfg.lr)
    replay = EpisodeReplay(capacity=acfg.replay_capacity)

    rng_np = np.random.default_rng(cfg.seed)
    rng_py = random.Random(cfg.seed)

    # ── Resume ─────────────────────────────────────────────────────────────
    env_step: int = 0
    updates_done: int = 0
    wall_seconds_prev: float = 0.0
    resuming = False
    _td_loss_ema_resume: float = 0.0
    _grad_norm_ema_resume: float = 0.0
    _episode_count_resume: int = 0

    train_state_path = out_dir / "train_state.json"
    if resume and train_state_path.exists():
        state = json.loads(train_state_path.read_text())
        env_step = int(state["env_step"])
        wall_seconds_prev = float(state.get("wall_seconds", 0.0))
        updates_done = int(state.get("updates_done", 0))
        _td_loss_ema_resume = float(state.get("td_loss_ema", 0.0))
        _grad_norm_ema_resume = float(state.get("grad_norm_ema", 0.0))
        _episode_count_resume = int(state.get("episode_count", 0))

        # Restore RNG states when available
        if "rng_state_numpy" in state:
            _rng_tmp = np.random.default_rng()
            _rng_tmp.__setstate__(state["rng_state_numpy"])  # type: ignore[attr-defined]
            rng_np = _rng_tmp
        if "rng_state_random" in state:
            # random.setstate expects (version, internalstate_tuple, gauss_next)
            # JSON round-trip turns tuples -> lists, so we need to reconstruct.
            raw = state["rng_state_random"]
            rng_py.setstate((int(raw[0]), tuple(int(x) for x in raw[1]), raw[2]))

        net_path = out_dir / "agent_net.pt"
        tgt_path = out_dir / "target_net.pt"
        opt_path = out_dir / "optimizer.pt"
        if net_path.exists():
            online.load_state_dict(
                torch.load(net_path, map_location=device, weights_only=True)
            )
        if tgt_path.exists():
            target.load_state_dict(
                torch.load(tgt_path, map_location=device, weights_only=True)
            )
        if opt_path.exists():
            opt.load_state_dict(
                torch.load(opt_path, map_location=device, weights_only=False)
            )
        resuming = env_step > 0

    # ── Idempotent exit when already done ─────────────────────────────────
    if env_step >= acfg.env_steps:
        jsonl_path = out_dir / "train_log.jsonl"
        if jsonl_path.exists():
            lines = jsonl_path.read_text().strip().splitlines()
            if lines:
                return json.loads(lines[-1])
        return {"env_step": env_step}

    # ── Training env ────────────────────────────────────────────────────────
    train_env = RestoreEnv(
        images=images,
        tools=tools,
        device=device,
        seed=cfg.seed + 1,
        training_mode=True,
    )

    # ── Val envs (fixed seeds per stability_spec §1: seed 17) ─────────────
    val_envs = _build_val_envs(
        images=val_images,
        tools=tools,
        device=device,
        n=acfg.eval_pairs,
        base_seed=17,
    )

    # ── Baselines (computed once, cached) ─────────────────────────────────
    baselines_path = out_dir / "agent_baselines.json"
    if baselines_path.exists():
        baselines = json.loads(baselines_path.read_text())
    else:
        baseline_seeds = list(range(17, 17 + max(acfg.eval_pairs, 256)))
        baselines = compute_baselines(
            tools=tools,
            images=val_images,
            device=device,
            seeds=baseline_seeds,
            n=min(acfg.eval_pairs, 256),
        )
        _atomic_save_json(baselines, baselines_path)

    # ── Replay refill on resume (infra_spec §2) ────────────────────────────
    if resuming:
        refill_steps = min(50_000, acfg.env_steps // 4)
        refill_eps = epsilon_at(
            env_step,
            warmup=acfg.warmup_steps,
            decay_steps=float(acfg.epsilon_anneal_steps),
            eps_end=acfg.epsilon_end,
        )
        _replay_refill(
            env=train_env,
            online=online,
            replay=replay,
            rng_py=rng_py,
            device=device,
            refill_steps=refill_steps,
            eps=refill_eps,
        )

    # ── EMA state (restored from checkpoint if resuming) ──────────────────
    td_loss_ema: float = _td_loss_ema_resume
    grad_norm_ema: float = _grad_norm_ema_resume
    _EMA_ALPHA: float = 0.01

    # ── Rolling episode stats ─────────────────────────────────────────────
    recent_rewards: list[float] = []
    recent_early_term: list[bool] = []
    episode_count: int = _episode_count_resume  # total since start (for train_every schedule)

    # ── Chunk-level NaN counter (cumulative across chunks) ─────────────────
    nan_loss_count: int = 0

    final_metrics: dict[str, Any] = {}

    # ══════════════════════════ CHUNK LOOP ════════════════════════════════
    while env_step < acfg.env_steps:
        chunk_start_step = env_step
        chunk_start_wall = time.monotonic()
        td_loss_max_chunk: float = 0.0

        # ── Inner env-step loop for this chunk ──────────────────────────
        # NOTE: a chunk may overshoot by up to MAX_STEPS-1 env steps because episodes are atomic.
        while (env_step - chunk_start_step) < acfg.chunk_size and env_step < acfg.env_steps:
            eps = epsilon_at(
                env_step,
                warmup=acfg.warmup_steps,
                decay_steps=float(acfg.epsilon_anneal_steps),
                eps_end=acfg.epsilon_end,
            )
            cur_lr = lr_at(env_step, lr_start=acfg.lr)

            # Collect one episode
            ep_info = collect_episode(
                env=train_env,
                net=online,
                eps=eps,
                rng=rng_py,
                device=device,
                replay=replay,
            )
            env_step += ep_info["length"]
            episode_count += 1

            recent_rewards.append(float(ep_info["reward"]))
            recent_early_term.append(bool(ep_info["early_term"]))
            if len(recent_rewards) > 1000:
                recent_rewards.pop(0)
                recent_early_term.pop(0)

            # Training update (past warmup + enough buffer + on schedule)
            past_warmup = env_step >= acfg.warmup_steps
            enough_eps = replay.num_episodes >= acfg.batch_episodes
            on_schedule = (episode_count % acfg.train_every) == 0

            if past_warmup and enough_eps and on_schedule:
                step_result = dqn_train_step(
                    online=online,
                    target=target,
                    opt=opt,
                    replay=replay,
                    rng=rng_np,
                    device=device,
                    batch_episodes=acfg.batch_episodes,
                    gamma=acfg.gamma,
                    use_double_dqn=acfg.use_double_dqn,
                    lr=cur_lr,
                    grad_clip_norm=acfg.grad_clip_norm,
                )
                loss_val = float(step_result["loss"])
                grad_val = float(step_result["grad_norm"])

                if math.isnan(loss_val) or math.isinf(loss_val):
                    nan_loss_count += 1
                    opt.zero_grad()
                    # HALT before any checkpoint write: the bad opt.step already
                    # poisoned the in-memory weights; saving them would overwrite
                    # the only clean rolling checkpoint on disk. Leave the last
                    # good checkpoint intact and signal via status.json + exit.
                    _atomic_save_json(
                        {
                            "env_step": env_step,
                            "nan_halt": True,
                            "nan_loss_count": nan_loss_count,
                            "session_id": session_id,
                            "updated_utc": _utc_iso(),
                        },
                        out_dir / "status.json",
                    )
                    raise RuntimeError(
                        f"NaN/inf loss at env_step={env_step} — halting WITHOUT "
                        "checkpointing (last good checkpoint preserved). "
                        "Resume restores pre-NaN state; see stability_spec F5."
                    )
                else:
                    if td_loss_ema == 0.0 and grad_norm_ema == 0.0:
                        td_loss_ema = loss_val
                        grad_norm_ema = grad_val
                    else:
                        td_loss_ema = _EMA_ALPHA * loss_val + (1 - _EMA_ALPHA) * td_loss_ema
                        grad_norm_ema = _EMA_ALPHA * grad_val + (1 - _EMA_ALPHA) * grad_norm_ema
                    td_loss_max_chunk = max(td_loss_max_chunk, loss_val)
                    updates_done += 1

                    # Target sync on update count schedule
                    if (updates_done % acfg.target_sync_every) == 0:
                        target.load_state_dict(online.state_dict())

        # ── End-of-chunk: evaluate, log, checkpoint ──────────────────────
        chunk_wall = time.monotonic() - chunk_start_wall
        total_wall = (time.monotonic() - wall_start) + wall_seconds_prev

        eps_now = epsilon_at(
            env_step,
            warmup=acfg.warmup_steps,
            decay_steps=float(acfg.epsilon_anneal_steps),
            eps_end=acfg.epsilon_end,
        )
        lr_now = lr_at(env_step, lr_start=acfg.lr)

        eval_metrics = evaluate(net=online, envs=val_envs, device=device)

        mean_ep_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        early_term_rate = float(np.mean(recent_early_term)) if recent_early_term else 0.0
        eps_per_sec = float(episode_count / max(total_wall, 1e-9))

        chunk_metrics: dict[str, Any] = {
            # ── stability_spec §1 fields ───────────────────────────────
            "env_step": env_step,
            "wall_sec": total_wall,
            "episodes_per_sec": eps_per_sec,
            "epsilon": eps_now,
            "lr": lr_now,
            "mean_ep_reward": mean_ep_reward,
            "val_psnr_gain_db": eval_metrics["val_psnr_gain_db"],
            "action_hist": eval_metrics["action_hist"],
            "entropy": eval_metrics["entropy"],
            "action_hist_entropy": eval_metrics["entropy"],  # alias for stability_spec naming
            "stop_rate_t1": eval_metrics["stop_rate_t1"],
            "stop_rate_t2": eval_metrics["stop_rate_t2"],
            "stop_rate_t3": eval_metrics["stop_rate_t3"],
            "mean_max_q": eval_metrics["mean_max_q"],
            "mean_q_stop": eval_metrics["mean_q_stop"],
            "mean_q_best_tool": eval_metrics["mean_q_best_tool"],
            "q_ratio_stop_best": eval_metrics["q_ratio_stop_best"],
            "td_loss_ema": td_loss_ema,
            "td_loss_max_last_ckpt": td_loss_max_chunk,
            "grad_norm_ema": grad_norm_ema,
            "replay_size": replay.num_transitions,
            "nan_loss_count": nan_loss_count,
            "early_term_rate": early_term_rate,
            # ── runner-owned extras ────────────────────────────────────
            "updates_done": updates_done,
            "episodes_total": episode_count,
            "chunk_sec": chunk_wall,
        }
        final_metrics = chunk_metrics

        # Append metrics line to jsonl
        _append_jsonl(chunk_metrics, out_dir / "train_log.jsonl")

        # Write status heartbeat (infra_spec §5)
        _atomic_save_json(
            {
                "env_step": env_step,
                "epsilon": eps_now,
                "lr": lr_now,
                "last_val_psnr": eval_metrics["val_psnr_gain_db"],
                "buffer_fill": replay.num_transitions,
                "updates_done": updates_done,
                "wall_sec_total": total_wall,
                "last_chunk_sec": chunk_wall,
                "session_id": session_id,
                "updated_utc": _utc_iso(),
            },
            out_dir / "status.json",
        )

        # Atomic checkpoints
        _atomic_save_torch(online.state_dict(), out_dir / "agent_net.pt")
        _atomic_save_torch(target.state_dict(), out_dir / "target_net.pt")
        _atomic_save_torch(opt.state_dict(), out_dir / "optimizer.pt")

        # train_state.json  (epsilon/lr intentionally omitted — derived from env_step)
        _atomic_save_json(
            {
                "env_step": env_step,
                "wall_seconds": total_wall,
                "updates_done": updates_done,
                "td_loss_ema": td_loss_ema,
                "grad_norm_ema": grad_norm_ema,
                "episode_count": episode_count,
                "rng_state_numpy": rng_np.__getstate__(),  # type: ignore[attr-defined]
                "rng_state_random": list(rng_py.getstate()),
            },
            out_dir / "train_state.json",
        )

        # Periodic snapshot (divergence insurance): the rolling checkpoint above
        # is the ONLY restore point — a soft divergence would overwrite it with
        # contaminated weights before any guard fires. Keep an immutable copy
        # every _SNAPSHOT_EVERY env steps (~1.5 MB each, 8 per full run).
        bucket = env_step // _SNAPSHOT_EVERY
        if bucket > 0:
            snap_dir = out_dir / "snapshots" / f"step_{bucket * _SNAPSHOT_EVERY // 1000}k"
            if not snap_dir.exists():
                snap_dir.mkdir(parents=True)
                for fname in (
                    "agent_net.pt",
                    "target_net.pt",
                    "optimizer.pt",
                    "train_state.json",
                ):
                    shutil.copy2(out_dir / fname, snap_dir / fname)

    return final_metrics
