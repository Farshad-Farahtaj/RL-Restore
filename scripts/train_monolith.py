"""Train the monolithic end-to-end restoration CNN (P3.E4 / B5).

Single DnCNN-style network, capacity matched to the 12-tool toolbox, trained on
the FULL severity family (docs/plan3/monolith_spec.md). Reuses the tools'
early-stopping methodology (MIN_IMPROVEMENT_DB on overall val GAIN, patience,
significant_best, best.pt/last.pt schema, optimizer-mismatch resume guard) and
the agent runner's resumable-Colab discipline (atomic tmp+rename, status.json
heartbeat, train_log.jsonl, NaN-halt WITHOUT overwriting best.pt, snapshots).

The training loop is factored into run_monolith_training(...) so tests can drive
it directly with a tiny config.

CLI:
    uv run python scripts/train_monolith.py [--config configs/monolith.yaml]
        [--device cuda] [--out-dir checkpoints/monolith/<name>] [--no-resume]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from rlrestore.baselines.monolith import MonoRestoreCNN, count_parameters
from rlrestore.common.config import MonolithCfg, load_config
from rlrestore.common.device import get_device
from rlrestore.common.metrics import psnr
from rlrestore.common.seed import set_seed
from rlrestore.data.div2k import image_paths
from rlrestore.data.mixed_dataset import MixedPairDataset
from rlrestore.data.pipeline import make_training_pair
from rlrestore.tools.dataset import _to_tensor, build_pyramid
from rlrestore.tools.train import load_images

MIN_IMPROVEMENT_DB = 0.02  # below this, an "improvement" is measurement noise
_SNAPSHOT_EVERY = 10  # epochs between immutable checkpoint snapshots


# ──────────────────────────── atomic I/O helpers ─────────────────────────────


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


def _atomic_save_json(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _append_jsonl(line: dict[str, Any], path: Path) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────── loss ───────────────────────────────────────────


def charbonnier_loss(pred: torch.Tensor, clean: torch.Tensor, eps: float) -> torch.Tensor:
    """sqrt((pred-clean)^2 + eps^2).mean() — robust L1-like loss (EDSR-era)."""
    return torch.sqrt((pred - clean) ** 2 + eps * eps).mean()


# ──────────────────────────── validation ─────────────────────────────────────


def _build_val_set(
    val_images: list[np.ndarray],
    severities: tuple[str, ...],
    val_pairs: int,
    seed: int,
    pyramid: list | None = None,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Fixed (clean, degraded, severity) val triples, built ONCE.

    Round-robins severities across the val_pairs so each class is represented
    evenly; every triple is fully determined by (seed, i) so the val set is
    byte-identical across epochs and resumes.
    """
    triples: list[tuple[np.ndarray, np.ndarray, str]] = []
    for i in range(val_pairs):
        rng = np.random.default_rng((seed, i))
        img_idx = int(rng.integers(len(val_images)))
        severity = severities[i % len(severities)]
        src = pyramid[img_idx][0] if pyramid is not None else val_images[img_idx]
        clean, degraded = make_training_pair(src, severity, rng)
        triples.append((clean, degraded, severity))
    return triples


@torch.no_grad()
def _validate(
    model: MonoRestoreCNN,
    val_set: list[tuple[np.ndarray, np.ndarray, str]],
    severities: tuple[str, ...],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[float, dict[str, float]]:
    """Overall + per-severity mean PSNR GAIN (psnr(model(degraded)) - psnr(degraded)).

    Returns (overall_gain_db, {severity: gain_db}).
    """
    model.eval()
    per_class: dict[str, list[float]] = {s: [] for s in severities}
    try:
        for start in range(0, len(val_set), batch_size):
            chunk = val_set[start : start + batch_size]
            degraded = torch.stack([_to_tensor(d) for _, d, _ in chunk]).to(device)
            restored = model(degraded).cpu().numpy()
            for (clean, degraded_np, sev), r in zip(chunk, restored):
                rest_hwc = r.transpose(1, 2, 0)
                gain = psnr(rest_hwc, clean) - psnr(degraded_np, clean)
                per_class[sev].append(gain)
    finally:
        model.train()
    class_means = {s: float(np.mean(v)) if v else 0.0 for s, v in per_class.items()}
    all_gains = [g for v in per_class.values() for g in v]
    overall = float(np.mean(all_gains)) if all_gains else 0.0
    return overall, class_means


# ──────────────────────────── train state ────────────────────────────────────


@dataclass
class MonolithTrainState:
    """epoch: completed epochs. resumed_from: start_epoch before this call.
    early_stopped: whether early stopping fired. best_val_gain: best overall gain."""

    epoch: int
    best_val_gain: float
    best_epoch: int
    resumed_from: int
    early_stopped: bool


# ──────────────────────────── run_monolith_training ──────────────────────────


def run_monolith_training(
    *,
    mcfg: MonolithCfg,
    images: list[np.ndarray],
    val_images: list[np.ndarray],
    out_dir: Path | str,
    seed: int,
    device: torch.device,
    resume: bool = True,
    loader_workers: int = 0,
) -> MonolithTrainState:
    """Resumable, early-stopping training loop for the monolithic CNN.

    Mirrors tools.train.train_tool (early stopping) and agent.runner (atomic
    I/O, status heartbeat, NaN-halt, snapshots). Validation metric is overall
    PSNR GAIN on a FIXED val set spanning all severities.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    severities = tuple(mcfg.severities)
    optimizer = mcfg.optimizer
    session_id = str(uuid.uuid4())[:8]
    wall_start = time.monotonic()

    model = MonoRestoreCNN(depth=mcfg.depth, width=mcfg.width).to(device)
    n_params = count_parameters(model)

    # Build pyramids ONCE (shared across epochs; copy-on-write with forked
    # workers on Linux). OFF by default on 16 GB machines.
    cache = os.environ.get("RLR_CACHE_PYRAMID", "0") == "1"
    train_pyr = build_pyramid(images) if cache else None
    val_pyr = build_pyramid(val_images) if cache else None

    if optimizer == "adam":
        opt: torch.optim.Optimizer = torch.optim.Adam(
            model.parameters(), lr=mcfg.lr, weight_decay=mcfg.weight_decay
        )
    else:  # documented fallback; spec uses adam
        opt = torch.optim.SGD(
            model.parameters(), lr=mcfg.lr, momentum=0.9, weight_decay=mcfg.weight_decay
        )

    # Fixed val set — built ONCE, byte-identical across epochs/resumes.
    val_set = _build_val_set(val_images, severities, mcfg.val_pairs, seed + 10_000, val_pyr)

    # ── Resume (+ idempotent exit when already done / early-stopped) ──────
    start_epoch, best = 0, -float("inf")
    best_epoch = 0
    significant = -float("inf")
    wall_prev = 0.0
    last = out_dir / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        ck_opt = ck.get("optimizer_type")
        if ck_opt is not None and ck_opt != optimizer:
            raise SystemExit(
                f"checkpoint used optimizer={ck_opt!r} but config says {optimizer!r}; "
                "delete last.pt to retrain"
            )
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]
        best = ck["best_val"]
        best_epoch = ck.get("best_epoch", start_epoch)
        significant = ck.get("significant_best", best)
        wall_prev = float(ck.get("wall_sec_total", 0.0))
        resumed_from = start_epoch
        if ck.get("early_stopped") or start_epoch >= mcfg.epochs:
            print(
                f"monolith already finished at epoch {start_epoch} "
                f"(early_stopped={ck.get('early_stopped', False)}); nothing to do"
            )
            return MonolithTrainState(
                epoch=start_epoch,
                best_val_gain=best,
                best_epoch=best_epoch,
                resumed_from=resumed_from,
                early_stopped=bool(ck.get("early_stopped", False)),
            )
    else:
        resumed_from = start_epoch

    completed = start_epoch
    early_stopped = False
    for epoch in range(start_epoch, mcfg.epochs):
        epoch_wall_start = time.monotonic()
        cur_lr = mcfg.lr * (mcfg.lr_drop_factor ** (epoch // mcfg.lr_drop_every))
        for g in opt.param_groups:
            g["lr"] = cur_lr

        ds = MixedPairDataset(
            images,
            seed=seed + epoch,
            length=mcfg.steps_per_epoch * mcfg.batch_size,
            severities=severities,
            pyramid=train_pyr,
        )
        dl = DataLoader(ds, batch_size=mcfg.batch_size, num_workers=loader_workers)
        model.train()
        for degraded, clean in dl:
            degraded, clean = degraded.to(device), clean.to(device)
            pred = model(degraded)
            if mcfg.loss == "mse":
                loss = torch.nn.functional.mse_loss(pred, clean)
            else:
                loss = charbonnier_loss(pred, clean, mcfg.eps)

            # NaN-halt: the bad step has NOT run yet, but a non-finite loss
            # backward would poison weights. Halt BEFORE any checkpoint write so
            # the last good best.pt/last.pt on disk stay clean (agent discipline).
            if not torch.isfinite(loss):
                total_wall = (time.monotonic() - wall_start) + wall_prev
                _atomic_save_json(
                    {
                        "epoch": epoch,
                        "nan_halt": True,
                        "session_id": session_id,
                        "updated_utc": _utc_iso(),
                        "wall_sec_total": total_wall,
                    },
                    out_dir / "status.json",
                )
                raise RuntimeError(
                    f"NaN/inf loss at epoch {epoch} — halting WITHOUT checkpointing "
                    "(last good checkpoint preserved; resume restores pre-NaN state)."
                )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), mcfg.grad_clip_norm)
            opt.step()

        overall_gain, class_gains = _validate(model, val_set, severities, device)
        completed = epoch + 1
        chunk_sec = time.monotonic() - epoch_wall_start
        total_wall = (time.monotonic() - wall_start) + wall_prev
        print(
            f"monolith epoch {completed}/{mcfg.epochs} "
            f"val_gain={overall_gain:.3f} dB lr={cur_lr:g} | "
            + " ".join(f"{s}={class_gains[s]:.2f}" for s in severities)
        )

        # best.pt tracks the TRUE max overall gain (every tiny improvement).
        if overall_gain > best:
            best = overall_gain
            _atomic_save(
                {"model": model.state_dict(), "val_psnr": overall_gain},
                out_dir / "best.pt",
            )
        # Patience resets only on MEANINGFUL improvement (>= MIN_IMPROVEMENT_DB);
        # micro-creep at the measurement-noise level must not keep training alive.
        if overall_gain > significant + MIN_IMPROVEMENT_DB:
            significant = overall_gain
            best_epoch = completed
        early_stopped = mcfg.patience > 0 and completed - best_epoch >= mcfg.patience

        # status.json heartbeat
        _atomic_save_json(
            {
                "epoch": completed,
                "last_val_psnr": overall_gain,
                "per_class": class_gains,
                "lr": cur_lr,
                "wall_sec_total": total_wall,
                "last_chunk_sec": chunk_sec,
                "session_id": session_id,
                "updated_utc": _utc_iso(),
            },
            out_dir / "status.json",
        )
        # train_log.jsonl (one line per epoch)
        _append_jsonl(
            {
                "epoch": completed,
                "val_gain_db": overall_gain,
                "per_class_gain_db": class_gains,
                "lr": cur_lr,
                "best_val": best,
                "best_epoch": best_epoch,
                "wall_sec_total": total_wall,
                "chunk_sec": chunk_sec,
            },
            out_dir / "train_log.jsonl",
        )
        # last.pt (rolling resume point)
        _atomic_save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": completed,
                "best_val": best,
                "best_epoch": best_epoch,
                "significant_best": significant,
                "early_stopped": early_stopped,
                "optimizer_type": optimizer,
                "wall_sec_total": total_wall,
            },
            last,
        )
        # Immutable snapshot every _SNAPSHOT_EVERY epochs (divergence insurance).
        if completed % _SNAPSHOT_EVERY == 0:
            snap_dir = out_dir / "snapshots" / f"epoch_{completed}"
            if not snap_dir.exists():
                snap_dir.mkdir(parents=True)
                for fname in ("best.pt", "last.pt"):
                    if (out_dir / fname).exists():
                        shutil.copy2(out_dir / fname, snap_dir / fname)

        if early_stopped:
            print(
                f"monolith early stop at epoch {completed} "
                f"(best was epoch {best_epoch}, patience {mcfg.patience})"
            )
            break

    print(
        f"done: best overall val gain {best:.3f} dB at epoch {best_epoch} "
        f"({n_params:,} params)"
    )
    return MonolithTrainState(
        epoch=completed,
        best_val_gain=best,
        best_epoch=best_epoch,
        resumed_from=resumed_from,
        early_stopped=early_stopped,
    )


# ──────────────────────────── CLI ────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train the monolithic end-to-end restoration CNN (B5)"
    )
    ap.add_argument("--config", default="configs/monolith.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="default checkpoints/monolith/<config name>",
    )
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if cfg.monolith is None:
        raise SystemExit(f"config {args.config} has no 'monolith' section")
    device = get_device(args.device) if args.device else get_device()
    try:
        torch.zeros(1, device=device)
    except (RuntimeError, AssertionError) as exc:
        raise SystemExit(f"device '{device}' unavailable on this machine: {exc}") from exc

    out_dir = Path(args.out_dir) if args.out_dir else Path("checkpoints/monolith") / cfg.name

    images = load_images(image_paths(cfg.data.root, "train"), cfg.data.max_train_images)
    val_images = load_images(image_paths(cfg.data.root, "val"), cfg.data.max_train_images or 20)
    print(f"Loaded {len(images)} train / {len(val_images)} val images; device={device}")

    state = run_monolith_training(
        mcfg=cfg.monolith,
        images=images,
        val_images=val_images,
        out_dir=out_dir,
        seed=cfg.seed,
        device=device,
        resume=not args.no_resume,
        # 0 on macOS (spawn workers each pickle the full image list); set
        # RLR_LOADER_WORKERS=8 on Linux (fork shares memory, e.g. Colab).
        loader_workers=int(os.environ.get("RLR_LOADER_WORKERS", "0")),
    )
    print(
        f"final: epoch {state.epoch}, best {state.best_val_gain:.3f} dB "
        f"@ epoch {state.best_epoch}, early_stopped={state.early_stopped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
