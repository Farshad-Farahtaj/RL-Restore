"""Tiny CPU smoke for the monolith training loop (run_monolith_training).

Drives the importable loop directly with a depth=2/width=8 net for 1 epoch on
3 synthetic images, then asserts the artifact set and resume idempotency.
"""

import json

import numpy as np
import torch
from scripts.train_monolith import MonolithTrainState, run_monolith_training

from rlrestore.common.config import MonolithCfg


def _images(n=3, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    base = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    return [
        np.clip(base + 0.1 * rng.standard_normal(base.shape), 0, 1).astype(np.float32)
        for _ in range(n)
    ]


def _tiny_cfg(epochs=1, patience=0):
    return MonolithCfg(
        loss="charbonnier",
        eps=1e-3,
        optimizer="adam",
        lr=1e-3,
        lr_drop_every=5,
        lr_drop_factor=0.5,
        weight_decay=1e-4,
        grad_clip_norm=0.5,
        batch_size=2,
        steps_per_epoch=2,
        epochs=epochs,
        patience=patience,
        severities=["mild", "moderate", "severe"],
        robust_extras=False,
        val_pairs=2,
        depth=2,
        width=8,
    )


def test_smoke_writes_artifacts(tmp_path):
    state = run_monolith_training(
        mcfg=_tiny_cfg(epochs=1),
        images=_images(),
        val_images=_images(),
        out_dir=tmp_path,
        seed=0,
        device=torch.device("cpu"),
    )
    assert isinstance(state, MonolithTrainState)
    assert state.epoch == 1
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "status.json").exists()
    assert (tmp_path / "train_log.jsonl").exists()

    status = json.loads((tmp_path / "status.json").read_text())
    for key in ("epoch", "last_val_psnr", "per_class", "lr", "wall_sec_total",
                "last_chunk_sec", "session_id", "updated_utc"):
        assert key in status, f"status.json missing {key}"
    assert set(status["per_class"]) == {"mild", "moderate", "severe"}

    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert set(best) == {"model", "val_psnr"}
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "opt", "epoch", "best_val", "best_epoch",
                "significant_best", "early_stopped", "optimizer_type"):
        assert key in last, f"last.pt missing {key}"
    assert last["optimizer_type"] == "adam"

    # train_log has exactly one epoch line
    lines = (tmp_path / "train_log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["epoch"] == 1


def test_resume_extends_and_is_idempotent(tmp_path):
    s1 = run_monolith_training(
        mcfg=_tiny_cfg(epochs=1),
        images=_images(),
        val_images=_images(),
        out_dir=tmp_path,
        seed=0,
        device=torch.device("cpu"),
    )
    assert s1.epoch == 1 and s1.resumed_from == 0

    # Resume to 2 epochs: must pick up from last.pt (resumed_from == 1).
    s2 = run_monolith_training(
        mcfg=_tiny_cfg(epochs=2),
        images=_images(),
        val_images=_images(),
        out_dir=tmp_path,
        seed=0,
        device=torch.device("cpu"),
    )
    assert s2.epoch == 2 and s2.resumed_from == 1

    # train_log now has 2 lines (one appended per epoch across runs).
    lines = (tmp_path / "train_log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2

    # Re-running at the same epoch budget is a no-op (already done).
    s3 = run_monolith_training(
        mcfg=_tiny_cfg(epochs=2),
        images=_images(),
        val_images=_images(),
        out_dir=tmp_path,
        seed=0,
        device=torch.device("cpu"),
    )
    assert s3.epoch == 2
    # No extra epoch line appended on the idempotent re-run.
    lines2 = (tmp_path / "train_log.jsonl").read_text().strip().splitlines()
    assert len(lines2) == 2


def test_optimizer_mismatch_guard(tmp_path):
    import pytest

    run_monolith_training(
        mcfg=_tiny_cfg(epochs=1),
        images=_images(),
        val_images=_images(),
        out_dir=tmp_path,
        seed=0,
        device=torch.device("cpu"),
    )
    mismatched = _tiny_cfg(epochs=2)
    object.__setattr__(mismatched, "optimizer", "sgd")  # frozen dataclass
    with pytest.raises(SystemExit, match="optimizer"):
        run_monolith_training(
            mcfg=mismatched,
            images=_images(),
            val_images=_images(),
            out_dir=tmp_path,
            seed=0,
            device=torch.device("cpu"),
        )
