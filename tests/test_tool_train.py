import numpy as np
import torch

from rlrestore.tools.specs import TOOL_SPECS
from rlrestore.tools.train import TrainState, train_tool


def _images(n=2):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    base = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    return [
        np.clip(base + 0.1 * rng.standard_normal(base.shape), 0, 1).astype(np.float32)
        for _ in range(n)
    ]


def test_train_tool_runs_and_improves_val(tmp_path):
    state = train_tool(
        spec=TOOL_SPECS[4],          # mild noise: easiest to learn fast
        images=_images(),
        val_images=_images(),
        out_dir=tmp_path,
        epochs=2, steps_per_epoch=25, batch_size=8,
        lr=0.01, lr_drop_every=5, lr_drop_factor=0.1,
        momentum=0.9, weight_decay=1e-4, grad_clip_norm=0.4,
        robust_extras=False, optimizer="adam",   # adam: stable for a 50-step test
        val_pairs=8, seed=0, device=torch.device("cpu"),
        loader_workers=0,
    )
    assert isinstance(state, TrainState)
    assert state.epoch == 2
    assert (tmp_path / "last.pt").exists() and (tmp_path / "best.pt").exists()
    assert state.best_val_psnr > 0


def test_train_tool_resumes(tmp_path):
    kwargs = dict(
        spec=TOOL_SPECS[4], images=_images(), val_images=_images(), out_dir=tmp_path,
        steps_per_epoch=5, batch_size=4, lr=0.01, lr_drop_every=5, lr_drop_factor=0.1,
        momentum=0.9, weight_decay=1e-4, grad_clip_norm=0.4, robust_extras=False,
        optimizer="adam", val_pairs=4, seed=0, device=torch.device("cpu"),
        loader_workers=0,
    )
    s1 = train_tool(epochs=1, **kwargs)
    s2 = train_tool(epochs=2, **kwargs)  # picks up from last.pt
    assert s1.epoch == 1 and s2.epoch == 2
    assert s2.resumed_from == 1


def test_train_tool_early_stops_on_plateau(tmp_path, monkeypatch):
    import rlrestore.tools.train as tr

    monkeypatch.setattr(tr, "_validate", lambda *a, **k: 30.0)  # constant: best at epoch 1
    state = tr.train_tool(
        spec=TOOL_SPECS[4], images=_images(), val_images=_images(), out_dir=tmp_path,
        epochs=10, steps_per_epoch=3, batch_size=4,
        lr=0.01, lr_drop_every=5, lr_drop_factor=0.1,
        momentum=0.9, weight_decay=1e-4, grad_clip_norm=0.4,
        robust_extras=False, optimizer="adam",
        val_pairs=4, seed=0, device=torch.device("cpu"), patience=2,
    )
    # best at epoch 1; epochs 2 and 3 don't improve -> stop at epoch 3
    assert state.epoch == 3
    ck = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert ck["early_stopped"] is True and ck["best_epoch"] == 1


def test_early_stop_ignores_microscopic_improvements(tmp_path, monkeypatch):
    import rlrestore.tools.train as tr

    calls = {"n": 0}

    def creeping_validate(*a, **k):
        calls["n"] += 1
        return 30.0 + calls["n"] * 0.001  # +0.001 dB per epoch: measurement noise

    monkeypatch.setattr(tr, "_validate", creeping_validate)
    state = tr.train_tool(
        spec=TOOL_SPECS[4], images=_images(), val_images=_images(), out_dir=tmp_path,
        epochs=12, steps_per_epoch=3, batch_size=4,
        lr=0.01, lr_drop_every=5, lr_drop_factor=0.1,
        momentum=0.9, weight_decay=1e-4, grad_clip_norm=0.4,
        robust_extras=False, optimizer="adam",
        val_pairs=4, seed=0, device=torch.device("cpu"), patience=3,
    )
    # epoch 1 is the only MEANINGFUL improvement; +0.001 creep must not reset patience
    assert state.epoch == 4
    ck = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert ck["early_stopped"] is True and ck["best_epoch"] == 1
    assert ck["best_val_psnr"] > 30.003  # best.pt still tracks every tiny gain
