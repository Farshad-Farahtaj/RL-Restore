"""Train one repair tool. Paper recipe: MSE, SGD(momentum .9, wd 1e-4),
LR 0.1 x0.1 on a schedule, grad-norm clip 0.4 (VDSR-style), batch 64.
Fallback optimizer 'adam' (lr 1e-3) if SGD diverges - documented decision.

CLI: uv run python -m rlrestore.tools.train --tool 0 --config configs/smoke.yaml
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rlrestore.common.config import load_config
from rlrestore.common.device import get_device
from rlrestore.common.metrics import psnr
from rlrestore.common.seed import set_seed
from rlrestore.data.div2k import image_paths
from rlrestore.tools.dataset import ToolPairDataset, build_pyramid
from rlrestore.tools.models import build_tool
from rlrestore.tools.specs import TOOL_SPECS, ToolSpec

MIN_IMPROVEMENT_DB = 0.02  # below this, an "improvement" is measurement noise


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.rename(path)


@dataclass
class TrainState:
    """epoch: completed epochs (== epochs arg on success).
    resumed_from: start_epoch before this call (0 = fresh start)."""

    epoch: int
    best_val_psnr: float
    resumed_from: int  # 0 = fresh


def load_images(paths, limit=0):
    import cv2

    if limit:
        paths = paths[:limit]
    out = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"cv2.imread could not read: {p}")
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # uint8 RGB, ~4x less RAM
    return out


@torch.no_grad()
def _validate(model, val_images, spec, robust_extras, val_pairs, seed, device, batch_size=32,
              pyramid=None):
    ds = ToolPairDataset(val_images, spec, seed=seed + 10_000, length=val_pairs,
                         robust_extras=robust_extras, pyramid=pyramid)
    model.eval()
    try:
        scores = []
        for start in range(0, len(ds), batch_size):
            items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            degraded = torch.stack([d for d, _ in items]).to(device)
            clean = torch.stack([c for _, c in items])
            restored = model(degraded).cpu()
            for r, c in zip(restored, clean):
                scores.append(psnr(r.numpy().transpose(1, 2, 0), c.numpy().transpose(1, 2, 0)))
    finally:
        model.train()
    return float(np.mean(scores))


def train_tool(*, spec: ToolSpec, images, val_images, out_dir, epochs, steps_per_epoch,
               batch_size, lr, lr_drop_every, lr_drop_factor, momentum, weight_decay,
               grad_clip_norm, robust_extras, optimizer, val_pairs, seed, device,
               log_dir=None, loader_workers: int = 0, patience: int = 0) -> TrainState:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed + spec.index)
    model = build_tool(spec).to(device)

    # Build pyramids ONCE per tool (not per epoch/val call); shared by every
    # dataset below and copy-on-write shared with forked workers on Linux.
    cache = os.environ.get("RLR_CACHE_PYRAMID", "0") == "1"
    train_pyr = build_pyramid(images) if cache else None
    val_pyr = build_pyramid(val_images) if cache else None
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

    start_epoch, best = 0, -1.0
    last = out_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        ck_opt = ck.get("optimizer_type")
        if ck_opt is not None and ck_opt != optimizer:
            raise SystemExit(
                f"checkpoint used optimizer={ck_opt!r} but config says {optimizer!r}; "
                "delete last.pt to retrain"
            )
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best = ck["epoch"], ck["best_val_psnr"]
        best_epoch = ck.get("best_epoch", start_epoch)  # old checkpoints: be conservative
        significant = ck.get("significant_best", best)
    else:
        best_epoch = 0
        significant = -1.0
    resumed_from = start_epoch
    completed = start_epoch

    writer = SummaryWriter(log_dir or f"runs/tools/tool{spec.index:02d}")
    try:
        for epoch in range(start_epoch, epochs):
            cur_lr = lr * (lr_drop_factor ** (epoch // lr_drop_every))
            if optimizer == "sgd":
                for g in opt.param_groups:
                    g["lr"] = cur_lr
            ds = ToolPairDataset(
                images, spec, seed=seed + epoch,
                length=steps_per_epoch * batch_size,
                robust_extras=robust_extras, pyramid=train_pyr,
            )
            dl = DataLoader(ds, batch_size=batch_size, num_workers=loader_workers)
            for step, (degraded, clean) in enumerate(dl):
                degraded, clean = degraded.to(device), clean.to(device)
                loss = torch.nn.functional.mse_loss(model(degraded), clean)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()
                if step % 50 == 0:
                    writer.add_scalar("loss", loss.item(), epoch * steps_per_epoch + step)
            val = _validate(model, val_images, spec, robust_extras, val_pairs, seed, device,
                            pyramid=val_pyr)
            writer.add_scalar("val_psnr", val, epoch + 1)
            print(
                f"tool{spec.index:02d} epoch {epoch + 1}/{epochs}"
                f" val_psnr={val:.2f} lr={cur_lr:g}"
            )
            if val > best:
                best = val
                _atomic_save({"model": model.state_dict(), "spec_index": spec.index,
                              "val_psnr": val}, out_dir / "best.pt")
            # Patience resets only on MEANINGFUL improvement: micro-creep at the
            # measurement-noise level (<0.02 dB) must not keep training alive.
            if val > significant + MIN_IMPROVEMENT_DB:
                significant = val
                best_epoch = epoch + 1
            completed = epoch + 1
            stopping = patience > 0 and completed - best_epoch >= patience
            _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                          "epoch": completed, "best_val_psnr": best,
                          "best_epoch": best_epoch, "significant_best": significant,
                          "early_stopped": stopping,
                          "optimizer_type": optimizer}, last)
            if stopping:
                print(f"tool{spec.index:02d} early stop at epoch {completed}"
                      f" (best was epoch {best_epoch}, patience {patience})")
                break
    finally:
        writer.close()
    return TrainState(epoch=completed, best_val_psnr=best, resumed_from=resumed_from)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", type=int, required=True, choices=range(12))
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device)
    # Amendment 3: fail fast on unavailable explicitly-requested devices
    try:
        torch.zeros(1, device=device)
    except (RuntimeError, AssertionError) as exc:
        raise SystemExit(f"device '{device}' unavailable on this machine: {exc}") from exc
    spec = TOOL_SPECS[args.tool]
    images = load_images(image_paths(cfg.data.root, "train"), cfg.data.max_train_images)
    val_images = load_images(image_paths(cfg.data.root, "val"), cfg.data.max_train_images or 20)
    t = cfg.tools
    state = train_tool(
        spec=spec, images=images, val_images=val_images,
        out_dir=Path("checkpoints/tools") / cfg.name / f"tool{spec.index:02d}",
        epochs=t.epochs, steps_per_epoch=t.steps_per_epoch, batch_size=t.batch_size,
        lr=t.lr, lr_drop_every=t.lr_drop_every, lr_drop_factor=t.lr_drop_factor,
        momentum=t.momentum, weight_decay=t.weight_decay, grad_clip_norm=t.grad_clip_norm,
        robust_extras=t.robust_extras, optimizer=t.optimizer,
        val_pairs=cfg.data.val_pairs, seed=cfg.seed, device=device,
        log_dir=f"runs/{cfg.name}/tool{spec.index:02d}",
        # 0 on macOS: spawn-based DataLoader workers each get a full pickled COPY
        # of the ~6 GB image list -> 18+ GB on a 16 GB machine (observed swap
        # thrash). On Linux (fork shares memory, e.g. Colab) set
        # RLR_LOADER_WORKERS=8 to make loading parallel.
        loader_workers=int(os.environ.get("RLR_LOADER_WORKERS", "0")),
        # 0 = off; each tool stops when val PSNR hasn't improved for N epochs.
        patience=int(os.environ.get("RLR_PATIENCE", "0")),
    )
    print(f"done: best val PSNR {state.best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
