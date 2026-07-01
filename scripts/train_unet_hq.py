"""Train the HQ U-Net restoration model (the "improve on the paper" upgrade).

A no-BatchNorm U-Net (~0.63M params) trained with L1 on the FULL mixed-degradation
family (the same MixedPairDataset the monolith uses). It is a NEW, additive path:
nothing about the agent, the 12 tools, or the DnCNN monolith changes — this writes
to checkpoints/unet_hq/ and is exposed as a separate "HQ Restore" method.

Reuses train_monolith.py's resumable-Colab discipline (atomic best.pt/last.pt,
status.json heartbeat, train_log.jsonl, NaN-halt, snapshots) so a dead Colab
session resumes cleanly. Adds a live tqdm progress bar, an optional per-epoch
LPIPS readout (the upgrade's headline metric), and a sample before/after montage
each few epochs so quality can be watched climbing.

CLI (Colab A100):
    uv run python scripts/train_unet_hq.py --device cuda --epochs 40 \
        --steps-per-epoch 2000 --width 48 --loss l1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for train_monolith helpers
from train_monolith import (  # noqa: E402  (reuse, don't duplicate)
    MIN_IMPROVEMENT_DB,
    _SNAPSHOT_EVERY,
    _append_jsonl,
    _atomic_save,
    _atomic_save_json,
    _build_val_set,
    _utc_iso,
    _validate,
    charbonnier_loss,
)

from rlrestore.baselines.unet_restore import UNetRestore, count_parameters  # noqa: E402
from rlrestore.common.device import get_device  # noqa: E402
from rlrestore.common.seed import set_seed  # noqa: E402
from rlrestore.data.degradations import gaussian_blur, gaussian_noise, jpeg_compress  # noqa: E402
from rlrestore.data.div2k import image_paths  # noqa: E402
from rlrestore.data.mixed_dataset import MixedPairDataset  # noqa: E402
from rlrestore.tools.dataset import _to_tensor  # noqa: E402
from rlrestore.tools.train import load_images  # noqa: E402

SEVERITIES = ("mild", "moderate", "severe")


def _make_loss(kind: str):
    if kind == "mse":
        return lambda p, c: torch.nn.functional.mse_loss(p, c)
    if kind == "charbonnier":
        return lambda p, c: charbonnier_loss(p, c, 1e-3)
    return lambda p, c: torch.nn.functional.l1_loss(p, c)  # default: l1 (smoke winner)


def _try_lpips(device):
    try:
        import lpips  # noqa: PLC0415
        return lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception:
        return None


@torch.no_grad()
def _val_lpips(model, val_set, lp, device, n=48):
    if lp is None:
        return None
    model.eval()
    vals = []
    for clean, degraded, _ in val_set[:n]:
        d = _to_tensor(degraded)[None].to(device)
        r = model(d).clamp(0, 1)
        c = _to_tensor(clean)[None].to(device)
        vals.append(float(lp(r * 2 - 1, c * 2 - 1).item()))
    model.train()
    return float(np.mean(vals)) if vals else None


@torch.no_grad()
def _save_montage(model, val_images, out_path, device, n=3, seed=123):
    """clean | degraded(moderate) | restored, for a few full crops — visual monitor."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    cs = 256
    rows = []
    for k in range(min(n, len(val_images))):
        a = val_images[k]
        a = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a
        y0 = int(rng.integers(0, max(1, a.shape[0] - cs)))
        x0 = int(rng.integers(0, max(1, a.shape[1] - cs)))
        clean = np.ascontiguousarray(a[y0 : y0 + cs, x0 : x0 + cs])
        deg = np.clip(jpeg_compress(gaussian_noise(gaussian_blur(clean, 1.2), 22.0, rng), 42), 0, 1)
        t = torch.from_numpy(deg).permute(2, 0, 1)[None].to(device)
        rest = model(t).clamp(0, 1).cpu()[0].permute(1, 2, 0).numpy()
        rows.append((clean, deg, rest))
    model.train()
    pad, lab = 8, 18
    h = rows[0][0].shape[0]
    mont = Image.new("RGB", (3 * cs + 4 * pad, len(rows) * (h + lab + pad) + pad), (18, 20, 28))
    dr = ImageDraw.Draw(mont)
    for ri, (clean, deg, rest) in enumerate(rows):
        cy = pad + ri * (h + lab + pad)
        for ci, (im, name) in enumerate([(clean, "clean"), (deg, "degraded"), (rest, "HQ restored")]):
            mont.paste(Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)), (pad + ci * (cs + pad), cy + lab))
            dr.text((pad + ci * (cs + pad), cy), name, fill=(235, 235, 245))
    mont.save(out_path)


def run_unet_training(*, images, val_images, out_dir, device, args) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    session_id = str(uuid.uuid4())[:8]
    wall_start = time.monotonic()

    model = UNetRestore(width=args.width).to(device)
    n_params = count_parameters(model)
    loss_fn = _make_loss(args.loss)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lp = _try_lpips(device)

    val_set = _build_val_set(val_images, SEVERITIES, args.val_pairs, args.seed + 10_000)

    start_epoch, best, best_epoch, significant, wall_prev = 0, -float("inf"), 0, -float("inf"), 0.0
    last = out_dir / "last.pt"
    if not args.no_resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]; best = ck["best_val"]; best_epoch = ck.get("best_epoch", start_epoch)
        significant = ck.get("significant_best", best); wall_prev = float(ck.get("wall_sec_total", 0.0))
        if ck.get("early_stopped") or start_epoch >= args.epochs:
            print(f"unet_hq already finished at epoch {start_epoch}; nothing to do")
            return {"epoch": start_epoch, "best_val_gain": best, "best_epoch": best_epoch}
        print(f"resumed unet_hq from epoch {start_epoch} (best {best:.3f} dB)")

    print(f"UNetRestore width={args.width} ({n_params:,} params) loss={args.loss} "
          f"lpips={'on' if lp else 'off'} device={device}")
    completed, early_stopped = start_epoch, False
    for epoch in range(start_epoch, args.epochs):
        ep_start = time.monotonic()
        cur_lr = args.lr * (args.lr_drop_factor ** (epoch // args.lr_drop_every))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        ds = MixedPairDataset(images, seed=args.seed + epoch,
                              length=args.steps_per_epoch * args.batch, severities=SEVERITIES)
        dl = DataLoader(ds, batch_size=args.batch,
                        num_workers=int(os.environ.get("RLR_LOADER_WORKERS", "0")))
        model.train()
        running = 0.0
        pbar = tqdm(dl, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False, dynamic_ncols=True)
        for i, (degraded, clean) in enumerate(pbar):
            degraded, clean = degraded.to(device), clean.to(device)
            loss = loss_fn(model(degraded), clean)
            if not torch.isfinite(loss):
                _atomic_save_json({"epoch": epoch, "nan_halt": True, "session_id": session_id,
                                   "updated_utc": _utc_iso()}, out_dir / "status.json")
                raise RuntimeError(f"NaN/inf loss at epoch {epoch} — halting WITHOUT checkpointing")
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
            running += loss.item()
            if i % 25 == 0:
                pbar.set_postfix(loss=f"{running / (i + 1):.4f}", lr=f"{cur_lr:.1e}")

        overall_gain, class_gains = _validate(model, val_set, SEVERITIES, device)
        val_lpips = _val_lpips(model, val_set, lp, device)
        completed = epoch + 1
        total_wall = (time.monotonic() - wall_start) + wall_prev
        msg = (f"epoch {completed}/{args.epochs}  val_gain={overall_gain:.3f} dB  "
               + " ".join(f"{s}={class_gains[s]:.2f}" for s in SEVERITIES))
        if val_lpips is not None:
            msg += f"  LPIPS={val_lpips:.4f}"
        print(msg, flush=True)

        if overall_gain > best:
            best = overall_gain
            _atomic_save({"model": model.state_dict(), "val_psnr": overall_gain,
                          "val_lpips": val_lpips, "width": args.width, "loss": args.loss},
                         out_dir / "best.pt")
        if overall_gain > significant + MIN_IMPROVEMENT_DB:
            significant = overall_gain; best_epoch = completed
        early_stopped = args.patience > 0 and completed - best_epoch >= args.patience

        _atomic_save_json({"epoch": completed, "last_val_psnr": overall_gain, "val_lpips": val_lpips,
                           "per_class": class_gains, "lr": cur_lr, "wall_sec_total": total_wall,
                           "last_chunk_sec": time.monotonic() - ep_start, "session_id": session_id,
                           "updated_utc": _utc_iso()}, out_dir / "status.json")
        _append_jsonl({"epoch": completed, "val_gain_db": overall_gain, "val_lpips": val_lpips,
                       "per_class_gain_db": class_gains, "lr": cur_lr, "best_val": best,
                       "best_epoch": best_epoch, "wall_sec_total": total_wall}, out_dir / "train_log.jsonl")
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": completed,
                      "best_val": best, "best_epoch": best_epoch, "significant_best": significant,
                      "early_stopped": early_stopped, "wall_sec_total": total_wall}, last)
        if completed % _SNAPSHOT_EVERY == 0 or completed == args.epochs:
            try:
                _save_montage(model, val_images, out_dir / f"sample_epoch_{completed}.png", device)
            except Exception as e:  # montage is monitoring-only; never kill training for it
                print(f"  (montage skipped: {e})")
        if early_stopped:
            print(f"early stop at epoch {completed} (best epoch {best_epoch}, patience {args.patience})")
            break

    print(f"done: best val gain {best:.3f} dB @ epoch {best_epoch} ({n_params:,} params)")
    return {"epoch": completed, "best_val_gain": best, "best_epoch": best_epoch}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the HQ U-Net restoration model (additive upgrade)")
    ap.add_argument("--data-root", default="data/DIV2K")
    ap.add_argument("--out-dir", default="checkpoints/unet_hq")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-drop-every", type=int, default=15)
    ap.add_argument("--lr-drop-factor", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--loss", choices=["l1", "charbonnier", "mse"], default="l1")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-pairs", type=int, default=150)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    device = get_device(args.device) if args.device else get_device()
    try:
        torch.zeros(1, device=device)
    except (RuntimeError, AssertionError) as exc:
        raise SystemExit(f"device '{device}' unavailable: {exc}") from exc

    images = load_images(image_paths(args.data_root, "train"), args.max_train)
    val_images = load_images(image_paths(args.data_root, "val"), args.max_train or 20)
    print(f"Loaded {len(images)} train / {len(val_images)} val images; device={device}")
    run_unet_training(images=images, val_images=val_images, out_dir=args.out_dir, device=device, args=args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
