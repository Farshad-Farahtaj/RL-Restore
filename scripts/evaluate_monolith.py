"""Held-out TEST evaluation of the monolithic CNN (P3.E4 / B5).

Sibling of scripts/evaluate_agent.py — it does NOT touch that script. Rebuilds
the byte-identical TEST episodes the agent eval uses (RestoreEnv with the same
seed scheme mild 10000.. / moderate 20000.. / severe 30000.., training_mode=False,
default uint8 path), then scores the monolith with a SINGLE unclamped forward:

    gain = psnr(model(degraded), clean) - psnr(degraded, clean)   per episode

mean per class + unweighted overall. Emits one extra "B5: single end-to-end CNN"
row in the same table format alongside the agent/baseline rows, plus a protocol
block (checkpoint sha256[:16], param count, device, date, per-class, seed scheme).

RestoreEnv needs the 12 tools to construct even though the monolith uses none of
them — its reset() builds the (clean, degraded) pair before any tool is applied,
so we load the real tools purely for pair generation.

This script is reporting, not a gate: it always exits 0.

Usage:
    uv run python scripts/evaluate_monolith.py
        [--checkpoint checkpoints/monolith/monolith/best.pt]
        [--config configs/monolith.yaml] [--tools-name full] [--device cpu]
        [--per-class 500] [--out-dir reports/eval_monolith]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rlrestore.agent.env import RestoreEnv, load_tools
from rlrestore.baselines.monolith import MonoRestoreCNN, count_parameters
from rlrestore.common.config import load_config
from rlrestore.common.device import get_device
from rlrestore.common.metrics import psnr
from rlrestore.data.div2k import image_paths
from rlrestore.tools.train import load_images

SEVERITIES = ("mild", "moderate", "severe")
SEED_BASE = {"mild": 10_000, "moderate": 20_000, "severe": 30_000}


# ─────────────────────────── per-class evaluation ────────────────────────────


@torch.no_grad()
def evaluate_monolith_on_class(
    model: MonoRestoreCNN,
    tools: list[nn.Module],
    images: list[np.ndarray],
    device: torch.device,
    severity: str,
    seeds: list[int],
) -> dict[str, Any]:
    """Mean monolith PSNR gain on one severity class.

    Episodes are fully determined by (images, seeds, severity): each RestoreEnv
    reset() rebuilds the identical (clean, degraded) pair the agent eval scores,
    so the comparison is on byte-identical episodes. Single unclamped forward,
    no chain.

    Returns {"psnr_gain_db", "per_episode_gain_db" (summary stats), "n_episodes"}.
    """
    model.eval()
    gains: list[float] = []
    for seed in seeds:
        env = RestoreEnv(
            images=images,
            tools=tools,
            device=device,
            seed=seed,
            training_mode=False,
            severity=severity,
        )
        obs = env.reset()
        degraded, _ = obs  # current image == the degraded patch before any tool
        clean = env.clean_image
        x = torch.from_numpy(degraded).permute(2, 0, 1).unsqueeze(0).to(device)
        restored = model(x).squeeze(0).permute(1, 2, 0).cpu().numpy()
        gain = psnr(restored, clean) - psnr(degraded, clean)
        gains.append(float(gain))
    mean_gain = float(np.mean(gains)) if gains else 0.0
    return {
        "psnr_gain_db": mean_gain,
        "gain_std_db": float(np.std(gains)) if gains else 0.0,
        "n_episodes": len(seeds),
    }


# ─────────────────────────── report rendering ────────────────────────────────


def render_markdown(
    per_class: dict[str, dict[str, Any]],
    overall_gain: float,
    protocol: dict[str, Any],
) -> str:
    """One-row B5 table (cols = severity classes + overall) plus a protocol block."""
    cells = [f"{per_class[cls]['psnr_gain_db']:.2f}" for cls in SEVERITIES]
    n_params = protocol["param_count"]
    b5_row = (
        f"| B5: single end-to-end CNN ({n_params:,} params) | "
        + " | ".join(cells)
        + f" | {overall_gain:.2f} |"
    )
    lines = [
        "# Monolithic-CNN TEST evaluation (DIV2K 751-800)",
        "",
        "PSNR gain in dB over the degraded input (single unclamped forward, no chain).",
        "Episodes are byte-identical to scripts/evaluate_agent.py (same seed scheme).",
        "",
        "| Method | mild | moderate | severe | overall |",
        "|---|---|---|---|---|",
        b5_row,
        "",
        "## Protocol",
        "",
        f"- {protocol['n_per_class']} episodes per class on the "
        f"{protocol['test_split']} TEST split (never seen during training).",
        f"- Seeds: {protocol['seeds_scheme']} — deterministic and disjoint; "
        "identical episodes to the agent evaluation per class.",
        f"- Monolith: {n_params:,} params, single forward, output unclamped "
        "(matches metrics.psnr convention).",
        f"- Checkpoint sha256[:16] = `{protocol['checkpoint_sha256']}`, "
        f"device = {protocol['device']}, date = {protocol['date']}.",
        f"- Image dtype: {protocol['image_dtype']}.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────── CLI ──────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Held-out TEST evaluation: monolithic CNN (B5)"
    )
    p.add_argument(
        "--checkpoint",
        default="checkpoints/monolith/monolith/best.pt",
        help="MonoRestoreCNN best.pt checkpoint ({model, val_psnr})",
    )
    p.add_argument("--config", default="configs/monolith.yaml")
    p.add_argument(
        "--tools-name",
        default="full",
        help="checkpoints/tools/<name> to load the 12 tools (pair generation only)",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--per-class", type=int, default=500, help="Episodes per severity class"
    )
    p.add_argument("--out-dir", default="reports/eval_monolith")
    p.add_argument(
        "--float-images",
        action="store_true",
        help="Diagnostic: convert TEST images to float32 [0,1] before episode "
        "synthesis. Default OFF for training parity (uint8 path).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    device = get_device(args.device) if args.device else get_device()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load TEST images (same data path as training) ───────────────────────
    test_images: list[np.ndarray] = load_images(image_paths(cfg.data.root, "test"))
    if args.float_images:
        test_images = [im.astype(np.float32) / 255.0 for im in test_images]
        print("diagnostic: TEST images converted to float32 [0,1]")
    print(f"Loaded {len(test_images)} TEST images (DIV2K 751-800)")

    # ── Tools (pair generation only) ────────────────────────────────────────
    tools = load_tools(args.tools_name, device)
    print(f"Loaded {len(tools)} tools from checkpoints/tools/{args.tools_name}")

    # ── Monolith checkpoint ─────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    depth = cfg.monolith.depth if cfg.monolith else 14
    width = cfg.monolith.width if cfg.monolith else 64
    model = MonoRestoreCNN(depth=depth, width=width).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = count_parameters(model)
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16]
    print(f"Loaded monolith {ckpt_path} ({n_params:,} params, sha256[:16]={ckpt_sha})")

    # ── Per-class evaluation ────────────────────────────────────────────────
    per_class: dict[str, dict[str, Any]] = {}
    for severity in SEVERITIES:
        base = SEED_BASE[severity]
        seeds = list(range(base, base + args.per_class))
        print(f"[{severity}] {args.per_class} episodes, seeds {base}..{seeds[-1]} ...")
        per_class[severity] = evaluate_monolith_on_class(
            model=model,
            tools=tools,
            images=test_images,
            device=device,
            severity=severity,
            seeds=seeds,
        )
        print(f"[{severity}] monolith {per_class[severity]['psnr_gain_db']:+.2f} dB")

    overall_gain = float(np.mean([per_class[c]["psnr_gain_db"] for c in SEVERITIES]))
    protocol = {
        "n_per_class": args.per_class,
        "seeds_scheme": "mild 10000.., moderate 20000.., severe 30000..",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "param_count": n_params,
        "test_split": "DIV2K 751-800",
        "image_dtype": "float32[0,1]" if args.float_images else "uint8 (training parity)",
        "device": str(device),
        "date": datetime.now(timezone.utc).isoformat(),
    }

    # ── Write reports ───────────────────────────────────────────────────────
    result = {
        "per_class": per_class,
        "overall": {"monolith_psnr_gain_db": overall_gain},
        "protocol": protocol,
    }
    json_path = out_dir / "monolith_eval_test.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {json_path}")

    md = render_markdown(per_class, overall_gain, protocol)
    md_path = out_dir / "monolith_eval_test.md"
    md_path.write_text(md)
    print(f"wrote {md_path}")

    print()
    print(
        f"| B5: single end-to-end CNN ({n_params:,} params) | "
        + " | ".join(f"{per_class[c]['psnr_gain_db']:.2f}" for c in SEVERITIES)
        + f" | {overall_gain:.2f} |"
    )
    return 0  # reporting, not a gate


if __name__ == "__main__":
    sys.exit(main())
