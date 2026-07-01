"""CLI for chunked DQN agent training.

Usage examples
--------------
  uv run python scripts/train_agent.py --config configs/smoke.yaml --smoke
  uv run python scripts/train_agent.py --config configs/full.yaml --device mps

Flags
-----
--config   Path to yaml config file (must contain an `agent:` section).
--device   torch device string; default: auto-detect (cuda > mps > cpu).
--out-dir  Checkpoint / log directory. Default: checkpoints/agent/<cfg.name>.
--smoke    After training completes, run stability_spec §2 smoke checks
           (S1-S7) on the final metrics; exit nonzero listing failures.
--no-resume  Start fresh even if a checkpoint exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from rlrestore.agent.env import load_tools
from rlrestore.agent.runner import run_training
from rlrestore.agent.train import check_smoke_thresholds
from rlrestore.common.config import load_config
from rlrestore.common.device import get_device
from rlrestore.data.div2k import image_paths
from rlrestore.tools.train import load_images


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL-Restore DQN agent")
    p.add_argument("--config", required=True, help="Path to yaml config file")
    p.add_argument("--device", default=None, help="Torch device (default: auto)")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: checkpoints/agent/<name>)",
    )
    p.add_argument(
        "--tools-name",
        default="full",
        help="checkpoints/tools/<name> to load the 12 tools from (default: full — "
        "the real toolbox; smoke runs must exercise real tools too)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke gate (S1-S7) on final metrics; exit nonzero on failure",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh even if a checkpoint exists",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg = load_config(args.config)
    assert cfg.agent is not None, (
        f"Config '{args.config}' has no `agent:` section. Add one before training."
    )

    device = get_device(args.device) if args.device else get_device()

    out_dir = Path(
        args.out_dir if args.out_dir else f"checkpoints/agent/{cfg.name}"
    )

    # ── Load data ──────────────────────────────────────────────────────────
    data_root = cfg.data.root
    train_paths = image_paths(data_root, "train")
    val_paths = image_paths(data_root, "val")

    limit = cfg.data.max_train_images or 0
    images: list[np.ndarray] = load_images(train_paths, limit=int(limit))
    val_images: list[np.ndarray] = load_images(val_paths)

    print(f"Loaded {len(images)} train images, {len(val_images)} val images")

    # ── Load tools ─────────────────────────────────────────────────────────
    tools = load_tools(args.tools_name, device)
    print(f"Loaded {len(tools)} tools")

    # ── Run training ───────────────────────────────────────────────────────
    resume = not args.no_resume
    final_metrics = run_training(
        cfg=cfg,
        tools=tools,
        images=images,
        val_images=val_images,
        device=device,
        out_dir=out_dir,
        resume=resume,
    )

    print(f"Training done — env_step={final_metrics.get('env_step')}")
    print(f"  val_psnr_gain_db = {final_metrics.get('val_psnr_gain_db', 'N/A'):.4f}")
    print(f"  mean_ep_reward   = {final_metrics.get('mean_ep_reward', 'N/A'):.4f}")

    # ── Smoke gate ─────────────────────────────────────────────────────────
    if args.smoke:
        failures = check_smoke_thresholds(final_metrics)
        if failures:
            print("\nSMOKE GATE FAILED:")
            for f in failures:
                print(f"  {f}")
            return 1
        print("\nSMOKE GATE PASSED (S1-S7)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
