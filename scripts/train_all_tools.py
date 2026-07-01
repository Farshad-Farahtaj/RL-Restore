"""Train all 12 tools sequentially with skip-if-done and resume.

Usage:
  uv run python scripts/train_all_tools.py --config configs/smoke.yaml          # pipeline check
  uv run python scripts/train_all_tools.py --config configs/full.yaml           # overnight
  uv run python scripts/train_all_tools.py --config configs/full.yaml --dry-run # list jobs
Designed to be interrupted at any time; rerun continues where it stopped.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from rlrestore.common.config import load_config
from rlrestore.tools.specs import TOOL_SPECS


def is_done(cfg, idx: int) -> bool:
    last = Path("checkpoints/tools") / cfg.name / f"tool{idx:02d}" / "last.pt"
    if not last.exists():
        return False
    try:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        return bool(ck.get("early_stopped", False)) or ck["epoch"] >= cfg.tools.epochs
    except Exception:
        print(f"tool{idx:02d}: unreadable last.pt, will retrain")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    jobs = [s.index for s in TOOL_SPECS if not is_done(cfg, s.index)]
    print(f"config={cfg.name}: {12 - len(jobs)}/12 done, {len(jobs)} to train: {jobs}")
    if args.dry_run:
        return
    for idx in jobs:
        print(f"=== tool {idx:02d} ===")
        r = subprocess.run([sys.executable, "-m", "rlrestore.tools.train",
                            "--tool", str(idx), "--config", args.config])
        if r.returncode != 0:
            sys.exit(f"tool {idx:02d} failed (exit {r.returncode}); rerun to resume")
    print("all 12 tools trained")


if __name__ == "__main__":
    main()
