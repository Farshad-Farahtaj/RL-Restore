"""Joint fine-tuning of the toolbox — RL-Restore Algorithm 1 (P3.E5).

Drives :func:`rlrestore.agent.joint_finetune.run_joint_finetune`: a FROZEN
trained agent routes the 12 trainable tools, the chain's final MSE vs clean
fine-tunes the tools (agent untouched), per-tool gradient averaged by use-count.
Originals in ``checkpoints/tools/<in-tools-name>/`` are NEVER overwritten; the
fine-tuned toolbox lands in ``checkpoints/tools/<out-tools-name>/`` as a drop-in
for ``evaluate_agent.py --tools-name <out-tools-name>``.

Resumable-Colab discipline (mirrors scripts/train_monolith.py + agent/runner.py):
atomic checkpoints, status.json heartbeat, train_log.jsonl per chunk, system-val
early stopping, NaN-halt without overwriting the best tools, snapshots, resume.

CLI::

    uv run python scripts/train_joint_finetune.py
        [--agent-ckpt checkpoints/agent_full_v2/agent_net.pt]
        [--config configs/full.yaml] [--in-tools-name full]
        [--out-tools-name full_ft] [--lr 1e-4] [--iters 200000]
        [--batch 32] [--chunk 5000] [--severities mild,moderate,severe]
        [--device cpu] [--out-dir checkpoints/joint_ft] [--no-resume]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from rlrestore.agent.joint_finetune import JointFTState, run_joint_finetune
from rlrestore.agent.net import AgentNet
from rlrestore.common.config import load_config
from rlrestore.common.device import get_device
from rlrestore.data.div2k import image_paths
from rlrestore.tools.train import load_images


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint fine-tuning of the toolbox (RL-Restore Algorithm 1)"
    )
    p.add_argument(
        "--agent-ckpt",
        default="checkpoints/agent_full_v2/agent_net.pt",
        help="Frozen AgentNet state-dict checkpoint",
    )
    p.add_argument(
        "--config",
        default="configs/full.yaml",
        help="Config yaml (used for the data root and seed)",
    )
    p.add_argument(
        "--in-tools-name",
        default="full",
        help="checkpoints/tools/<name> to initialise the tools from (never written)",
    )
    p.add_argument(
        "--out-tools-name",
        default="full_ft",
        help="checkpoints/tools/<name> to write the fine-tuned tools to",
    )
    p.add_argument("--lr", type=float, default=1e-4, help="Adam lr (paper alpha)")
    p.add_argument(
        "--iters", type=int, default=200_000, help="Iteration cap (paper 2e5)"
    )
    p.add_argument(
        "--batch", type=int, default=32, help='Images (chains) per iter ("M=64")'
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=5_000,
        help="Iterations per validate+checkpoint chunk",
    )
    p.add_argument(
        "--severities",
        default="mild,moderate,severe",
        help="Comma-separated severity classes sampled uniformly",
    )
    p.add_argument("--device", default=None, help="Torch device (default: auto)")
    p.add_argument(
        "--out-dir",
        default="checkpoints/joint_ft",
        help="Directory for optimiser checkpoint, logs, status, snapshots",
    )
    p.add_argument(
        "--val-pairs",
        type=int,
        default=128,
        help="Fixed system-val episodes across severities (default 128)",
    )
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.5,
        help="Global grad-norm clip after use-count averaging (default 0.5)",
    )
    p.add_argument(
        "--patience-chunks",
        type=int,
        default=6,
        help="Early-stop patience in chunks (default 6)",
    )
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    device = get_device(args.device) if args.device else get_device()
    if device.type == "mps":
        # The agent's LSTM crashes on MPS (native MPSNDArray assertion). It only
        # runs in no_grad for chain selection, but PyTorch still routes it
        # through the MPS LSTM kernel, so fall back to CPU like evaluate_agent.py.
        print("note: LSTM crashes on MPS — falling back to cpu")
        device = torch.device("cpu")
    try:
        torch.zeros(1, device=device)
    except (RuntimeError, AssertionError) as exc:
        raise SystemExit(f"device '{device}' unavailable: {exc}") from exc

    cfg = load_config(args.config)
    severities = tuple(s.strip() for s in args.severities.split(",") if s.strip())

    # Same data path as training/eval (scripts/train_agent.py, evaluate_agent.py).
    images = load_images(image_paths(cfg.data.root, "train"), cfg.data.max_train_images)
    val_images = load_images(image_paths(cfg.data.root, "val"))
    print(
        f"Loaded {len(images)} train / {len(val_images)} val images; "
        f"device={device}; severities={severities}"
    )

    # Frozen agent.
    agent = AgentNet().to(device)
    agent.load_state_dict(
        torch.load(args.agent_ckpt, map_location=device, weights_only=True)
    )
    print(f"Loaded frozen agent from {args.agent_ckpt}")

    if args.out_tools_name == args.in_tools_name:
        raise SystemExit(
            f"--out-tools-name must differ from --in-tools-name "
            f"(both {args.in_tools_name!r}); originals must never be overwritten"
        )

    state: JointFTState = run_joint_finetune(
        agent=agent,
        images=images,
        val_images=val_images,
        out_dir=Path(args.out_dir),
        out_tools_name=args.out_tools_name,
        seed=cfg.seed,
        device=device,
        iters=args.iters,
        batch_size=args.batch,
        chunk_size=args.chunk,
        lr=args.lr,
        grad_clip_norm=args.grad_clip_norm,
        severities=severities,
        in_tools_name=args.in_tools_name,
        val_pairs=args.val_pairs,
        patience_chunks=args.patience_chunks,
        resume=not args.no_resume,
    )
    print(
        f"final: {state.iters} iters, best system val gain "
        f"{state.best_system_gain:+.3f} dB @ chunk {state.best_chunk}, "
        f"early_stopped={state.early_stopped}"
    )
    print(
        f"fine-tuned tools written to checkpoints/tools/{args.out_tools_name}/ "
        f"(originals in {args.in_tools_name}/ untouched)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
