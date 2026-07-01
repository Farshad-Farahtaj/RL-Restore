"""Held-out TEST evaluation of the trained DQN agent vs baselines B0-B4.

Protocol (P3.E1)
----------------
For each severity class (mild / moderate / severe):
  - Build a fixed, deterministic, per-class-disjoint episode set from the
    DIV2K TEST split (images 751-800): mild seeds 10_000.., moderate 20_000..,
    severe 30_000.. (training_mode=False).
  - Greedy agent metrics via rlrestore.agent.train.evaluate (gain, action
    histogram, entropy, stop rates) plus mean chain length and top-3 tools.
  - Baselines B0-B4 via rlrestore.agent.train.compute_baselines on the SAME
    seeds (identical episodes).
  - Example toolchain strips: [degraded | per tool step ... | clean] PNGs.

Data path note: images are fed through the SAME load_images path the training
CLI (scripts/train_agent.py) uses, so the evaluation episode distribution
matches the distribution the checkpoint was trained on.

Outputs (under --out-dir, default reports/eval_test):
  agent_eval_test.json — per_class / overall / protocol blocks
  agent_eval_test.md   — readable comparison table
  strips/<class>_<seed>.png — example toolchain strips

This script is reporting, not a gate: it always exits 0.

Usage:
    uv run python scripts/evaluate_agent.py [--checkpoint ...] [--config ...]
        [--tools-name full] [--device cpu] [--per-class 500]
        [--n-random-chains 100] [--strips 6] [--out-dir reports/eval_test]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from rlrestore.agent.env import RestoreEnv, load_tools  # noqa: E402
from rlrestore.agent.net import AgentNet  # noqa: E402
from rlrestore.agent.train import compute_baselines, evaluate  # noqa: E402
from rlrestore.common.config import load_config  # noqa: E402
from rlrestore.common.device import get_device  # noqa: E402
from rlrestore.common.metrics import psnr  # noqa: E402
from rlrestore.data.div2k import image_paths  # noqa: E402
from rlrestore.tools.specs import TOOL_SPECS  # noqa: E402
from rlrestore.tools.train import load_images  # noqa: E402

# ─────────────────────────── protocol constants ──────────────────────────────

SEVERITIES = ("mild", "moderate", "severe")
SEED_BASE = {"mild": 10_000, "moderate": 20_000, "severe": 30_000}
_STOP = 12

# Tool naming derived from TOOL_SPECS: indices 0-3 deblur, 4-7 denoise,
# 8-11 dejpeg; band 1 (weakest damage) -> band 4 (strongest) within each kind.
_KIND_LABEL = {"blur": "deblur", "noise": "denoise", "jpeg": "dejpeg"}


def tool_name(idx: int) -> str:
    """Human-readable tool name, e.g. tool07 -> 'denoise4 (37.5-50.0)'."""
    spec = TOOL_SPECS[idx]
    band = idx % 4 + 1
    return f"{_KIND_LABEL[spec.kind]}{band} ({spec.lo:g}-{spec.hi:g})"


def tool_name_short(idx: int) -> str:
    """Short tool name for figure titles, e.g. tool07 -> 'denoise4'."""
    return f"{_KIND_LABEL[TOOL_SPECS[idx].kind]}{idx % 4 + 1}"


# ─────────────────────────── toolchain strips ────────────────────────────────


def _greedy_frames(
    net: AgentNet,
    env: RestoreEnv,
    device: torch.device,
) -> list[tuple[str, np.ndarray, float | None]]:
    """Greedy rollout capturing (title, image, psnr) panels for one episode.

    Panels: degraded start, one per tool action taken (STOP noted on the
    preceding panel), then the clean ground truth.
    """
    img_hwc, prev_oh = env.reset()
    clean = env.clean_image
    panels: list[tuple[str, np.ndarray, float | None]] = [
        ("degraded", img_hwc.copy(), psnr(img_hwc, clean))
    ]

    hx: tuple[Tensor, Tensor] | None = None
    net.eval()
    with torch.no_grad():
        while True:
            img_nchw = (
                torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            prev_oh_t = torch.from_numpy(prev_oh).unsqueeze(0).to(device)
            q, hx = net.forward_step(img_nchw, prev_oh_t, hx)
            action = int(q.argmax(dim=-1).item())

            obs_next, _reward, terminated, info = env.step(action)

            if action == _STOP:
                title, img, p = panels[-1]
                panels[-1] = (f"{title} +STOP", img, p)
                break

            img_hwc, prev_oh = obs_next
            panels.append(
                (tool_name_short(action), img_hwc.copy(), float(info["current_psnr"]))
            )
            if terminated:
                break

    panels.append(("ground truth", clean.copy(), None))
    return panels


def _upscale3x(img: np.ndarray) -> np.ndarray:
    """3x nearest-neighbour upscale of a HWC image for strip visibility."""
    return np.repeat(np.repeat(img, 3, axis=0), 3, axis=1)


def save_strip(
    panels: list[tuple[str, np.ndarray, float | None]],
    out_path: Path,
    suptitle: str,
) -> None:
    """Save one horizontal toolchain strip PNG (63x63 panels upscaled 3x)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0), squeeze=False)
    for ax, (title, img, p) in zip(axes[0], panels):
        ax.imshow(_upscale3x(np.clip(img, 0.0, 1.0)), interpolation="nearest")
        label = title if p is None else f"{title}\n{p:.2f} dB"
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)


# ─────────────────────────── per-class evaluation ────────────────────────────


def evaluate_on_class(
    net: AgentNet,
    tools: list[nn.Module],
    images: list[np.ndarray],
    device: torch.device,
    severity: str,
    seeds: list[int],
    n_random_chains: int = 100,
    n_strips: int = 0,
    strips_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate agent + baselines B0-B4 on one severity class.

    Episodes are fully determined by (images, seeds, severity): the agent
    eval, the baselines, and the strips all rebuild identical episodes from
    the same seeds.

    Parameters
    ----------
    net             : Trained AgentNet (eval mode assumed).
    tools           : Exactly 12 frozen tool nn.Modules.
    images          : Image pool (TEST split in production).
    device          : Torch device.
    severity        : "mild" | "moderate" | "severe".
    seeds           : One deterministic seed per episode.
    n_random_chains : Random chains per episode for B2.
    n_strips        : Number of example toolchain strips to render.
    strips_dir      : Output directory for strips (required if n_strips > 0).

    Returns
    -------
    JSON-able dict with keys "agent" and "baselines".
    """
    envs = [
        RestoreEnv(
            images=images,
            tools=tools,
            device=device,
            seed=s,
            training_mode=False,
            severity=severity,
        )
        for s in seeds
    ]

    # ── Agent: greedy rollout metrics ───────────────────────────────────────
    agent_metrics = evaluate(net=net, envs=envs, device=device)

    # Chain length = number of tool actions (0-3). STOP at t1/t2/t3 means
    # 0/1/2 tool actions; episodes without STOP run the full 3 tool steps.
    sr1 = agent_metrics["stop_rate_t1"]
    sr2 = agent_metrics["stop_rate_t2"]
    sr3 = agent_metrics["stop_rate_t3"]
    mean_chain_length = 1.0 * sr2 + 2.0 * sr3 + 3.0 * (1.0 - sr1 - sr2 - sr3)

    tool_hist = agent_metrics["action_hist"][:12]
    top3 = sorted(range(12), key=lambda i: tool_hist[i], reverse=True)[:3]
    top_tools = [
        {"index": i, "name": tool_name(i), "count": int(tool_hist[i])} for i in top3
    ]

    agent: dict[str, Any] = {
        "psnr_gain_db": agent_metrics["val_psnr_gain_db"],
        "action_hist": agent_metrics["action_hist"],
        "entropy": agent_metrics["entropy"],
        "stop_rate_t1": sr1,
        "stop_rate_t2": sr2,
        "stop_rate_t3": sr3,
        "mean_max_q": agent_metrics["mean_max_q"],
        "mean_chain_length": float(mean_chain_length),
        "top_tools": top_tools,
        "n_episodes": len(seeds),
    }

    # ── Baselines on the SAME seeds (identical episodes) ────────────────────
    baselines = compute_baselines(
        tools=tools,
        images=images,
        device=device,
        seeds=seeds,
        n=len(seeds),
        n_random_chains=n_random_chains,
        severity=severity,
    )

    # ── Example toolchain strips ────────────────────────────────────────────
    if n_strips > 0:
        assert strips_dir is not None, "strips_dir required when n_strips > 0"
        for seed in seeds[:n_strips]:
            env = RestoreEnv(
                images=images,
                tools=tools,
                device=device,
                seed=seed,
                training_mode=False,
                severity=severity,
            )
            panels = _greedy_frames(net, env, device)
            start_psnr = panels[0][2] or 0.0
            tool_panels = [p for p in panels[1:-1] if p[2] is not None]
            end_psnr = tool_panels[-1][2] if tool_panels else start_psnr
            gain = float(end_psnr) - float(start_psnr)
            save_strip(
                panels,
                strips_dir / f"{severity}_{seed}.png",
                suptitle=f"{severity} | seed {seed} | agent gain {gain:+.2f} dB",
            )

    return {"agent": agent, "baselines": baselines}


# ─────────────────────────── report rendering ────────────────────────────────


def _overall(per_class: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Unweighted means across the three severity classes."""

    def mean(path: list[str]) -> float:
        vals = []
        for cls in SEVERITIES:
            node: Any = per_class[cls]
            for key in path:
                node = node[key]
            vals.append(float(node))
        return float(np.mean(vals))

    return {
        "agent_psnr_gain_db": mean(["agent", "psnr_gain_db"]),
        "agent_mean_chain_length": mean(["agent", "mean_chain_length"]),
        "agent_stop_rate_t1": mean(["agent", "stop_rate_t1"]),
        "agent_stop_rate_t2": mean(["agent", "stop_rate_t2"]),
        "agent_stop_rate_t3": mean(["agent", "stop_rate_t3"]),
        "B0_do_nothing": mean(["baselines", "B0_do_nothing"]),
        "B1_best_single_tool": mean(["baselines", "B1_best_single_tool"]),
        "B2_random_chain_mean": mean(["baselines", "B2_random_chain_mean"]),
        "B3_greedy_oracle": mean(["baselines", "B3_greedy_oracle"]),
        "B4_oracle_with_stop": mean(["baselines", "B4_oracle_with_stop"]),
    }


def render_markdown(
    per_class: dict[str, dict[str, Any]],
    overall: dict[str, float],
    protocol: dict[str, Any],
) -> str:
    """Render the comparison table (rows = methods, cols = severity classes)."""

    def row(label: str, fn) -> str:
        cells = [f"{fn(per_class[cls]):.2f}" for cls in SEVERITIES]
        return f"| {label} | " + " | ".join(cells) + f" | {fn(None):.2f} |"

    # B1 row label carries the per-class best tool index/name (may differ
    # per class), so build it cell by cell instead.
    b1_cells = []
    for cls in SEVERITIES:
        gains = per_class[cls]["baselines"]["B1_tool_gains"]
        best = int(np.argmax(gains))
        b1_cells.append(f"{gains[best]:.2f} (tool{best:02d} {tool_name_short(best)})")
    b1_row = (
        "| B1 best single tool | "
        + " | ".join(b1_cells)
        + f" | {overall['B1_best_single_tool']:.2f} |"
    )

    def b2_cell(cls: str) -> str:
        b = per_class[cls]["baselines"]
        return f"{b['B2_random_chain_mean']:.2f} ± {b['B2_random_chain_std']:.2f}"

    def stop_cell(cls: str) -> str:
        a = per_class[cls]["agent"]
        return (
            f"{a['stop_rate_t1']:.2f} / {a['stop_rate_t2']:.2f} / "
            f"{a['stop_rate_t3']:.2f}"
        )

    lines = [
        "# Agent TEST evaluation (DIV2K 751-800)",
        "",
        "PSNR gain in dB over the degraded input (higher is better).",
        "",
        "| Method | mild | moderate | severe | overall |",
        "|---|---|---|---|---|",
        row(
            "B0 do nothing",
            lambda d: overall["B0_do_nothing"]
            if d is None
            else d["baselines"]["B0_do_nothing"],
        ),
        "| B2 random 3-chain | "
        + " | ".join(b2_cell(cls) for cls in SEVERITIES)
        + f" | {overall['B2_random_chain_mean']:.2f} |",
        b1_row,
        row(
            "Agent (greedy DQN)",
            lambda d: overall["agent_psnr_gain_db"]
            if d is None
            else d["agent"]["psnr_gain_db"],
        ),
        row(
            "B3 greedy oracle",
            lambda d: overall["B3_greedy_oracle"]
            if d is None
            else d["baselines"]["B3_greedy_oracle"],
        ),
        row(
            "B4 oracle + STOP",
            lambda d: overall["B4_oracle_with_stop"]
            if d is None
            else d["baselines"]["B4_oracle_with_stop"],
        ),
        "| Agent stop rate t1/t2/t3 | "
        + " | ".join(stop_cell(cls) for cls in SEVERITIES)
        + f" | {overall['agent_stop_rate_t1']:.2f} / "
        + f"{overall['agent_stop_rate_t2']:.2f} / "
        + f"{overall['agent_stop_rate_t3']:.2f} |",
        row(
            "Agent mean chain length",
            lambda d: overall["agent_mean_chain_length"]
            if d is None
            else d["agent"]["mean_chain_length"],
        ),
        "",
        "## Protocol",
        "",
        f"- {protocol['n_per_class']} episodes per class on the "
        f"{protocol['test_split']} TEST split (never seen by tools or agent).",
        f"- Seeds: {protocol['seeds_scheme']} — deterministic and disjoint; "
        "agent and baselines share identical episodes per class.",
        f"- B2 uses {protocol['n_random_chains']} random 3-chains per episode.",
        f"- Checkpoint sha256[:16] = `{protocol['checkpoint_sha256']}`, "
        f"device = {protocol['device']}, date = {protocol['date']}.",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────── CLI ──────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Held-out TEST evaluation: trained agent vs baselines B0-B4"
    )
    p.add_argument(
        "--checkpoint",
        default="checkpoints/agent_full/agent_net.pt",
        help="AgentNet state-dict checkpoint",
    )
    p.add_argument(
        "--config",
        default="configs/full.yaml",
        help="Config yaml (used for the data root)",
    )
    p.add_argument(
        "--tools-name",
        default="full",
        help="checkpoints/tools/<name> to load the 12 tools from",
    )
    p.add_argument("--device", default=None, help="Torch device (default: auto)")
    p.add_argument(
        "--per-class",
        type=int,
        default=500,
        help="Episodes per severity class (default 500)",
    )
    p.add_argument(
        "--n-random-chains",
        type=int,
        default=100,
        help="Random 3-chains per episode for B2 (default 100)",
    )
    p.add_argument(
        "--strips",
        type=int,
        default=6,
        help="Example toolchain strips per class (default 6)",
    )
    p.add_argument(
        "--out-dir",
        default="reports/eval_test",
        help="Output directory (default reports/eval_test)",
    )
    p.add_argument(
        "--float-images",
        action="store_true",
        help="Diagnostic: convert TEST images to float32 [0,1] before episode "
        "synthesis. Default OFF for training parity: load_images returns uint8 "
        "and make_training_pair clips raw values to [0,1], so the default path "
        "saturates clean patches to ~white exactly as during training.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    device = get_device(args.device) if args.device else get_device()
    if device.type == "mps":
        # infra_spec §6 correction: PyTorch LSTM crashes on MPS (native
        # MPSNDArray assertion). Evaluation falls back to CPU.
        print("note: LSTM crashes on MPS — falling back to cpu for evaluation")
        device = torch.device("cpu")

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load TEST images (same data path as training: scripts/train_agent.py)
    test_paths = image_paths(cfg.data.root, "test")
    test_images: list[np.ndarray] = load_images(test_paths)
    if args.float_images:
        test_images = [im.astype(np.float32) / 255.0 for im in test_images]
        print("diagnostic: TEST images converted to float32 [0,1]")
    print(f"Loaded {len(test_images)} TEST images (DIV2K 751-800)")

    # ── Load tools and agent ────────────────────────────────────────────────
    tools = load_tools(args.tools_name, device)
    print(f"Loaded {len(tools)} tools from checkpoints/tools/{args.tools_name}")

    ckpt_path = Path(args.checkpoint)
    net = AgentNet().to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    net.eval()
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16]
    print(f"Loaded agent {ckpt_path} (sha256[:16]={ckpt_sha})")

    # ── Per-class evaluation ────────────────────────────────────────────────
    per_class: dict[str, dict[str, Any]] = {}
    for severity in SEVERITIES:
        base = SEED_BASE[severity]
        seeds = list(range(base, base + args.per_class))
        print(
            f"[{severity}] {args.per_class} episodes, seeds {base}..{seeds[-1]} ..."
        )
        per_class[severity] = evaluate_on_class(
            net=net,
            tools=tools,
            images=test_images,
            device=device,
            severity=severity,
            seeds=seeds,
            n_random_chains=args.n_random_chains,
            n_strips=args.strips,
            strips_dir=out_dir / "strips",
        )
        a = per_class[severity]["agent"]
        b = per_class[severity]["baselines"]
        print(
            f"[{severity}] agent {a['psnr_gain_db']:+.2f} dB | "
            f"B1 {b['B1_best_single_tool']:+.2f} | B3 {b['B3_greedy_oracle']:+.2f}"
        )

    overall = _overall(per_class)
    protocol = {
        "n_per_class": args.per_class,
        "n_random_chains": args.n_random_chains,
        "seeds_scheme": "mild 10000.., moderate 20000.., severe 30000..",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "test_split": "DIV2K 751-800",
        "image_dtype": "float32[0,1]" if args.float_images else "uint8 (training parity)",
        "device": str(device),
        "date": datetime.now(timezone.utc).isoformat(),
    }

    # ── Write reports ───────────────────────────────────────────────────────
    result = {"per_class": per_class, "overall": overall, "protocol": protocol}
    json_path = out_dir / "agent_eval_test.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {json_path}")

    md = render_markdown(per_class, overall, protocol)
    md_path = out_dir / "agent_eval_test.md"
    md_path.write_text(md)
    print(f"wrote {md_path}")
    print(f"wrote {args.strips} strips/class under {out_dir / 'strips'}")

    print()
    print(md)
    return 0  # reporting, not a gate


if __name__ == "__main__":
    sys.exit(main())
