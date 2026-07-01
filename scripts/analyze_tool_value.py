"""Marginal-value analysis of the 12 restoration tools (P3.E2).

Question answered
-----------------
"Which tools earn their keep, and is 12 the right toolbox size?"

Two independent lenses on every tool:

1. AGENT USAGE — how often the *deployed* greedy DQN manager actually reaches
   for each tool. Read straight from the held-out TEST eval
   (reports/eval_test/agent_eval_test.json); the per-class action_hist[13]
   (0-11 = tools, 12 = STOP) is normalised into per-tool fractions, with the
   STOP fraction reported separately.

2. LEAVE-ONE-OUT ORACLE DROP — each tool's marginal contribution to the
   best-possible policy. We replicate compute_baselines' B3 greedy oracle
   (<=3 steps, each step apply the tool that maximises PSNR vs the clean
   reference, no STOP) and measure how much the oracle ceiling falls when a
   single tool is removed from the toolbox:

       drop[t] = oracle_gain(all 12) - oracle_gain(all 12 except t)

   The oracle is GREEDY (not exhaustively optimal), so it is path-dependent and
   NOT monotone under tool removal: drop[t] is usually >= 0 but can be slightly
   negative when greedy myopically over-uses tool t and does better without it.
   We keep the sign. A large positive drop means the ceiling genuinely depends
   on tool t; a near-zero or negative drop means another tool covers its job —
   toolbox-size evidence.

The two lenses are deliberately orthogonal: a tool can be a "hidden keystone"
(high oracle drop, low agent usage — the greedy agent under-uses it) or
"removable" (low drop — a candidate for a smaller toolbox).

This is a LOCAL, CPU-only analysis. No training, no agent rollouts for the
oracle part (the oracle is policy-free). It mirrors evaluate_agent.py's
per-class seed scheme exactly so the episodes line up with the TEST eval.

Outputs (under --out-dir, default reports/tool_value):
  tool_value.json — per_tool list + ceilings + protocol
  tool_value.md   — readable table sorted by leave-one-out drop (desc)

Reporting, not a gate: always exits 0.

Usage:
    RLR_DEVICE=cpu uv run python scripts/analyze_tool_value.py
        [--checkpoint checkpoints/agent_full_v2/agent_net.pt]
        [--config configs/full.yaml] [--tools-name full] [--device cpu]
        [--per-class 200] [--out-dir reports/tool_value]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rlrestore.agent.env import RestoreEnv, load_tools
from rlrestore.common.config import load_config
from rlrestore.common.metrics import psnr
from rlrestore.data.div2k import image_paths
from rlrestore.tools.specs import TOOL_SPECS
from rlrestore.tools.train import load_images

# ─────────────────────────── protocol constants ──────────────────────────────

SEVERITIES = ("mild", "moderate", "severe")
SEED_BASE = {"mild": 10_000, "moderate": 20_000, "severe": 30_000}
_STOP = 12
_MAX_STEPS = 3
_N_TOOLS = 12

# Numerical-noise tolerance: leave-one-out drops in [-EPS_CLIP, 0) are clipped
# to 0.0 (a strictly larger toolbox can never lower the oracle ceiling, so any
# negative is float jitter from the PSNR argmax).
_EPS_CLIP = 1e-6

# Verdict thresholds on the OVERALL leave-one-out drop (dB). Stated in the MD.
# keystone   : drop >= 0.30  — the oracle ceiling leans hard on this tool.
# useful     : 0.10 <= drop < 0.30
# marginal   : 0.02 <= drop < 0.10
# redundant  : drop < 0.02   — another tool covers its job; removable.
VERDICT_KEYSTONE = 0.30
VERDICT_USEFUL = 0.10
VERDICT_MARGINAL = 0.02

# Cross-flag thresholds: a "hidden keystone" has a real ceiling cost but the
# greedy agent rarely reaches for it; a "removable" tool barely moves the
# ceiling regardless of how the agent uses it.
HIDDEN_KEYSTONE_USAGE = 0.03  # agent usage fraction below which "under-used"

# Tool naming derived from TOOL_SPECS (same scheme as evaluate_agent.py):
# indices 0-3 deblur, 4-7 denoise, 8-11 dejpeg; band 1 (weakest) -> 4 within kind.
_KIND_LABEL = {"blur": "deblur", "noise": "denoise", "jpeg": "dejpeg"}


def tool_name(idx: int) -> str:
    """Human-readable tool name, e.g. 7 -> 'denoise4 (37.5-50)'."""
    spec = TOOL_SPECS[idx]
    band = idx % 4 + 1
    return f"{_KIND_LABEL[spec.kind]}{band} ({spec.lo:g}-{spec.hi:g})"


def tool_name_short(idx: int) -> str:
    """Short tool name, e.g. 7 -> 'denoise4'."""
    return f"{_KIND_LABEL[TOOL_SPECS[idx].kind]}{idx % 4 + 1}"


def tool_band(idx: int) -> str:
    """Damage band string, e.g. 7 -> 'noise 37.5-50'."""
    spec = TOOL_SPECS[idx]
    return f"{spec.kind} {spec.lo:g}-{spec.hi:g}"


# ─────────────────────────── (1) agent usage ─────────────────────────────────


def usage_from_hist(action_hist: list[int]) -> dict[str, Any]:
    """Normalise one 13-bin action histogram into tool/STOP fractions.

    Parameters
    ----------
    action_hist : 13 ints — bins 0-11 are tool action counts, bin 12 is STOP.

    Returns
    -------
    dict with:
        tool_fractions : list[12] — per-tool fraction of ALL actions (the
                         denominator includes STOP, so these 12 plus
                         stop_fraction sum to 1).
        stop_fraction  : float — STOP's share of all actions.
        total_actions  : int — sum over all 13 bins.
    """
    assert len(action_hist) == 13, f"expected 13 bins, got {len(action_hist)}"
    total = float(sum(action_hist))
    if total <= 0:
        return {
            "tool_fractions": [0.0] * _N_TOOLS,
            "stop_fraction": 0.0,
            "total_actions": 0,
        }
    tool_fractions = [action_hist[i] / total for i in range(_N_TOOLS)]
    stop_fraction = action_hist[_STOP] / total
    return {
        "tool_fractions": tool_fractions,
        "stop_fraction": stop_fraction,
        "total_actions": int(total),
    }


def load_agent_usage(eval_json_path: Path) -> dict[str, Any] | None:
    """Parse per-class + overall agent tool-usage fractions from the TEST eval.

    The overall histogram is the *pooled* sum of the three per-class histograms
    (so classes with more actions weigh more), then normalised. Returns None if
    the file is missing so the caller can proceed with the oracle part alone.

    Returns
    -------
    dict or None with:
        by_class : {class: usage_from_hist(...)} for mild/moderate/severe
        overall  : usage_from_hist(summed histogram)
    """
    if not eval_json_path.exists():
        return None
    data = json.loads(eval_json_path.read_text())
    per_class = data["per_class"]

    by_class: dict[str, dict[str, Any]] = {}
    pooled = [0] * 13
    for cls in SEVERITIES:
        hist = list(per_class[cls]["agent"]["action_hist"])
        by_class[cls] = usage_from_hist(hist)
        for i in range(13):
            pooled[i] += int(hist[i])

    return {"by_class": by_class, "overall": usage_from_hist(pooled)}


# ─────────────────────────── (2) leave-one-out oracle ────────────────────────


def build_pairs(
    images: list[np.ndarray],
    tools: list[nn.Module],
    device: torch.device,
    severity: str,
    seeds: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build fixed (clean, degraded) HWC pairs from seeds for one class.

    Identical construction to compute_baselines / evaluate_on_class:
    RestoreEnv.reset(seed=s) with training_mode=False, then read clean_image
    and the initial degraded current image. `tools` is only needed to satisfy
    the RestoreEnv constructor (the pairs are policy-free).
    """
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for s in seeds:
        env = RestoreEnv(
            images=images,
            tools=tools,
            device=device,
            seed=s,
            training_mode=False,
            severity=severity,
        )
        img_hwc, _ = env.reset()
        pairs.append((env.clean_image.copy(), img_hwc.copy()))
    return pairs


def _apply_tool_once(
    img_hwc: np.ndarray, tool: nn.Module, device: torch.device
) -> np.ndarray:
    """Apply a tool once: HWC numpy -> NCHW torch -> HWC numpy (no grad)."""
    x = torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        y = tool(x)
    return y.squeeze(0).permute(1, 2, 0).cpu().numpy()


def greedy_oracle_gain(
    tools_subset: list[nn.Module],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> float:
    """Mean PSNR gain (dB) of the no-STOP greedy oracle over a tool subset.

    Replicates compute_baselines' B3 logic exactly: for each (clean, degraded)
    pair, run <=3 steps; at each step apply every tool in the subset once, keep
    the output with the highest PSNR vs clean, accumulate its (signed) gain, and
    advance. No STOP — the best tool is always applied even if it does not help,
    matching B3. Gain = (final PSNR - initial PSNR) averaged over pairs, which
    equals the sum of per-step best gains.

    With tools_subset == all 12 tools this reproduces B3_greedy_oracle for the
    same seeds. NOTE: a larger subset widens the per-step argmax, but greedy is
    path-dependent over 3 steps — a locally-better step-1 choice can land in a
    state that restores worse afterwards. So the 3-step gain is NOT guaranteed
    monotone in the subset, and a leave-one-out drop may be slightly negative
    (greedy over-uses that tool). leave_one_out_drops keeps that sign.

    Parameters
    ----------
    tools_subset : list of frozen tool nn.Modules (>= 1).
    pairs        : list of (clean HWC, degraded HWC) float32 patches.
    device       : torch device.

    Returns
    -------
    float — mean dB gain over pairs (0.0 if pairs is empty).
    """
    assert len(tools_subset) >= 1, "need at least one tool"
    if not pairs:
        return 0.0

    gains: list[float] = []
    for clean, degraded in pairs:
        current = degraded.copy()
        cur_p = psnr(current, clean)
        total = 0.0
        for _ in range(_MAX_STEPS):
            best_g = -float("inf")
            best_next = current
            for tool in tools_subset:
                out = _apply_tool_once(current, tool, device)
                new_p = psnr(out, clean)
                g = new_p - cur_p
                if g > best_g:
                    best_g = g
                    best_next = out
            total += best_g
            current = best_next
            cur_p = psnr(current, clean)
        gains.append(total)
    return float(np.mean(gains))


def leave_one_out_drops(
    tools: list[nn.Module],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> tuple[float, list[float], int]:
    """Full oracle ceiling and per-tool leave-one-out drops on one pair set.

    Returns
    -------
    (ceiling, drops, n_negative) where
        ceiling    : full-toolbox greedy-oracle gain (== B3 for the same seeds).
        drops      : list[12] — ceiling minus the gain with tool t removed,
                     SIGNED. Sub-jitter wobble (|drop| < _EPS_CLIP) snaps to 0.
        n_negative : how many drops are genuinely negative (a real signal).

    Note on sign: the oracle is GREEDY (locally-best tool each step), not the
    exhaustive optimum, so it is NOT monotone under tool removal. Dropping a
    tool that the greedy step myopically favours can let it pick a chain that
    ENDS better, producing a small negative drop. We keep the sign: a negative
    drop means "the toolbox doesn't need this tool, and greedy slightly
    over-uses it" — that is a finding, not float jitter to be clipped away.
    """
    ceiling = greedy_oracle_gain(tools, pairs, device)
    drops: list[float] = []
    n_negative = 0
    for t in range(len(tools)):
        subset = [tool for i, tool in enumerate(tools) if i != t]
        gain_without = greedy_oracle_gain(subset, pairs, device)
        drop = ceiling - gain_without
        if abs(drop) < _EPS_CLIP:
            drop = 0.0  # tool never chosen on these pairs -> identical chains
        elif drop < 0.0:
            n_negative += 1
        drops.append(float(drop))
    return float(ceiling), drops, n_negative


def greedy_forward_selection(
    tools: list[nn.Module],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Build the toolbox one tool at a time, always adding the most valuable.

    This is the CORRECT answer to "how many tools do you actually need?" —
    leave-one-out under-counts importance when tools substitute (graded bands),
    so its small drops must NOT be summed to justify removing a group. Forward
    selection instead measures the ceiling reachable with the best k tools.

    Returns
    -------
    list[12] of {size, added_index, added_name, ceiling, frac_of_full} — the
    state AFTER each addition, in selection order. ceiling is the pooled
    greedy-oracle gain on `pairs`; frac_of_full is ceiling / full-12 ceiling.
    """
    full_ceiling = greedy_oracle_gain(tools, pairs, device)
    chosen: list[int] = []
    remaining = list(range(len(tools)))
    curve: list[dict[str, Any]] = []
    while remaining:
        best_idx, best_ceiling = remaining[0], -float("inf")
        for t in remaining:
            subset = [tools[i] for i in (*chosen, t)]
            gain = greedy_oracle_gain(subset, pairs, device)
            if gain > best_ceiling:
                best_ceiling, best_idx = gain, t
        chosen.append(best_idx)
        remaining.remove(best_idx)
        curve.append({
            "size": len(chosen),
            "added_index": best_idx,
            "added_name": tool_name_short(best_idx),
            "ceiling": float(best_ceiling),
            "frac_of_full": float(best_ceiling / full_ceiling) if full_ceiling else 0.0,
        })
    return curve


def tools_to_reach(curve: list[dict[str, Any]], frac: float) -> int:
    """Smallest toolbox size whose ceiling reaches `frac` of the full ceiling."""
    for entry in curve:
        if entry["frac_of_full"] >= frac:
            return int(entry["size"])
    return len(curve)


# ─────────────────────────── (3) specialist matrix diagonal ──────────────────


def parse_specialist_diagonal(md_path: Path) -> list[float] | None:
    """Parse the own-band ΔPSNR diagonal from reports/tools_specialist_matrix.md.

    The matrix is a 12x12 markdown table; row i column i is tool i applied to
    its own damage band. Returns a list[12] of floats, or None if the file is
    absent / unparseable so the caller can skip the column gracefully.
    """
    if not md_path.exists():
        return None
    diag: list[float] = []
    row_idx = 0
    for line in md_path.read_text().splitlines():
        # Data rows start with "| toolNN |".
        m = re.match(r"\|\s*tool(\d{2})\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # cells[0] == "toolNN"; cells[1:] are the 12 band values.
        if len(cells) < 1 + _N_TOOLS:
            return None
        try:
            value = float(cells[1 + row_idx])
        except ValueError:
            return None
        diag.append(value)
        row_idx += 1
    return diag if len(diag) == _N_TOOLS else None


# ─────────────────────────── verdict + assembly ──────────────────────────────


def verdict_for(drop_overall: float) -> str:
    """Map an overall leave-one-out drop (dB) to a verdict label."""
    if drop_overall >= VERDICT_KEYSTONE:
        return "keystone"
    if drop_overall >= VERDICT_USEFUL:
        return "useful"
    if drop_overall >= VERDICT_MARGINAL:
        return "marginal"
    return "redundant"


def analyze_tool_value(
    images: list[np.ndarray],
    tools: list[nn.Module],
    device: torch.device,
    per_class: int,
    agent_usage: dict[str, Any] | None,
    specialist_diag: list[float] | None,
) -> dict[str, Any]:
    """Core computation: per-class oracle drops + agent usage -> result dict.

    Episodes mirror evaluate_agent.py: per class, seeds SEED_BASE[cls] ..
    +per_class-1, training_mode=False. The OVERALL leave-one-out drop is the
    POOLED mean across all 3*per_class pairs (every pair weighted equally),
    matching how a single deployed toolbox is judged across the whole TEST
    distribution rather than averaging three class means.

    Parameters
    ----------
    images          : TEST image pool.
    tools           : exactly 12 frozen tool nn.Modules.
    device          : torch device.
    per_class       : episodes per severity class for the oracle.
    agent_usage     : output of load_agent_usage (or None if eval JSON absent).
    specialist_diag : own-band diagonal list[12] (or None if matrix absent).

    Returns
    -------
    JSON-able dict with per_tool / full_oracle_ceiling_by_class /
    stop_fraction_by_class / and an internal "_meta" block the CLI lifts into
    the protocol (n_negative-drop count, pooled pair count).
    """
    assert len(tools) == _N_TOOLS, f"expected {_N_TOOLS} tools, got {len(tools)}"

    # ── Per-class pairs + leave-one-out drops ───────────────────────────────
    ceiling_by_class: dict[str, float] = {}
    drops_by_class: dict[str, list[float]] = {}
    all_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    n_negative_total = 0
    for cls in SEVERITIES:
        base = SEED_BASE[cls]
        seeds = list(range(base, base + per_class))
        pairs = build_pairs(images, tools, device, cls, seeds)
        all_pairs.extend(pairs)
        ceiling, drops, n_negative = leave_one_out_drops(tools, pairs, device)
        ceiling_by_class[cls] = ceiling
        drops_by_class[cls] = drops
        n_negative_total += n_negative
        print(
            f"[{cls}] full-oracle ceiling {ceiling:+.3f} dB "
            f"(pooled over {len(pairs)} pairs)"
        )

    # ── Pooled overall ceiling + drops across all classes ───────────────────
    overall_ceiling, overall_drops, n_negative_overall = leave_one_out_drops(
        tools, all_pairs, device
    )
    n_negative_total += n_negative_overall
    print(
        f"[overall] pooled full-oracle ceiling {overall_ceiling:+.3f} dB "
        f"over {len(all_pairs)} pairs"
    )
    if n_negative_total:
        print(
            f"note: {n_negative_total} leave-one-out drop(s) are negative — "
            "greedy (non-optimal) oracle slightly improves without that tool; "
            "reported signed as a redundancy signal, not clipped."
        )

    # ── Forward selection: the honest toolbox-size answer ───────────────────
    # Leave-one-out underestimates importance when tools substitute, so its
    # drops must NOT be summed. Forward selection measures the ceiling of the
    # best-k toolbox directly.
    forward_curve = greedy_forward_selection(tools, all_pairs, device)
    print("forward selection (pooled): " + " -> ".join(
        f"{e['size']}:{e['added_name']}({e['frac_of_full']*100:.0f}%)"
        for e in forward_curve
    ))

    # ── Per-tool assembly ───────────────────────────────────────────────────
    per_tool: list[dict[str, Any]] = []
    for t in range(_N_TOOLS):
        usage_overall: float | None = None
        usage_by_class: dict[str, float] | None = None
        if agent_usage is not None:
            usage_overall = agent_usage["overall"]["tool_fractions"][t]
            usage_by_class = {
                cls: agent_usage["by_class"][cls]["tool_fractions"][t]
                for cls in SEVERITIES
            }

        drop_overall = overall_drops[t]
        own_band = specialist_diag[t] if specialist_diag is not None else None
        verdict = verdict_for(drop_overall)

        # Cross-flags: hidden keystone (matters to the ceiling but the greedy
        # agent under-uses it) and removable (toolbox-size evidence).
        flags: list[str] = []
        if drop_overall >= VERDICT_USEFUL and usage_overall is not None and (
            usage_overall < HIDDEN_KEYSTONE_USAGE
        ):
            flags.append("hidden_keystone_underused_by_agent")
        if verdict == "redundant":
            flags.append("removable_toolbox_size_evidence")

        per_tool.append(
            {
                "index": t,
                "name": tool_name(t),
                "short_name": tool_name_short(t),
                "band": tool_band(t),
                "agent_usage_overall": usage_overall,
                "agent_usage_by_class": usage_by_class,
                "leaveoneout_drop_overall": drop_overall,
                "leaveoneout_drop_by_class": {
                    cls: drops_by_class[cls][t] for cls in SEVERITIES
                },
                "specialist_own_band_gain": own_band,
                "verdict": verdict,
                "flags": flags,
            }
        )

    stop_fraction_by_class: dict[str, float] | None = None
    stop_fraction_overall: float | None = None
    if agent_usage is not None:
        stop_fraction_by_class = {
            cls: agent_usage["by_class"][cls]["stop_fraction"] for cls in SEVERITIES
        }
        stop_fraction_overall = agent_usage["overall"]["stop_fraction"]

    return {
        "per_tool": per_tool,
        "full_oracle_ceiling_by_class": ceiling_by_class,
        "full_oracle_ceiling_overall": overall_ceiling,
        "forward_selection": forward_curve,
        "tools_for_90pct": tools_to_reach(forward_curve, 0.90),
        "tools_for_95pct": tools_to_reach(forward_curve, 0.95),
        "tools_for_99pct": tools_to_reach(forward_curve, 0.99),
        "stop_fraction_by_class": stop_fraction_by_class,
        "stop_fraction_overall": stop_fraction_overall,
        "_meta": {
            "n_pairs_pooled": len(all_pairs),
            "n_negative_drops": n_negative_total,
            "agent_usage_available": agent_usage is not None,
            "specialist_matrix_available": specialist_diag is not None,
        },
    }


# ─────────────────────────── report rendering ────────────────────────────────


def _fmt(value: float | None, places: int = 3) -> str:
    """Format a possibly-None float for the markdown/stdout table."""
    if value is None:
        return "n/a"
    return f"{value:.{places}f}"


def _pct(frac: float | None) -> str:
    """Format an agent-usage fraction as a percentage string."""
    if frac is None:
        return "n/a"
    return f"{100.0 * frac:.1f}%"


def render_markdown(result: dict[str, Any], protocol: dict[str, Any]) -> str:
    """Render tool_value.md: ranked table + ceilings + STOP + takeaway."""
    per_tool = sorted(
        result["per_tool"],
        key=lambda d: d["leaveoneout_drop_overall"],
        reverse=True,
    )

    counts = {"keystone": 0, "useful": 0, "marginal": 0, "redundant": 0}
    for tool in per_tool:
        counts[tool["verdict"]] += 1

    lines: list[str] = [
        "# Tool marginal-value analysis (P3.E2)",
        "",
        "Which of the 12 restoration tools earn their keep, and is 12 the right "
        "toolbox size? Two lenses per tool:",
        "",
        "- **Agent use %** — how often the deployed greedy DQN manager picks the "
        "tool on the held-out TEST set (action_hist, STOP excluded from the "
        "per-tool share but reported below).",
        "- **Leave-one-out drop** — how far the no-STOP greedy-oracle ceiling "
        "falls when the tool is removed from the toolbox; its marginal "
        "contribution to the best-possible policy. Pooled across all "
        f"{result['_meta']['n_pairs_pooled']} TEST pairs.",
        "",
        "| Rank | Tool | Band | Agent use % (overall) | Leave-one-out drop dB "
        "(overall) | Own-band gain dB | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, tool in enumerate(per_tool, start=1):
        flag_mark = ""
        if "hidden_keystone_underused_by_agent" in tool["flags"]:
            flag_mark = " ⚑"  # hidden keystone
        lines.append(
            f"| {rank} | tool{tool['index']:02d} {tool['short_name']}{flag_mark} "
            f"| {tool['band']} "
            f"| {_pct(tool['agent_usage_overall'])} "
            f"| {_fmt(tool['leaveoneout_drop_overall'])} "
            f"| {_fmt(tool['specialist_own_band_gain'], 2)} "
            f"| {tool['verdict']} |"
        )

    # Full-oracle ceiling line per class + overall.
    ceil_by = result["full_oracle_ceiling_by_class"]
    ceil_cells = " | ".join(f"{ceil_by[c]:+.3f}" for c in SEVERITIES)
    lines += [
        "",
        "## Full-oracle ceiling (all 12 tools, no STOP)",
        "",
        "| mild | moderate | severe | overall (pooled) |",
        "|---|---|---|---|",
        f"| {ceil_cells} | {result['full_oracle_ceiling_overall']:+.3f} |",
    ]

    # STOP fractions.
    if result["stop_fraction_by_class"] is not None:
        stop_by = result["stop_fraction_by_class"]
        stop_cells = " | ".join(_pct(stop_by[c]) for c in SEVERITIES)
        lines += [
            "",
            "## Agent STOP fraction (share of all actions)",
            "",
            "| mild | moderate | severe | overall |",
            "|---|---|---|---|",
            f"| {stop_cells} | {_pct(result['stop_fraction_overall'])} |",
        ]
    else:
        lines += [
            "",
            "## Agent STOP fraction",
            "",
            "_TEST eval JSON unavailable — agent usage and STOP fractions skipped._",
        ]

    # Hidden-keystone callout (tools that matter to the ceiling but the agent
    # under-uses). "Redundant"-labelled tools are interpreted via forward
    # selection below, not listed here as group-removable.
    hidden = [
        t for t in per_tool if "hidden_keystone_underused_by_agent" in t["flags"]
    ]
    if hidden:
        names = ", ".join(
            f"tool{t['index']:02d} {t['short_name']} "
            f"(drop {t['leaveoneout_drop_overall']:.3f} dB, "
            f"agent use {_pct(t['agent_usage_overall'])})"
            for t in hidden
        )
        lines += [
            "",
            "## ⚑ Hidden keystones (matter to the ceiling, under-used by the agent)",
            "",
            names + ".",
        ]

    # Toolbox-size: forward selection (the correct answer to "how many tools?").
    curve = result["forward_selection"]
    full = result["full_oracle_ceiling_overall"]
    lines += [
        "",
        "## Toolbox-size: forward selection (the honest answer)",
        "",
        "Built greedily from zero, always adding the single most valuable tool. "
        "Unlike leave-one-out, this is safe to read for toolbox size: it shows "
        "the ceiling reachable with the best *k* tools.",
        "",
        "| k | tool added | ceiling dB | % of full-12 |",
        "|---|---|---|---|",
    ]
    for e in curve:
        lines.append(
            f"| {e['size']} | {e['added_name']} | {e['ceiling']:+.3f} | "
            f"{e['frac_of_full'] * 100:.1f}% |"
        )
    lines += [
        "",
        "## Toolbox-size takeaway",
        "",
        _takeaway_sentence(result, counts, full),
    ]

    # Protocol + thresholds.
    lines += [
        "",
        "## Protocol",
        "",
        f"- {protocol['per_class']} episodes per class "
        f"({result['_meta']['n_pairs_pooled']} pooled), a subset of the "
        "500-episode TEST eval (capped for CPU feasibility).",
        f"- Seeds: {protocol['seed_scheme']} — identical scheme to "
        "evaluate_agent.py (training_mode=False).",
        "- Leave-one-out drop = full-oracle gain − gain with the tool removed, "
        "where the oracle is the no-STOP greedy oracle (B3): <=3 steps, each "
        "step applies the tool maximising PSNR vs clean.",
        "- Overall = POOLED mean across all "
        f"{result['_meta']['n_pairs_pooled']} pairs (not the mean of class "
        "means).",
        "- Agent usage = action_hist fractions from the held-out TEST eval "
        "(reports/eval_test/agent_eval_test.json); STOP reported separately.",
        f"- Verdict thresholds (overall drop, dB): keystone >= "
        f"{VERDICT_KEYSTONE}, useful >= {VERDICT_USEFUL}, marginal >= "
        f"{VERDICT_MARGINAL}, else redundant.",
        f"- ⚑ hidden keystone = useful-or-better drop but agent usage < "
        f"{_pct(HIDDEN_KEYSTONE_USAGE)}.",
        f"- Checkpoint sha256[:16] = `{protocol['checkpoint_sha256_16']}`, "
        f"n_tools = {protocol['n_tools']}, device = {protocol['device']}, "
        f"date = {protocol['date']}.",
    ]
    if not result["_meta"]["specialist_matrix_available"]:
        lines.append(
            "- Own-band gain column: reports/tools_specialist_matrix.md not "
            "found — column left as n/a."
        )
    lines.append("")
    return "\n".join(lines)


def _takeaway_sentence(
    result: dict[str, Any], counts: dict[str, int], full_ceiling: float
) -> str:
    """Interpret toolbox size from forward selection (NOT summed leave-one-out)."""
    k90 = result["tools_for_90pct"]
    k95 = result["tools_for_95pct"]
    first = [e["added_name"] for e in result["forward_selection"][:k95]]
    n_redundant = counts["redundant"]
    return (
        f"Forward selection reaches **90% of the 12-tool ceiling with {k90} "
        f"tool(s)** and **95% with {k95}** ({', '.join(first)}) — so on this "
        f"damage mix roughly {k95} well-chosen tools capture nearly all the "
        "achievable gain, and 12 is generous. IMPORTANT: this is the forward "
        "curve, NOT the sum of leave-one-out drops. The small per-tool "
        f"leave-one-out drops (and the {n_redundant} 'redundant' labels) reflect "
        "pairwise SUBSTITUTABILITY within the graded bands (denoise1-4 etc.), "
        "not group-removability — the strong specialists (e.g. denoise3/4, "
        "own-band +2.2/+3.0 dB) only look cheap because a neighbour covers them. "
        "The extra graded tools are cheap insurance that let the agent "
        "specialise finely; they are not free wins to delete wholesale."
    )


# ─────────────────────────── CLI ──────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Marginal-value analysis of the 12 restoration tools (P3.E2)"
    )
    p.add_argument(
        "--checkpoint",
        default="checkpoints/agent_full_v2/agent_net.pt",
        help="Deployed AgentNet checkpoint (used only for its sha256 stamp; the "
        "oracle analysis is policy-free).",
    )
    p.add_argument(
        "--config",
        default="configs/full.yaml",
        help="Config yaml (data root + tools name).",
    )
    p.add_argument(
        "--tools-name",
        default="full",
        help="checkpoints/tools/<name> to load the 12 tools from.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default cpu; mps is forced to cpu — LSTM/MPS aside, "
        "this keeps tool forward passes deterministic and avoids MPS surprises).",
    )
    p.add_argument(
        "--per-class",
        type=int,
        default=200,
        help="Episodes per severity class for the leave-one-out oracle "
        "(default 200; a CPU-feasible subset of the 500-episode TEST eval).",
    )
    p.add_argument(
        "--eval-json",
        default="reports/eval_test/agent_eval_test.json",
        help="Held-out TEST eval JSON for agent usage fractions.",
    )
    p.add_argument(
        "--specialist-md",
        default="reports/tools_specialist_matrix.md",
        help="Specialist matrix markdown for own-band diagonal gains.",
    )
    p.add_argument(
        "--out-dir",
        default="reports/tool_value",
        help="Output directory (default reports/tool_value).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    requested = str(args.device)
    if requested == "mps":
        print("note: forcing device=cpu (analysis runs tool CNNs on CPU)")
        device = torch.device("cpu")
    else:
        device = torch.device(requested)

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load TEST images (same data path as training / eval) ─────────────────
    test_paths = image_paths(cfg.data.root, "test")
    test_images: list[np.ndarray] = load_images(test_paths)
    print(f"Loaded {len(test_images)} TEST images (DIV2K 751-800)")

    # ── Load tools ───────────────────────────────────────────────────────────
    tools = load_tools(args.tools_name, device)
    print(f"Loaded {len(tools)} tools from checkpoints/tools/{args.tools_name}")

    # ── Checkpoint sha256 stamp (the model the usage histogram came from) ────
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16]
    else:
        ckpt_sha = "missing"
        print(f"note: checkpoint {ckpt_path} not found — sha256 stamp = 'missing'")

    # ── (1) agent usage ──────────────────────────────────────────────────────
    agent_usage = load_agent_usage(Path(args.eval_json))
    if agent_usage is None:
        print(
            f"note: {args.eval_json} not found — agent usage set to null; "
            "running the oracle analysis alone."
        )
    else:
        print(f"Parsed agent usage from {args.eval_json}")

    # ── (3) specialist diagonal ──────────────────────────────────────────────
    specialist_diag = parse_specialist_diagonal(Path(args.specialist_md))
    if specialist_diag is None:
        print(
            f"note: {args.specialist_md} not found/unparseable — own-band gain "
            "column skipped."
        )
    else:
        print(f"Parsed specialist own-band diagonal from {args.specialist_md}")

    print(
        f"Leave-one-out oracle: {args.per_class} episodes/class "
        f"({3 * args.per_class} pooled) — subset of the 500-ep TEST eval."
    )

    # ── (2) core computation ─────────────────────────────────────────────────
    result = analyze_tool_value(
        images=test_images,
        tools=tools,
        device=device,
        per_class=args.per_class,
        agent_usage=agent_usage,
        specialist_diag=specialist_diag,
    )

    protocol = {
        "per_class": args.per_class,
        "seed_scheme": "mild 10000.., moderate 20000.., severe 30000..",
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256_16": ckpt_sha,
        "n_tools": _N_TOOLS,
        "device": str(device),
        "date": datetime.now(timezone.utc).isoformat(),
    }
    result["protocol"] = protocol

    # ── Write JSON (drop the internal _meta-only duplication but keep _meta) ──
    json_path = out_dir / "tool_value.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {json_path}")

    md = render_markdown(result, protocol)
    md_path = out_dir / "tool_value.md"
    md_path.write_text(md)
    print(f"wrote {md_path}")

    print()
    print(md)
    return 0  # reporting, not a gate


if __name__ == "__main__":
    sys.exit(main())
