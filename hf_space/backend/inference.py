"""Restoration rollouts: the agent toolchain and the monolith single pass.

All model math is REUSED from ``rlrestore`` (tool forward, AgentNet.forward_step,
MonoRestoreCNN.forward, metrics.psnr) — nothing here re-implements it. This
module only runs the rollout and shapes the JSON response.

Two image tracks per agent step (webapp_spec "63x63 policy crop, full-image
tools"): the policy sees a 63x63 CENTER-CROP of the current full image; the
CHOSEN tool is applied to the FULL (<=512px) image.

Quality:
- reference given (simulator path) -> full-reference PSNR (peak 1.0, unclamped),
  computed on the UNCLAMPED working image to match how the models were evaluated.
- no reference (uploads) -> the no-reference sharpness proxy.
Display images are always clamped to [0,1] (``restore_for_display`` convention).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from rlrestore.tools.specs import TOOL_SPECS

from .imaging import center_crop_63, encode_dataurl, from_nchw, resize_long_edge, to_nchw
from .models import METHOD_TOOLS_KEY, ModelRegistry
from .quality import NO_REF_LABEL, REF_LABEL, no_ref_quality
from .quality import psnr as ref_psnr

STOP_ACTION = 12
MAX_STEPS = 3

# Human-readable tool naming, identical to scripts/evaluate_agent.py.
_FAMILY = {"blur": "deblur", "noise": "denoise", "jpeg": "dejpeg"}


def tool_family(idx: int) -> str:
    return _FAMILY[TOOL_SPECS[idx].kind]


def tool_band(idx: int) -> int:
    return idx % 4 + 1


def tool_name(idx: int) -> str:
    """e.g. tool07 -> 'denoise4 (37.5-50)' (matches evaluate_agent.tool_name)."""
    spec = TOOL_SPECS[idx]
    return f"{_FAMILY[spec.kind]}{tool_band(idx)} ({spec.lo:g}-{spec.hi:g})"


def _quality(img: np.ndarray, reference: np.ndarray | None) -> float:
    if reference is not None:
        return ref_psnr(img, reference)
    return no_ref_quality(img)


def quality_metric_label(reference: np.ndarray | None) -> str:
    return REF_LABEL if reference is not None else NO_REF_LABEL


def _apply_tool(tool: torch.nn.Module, img: np.ndarray, device: torch.device) -> np.ndarray:
    """Run a frozen tool on the FULL image. HWC->NCHW->tool->HWC, UNCLAMPED."""
    x = to_nchw(img, device)
    y = tool(x)
    return from_nchw(y)


def run_agent_chain(
    registry: ModelRegistry,
    img: np.ndarray,
    tools_key: str = "tools_full",
    reference: np.ndarray | None = None,
    fmt: str = "png",
) -> dict:
    """Greedy AgentNet rollout over the FULL image; <=3 tool steps then STOP.

    Returns the per-spec /api/restore response dict (minus ``method``/``input``,
    which the route fills in). ``img`` and ``reference`` are HWC float32 [0,1].
    """
    if registry.agent is None:
        raise RuntimeError("agent model not loaded")
    tools = registry.tools_for(tools_key)
    device = registry.device
    net = registry.agent

    t0 = time.perf_counter()

    work = np.ascontiguousarray(img, dtype=np.float32)
    prev_oh = np.zeros(STOP_ACTION, dtype=np.float32)  # 12-dim (STOP has no slot)
    hx: tuple[torch.Tensor, torch.Tensor] | None = None

    base_quality = _quality(work, reference)
    prev_quality = base_quality

    steps: list[dict] = []
    chain: list[int] = []

    for t in range(MAX_STEPS):
        crop = center_crop_63(work)
        img_nchw = to_nchw(crop, device)
        prev_oh_t = torch.from_numpy(prev_oh).unsqueeze(0).to(device)
        q, hx = net.forward_step(img_nchw, prev_oh_t, hx)
        q_values = q.squeeze(0).detach().cpu().tolist()
        action = int(np.argmax(q_values))

        if action == STOP_ACTION:
            steps.append(
                {
                    "index": t + 1,
                    "label": "STOP",
                    "action_index": STOP_ACTION,
                    "action_name": "stop",
                    "image": None,
                    "quality": prev_quality,
                    "quality_delta": 0.0,
                    "q_values": q_values,
                }
            )
            break

        # Apply the chosen tool to the FULL image (UNCLAMPED).
        work = _apply_tool(tools[action], work, device)
        cur_quality = _quality(work, reference)
        delta = cur_quality - prev_quality

        steps.append(
            {
                "index": t + 1,
                "label": tool_name(action),
                "action_index": action,
                "action_name": tool_name(action),
                "image": encode_dataurl(work, clamp=True, fmt=fmt),
                "quality": cur_quality,
                "quality_delta": delta,
                "q_values": q_values,
            }
        )
        chain.append(action)

        # Update recurrent state for the next decision.
        prev_oh = np.zeros(STOP_ACTION, dtype=np.float32)
        prev_oh[action] = 1.0
        prev_quality = cur_quality

    final_quality = prev_quality
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "has_reference": reference is not None,
        "quality_metric": quality_metric_label(reference),
        "steps": steps,
        "final_image": encode_dataurl(work, clamp=True, fmt=fmt),
        "summary": {
            "chain": chain,
            "n_tool_steps": len(chain),
            "total_quality_delta": final_quality - base_quality,
            "elapsed_ms": elapsed_ms,
        },
    }


def _run_single_pass(
    registry: ModelRegistry,
    model: torch.nn.Module,
    action_name: str,
    img: np.ndarray,
    reference: np.ndarray | None = None,
    fmt: str = "png",
) -> dict:
    """A single-pass CNN restoration (monolith or HQ U-Net), presented as a
    2-entry chain [Degraded(input), Restored(output)] for a uniform UI.
    ``action_index`` is -1 (not a toolbox action); ``q_values`` is omitted.
    """
    device = registry.device
    t0 = time.perf_counter()

    work = np.ascontiguousarray(img, dtype=np.float32)
    base_quality = _quality(work, reference)

    x = to_nchw(work, device)
    y = model(x)  # UNCLAMPED residual restoration
    restored = from_nchw(y)
    restored_quality = _quality(restored, reference)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    steps = [
        {
            "index": 0,
            "label": "Degraded",
            "action_index": -1,
            "action_name": "input",
            "image": encode_dataurl(work, clamp=True, fmt=fmt),
            "quality": base_quality,
            "quality_delta": 0.0,
            "q_values": None,
        },
        {
            "index": 1,
            "label": "Restored",
            "action_index": -1,
            "action_name": action_name,
            "image": encode_dataurl(restored, clamp=True, fmt=fmt),
            "quality": restored_quality,
            "quality_delta": restored_quality - base_quality,
            "q_values": None,
        },
    ]

    return {
        "has_reference": reference is not None,
        "quality_metric": quality_metric_label(reference),
        "steps": steps,
        "final_image": encode_dataurl(restored, clamp=True, fmt=fmt),
        "summary": {
            "chain": [],
            "n_tool_steps": 1,
            "total_quality_delta": restored_quality - base_quality,
            "elapsed_ms": elapsed_ms,
        },
    }


def run_monolith(registry, img, reference=None, fmt="png"):
    if registry.monolith is None:
        raise RuntimeError("monolith model not loaded")
    return _run_single_pass(registry, registry.monolith, "monolith", img, reference, fmt)


def run_unet_hq(registry, img, reference=None, fmt="png"):
    if registry.unet_hq is None:
        raise RuntimeError("unet_hq model not loaded")
    return _run_single_pass(registry, registry.unet_hq, "unet_hq", img, reference, fmt)


# ─────────────────────────── enhance (generative finisher) ───────────────────

ENHANCE_SCALE = 4
# Real-ESRGAN on the free 2-vCPU CPU is compute-bound (~11s per 256px-worth), so
# the cap is set by SPEED, not memory: 384 -> 1536px output in ~25s (a real bump
# over the old 1280, comfortably responsive). Tiling still bounds peak memory and
# would let bigger inputs run if this ever moves to a GPU.
MAX_ENHANCE_LONG_EDGE = 384
ENHANCE_TILE = 384  # >= the cap, so a capped input is one pass (no tiling overhead)
ENHANCE_TILE_PAD = 16  # overlap each tile reads, to hide seams (cropped after SR)


def _sr_tiled(model, x, scale=ENHANCE_SCALE, tile=ENHANCE_TILE, pad=ENHANCE_TILE_PAD):
    """Tiled 4x super-resolution. Splits the input into overlapping tiles, upscales
    each, and stitches the (un-padded) centres back together. Peak memory depends
    on the tile size, not the whole image — which is what lets a big image run on CPU.
    """
    _, _, h, w = x.shape
    out = torch.zeros((1, 3, h * scale, w * scale), dtype=x.dtype)
    n_x = (w + tile - 1) // tile
    n_y = (h + tile - 1) // tile
    for iy in range(n_y):
        for ix in range(n_x):
            x0, y0 = ix * tile, iy * tile
            x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
            # pad the tile on each side (clamped to the image) so seams have context
            xp0, yp0 = max(x0 - pad, 0), max(y0 - pad, 0)
            xp1, yp1 = min(x1 + pad, w), min(y1 + pad, h)
            o = model(x[:, :, yp0:yp1, xp0:xp1])
            # crop the padding (x scale) and place the real tile into the canvas
            sx0, sy0 = (x0 - xp0) * scale, (y0 - yp0) * scale
            sx1, sy1 = sx0 + (x1 - x0) * scale, sy0 + (y1 - y0) * scale
            out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = o[:, :, sy0:sy1, sx0:sx1]
    return out


def run_enhance(registry: ModelRegistry, img: np.ndarray, fmt: str = "png") -> dict:
    """Real-ESRGAN 4x super-resolution finisher, applied AFTER any restorer.

    Synthesizes the crisp high-frequency detail that pixel-loss (L1/MSE) models
    regress away. ``img`` is HWC float32 [0,1] (typically a method's restored
    output). The input is capped to ``MAX_ENHANCE_LONG_EDGE`` to bound CPU
    time/memory, then upscaled 4x. No PSNR is reported: detail is *synthesized*,
    not recovered, so a reference metric would be misleading.
    """
    if registry.realesrgan is None:
        raise RuntimeError("enhance model not loaded")
    device = registry.device
    t0 = time.perf_counter()

    src = resize_long_edge(img, MAX_ENHANCE_LONG_EDGE)
    in_h, in_w = src.shape[:2]
    with torch.no_grad():
        out = from_nchw(_sr_tiled(registry.realesrgan, to_nchw(src, device)))
    out_h, out_w = out.shape[:2]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "enhanced_image": encode_dataurl(out, clamp=True, fmt=fmt),
        "scale": ENHANCE_SCALE,
        "input": {"w": in_w, "h": in_h},
        "output": {"w": out_w, "h": out_h},
        "elapsed_ms": elapsed_ms,
    }


def run_method(
    registry: ModelRegistry,
    method: str,
    img: np.ndarray,
    reference: np.ndarray | None = None,
    fmt: str = "png",
) -> dict:
    """Dispatch to the rollout for ``method`` (agent/agent_ft/monolith/unet_hq)."""
    if method == "monolith":
        return run_monolith(registry, img, reference=reference, fmt=fmt)
    if method == "unet_hq":
        return run_unet_hq(registry, img, reference=reference, fmt=fmt)
    tools_key = METHOD_TOOLS_KEY.get(method)
    if tools_key is None:
        raise KeyError(f"unknown method {method!r}")
    return run_agent_chain(registry, img, tools_key=tools_key, reference=reference, fmt=fmt)
