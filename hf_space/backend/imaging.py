"""Image I/O and geometry helpers for the demo backend.

Canonical in-memory image is HWC float32 in [0, 1] RGB — the SAME contract the
``rlrestore`` pipeline / tools use (uint8/255). Conversions to NCHW torch follow
``agent/env.py`` exactly (``from_numpy(hwc).permute(2,0,1).unsqueeze(0)``).

Resize uses OpenCV bicubic (``INTER_CUBIC``), matching ``data/pipeline.rescale``;
PNG/WebP encoding uses OpenCV so the runtime needs no extra image libs beyond
Pillow (decode) + opencv-python-headless (already a repo dependency).
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image

__all__ = [
    "decode",
    "resize_long_edge",
    "to_nchw",
    "from_nchw",
    "encode_dataurl",
    "center_crop_63",
    "POLICY_CROP",
]

POLICY_CROP = 63  # AgentNet's Linear(384,...) is hard-wired to a 63x63 input.


def decode(data: bytes) -> np.ndarray:
    """Decode image bytes -> HWC float32 [0,1] RGB.

    Uses Pillow with ``.convert("RGB")`` so palette/grayscale/RGBA/CMYK and EXIF
    orientation are all normalised to 3-channel RGB before we touch the array.
    """
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr)


def resize_long_edge(img: np.ndarray, max_long_edge: int) -> np.ndarray:
    """Bicubic downscale so the long edge <= ``max_long_edge``. Never upscales.

    Returns the input unchanged (a copy is not forced) when already small enough.
    """
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return out.astype(np.float32)


def to_nchw(img: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
    """HWC float32 -> (1,3,H,W) float32 torch tensor (env.py convention)."""
    t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
    t = t.permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t


def from_nchw(t: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) torch -> HWC float32 numpy (UNCLAMPED, env.py convention)."""
    return t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def center_crop_63(img: np.ndarray) -> np.ndarray:
    """Center 63x63 crop for the policy. Reflect-pads if a side is < 63.

    Tools are fully convolutional and run on the full image; only the AgentNet
    policy needs a fixed 63x63 view (webapp_spec "63x63 policy crop").
    """
    h, w = img.shape[:2]
    pad_h = max(0, POLICY_CROP - h)
    pad_w = max(0, POLICY_CROP - w)
    if pad_h or pad_w:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img = np.pad(
            img, ((top, bottom), (left, right), (0, 0)), mode="reflect"
        )
        h, w = img.shape[:2]
    top = (h - POLICY_CROP) // 2
    left = (w - POLICY_CROP) // 2
    crop = img[top : top + POLICY_CROP, left : left + POLICY_CROP]
    return np.ascontiguousarray(crop, dtype=np.float32)


def encode_dataurl(img: np.ndarray, clamp: bool = True, fmt: str = "png") -> str:
    """HWC float [0,1]ish -> ``data:image/<fmt>;base64,...`` URL.

    Tool/monolith outputs are unclamped by design; for *display* we clamp to
    [0,1] (matches ``restore_for_display``). Set ``clamp=False`` only when the
    caller has already clamped.
    """
    arr = np.asarray(img, dtype=np.float32)
    if clamp:
        arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ext = ".webp" if fmt == "webp" else ".png"
    mime = "image/webp" if fmt == "webp" else "image/png"
    ok, buf = cv2.imencode(ext, bgr)
    if not ok:  # pragma: no cover - cv2 encode failure is not expected
        raise RuntimeError(f"cv2.imencode failed for format {fmt!r}")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"
