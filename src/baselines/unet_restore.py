"""UNetRestore — a compact no-BatchNorm U-Net for the high-quality (HQ) upgrade.

This is the "improve on the paper" extension: a capacity bump over the 12 tiny
tool CNNs (21-51k params) and the DnCNN monolith (448k), trained with an L1
objective (perceptually sharper than the paper's MSE) on the full
mixed-degradation family. ~0.63M params at width 48 (smoke-validated: it dropped
held-out LPIPS 0.226 -> 0.163 vs the tiny tool, with HIGHER artifact-fidelity, so
it recovers real detail rather than hallucinating).

Design choices, each tied to evidence (docs/plan4/tool_upgrade_spec.md):
- No BatchNorm: the deep BN monolith failed to converge under a plain Adam run;
  a no-BN U-Net trains stably with Adam.
- Global residual (out = in + correction): same interface as the tools/monolith,
  so it drops into the same eval and serving paths.
- Reflect-pads internally to a multiple of 4, so it accepts the 63x63 training
  patches AND arbitrary full-image eval sizes without skip-connection mismatch.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class UNetRestore(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        w = width
        self.e1 = nn.Conv2d(3, w, 3, padding=1)
        self.e2 = nn.Conv2d(w, 2 * w, 3, stride=2, padding=1)
        self.e3 = nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(4 * w, 4 * w, 3, padding=1)
        self.d3 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.d2 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.out = nn.Conv2d(w, 3, 3, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4  # pad up to a multiple of 4 for clean 2x/2x
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        a = self.act(self.e1(xp))              # full res
        b = self.act(self.e2(a))               # /2
        c = self.act(self.e3(b))               # /4
        c = self.act(self.mid(c))
        d = self.act(self.d3(c)) + b           # /2  (+ skip)
        e = self.act(self.d2(d)) + a           # full (+ skip)
        corr = self.out(e)[..., :h, :w]        # crop back to the input size
        return x + corr                        # global residual, unclamped


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
