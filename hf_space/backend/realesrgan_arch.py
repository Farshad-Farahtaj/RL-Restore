"""RRDBNet generator (Real-ESRGAN x4) for the optional 'Enhance & Upscale' finisher.

Standard ESRGAN / Real-ESRGAN generator (Wang et al., ECCVW 2018 / ICCVW 2021).
Defined inline — with the SAME parameter names as the official checkpoint — so
``RealESRGAN_x4plus.pth`` (~64 MB) loads directly, without pulling in the
build-broken ``basicsr`` / ``realesrgan`` packages. Upstream code + weights are
BSD-3-Clause (github.com/xinntao/Real-ESRGAN).

Serving-only: a 4x super-resolution pass applied AFTER a restorer, to synthesize
the crisp high-frequency detail that pixel-loss (L1/MSE) models regress away.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ResidualDenseBlock(nn.Module):
    """Five-conv dense block with residual scaling 0.2 (ESRGAN RDB)."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (three RDBs, residual scaling 0.2)."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN x4 generator. Defaults match ``RealESRGAN_x4plus.pth`` (~16.7M params)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsampling: 2x then 2x = 4x
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_realesrgan_x4(ckpt_path: Path, device: torch.device) -> RRDBNet:
    """Build RRDBNet and load official x4plus weights (under ``params_ema``/``params``)."""
    model = RRDBNet()
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "params_ema" in sd:
        sd = sd["params_ema"]
    elif isinstance(sd, dict) and "params" in sd:
        sd = sd["params"]
    model.load_state_dict(sd, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model
