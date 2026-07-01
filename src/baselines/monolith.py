"""Monolithic end-to-end restoration CNN (P3.E4 baseline B5).

A single DnCNN-style network that maps a degraded patch directly to its clean
estimate, the headline control against the "12 frozen tools + DQN dispatcher"
system (docs/plan3/monolith_spec.md). Architecture (depth D, width W):

    Conv(in→W, 3×3, p1) + ReLU                          # head
    (D-2) × [Conv(W→W, 3×3, p1) + BN(W) + ReLU]         # body
    Conv(W→in, 3×3, p1)                                  # tail
    out = x - body(x)                                    # global residual

Output is UNCLAMPED (matches metrics.psnr + the tool convention, which let
outputs exceed [0,1]); clamp to [0,1] only for display via restore_for_display.

Parameter arithmetic (BN running buffers are NOT parameters):
  head  Conv(in→W,3×3)      = (in·9 + 1)·W
  body  (D-2) × [Conv(W→W,3×3) + BN(W)]
                            = (D-2) · [ (W·9 + 1)·W  +  2·W ]
  tail  Conv(W→in,3×3)      = (W·9 + 1)·in

  D=14, W=64, in=3:
    head = (3·9+1)·64                    = 28·64        =   1,792
    body = 12 · [ (64·9+1)·64 + 2·64 ]   = 12·37,056    = 444,672
    tail = (64·9+1)·3                     = 577·3        =   1,731
    total                                                = 448,195  (102.8% of toolbox)

  D=14, W=48, in=3:
    head = (3·9+1)·48                    = 28·48        =   1,344
    body = 12 · [ (48·9+1)·48 + 2·48 ]   = 12·20,880    = 250,560
    tail = (48·9+1)·3                     = 433·3        =   1,299
    total                                                = 253,203  (~58% of toolbox)
"""

from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters (BN running buffers excluded)."""
    return sum(p.numel() for p in model.parameters())


class MonoRestoreCNN(nn.Module):
    """Single end-to-end restoration CNN with a global residual.

    Parameters
    ----------
    depth : int
        Total convolutional layers D (head + body + tail). The body has D-2
        BN-bearing layers. Default 14 (29×29 receptive field, covers the 21×21
        blur kernel with margin).
    width : int
        Channel count W of the head/body. Default 64.
    in_channels : int
        Image channels. Default 3 (RGB).
    """

    def __init__(self, depth: int = 14, width: int = 64, in_channels: int = 3) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2 (head + tail), got {depth}")
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(width, width, 3, padding=1))
            layers.append(nn.BatchNorm2d(width))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(width, in_channels, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restored estimate, UNCLAMPED. ``out = x - body(x)`` (residual = noise)."""
        return x - self.body(x)

    def restore_for_display(self, x: torch.Tensor) -> torch.Tensor:
        """forward(x) clamped to [0,1] — for visualisation only, never for PSNR."""
        return self.forward(x).clamp(0.0, 1.0)
