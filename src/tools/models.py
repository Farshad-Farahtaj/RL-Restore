"""Tool CNNs, exactly matching the original's shipped checkpoints.

small: 21,827 params; large: 50,851 params (both verified against the TF
checkpoint files). Both are fully convolutional with a global residual:
output = input + correction. NCHW float tensors in [0,1] (not clipped).
Weight init: PyTorch default (Kaiming-uniform), not the TF original's
Glorot-uniform — topology and param counts are exact, init deliberately is not.
"""

import torch
import torch.nn.functional as F
from torch import nn

from rlrestore.tools.specs import ToolSpec


class SmallToolCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 9, padding=4)
        self.conv2 = nn.Conv2d(32, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 3, 5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        return x + self.conv3(h)


class LargeToolCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.conv1_2 = nn.Conv2d(64, 32, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2a = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2c = nn.Conv2d(32, 32, 3, padding=1)
        self.conv2d_ = nn.Conv2d(32, 32, 3, padding=1)  # trailing _ avoids nn.Conv2d clash
        self.conv2_2 = nn.Conv2d(32, 64, 1)
        self.conv3 = nn.Conv2d(64, 3, 5, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.conv1_2(F.relu(self.conv1(x)))     # pre-ReLU value kept for skip
        h = F.relu(b0)
        h = F.relu(self.conv2(h))
        s2 = b0 + self.conv2a(h)                     # residual block 1
        h = F.relu(s2)
        h = F.relu(self.conv2c(h))
        s3 = s2 + self.conv2d_(h)                    # residual block 2
        h = F.relu(self.conv2_2(s3))
        return x + self.conv3(h)                     # global residual


def build_tool(spec: ToolSpec) -> nn.Module:
    return SmallToolCNN() if spec.arch == "small" else LargeToolCNN()
