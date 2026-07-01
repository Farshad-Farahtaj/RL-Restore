"""AgentNet: DQN+LSTM network for RL-Restore.

Spec: docs/plan2/fidelity_spec.md §2.

Architecture
------------
conv1  9x9  3->32  s2 p4  (ReLU)   63x63 -> 32x32
conv2  5x5 32->24  s2 p2  (ReLU)   32x32 -> 16x16
conv3  5x5 24->24  s2 p2  (ReLU)   16x16 ->  8x8
conv4  5x5 24->24  s2 p2  (ReLU)    8x8  ->  4x4
flatten -> 384 (24*4*4)
fc     384 -> 32  (ReLU)
concat prev_action (12-dim one-hot) -> 44
LSTM   44 -> 50  num_layers=1  batch_first=False
q_head 50 -> 13  (linear)

Two calling conventions
-----------------------
Acting (single step):
    forward_step(img_1x3x63x63, prev_oh_1x12, hx) -> (q_1x13, hx_new)
    hx = None is treated as all-zeros.

Training (full episode batch):
    forward_episode_batch(imgs_GxTx3x63x63, prev_oh_GxTx12) -> q_(G*T)x13
    LSTM always starts from zeros (episodes are complete from the start).

Pitfall notes
-------------
P1: LSTM state is owned by the CALLER.  hx=None -> zeros at episode start.
P2: prev_action is 12-dim (STOP produces all-zeros; it has no slot).
    lstm.input_size == 44 == 32 + 12.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AgentNet(nn.Module):
    """DQN+LSTM network per fidelity_spec §2.

    Parameters
    ----------
    (none — all hyper-params are hard-coded to spec values)
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Convolutional tower ──────────────────────────────────────────────
        # Spatial trace: 63 -> 32 -> 16 -> 8 -> 4
        # Flatten: 24 * 4 * 4 = 384
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=2, padding=4),   # 63->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 24, kernel_size=5, stride=2, padding=2),  # 32->16
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=5, stride=2, padding=2),  # 16->8
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=5, stride=2, padding=2),  # 8->4
            nn.ReLU(inplace=True),
        )

        # ── Fully-connected reduction ────────────────────────────────────────
        self.fc = nn.Linear(384, 32)          # in_features MUST be 384

        # ── LSTM ─────────────────────────────────────────────────────────────
        # input_size = 32 (fc out) + 12 (prev_action) = 44  (P2)
        self.lstm = nn.LSTM(
            input_size=44,
            hidden_size=50,
            num_layers=1,
            batch_first=False,  # expects (T, N, input_size) — we control layout
        )

        # ── Q-value head ─────────────────────────────────────────────────────
        self.q_head = nn.Linear(50, 13)       # out_features MUST be 13

    # ──────────────────────────────── forward helpers ────────────────────────

    def _encode(self, img: Tensor) -> Tensor:
        """img (N, 3, 63, 63) -> visual embedding (N, 32).

        N can be any batch dimension (e.g. 1 for acting, G*T for training).
        """
        features = self.conv(img)            # (N, 24, 4, 4)
        flat = features.flatten(1)           # (N, 384)
        return torch.relu(self.fc(flat))     # (N, 32)

    # ──────────────────────────────── acting interface ───────────────────────

    def forward_step(
        self,
        img: Tensor,        # (1, 3, 63, 63)
        prev_oh: Tensor,    # (1, 12)
        hx: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Single-step forward for acting.

        Parameters
        ----------
        img    : (1, 3, 63, 63) float32
        prev_oh: (1, 12) float32 — MUST be 12-dim (P2).  Pass zeros at t=0.
        hx     : LSTM state (h, c) each (1, 1, 50), or None (zeros).

        Returns
        -------
        q      : (1, 13) Q-values
        hx_new : new LSTM state (h, c) each (1, 1, 50)
        """
        # Validate prev_oh dimension (P2 guard)
        if prev_oh.shape[-1] != 12:
            raise ValueError(
                f"prev_oh must have last dim 12 (P2), got {prev_oh.shape[-1]}"
            )

        vis = self._encode(img)                   # (1, 32)
        x = torch.cat([vis, prev_oh], dim=-1)     # (1, 44)

        # LSTM expects (T=1, N=1, 44) with batch_first=False
        x_seq = x.unsqueeze(0)                    # (1, 1, 44)

        lstm_out, hx_new = self.lstm(x_seq, hx)   # (1, 1, 50), hx_new
        q = self.q_head(lstm_out.squeeze(0))       # (1, 50) -> (1, 13)

        return q, hx_new

    # ──────────────────────────────── training interface ─────────────────────

    def forward_episode_batch(
        self,
        imgs: Tensor,      # (G, T, 3, 63, 63)
        prev_oh: Tensor,   # (G, T, 12)
    ) -> Tensor:
        """Full episode batch forward for training.

        The LSTM is always unrolled from zeros (episodes start fresh).
        Output shape: (G*T, 13) — flat over episodes and timesteps, row-major
        in (G, T) order so that [g*T + t] corresponds to episode g, step t.

        Parameters
        ----------
        imgs   : (G, T, 3, 63, 63) float32
        prev_oh: (G, T, 12) float32

        Returns
        -------
        q : (G*T, 13)
        """
        # Validate prev_oh dimension (P2 guard)
        if prev_oh.shape[-1] != 12:
            raise ValueError(
                f"prev_oh must have last dim 12 (P2), got {prev_oh.shape[-1]}"
            )

        G, T, C, H, W = imgs.shape               # e.g. (32, 3, 3, 63, 63)

        # Encode all images simultaneously: merge G and T into a flat batch
        imgs_flat = imgs.reshape(G * T, C, H, W)  # (G*T, 3, 63, 63)
        vis = self._encode(imgs_flat)              # (G*T, 32)

        # Attach prev_action
        oh_flat = prev_oh.reshape(G * T, 12)      # (G*T, 12)
        x_flat = torch.cat([vis, oh_flat], dim=-1) # (G*T, 44)

        # Reshape for LSTM: (T, G, 44) — batch_first=False means (T, N, input)
        # We want to unroll T steps across G episodes in parallel.
        x_seq = x_flat.reshape(G, T, 44).permute(1, 0, 2)  # (T, G, 44)
        x_seq = x_seq.contiguous()

        # LSTM unroll from zeros (no explicit hx -> PyTorch defaults to zeros)
        lstm_out, _ = self.lstm(x_seq)            # (T, G, 50)

        # Q head
        q_seq = self.q_head(lstm_out)             # (T, G, 13)

        # Reshape to (G*T, 13) in (G, T) row-major order:
        # Transpose back to (G, T, 13) then flatten
        q_out = q_seq.permute(1, 0, 2).contiguous().reshape(G * T, 13)

        return q_out

    # ──────────────────────────────── utility ────────────────────────────────

    def n_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
