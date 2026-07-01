"""RestoreEnv: gym-style MDP for the RL-Restore DQN agent.

Spec: docs/plan2/fidelity_spec.md §1.

Episode flow
------------
- reset(seed): pick a random image from the training pool, call
  make_training_pair to get (clean, degraded) 63x63 float32 HWC patches,
  record initial/current PSNR.
- step(action):
    action 0-11: run frozen tool i on `current` (NCHW torch, no_grad);
                 reward = psnr(next, clean) - current_psnr;
                 terminate when step_count >= 3 (reward still given — P4).
    action 12 (STOP): identity, reward = 0.0, terminated = True.
  Training-only early abort: if after a tool step current_psnr < initial_psnr,
  terminate immediately (negative reward still given).
- obs = (current.copy(), prev_action_onehot.copy()):
    prev_action_onehot is 12-dim; zeros at t=0 and after STOP.

Tool loading
------------
load_tools(cfg_name, device, ckpt_root=Path("checkpoints/tools"))
  Reads checkpoints/tools/<cfg_name>/toolNN/best.pt for N=0..11.
  Format: {"model": state_dict, ...}  (same as tools/train.py _atomic_save).
  Returns list of 12 nn.Module in eval mode with all grads frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rlrestore.common.metrics import psnr
from rlrestore.data.pipeline import make_training_pair
from rlrestore.data.recipes import SEVERITY_LEVELS
from rlrestore.tools.models import build_tool
from rlrestore.tools.specs import TOOL_SPECS

_N_ACTIONS = 12  # number of tools
_STOP = 12
_MAX_STEPS = 3


# ─────────────────────────────── load_tools ──────────────────────────────────


def load_tools(
    cfg_name: str,
    device: torch.device,
    ckpt_root: Path | str = Path("checkpoints/tools"),
) -> list[nn.Module]:
    """Load 12 frozen tool CNNs from best.pt checkpoints.

    Args:
        cfg_name:  config name sub-directory (e.g. "full").
        device:    torch device to map weights to.
        ckpt_root: root directory; default "checkpoints/tools".

    Returns:
        List of 12 nn.Module, each in eval mode with requires_grad=False.
    """
    assert len(TOOL_SPECS) == 12, "toolbox size drifted"
    ckpt_root = Path(ckpt_root)
    tools: list[nn.Module] = []
    for i, spec in enumerate(TOOL_SPECS):
        path = ckpt_root / cfg_name / f"tool{i:02d}" / "best.pt"
        model = build_tool(spec)
        ck = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.to(device).eval().requires_grad_(False)
        tools.append(model)
    return tools


# ─────────────────────────────── RestoreEnv ──────────────────────────────────


class RestoreEnv:
    """Minimal gym-style environment for image-restoration DQN.

    Parameters
    ----------
    images : list[np.ndarray]
        Pool of training images (float32 HWC); env picks randomly on reset.
    tools : list[nn.Module]
        Exactly 12 frozen tool modules.  Must accept NCHW float32 tensors and
        return NCHW float32 tensors (outputs may exceed [0, 1] by design).
    device : torch.device
        Device for tool forward passes.
    seed : int
        Initial RNG seed.  reset(seed=s) reseeds to s.
    training_mode : bool
        If True, early abort fires when PSNR drops below initial.
        If False (validation), early abort is suppressed.
    severity : str
        Damage severity class passed to make_training_pair on every reset.
        One of "mild" | "moderate" | "severe" (default "moderate", the
        training distribution).
    """

    def __init__(
        self,
        images: list[np.ndarray],
        tools: list[nn.Module],
        device: torch.device,
        seed: int = 0,
        training_mode: bool = True,
        severity: str = "moderate",
    ) -> None:
        if len(tools) != _N_ACTIONS:
            raise ValueError(f"Exactly {_N_ACTIONS} tools required, got {len(tools)}")
        assert severity in SEVERITY_LEVELS, (
            f"severity must be one of {sorted(SEVERITY_LEVELS)}, got {severity!r}"
        )
        self._images = images
        self._tools = tools
        self._device = device
        self._training_mode = training_mode
        self._severity = severity
        self._rng = np.random.default_rng(seed)

        # Episode state — initialised in reset()
        self._clean: np.ndarray | None = None
        self._current: np.ndarray | None = None
        self._initial_psnr: float = 0.0
        self._current_psnr: float = 0.0
        self._step_count: int = 0
        self._prev_action_onehot: np.ndarray = np.zeros(_N_ACTIONS, dtype=np.float32)
        self._terminated: bool = False

    # ------------------------------------------------------------------ reset

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Start a new episode.

        Returns
        -------
        obs : (current HWC float32, prev_action_onehot float32 shape (12,))
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Pick a random image and produce a (clean, degraded) patch pair
        idx = int(self._rng.integers(len(self._images)))
        img = self._images[idx]
        self._clean, self._current = make_training_pair(img, self._severity, self._rng)

        self._initial_psnr = psnr(self._current, self._clean)
        self._current_psnr = self._initial_psnr
        self._step_count = 0
        self._prev_action_onehot = np.zeros(_N_ACTIONS, dtype=np.float32)
        self._terminated = False

        return self._obs()

    # ------------------------------------------------------------------ step

    def step(
        self, action: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, bool, dict[str, Any]]:
        """Apply action, return (obs, reward, terminated, info).

        action 0-11: apply frozen tool.
        action 12  : STOP — reward=0.0, terminated=True.
        """
        if self._clean is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated:
            raise RuntimeError("step() called on terminated episode; call reset()")

        # ── STOP ────────────────────────────────────────────────────────────
        if action == _STOP:
            self._prev_action_onehot = np.zeros(_N_ACTIONS, dtype=np.float32)
            self._terminated = True
            info: dict[str, Any] = {
                "stop": True,
                "step_count": self._step_count,
                "current_psnr": self._current_psnr,
                "initial_psnr": self._initial_psnr,
            }
            return self._obs(), 0.0, True, info

        # ── Tool action ──────────────────────────────────────────────────────
        if not (0 <= action < _N_ACTIONS):
            raise ValueError(f"Invalid action {action}; must be 0-{_STOP}")

        # Build one-hot for this action
        self._prev_action_onehot = np.zeros(_N_ACTIONS, dtype=np.float32)
        self._prev_action_onehot[action] = 1.0

        # Apply tool: HWC numpy → NCHW torch → NCHW torch → HWC numpy
        x = torch.from_numpy(self._current).permute(2, 0, 1).unsqueeze(0).to(self._device)
        tool = self._tools[action]
        with torch.no_grad():
            y = tool(x)
        next_img: np.ndarray = y.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Reward = PSNR change (outputs NOT clipped, matching original)
        next_psnr = psnr(next_img, self._clean)
        reward = float(next_psnr - self._current_psnr)

        # Update state
        self._current = next_img
        self._current_psnr = next_psnr
        self._step_count += 1

        # ── Termination check ────────────────────────────────────────────────
        # P4: reward at forced terminal must NOT be zeroed → we compute reward
        #     BEFORE checking termination and always return it.
        terminated = False

        # Training-only early abort: current_psnr dropped below initial
        if self._training_mode and self._current_psnr < self._initial_psnr:
            terminated = True

        # Forced terminal after MAX_STEPS tool actions
        if self._step_count >= _MAX_STEPS:
            terminated = True

        if terminated:
            self._terminated = True

        info = {
            "step_count": self._step_count,
            "current_psnr": self._current_psnr,
            "initial_psnr": self._initial_psnr,
        }
        return self._obs(), reward, terminated, info

    # ----------------------------------------------------------------- helpers

    def _obs(self) -> tuple[np.ndarray, np.ndarray]:
        return (self._current.copy(), self._prev_action_onehot.copy())

    # ── expose internals needed by some tests ──────────────────────────────
    @property
    def training_mode(self) -> bool:
        return self._training_mode

    @property
    def severity(self) -> str:
        return self._severity

    @property
    def clean_image(self) -> np.ndarray:
        """Read-only view of the clean reference image for the current episode."""
        if self._clean is None:
            raise RuntimeError("reset() must be called before accessing clean_image")
        return self._clean
