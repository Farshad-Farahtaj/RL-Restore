"""ModelRegistry: load every restoration model ONCE, frozen, on CPU.

Built once in the FastAPI lifespan and stashed on ``app.state``. Loading mirrors
the verified checkpoint formats (webapp_spec "Model load"):

- agent  : bare state_dict -> ``AgentNet().load_state_dict(.., weights_only=True)``
- tools  : ``load_tools(name, device)`` (reads ``{"model": sd}`` per tool)
- monolith: ``{"model": sd, "val_psnr":..}`` -> ``MonoRestoreCNN(14,64)``,
            ``weights_only=False``

ANY method whose checkpoint is missing is omitted gracefully: ``load()`` never
raises for an absent optional checkpoint, and ``available_methods`` reflects only
what actually loaded. The agent + the "full" toolbox are treated as required
(the demo's headline path); if those are missing we surface a clear error.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from rlrestore.agent.env import load_tools
from rlrestore.agent.net import AgentNet
from rlrestore.baselines.monolith import MonoRestoreCNN
from rlrestore.baselines.unet_restore import UNetRestore

from .realesrgan_arch import RRDBNet, load_realesrgan_x4

log = logging.getLogger("rlrestore.backend.models")

# Checkpoint root resolution, in priority order:
#   1. $RLR_CKPT_ROOT (the Docker/HF-Space sets this to the bundled checkpoints/).
#   2. A `checkpoints/` next to hf_space/ (standalone Space layout).
#   3. The repo root's checkpoints/ (local dev from inside the research repo).
_SPACE_ROOT = Path(__file__).resolve().parents[1]  # hf_space/
_REPO_ROOT = Path(__file__).resolve().parents[2]  # research repo root (local dev)


def _resolve_ckpt_root() -> Path:
    env = os.environ.get("RLR_CKPT_ROOT")
    if env:
        return Path(env)
    space_local = _SPACE_ROOT / "checkpoints"
    if space_local.is_dir():
        return space_local
    return _REPO_ROOT / "checkpoints"


_CKPT_ROOT = _resolve_ckpt_root()
AGENT_CKPT = _CKPT_ROOT / "agent_full_v2" / "agent_net.pt"
TOOLS_ROOT = _CKPT_ROOT / "tools"
MONOLITH_CKPT = _CKPT_ROOT / "monolith" / "monolith" / "best.pt"
UNET_HQ_CKPT = _CKPT_ROOT / "unet_hq" / "best.pt"
REALESRGAN_CKPT = _CKPT_ROOT / "realesrgan" / "RealESRGAN_x4plus.pth"

MONOLITH_DEPTH = 14
MONOLITH_WIDTH = 64

# Public method ids -> the tools key they roll out against.
METHOD_TOOLS_KEY = {"agent": "tools_full", "agent_ft": "tools_full_ft"}


@dataclass
class ModelRegistry:
    """Holds frozen models + metadata. Construct then call ``load()``."""

    device: torch.device
    agent_ckpt: Path = AGENT_CKPT
    tools_root: Path = TOOLS_ROOT
    monolith_ckpt: Path = MONOLITH_CKPT
    unet_hq_ckpt: Path = UNET_HQ_CKPT
    realesrgan_ckpt: Path = REALESRGAN_CKPT

    agent: AgentNet | None = field(default=None, init=False)
    tools_full: list[nn.Module] | None = field(default=None, init=False)
    tools_full_ft: list[nn.Module] | None = field(default=None, init=False)
    monolith: MonoRestoreCNN | None = field(default=None, init=False)
    monolith_val_psnr: float | None = field(default=None, init=False)
    unet_hq: UNetRestore | None = field(default=None, init=False)
    unet_hq_val_psnr: float | None = field(default=None, init=False)
    realesrgan: RRDBNet | None = field(default=None, init=False)
    available_methods: list[str] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ load
    def load(self) -> "ModelRegistry":
        """Load all present checkpoints. Idempotent enough for a single call."""
        self._load_tools_full()
        self._load_agent()
        self._load_tools_full_ft()
        self._load_monolith()
        self._load_unet_hq()
        self._load_realesrgan()
        self._recompute_methods()
        log.info(
            "ModelRegistry loaded on %s; methods=%s",
            self.device,
            self.available_methods,
        )
        return self

    # ------------------------------------------------------------ components
    def _load_tools_full(self) -> None:
        path = self.tools_root / "full" / "tool00" / "best.pt"
        if not path.exists():
            log.warning("tools/full missing (%s); agent path unavailable", path)
            return
        self.tools_full = load_tools("full", self.device, ckpt_root=self.tools_root)
        log.info("loaded 12 'full' tools")

    def _load_tools_full_ft(self) -> None:
        path = self.tools_root / "full_ft" / "tool00" / "best.pt"
        if not path.exists():
            log.info("tools/full_ft absent (%s); agent_ft omitted", path)
            return
        self.tools_full_ft = load_tools(
            "full_ft", self.device, ckpt_root=self.tools_root
        )
        log.info("loaded 12 'full_ft' tools")

    def _load_agent(self) -> None:
        if not self.agent_ckpt.exists():
            log.warning("agent checkpoint missing (%s)", self.agent_ckpt)
            return
        net = AgentNet()
        sd = torch.load(self.agent_ckpt, map_location=self.device, weights_only=True)
        net.load_state_dict(sd)
        net.to(self.device).eval().requires_grad_(False)
        self.agent = net
        log.info("loaded agent (%d params)", sum(p.numel() for p in net.parameters()))

    def _load_monolith(self) -> None:
        if not self.monolith_ckpt.exists():
            log.info("monolith checkpoint absent (%s); monolith omitted", self.monolith_ckpt)
            return
        model = MonoRestoreCNN(depth=MONOLITH_DEPTH, width=MONOLITH_WIDTH)
        ck = torch.load(self.monolith_ckpt, map_location=self.device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.to(self.device).eval().requires_grad_(False)
        self.monolith = model
        val = ck.get("val_psnr")
        self.monolith_val_psnr = float(val) if val is not None else None
        log.info("loaded monolith (val_psnr=%s)", self.monolith_val_psnr)

    def _load_unet_hq(self) -> None:
        if not self.unet_hq_ckpt.exists():
            log.info("unet_hq checkpoint absent (%s); HQ Restore omitted", self.unet_hq_ckpt)
            return
        ck = torch.load(self.unet_hq_ckpt, map_location=self.device, weights_only=False)
        model = UNetRestore(width=int(ck.get("width", 48)))
        model.load_state_dict(ck["model"])
        model.to(self.device).eval().requires_grad_(False)
        self.unet_hq = model
        val = ck.get("val_psnr")
        self.unet_hq_val_psnr = float(val) if val is not None else None
        log.info("loaded unet_hq (val_psnr=%s)", self.unet_hq_val_psnr)

    def _load_realesrgan(self) -> None:
        if not self.realesrgan_ckpt.exists():
            log.info(
                "realesrgan checkpoint absent (%s); Enhance finisher omitted",
                self.realesrgan_ckpt,
            )
            return
        self.realesrgan = load_realesrgan_x4(self.realesrgan_ckpt, self.device)
        log.info(
            "loaded realesrgan x4 (%d params)",
            sum(p.numel() for p in self.realesrgan.parameters()),
        )

    # --------------------------------------------------------------- methods
    def _recompute_methods(self) -> None:
        methods: list[str] = []
        if self.agent is not None and self.tools_full is not None:
            methods.append("agent")
        if self.agent is not None and self.tools_full_ft is not None:
            methods.append("agent_ft")
        if self.monolith is not None:
            methods.append("monolith")
        if self.unet_hq is not None:
            methods.append("unet_hq")
        self.available_methods = methods

    # --------------------------------------------------------------- helpers
    @property
    def models_loaded(self) -> bool:
        return bool(self.available_methods)

    @property
    def enhance_available(self) -> bool:
        """The Real-ESRGAN 'Enhance & Upscale' finisher is loaded and usable."""
        return self.realesrgan is not None

    def tools_for(self, tools_key: str) -> list[nn.Module]:
        tools = getattr(self, tools_key, None)
        if tools is None:
            raise KeyError(f"tools '{tools_key}' not loaded")
        return tools

    def has_method(self, method: str) -> bool:
        return method in self.available_methods
