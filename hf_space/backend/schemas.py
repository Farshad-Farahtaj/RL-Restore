"""Pydantic response models for the /api surface (webapp_spec "API").

Images travel as base64 data-URL strings inside JSON (no binary endpoints).
Shapes mirror the spec exactly so the frontend contract is stable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ───────────────────────────────── health ────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str


# ───────────────────────────────── methods ───────────────────────────────────


class ToolInfo(BaseModel):
    index: int
    name: str  # human label, e.g. "denoise4 (37.5-50)"
    kind: str  # "blur" | "noise" | "jpeg"
    family: str  # "deblur" | "denoise" | "dejpeg"
    band: int  # 1..4 within the family
    lo: float
    hi: float
    arch: str  # "small" | "large"


class MethodInfo(BaseModel):
    id: str  # "agent" | "agent_ft" | "monolith"
    name: str
    description: str
    available: bool
    kind: str  # "agentic" | "single_pass"


class Limits(BaseModel):
    max_upload_mb: int
    max_long_edge: int
    max_steps: int


class MethodsResponse(BaseModel):
    methods: list[MethodInfo]
    tools: list[ToolInfo]
    severities: list[str]
    stop_action_index: int
    enhance_available: bool = False  # Real-ESRGAN "Enhance & Upscale" finisher loaded
    limits: Limits


# ───────────────────────────────── restore ───────────────────────────────────


class InputInfo(BaseModel):
    w: int
    h: int
    resized_from: list[int] | None = None  # [w, h] before the long-edge cap


class RestoreStep(BaseModel):
    index: int
    label: str
    action_index: int
    action_name: str
    image: str | None  # data URL, or None for the STOP frame
    quality: float
    quality_delta: float
    q_values: list[float] | None = None  # 13 floats (agent only)


class RestoreSummary(BaseModel):
    chain: list[int]  # action indices actually taken (tools only, no STOP)
    n_tool_steps: int
    total_quality_delta: float
    elapsed_ms: float


class RestoreResponse(BaseModel):
    method: str
    input: InputInfo
    has_reference: bool
    quality_metric: str  # "psnr" | "sharpness_proxy"
    steps: list[RestoreStep]
    final_image: str  # data URL of the final restored image
    summary: RestoreSummary


# ──────────────────────────────── simulate ───────────────────────────────────


class RecipeInfo(BaseModel):
    blur_sigma: float
    noise_sigma: float
    jpeg_quality: float
    levels: list[int] = Field(default_factory=list)  # (blur, noise, jpeg) 1..10


class SimulateResponse(BaseModel):
    severity: str
    seed: int
    recipe: RecipeInfo
    clean: str  # data URL
    degraded: str  # data URL
    psnr_degraded_vs_clean: float
    input: InputInfo
    restoration: RestoreResponse | None = None  # present iff then_restore requested


# ──────────────────────────────── enhance ────────────────────────────────────


class EnhanceResponse(BaseModel):
    """Real-ESRGAN 4x finisher result (no PSNR — detail is synthesized)."""

    enhanced_image: str  # data URL of the 4x upscaled, detail-synthesized image
    scale: int  # upscale factor (4)
    input: InputInfo  # size fed to the enhancer (after the long-edge cap)
    output: InputInfo  # size of the 4x output
    elapsed_ms: float
