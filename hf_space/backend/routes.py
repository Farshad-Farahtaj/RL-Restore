"""/api router: health, methods, restore, simulate.

Each handler validates the upload, runs the (blocking) rollout in a threadpool
behind the 1-permit semaphore, and returns a typed Pydantic response. Images are
base64 data URLs in the JSON body (no binary endpoints).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from rlrestore.tools.specs import TOOL_SPECS

from .deps import (
    MAX_LONG_EDGE,
    MAX_STEPS,
    MAX_UPLOAD_MB,
    get_registry,
    read_upload,
    run_blocking,
)
from .imaging import decode, resize_long_edge
from .inference import (
    STOP_ACTION,
    run_enhance,
    run_method,
    tool_band,
    tool_family,
    tool_name,
)
from .models import ModelRegistry
from .schemas import (
    EnhanceResponse,
    HealthResponse,
    Limits,
    MethodInfo,
    MethodsResponse,
    RestoreResponse,
    SimulateResponse,
    ToolInfo,
)
from .simulate import SEVERITIES, simulate

router = APIRouter(prefix="/api")

# Static method catalogue (availability is filled from the registry per request).
_METHOD_CATALOGUE = {
    "agent": {
        "name": "RL Agent",
        "kind": "agentic",
        "description": (
            "DQN+LSTM dispatcher that picks named restoration tools one at a "
            "time and shows its work. Most explainable (+2.93 dB mean)."
        ),
    },
    "agent_ft": {
        "name": "RL Agent (fine-tuned)",
        "kind": "agentic",
        "description": (
            "Same dispatcher, but its 12 tools were jointly fine-tuned on the "
            "mid-chain images they actually see (the paper's Algorithm 1). Best "
            "agent quality (+3.67 dB mean)."
        ),
    },
    "monolith": {
        "name": "Monolithic CNN",
        "kind": "single_pass",
        "description": (
            "Single end-to-end DnCNN-style network. Strong silent baseline — "
            "highest raw PSNR (+4.03 dB) but no step-by-step reasoning."
        ),
    },
    "unet_hq": {
        "name": "HQ Restore",
        "kind": "single_pass",
        "description": (
            "A bigger U-Net trained with an L1 objective — the 'improve on the "
            "paper' upgrade. Best held-out gain (+4.15 dB), strongest on severe damage."
        ),
    },
}


def _tools_payload() -> list[ToolInfo]:
    out: list[ToolInfo] = []
    for spec in TOOL_SPECS:
        out.append(
            ToolInfo(
                index=spec.index,
                name=tool_name(spec.index),
                kind=spec.kind,
                family=tool_family(spec.index),
                band=tool_band(spec.index),
                lo=spec.lo,
                hi=spec.hi,
                arch=spec.arch,
            )
        )
    return out


# ───────────────────────────────── health ────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=registry.models_loaded,
        device=str(registry.device),
    )


# ───────────────────────────────── methods ───────────────────────────────────


@router.get("/methods", response_model=MethodsResponse)
def methods(registry: ModelRegistry = Depends(get_registry)) -> MethodsResponse:
    available = set(registry.available_methods)
    method_infos = [
        MethodInfo(
            id=mid,
            name=meta["name"],
            description=meta["description"],
            kind=meta["kind"],
            available=mid in available,
        )
        for mid, meta in _METHOD_CATALOGUE.items()
        if mid in available  # only list methods whose checkpoints are present
    ]
    return MethodsResponse(
        methods=method_infos,
        tools=_tools_payload(),
        severities=SEVERITIES,
        stop_action_index=STOP_ACTION,
        enhance_available=registry.enhance_available,
        limits=Limits(
            max_upload_mb=MAX_UPLOAD_MB,
            max_long_edge=MAX_LONG_EDGE,
            max_steps=MAX_STEPS,
        ),
    )


# ───────────────────────────────── restore ───────────────────────────────────


@router.post("/restore", response_model=RestoreResponse)
async def restore(
    file: UploadFile = File(...),
    method: str = Form(...),
    reference: UploadFile | None = File(None),
    registry: ModelRegistry = Depends(get_registry),
) -> RestoreResponse:
    if not registry.has_method(method):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"method {method!r} unavailable; have {registry.available_methods}",
        )

    raw = await read_upload(file, field="file")
    ref_raw = await read_upload(reference, field="reference") if reference is not None else None

    def _work() -> dict:
        img_full = decode(raw)
        orig_h, orig_w = img_full.shape[:2]
        img = resize_long_edge(img_full, MAX_LONG_EDGE)
        h, w = img.shape[:2]
        resized_from = [orig_w, orig_h] if (h, w) != (orig_h, orig_w) else None

        ref_img = None
        if ref_raw is not None:
            ref_dec = decode(ref_raw)
            # Match the reference to the working resolution so PSNR is well-defined.
            ref_img = resize_long_edge(ref_dec, MAX_LONG_EDGE)
            if ref_img.shape[:2] != img.shape[:2]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="reference dimensions do not match the input image",
                )

        result = run_method(registry, method, img, reference=ref_img)
        result["method"] = method
        result["input"] = {"w": w, "h": h, "resized_from": resized_from}
        return result

    result = await run_blocking(_work)
    return RestoreResponse(**result)


# ───────────────────────────────── enhance ───────────────────────────────────


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance(
    file: UploadFile = File(...),
    registry: ModelRegistry = Depends(get_registry),
) -> EnhanceResponse:
    """Generative 4x finisher (Real-ESRGAN) applied to an already-restored image."""
    if not registry.enhance_available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="enhance (Real-ESRGAN) model is not loaded",
        )

    raw = await read_upload(file, field="file")

    def _work() -> dict:
        return run_enhance(registry, decode(raw))

    result = await run_blocking(_work)
    return EnhanceResponse(**result)


# ──────────────────────────────── simulate ───────────────────────────────────


@router.post("/simulate", response_model=SimulateResponse)
async def simulate_route(
    file: UploadFile = File(...),
    severity: str = Form("moderate"),
    seed: int = Form(0),
    then_restore: str | None = Form(None),
    registry: ModelRegistry = Depends(get_registry),
) -> SimulateResponse:
    if severity not in SEVERITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"severity must be one of {SEVERITIES}, got {severity!r}",
        )
    if then_restore is not None and not registry.has_method(then_restore):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"then_restore method {then_restore!r} unavailable",
        )

    raw = await read_upload(file, field="file")

    def _work() -> dict:
        clean_full = decode(raw)
        orig_h, orig_w = clean_full.shape[:2]
        clean = resize_long_edge(clean_full, MAX_LONG_EDGE)
        h, w = clean.shape[:2]
        resized_from = [orig_w, orig_h] if (h, w) != (orig_h, orig_w) else None

        sim = simulate(clean, severity, seed)
        clean_img = sim.pop("_clean_img")
        degraded_img = sim.pop("_degraded_img")
        sim["input"] = {"w": w, "h": h, "resized_from": resized_from}

        if then_restore is not None:
            restoration = run_method(
                registry, then_restore, degraded_img, reference=clean_img
            )
            restoration["method"] = then_restore
            restoration["input"] = {"w": w, "h": h, "resized_from": resized_from}
            sim["restoration"] = restoration

        return sim

    result = await run_blocking(_work)
    return SimulateResponse(**result)
