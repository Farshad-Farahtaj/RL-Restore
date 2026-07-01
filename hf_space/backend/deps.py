"""Request guards + inference serialisation for the /api routes.

- Upload validation: reject > MAX_UPLOAD_BYTES (413) and non-image types (415).
- A module-level ``asyncio.Semaphore(1)`` serialises inference so CPU torch
  never runs two rollouts at once (webapp_spec "serialize inference with a
  1-permit semaphore + threadpool"). Blocking work runs in a threadpool via
  ``run_blocking`` so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from .models import ModelRegistry

# Master limits (also surfaced via /api/methods).
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_LONG_EDGE = 512
MAX_STEPS = 3

# One in-flight inference at a time. Acquire AROUND the threadpool call.
INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

_ALLOWED_PREFIX = "image/"


def get_registry(request: Request) -> ModelRegistry:
    """FastAPI dependency: the frozen registry built in the lifespan."""
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    if registry is None:  # pragma: no cover - lifespan always sets it
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="models not loaded",
        )
    return registry


async def read_upload(file: UploadFile, field: str = "file") -> bytes:
    """Read an UploadFile with size + content-type guards.

    Raises 415 for a non-image content type and 413 if the body exceeds the cap.
    Reads in chunks so an oversized upload is rejected without buffering it all.
    """
    content_type = (file.content_type or "").lower()
    if content_type and not content_type.startswith(_ALLOWED_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"{field}: unsupported content type {content_type!r}; expected an image",
        )

    chunks: list[bytes] = []
    total = 0
    chunk_size = 1 << 20  # 1 MiB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{field}: upload exceeds {MAX_UPLOAD_MB} MB",
            )
        chunks.append(chunk)

    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field}: empty upload",
        )
    return b"".join(chunks)


async def run_blocking(func, *args, **kwargs):
    """Run blocking (CPU/torch) work in a threadpool under the 1-permit lock."""
    async with INFERENCE_SEMAPHORE:
        return await run_in_threadpool(func, *args, **kwargs)
