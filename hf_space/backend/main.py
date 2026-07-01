"""FastAPI entrypoint for the RL-Restore demo (single container, CPU-only).

Lifespan builds the frozen ``ModelRegistry`` ONCE and stores it on ``app.state``.
The /api router is included first; the built frontend (``../frontend/dist``) is
mounted LAST as a SPA so non-/api routes fall back to ``index.html``. The static
mount is guarded so the backend runs standalone (no frontend) for tests/CI.

Run: ``RLR_DEVICE=cpu uv run uvicorn backend.main:app`` (from ``hf_space/``) or
``uv run uvicorn hf_space.backend.main:app`` from the repo root.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import Response

from rlrestore.common.device import get_device

from .models import ModelRegistry
from .routes import router

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rlrestore.backend")

# hf_space/backend/main.py -> hf_space/frontend/dist
_FRONTEND_DIST = (Path(__file__).resolve().parent.parent / "frontend" / "dist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # CPU posture for a small, shared HF Space (webapp_spec "Models loaded ONCE").
    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(2)
    except Exception:  # pragma: no cover - defensive
        log.warning("torch.set_num_threads(2) failed", exc_info=True)

    device = get_device("cpu")  # RLR_DEVICE env still wins inside get_device
    log.info("building ModelRegistry on %s", device)
    registry = ModelRegistry(device=device).load()
    app.state.registry = registry
    log.info("startup complete; methods=%s", registry.available_methods)
    yield
    # Nothing to tear down (frozen in-memory models).


app = FastAPI(
    title="RL-Restore Demo API",
    summary="Legible, step-by-step image restoration (RL agent vs monolith).",
    lifespan=lifespan,
)

# /api routes FIRST.
app.include_router(router)


def _mount_frontend() -> None:
    """Mount the built SPA at / with an index.html fallback, if it exists."""
    if not _FRONTEND_DIST.is_dir():
        log.info("frontend dist not found (%s); serving API only", _FRONTEND_DIST)
        return

    index = _FRONTEND_DIST / "index.html"

    # Serve hashed build assets via StaticFiles (range requests, caching); the
    # catch-all below handles index.html + any other top-level file + SPA routes.
    assets_dir = _FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(request: Request, full_path: str) -> Response:
        # /api/* is handled by the router; anything else is the SPA. Serve a
        # real file if it exists (favicon, manifest, ...), else index.html.
        candidate = (_FRONTEND_DIST / full_path).resolve()
        if (
            full_path
            and _FRONTEND_DIST in candidate.parents
            and candidate.is_file()
        ):
            return FileResponse(candidate)
        return FileResponse(index)

    log.info("mounted frontend SPA from %s", _FRONTEND_DIST)


# Static SPA mount LAST (after /api routes), guarded for standalone runs.
_mount_frontend()
