# RL-Restore demo — single container for a HuggingFace Docker Space.
# Stage 1 builds the React frontend; stage 2 runs FastAPI (CPU) and serves it.

# ── Stage 1: build the React frontend ──────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /build
# Manifests first so npm ci is cached across code-only changes.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build            # → /build/dist

# ── Stage 2: Python runtime ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# cv2 (opencv) needs libglib even in headless builds; do apt as root first.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces run the container as a non-root user with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/user/app \
    RLR_CKPT_ROOT=/home/user/app/checkpoints \
    RLR_DEVICE=cpu \
    OMP_NUM_THREADS=2
WORKDIR /home/user/app

# CPU-only torch wheels (≈200 MB vs ≈2.5 GB CUDA). Deps pinned in requirements.txt.
COPY --chown=user backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r backend/requirements.txt

# App source + vendored rlrestore package + bundled checkpoints (~4 MB).
COPY --chown=user rlrestore/    ./rlrestore/
COPY --chown=user backend/      ./backend/
COPY --chown=user checkpoints/  ./checkpoints/
# Built SPA from stage 1 → the path the backend serves (../frontend/dist).
COPY --chown=user --from=frontend /build/dist ./frontend/dist

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
