---
title: RL-Restore
emoji: 🪄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Watch an AI repair photos one tool at a time
license: mit
---

# RL-Restore — interactive demo

A custom FastAPI + React app (no Gradio) that demonstrates a reimplementation of
**RL-Restore** (Yu et al., CVPR 2018). Upload or snap a photo and watch a
reinforcement-learning agent repair it **one tool at a time**, with the quality
meter climbing at each step. Compare it against a single end-to-end CNN.

**Open the direct app URL** (`https://<owner>-<space>.hf.space`) for camera capture
— the embedded preview iframe may block the camera.

## What's inside
- `backend/` — FastAPI; loads the models once and exposes `/api/{health,methods,restore,simulate}`.
- `frontend/` — React + Vite UI (the Docker build compiles it; `frontend/dist` is also committed).
- `rlrestore/` — the research package, vendored so the Space is self-contained.
- `checkpoints/` — the trained weights (agent + both toolboxes: 12 frozen + 12 jointly fine-tuned + monolith; ~5.5 MB).
- `Dockerfile` — multi-stage (Node build → Python CPU runtime), serves on port 7860.

## Run it locally
```bash
docker build -t rl-restore .
docker run --rm -u 1000 -p 7860:7860 rl-restore
# open http://localhost:7860
```

## Deploy to your own HuggingFace Space
1. Create a Space: https://huggingface.co/new-space → **SDK = Docker** (Blank), CPU Basic (free).
2. Get a **Write** token: https://huggingface.co/settings/tokens.
3. Push this `hf_space/` directory:
   ```bash
   git lfs install
   cd hf_space
   git init && git add . && git commit -m "RL-Restore demo Space"
   git branch -M main
   git remote add origin https://huggingface.co/spaces/<USERNAME>/<SPACE_NAME>
   git push -u origin main      # username = your HF name, password = the Write token
   ```
4. Watch **Logs → Build**, then **Container** until `Uvicorn running on http://0.0.0.0:7860`.
   The app is live at `https://<USERNAME>-<SPACE_NAME>.hf.space`. Every push rebuilds.

> The demo serves three methods, all bundled in `checkpoints/`: **agent** (frozen
> toolbox, +2.93 dB), **agent_ft** (the jointly fine-tuned toolbox — the headline,
> +3.67 dB), and **monolith** (+4.03 dB raw PSNR, no step-by-step reasoning).
