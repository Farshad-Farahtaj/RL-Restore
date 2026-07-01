# RL-Restore

A PyTorch reimplementation of **"Crafting a Toolchain for Image Restoration by
Deep Reinforcement Learning"** (Yu, Dong, Lin, Loy — CVPR 2018), with three
additions the original work didn't ship: a single-CNN baseline for an honest
comparison, the paper's unreleased joint fine-tuning (Algorithm 1), and an
interactive web demo.

The idea: real photos arrive with a *mix* of blur, noise, and JPEG artifacts, and
no single filter fixes all three. So we train 12 small specialist CNNs (each good
at one kind of damage at one strength) and a reinforcement-learning **agent** that
looks at a damaged photo and picks a short chain of tools to apply — in order,
adapting as it goes, and stopping when the photo is good enough.

## Headline results (held-out DIV2K test, PSNR gain over the damaged input)

| Method | mild | moderate | severe | overall |
|---|---|---|---|---|
| Best single tool | 1.51 | 2.34 | 3.16 | +2.34 |
| **RL agent** (DQN+LSTM, ≤3 tools) | 2.24 | 2.83 | 3.71 | **+2.93** |
| Greedy oracle (best-possible chain) | 2.73 | 3.13 | 3.86 | +3.24 |
| **Single CNN baseline** (448k params) | 3.38 | 3.91 | 4.79 | **+4.03** |

The agent reaches 90% of the oracle ceiling and nearly doubles the best single
tool. A single end-to-end CNN of equal size scores higher on raw PSNR — which is
the honest finding: the RL toolchain's value is **interpretability** (it shows its
work step by step), modularity, and per-image adaptivity, not peak fidelity.

**Joint fine-tuning** (the paper's unreleased Algorithm 1) then retrains the tools on
the mid-chain images they actually see. On the held-out test it lifts the *same* agent
from +2.93 to **+3.67** (a **+0.74 dB** gain, ~3× the paper's reported ~0.25) and
raises the toolbox's oracle ceiling to **+4.25** — above the single-CNN baseline. Full
numbers and protocol in [`docs/technical_documentation.md`](docs/technical_documentation.md)
and [`reports/eval_joint_ft/`](reports/eval_joint_ft/).

## Repository layout

```
src/rlrestore/        the package
  common/             seeding, device, PSNR, config
  data/               DIV2K, degradation operators, severity recipes, patch pipeline
  tools/              the 12 tool CNNs: architectures, dataset, trainer, specialist matrix
  agent/              RestoreEnv, AgentNet (DQN+LSTM), episode replay, training, runner,
                      joint_finetune (Algorithm 1)
  baselines/          MonoRestoreCNN (the single-CNN baseline)
scripts/              train/evaluate/analyze entry points
configs/              full.yaml, smoke.yaml, monolith.yaml, full_vanilla.yaml
tests/                ~200 tests
reports/              evaluation results (committed)
docs/                 technical reference + the paper configuration
hf_space/             the deployable web demo (FastAPI + React + Docker)
```

## Run it

```bash
uv sync                                          # venv + deps
uv run pytest                                    # the test suite
uv run python -m rlrestore.data.div2k download data/DIV2K   # DIV2K (~4 GB)

# train the 12 tools, then the agent (use a GPU for the agent)
uv run python scripts/train_all_tools.py --config configs/full.yaml
uv run python scripts/train_agent.py --config configs/full.yaml

# baselines and the improvement
uv run python scripts/train_monolith.py --config configs/monolith.yaml
uv run python scripts/train_joint_finetune.py --agent-ckpt checkpoints/agent_full_v2/agent_net.pt

# evaluate on the held-out test split
uv run python scripts/evaluate_agent.py --checkpoint checkpoints/agent_full_v2/agent_net.pt
uv run python scripts/evaluate_monolith.py
uv run python scripts/analyze_tool_value.py      # which tools earn their keep
```

Heavy training runs on an A100 via Colab with checkpoints synced to Google Drive;
everything else runs on a laptop CPU. `RLR_DEVICE`, `RLR_CACHE_PYRAMID`, and
`RLR_LOADER_WORKERS` tune the device, the image-pyramid cache, and DataLoader
workers.

Pre-trained weights ship under `hf_space/checkpoints/`, so the demo runs without any
training. DIV2K and the scripts above are only needed to reproduce the results from
scratch.

## The demo

`hf_space/` is a self-contained FastAPI + React app (no Gradio). Pick a method,
upload or snap a photo, and watch the agent restore it one tool at a time with the
quality meter climbing. To run it locally:

```bash
docker build -t rl-restore hf_space/ && docker run --rm -u 1000 -p 7860:7860 rl-restore
```

Deploy steps for a HuggingFace Docker Space are in [`hf_space/README.md`](hf_space/README.md).

## Documentation

- [`docs/technical_documentation.md`](docs/technical_documentation.md) — the engineer-facing reference: architectures, schedules, the full results, reproducibility.
- [`docs/paper_config.md`](docs/paper_config.md) — the exact degradation and training settings, matched to the original paper.

## Credits

Built by Parham Khosh Solat and Farshad Farahtaj for a computer-vision course.

- Based on Yu, Dong, Lin, Loy, "Crafting a Toolchain for Image Restoration by Deep
  Reinforcement Learning," CVPR 2018 ([arXiv:1804.03312](https://arxiv.org/abs/1804.03312)).
- The optional Enhance step uses Real-ESRGAN (Wang et al., 2021), BSD-3-Clause; the bundled
  weight `hf_space/checkpoints/realesrgan/RealESRGAN_x4plus.pth` is redistributed under that license.
- Our own code is released under the MIT License (see [`LICENSE`](LICENSE)).
