# RL-Restore — Technical Documentation

Engineer-facing reference for our PyTorch reimplementation of *Crafting a Toolchain for Image Restoration by Deep Reinforcement Learning* (Yu et al., CVPR 2018; arXiv:1804.03312), plus the two pieces we built on top: a monolithic-CNN control and the paper's unreleased joint fine-tuning (Algorithm 1).

This document describes what the code actually does — layer shapes, the environment contract, the replay invariants, the training schedules, the evaluation protocol — and reports the real measured numbers. For the plain-language walkthrough see `docs/professor_prep_eli5.md`; for the paper/official-code ground truth see `docs/paper_config.md`. Where our implementation deviates from the paper, the deviation is called out explicitly.

Every figure here is sourced from a file in the repo. The provenance table is in [§13](#13-where-the-numbers-come-from).

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Data pipeline](#2-data-pipeline)
3. [The 12 tools](#3-the-12-tools)
4. [The DQN agent](#4-the-dqn-agent)
5. [Evaluation and baselines](#5-evaluation-and-baselines)
6. [Results](#6-results)
7. [Toolbox sizing and tool value](#7-toolbox-sizing-and-tool-value)
8. [Joint fine-tuning (Algorithm 1)](#8-joint-fine-tuning-algorithm-1)
9. [The monolithic-CNN baseline](#9-the-monolithic-cnn-baseline)
10. [The web app](#10-the-web-app)
11. [Reproducibility](#11-reproducibility)
12. [Engineering decisions and notable fixes](#12-engineering-decisions-and-notable-fixes)
13. [Where the numbers come from](#13-where-the-numbers-come-from)

---

## 1. System overview

The task is blind restoration of images suffering *compound* degradation: Gaussian blur, additive Gaussian noise, and JPEG compression artefacts, all stacked on one image at once. A single-degradation network (a denoiser, a deblurrer) doesn't solve this well, because the degradations interact — deblurring sharpens, which amplifies residual noise, so the order in which you apply fixes matters. RL-Restore's answer is to train small specialist CNNs for each single degradation at each strength, then learn a policy that diagnoses an image and applies the right specialists in the right sequence.

The codebase is a Python package, `rlrestore` (under `src/`), with thin CLI drivers in `scripts/` and a separate demo web app in `hf_space/`. The pieces fit together like this:

```
DIV2K clean images (1–750 train / 751–800 test / 801–900 val)
        │
        ▼
data/  synthetic degradation:  blur → noise → JPEG  on float RGB [0,1]
        │  (63×63 patches, 3-scale pyramid, >50 dB skip)
        ▼
  (clean, degraded) 63×63 pair
        │
        ├──────────────► tools/   12 specialist CNNs trained per-band (frozen after training)
        │                          small 21,827 params · large 50,851 params
        │
        ├──────────────► agent/   RestoreEnv (MDP) ⇄ AgentNet (DQN+LSTM, 88,063 params)
        │                          episode-grouped replay · Double-DQN · ≤3 tool steps + STOP
        │                          → checkpoints/agent_full_v2/agent_net.pt
        │
        ├──────────────► agent/joint_finetune.py   frozen agent selects, tools fine-tune
        │                          (Algorithm 1) differentiable chain replay
        │                          → checkpoints/tools/full_ft/
        │
        └──────────────► baselines/monolith.py   single end-to-end DnCNN, 448,195 params
                                   (the headline control)

  evaluation (scripts/evaluate_agent.py, evaluate_monolith.py, analyze_tool_value.py)
        held-out TEST split · seed-matched episodes · PSNR gain vs B0–B4 baselines
```

The agent is a *controller* over a discrete action space of 13 choices — apply one of tools 0–11, or STOP. The tools are frozen during agent training. That decoupling is the whole architectural bet: you can swap, retrain, or add a tool without retraining the controller, and the controller's decisions are human-readable ("denoise2, then deblur1, then stop").

Three deviations from the paper run through everything and are flagged at each site: tools train for 20 epochs (early-stopped) rather than a fixed 80; images live in RAM as uint8 and are converted to float on access; the DQN uses Double-DQN and (late in training) gradient clipping, both of which we treat as ablatable. The full deviation list is in `docs/paper_config.md`.

---

## 2. Data pipeline

Source files: `src/rlrestore/data/{div2k,degradations,recipes,pipeline}.py`, `src/rlrestore/common/metrics.py`.

### 2.1 DIV2K and the splits

We use DIV2K (`data/div2k.py`), fetched directly from ETH Zürich. The paper's split is reproduced exactly:

| Split | Images | DIV2K source folder | Count |
|---|---|---|---|
| train | 1–750 | `DIV2K_train_HR` | 750 |
| test | 751–800 | `DIV2K_train_HR` | 50 |
| val | 801–900 | `DIV2K_valid_HR` | 100 |

`verify()` asserts these exact counts before any training. The test images (751–800) are never touched by tool training, agent training, or fine-tuning — they exist only for the held-out evaluation in [§5](#5-evaluation-and-baselines).

### 2.2 Degradation operators

All operators take and return float32 RGB, HWC, nominally in [0,1] (`data/degradations.py`). They mirror the original MATLAB `generate_train.m`:

- **Gaussian blur** — `cv2.GaussianBlur` with a fixed **21×21 kernel**, `sigmaX = sigmaY = σ`, `BORDER_REPLICATE`. σ < 1e-3 short-circuits to a copy. This was numerically checked against MATLAB's `fspecial('gaussian',21,σ) + imfilter(...,'replicate')` to ~7e-18 max-abs difference.
- **Gaussian noise** — zero-mean, σ specified on the **0–255 scale** (the kernel divides by 255 internally: `img + N(0, σ/255)`), then clipped to [0,1]. This matches `imnoise` behaviour.
- **JPEG** — a real encode/decode round-trip via `cv2.imencode('.jpg', …, [JPEG_QUALITY, q])` / `cv2.imdecode`. Note cv2 uses 4:2:0 chroma subsampling, so Q=100 is *not* per-pixel lossless on chroma; this is the same behaviour class as MATLAB `imwrite` and is adequate here.

The composition order is always **blur → noise → JPEG** (`apply_recipe` in `pipeline.py`), reflecting the physical pipeline of a real camera (optics blur → sensor noise → file compression).

### 2.3 Severity recipes and classes

Each degradation type has a 10-band grid (`recipes.py`):

- blur: `linspace(0, 5, 11)` → 10 bands
- noise: `linspace(0, 50, 11)` → 10 bands
- JPEG: `[100, 80, 60, 50, 40, 35, 30, 25, 20, 15, 10]` — descending, non-uniform

A recipe is a level triple `(blur_lvl, noise_lvl, jpeg_lvl)`, each 1..10. The **severity class is `sum(levels) − 2`**, partitioned into:

| Class | Level-sum band | Role |
|---|---|---|
| mild | `[9, 11]` | easiest |
| moderate | `[12, 17]` | **agent's training distribution** |
| severe | `[18, 20]` | hardest, most headroom |

`enumerate_level_rows` builds every `(k,m,n)` with `k ≤ m ≤ n` in the class's band, then adds all distinct permutations (matching the original's `noise_combination.m`). Sampling draws a row uniformly, then samples a continuous value uniformly *within* the chosen band — so triples with more distinct orderings get proportionally more weight, exactly as in the original. The agent trains on `moderate` only; the monolith and joint fine-tuning train on all three (see [§9](#9-the-monolithic-cnn-baseline)).

### 2.4 Patch sampling and the dtype contract

`make_training_pair(img, severity, rng)` (`pipeline.py`) produces one `(clean, degraded)` pair:

1. Pick a pyramid scale from `[1.0, 2/3, 1/3]` (bicubic, the original's 3-scale pyramid — note the paper text says "down-scaling by 2,3,4" but the released code uses these scales; code wins).
2. Random-crop a **63×63** patch.
3. Normalize to clean float [0,1]: **if the patch is uint8, divide by 255; if float, clip to [0,1].** This dtype branch is load-bearing — see [§12](#12-engineering-decisions-and-notable-fixes), the white-patch regression.
4. Apply the sampled recipe to get `degraded`.
5. **Resample if `PSNR(degraded, clean) > 50 dB`** (too smooth/clean to be a useful example), up to `max_tries=10`.

The whole training set is held in RAM as **uint8** and converted to float32/255 per crop. This matches the original's uint8-PNG → `im2double` path and is roughly 4× lighter (≈4–6 GB for 750 DIV2K images vs 17+ GB as float32 on a 16 GB machine).

The tool-training dataset (`tools/dataset.py`) has a `cache_pyramid` option (env var `RLR_CACHE_PYRAMID=1`) that pre-resizes each image to the three scales once, trading ~9 GB of RAM for removing the per-item resize cost — worthwhile on a GPU box where the dataloader is the bottleneck.

### 2.5 PSNR

`metrics.psnr(a, b) = 10·log10(1 / (MSE + 1e-10))`, peak = 1.0, computed over all channels (`common/metrics.py`). **Inputs are not clipped** — tool outputs can legitimately exceed [0,1], and clipping them before measuring would not match how the original computes the metric. The `1e-10` epsilon matches the original. Everywhere we report a *gain*: `PSNR(restored, clean) − PSNR(degraded, clean)`, because absolute PSNR is dominated by how badly the input was degraded.

---

## 3. The 12 tools

Source: `src/rlrestore/tools/{specs,models,dataset,train,validate}.py`.

### 3.1 The toolbox table

Twelve specialists — four per degradation family, four severity bands each (`tools/specs.py`, indices are the action indices the agent uses):

| Idx | Family | Band | Arch |
|---|---|---|---|
| 0 | blur σ | [0, 1.25] | small |
| 1 | blur σ | [1.25, 2.5] | small |
| 2 | blur σ | [2.5, 3.75] | large |
| 3 | blur σ | [3.75, 5.0] | large |
| 4 | noise σ (0–255) | [0, 12.5] | small |
| 5 | noise σ | [12.5, 25] | small |
| 6 | noise σ | [25, 37.5] | large |
| 7 | noise σ | [37.5, 50] | large |
| 8 | JPEG Q | [60, 100] | small |
| 9 | JPEG Q | [35, 60] | small |
| 10 | JPEG Q | [20, 35] | large |
| 11 | JPEG Q | [10, 20] | large |

The harder bands (heavier blur, heavier JPEG) get the bigger `large` architecture; more damage needs more capacity to undo. Action **12 is STOP** (no tool).

### 3.2 Architectures

Both tool classes are fully convolutional with a **global residual** — the network predicts a *correction added to the input* (`out = x + body(x)`), so the identity map is the trivial default and the network only spends capacity on the difference between degraded and clean. This is the VDSR/DnCNN lineage.

**SmallToolCNN** (`models.py`), 21,827 params:

```
conv1  9×9  3 → 32  (ReLU)
conv2  5×5  32 → 16 (ReLU)
conv3  5×5  16 → 3
out = x + conv3
```

**LargeToolCNN**, 50,851 params, with two internal residual blocks:

```
conv1   5×5  3 → 64  (ReLU)
conv1_2 1×1  64 → 32             # b0 = pre-ReLU value, kept for the skip
conv2   3×3  32 → 32  (ReLU)     ┐ residual block 1: s2 = b0 + conv2a(relu(conv2(relu(b0))))
conv2a  3×3  32 → 32             ┘
conv2c  3×3  32 → 32  (ReLU)     ┐ residual block 2: s3 = s2 + conv2d_(relu(conv2c(relu(s2))))
conv2d_ 3×3  32 → 32             ┘
conv2_2 1×1  32 → 64  (ReLU)
conv3   5×5  64 → 3
out = x + conv3                  # global residual
```

(`conv2d_` has a trailing underscore so the attribute name doesn't shadow `nn.Conv2d`.) The topology and exact parameter counts are pinned by tests against the original's shipped TF checkpoints. The weight *init* is PyTorch default (Kaiming-uniform), not the TF original's Glorot-uniform — a deliberate, documented deviation; the topology is exact, the init is not.

Toolbox total: **6 × 21,827 + 6 × 50,851 = 436,068 params**. (Verified programmatically.)

### 3.3 Per-tool training

`tools/train.py::train_tool`. The recipe follows the paper's "standard VDSR setting":

| Hyperparameter | Value (`configs/full.yaml`) | Paper |
|---|---|---|
| Loss | MSE | MSE |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 | same |
| LR | 0.1, ×0.1 every 5 epochs | 0.1, ×0.1 every 20 of 80 |
| Grad clip | 0.4 (VDSR-style) | clipping (unspecified) |
| Batch | 64 | 64 |
| Steps/epoch | 4,000 | 4,000 (3.2e5 / 80) |
| Epoch cap | 40 (safety rail) | fixed 80 |
| `robust_extras` | true | Sec 3.1 "+noise/+JPEG" trick |

**Epochs are the deliberate deviation.** The paper trains a fixed 80 epochs/tool (~4–6 h each on this hardware); we cap at 40 and let **early stopping** decide. The stopping rule (`tools/train.py`): track best validation PSNR on a fixed held-out set; patience is in epochs, and patience only resets on a **meaningful** improvement — `MIN_IMPROVEMENT_DB = 0.02`. Micro-creep below 0.02 dB does not keep training alive. The best snapshot is deployed (`best.pt`), not the final weights. When asked mid-project whether 20 epochs was enough, we raised the cap to 40 and resumed the four tools that were still creeping; all plateaued within 3–4 bonus epochs.

`robust_extras` (`tools/dataset.py::degrade_for_tool`) implements the paper's Sec 3.1 robustness trick — adding slight extra noise + JPEG to *all* tool training data so a tool is less brittle to off-distribution input. The paper reports this is worth ~+0.2 dB but doesn't state the magnitudes; our documented assumption (extra noise σ∈[0,5] then JPEG Q∈[80,100] after the band degradation) is in `paper_config.md`.

Phase-1 training ran on a Colab A100 (~90 min for all 12 tools with the pyramid cache and 8 fork workers); the M4 finished and proved the plateaus. Measured M4 throughput (`reports/benchmark.json`): small tool 44.4 it/s, large 14.5 it/s on MPS at batch 64.

### 3.4 The specialist matrix

`tools/validate.py::specialist_matrix` applies every tool to every band's pure damage — a 12×12 grade sheet (`reports/tools_specialist_matrix.md`). Pairs whose base PSNR already exceeds 50 dB are excluded (the too-clean rule). The numbers:

- **Diagonal mean +0.77 dB** (each tool on its own band), min −0.44 dB.
- **Off-diagonal mean −2.17 dB**, with the worst cells around −10 to −14 dB (a heavy deblurrer thrown at heavy noise).

Eleven of twelve tools clearly help their own band. The lone diagonal negative is tool04 (denoise1, −0.44 on lab-pure mild noise): a tool whose `robust_extras` exposure left it slightly worse on the cleanest noise band than doing nothing — a measured robustness-vs-purity tradeoff, not a broken tool. This matrix is the empirical motivation for the dispatcher: applying the wrong tool doesn't merely fail to help, it actively destroys the image, so *which* tool and *in what order* is the entire problem.

`select_eval_indices` is the shared helper that picks which degraded/clean pairs count toward a tool's measured gain (it drops the too-clean pairs); it's reused by the joint fine-tuning forgetting monitor.

---

## 4. The DQN agent

Source: `src/rlrestore/agent/{env,net,replay,train,runner}.py`. Design spec: `docs/plan2/{fidelity,stability,infra}_spec.md`.

### 4.1 AgentNet

`agent/net.py`. A DRQN (recurrent DQN): a conv encoder, an FC reduction, the previous action concatenated in, an LSTM for episode memory, and a linear Q-head.

```
input 63×63×3 (current patch)  +  prev_action one-hot (12-dim)
conv1  9×9  3 → 32   s2 p4  (ReLU)    63 → 32
conv2  5×5  32 → 24  s2 p2  (ReLU)    32 → 16
conv3  5×5  24 → 24  s2 p2  (ReLU)    16 → 8
conv4  5×5  24 → 24  s2 p2  (ReLU)     8 → 4
flatten                               24·4·4 = 384
fc     384 → 32  (ReLU)
concat prev_action(12)                32 + 12 = 44
LSTM   44 → 50   num_layers=1, batch_first=False
q_head 50 → 13   (linear)
```

Total **88,063 params** (verified). Two contracts are worth stating exactly, because both are guarded by tests and both were sources of subtle bugs:

- **The previous-action one-hot is 12-dimensional, not 13.** STOP has no slot; it's encoded as all-zeros, the same as the t=0 state. The LSTM `input_size` is asserted to be 44 = 32 + 12. (Pitfall P2.)
- **The LSTM hidden state is owned by the caller, not the network.** `forward_step(img, prev_oh, hx) → (q, hx_new)` advances it one step; `hx=None` means zeros (episode start). `forward_episode_batch(imgs_GxTx…, prev_oh_GxTx…) → q_(G·T)×13` always unrolls from zeros because training episodes are complete from their start. The caller resets `hx=None` per episode (Pitfall P1).

The conv tower processes any leading batch dimension (1 for acting, G·T for the batched-episode forward); the encoder is shared between the two calling conventions via `_encode`.

### 4.2 RestoreEnv

`agent/env.py`. A minimal gym-style MDP with no gym dependency.

- **`reset(seed)`** picks a random training image, calls `make_training_pair(img, severity, rng)` → `(clean, current)` 63×63 float HWC, and records `initial_psnr = current_psnr`.
- **State** kept internally: `clean`, `current`, `initial_psnr`, `current_psnr`, `step_count`, `prev_action_onehot`.
- **`obs`** = `(current.copy(), prev_action_onehot.copy())`. The one-hot is 12-dim, zeros at t=0 and after STOP.
- **`step(action)`**:
  - **Tool action 0–11**: convert `current` HWC→NCHW, run the frozen tool under `no_grad` (output **unclipped**), convert back. `reward = psnr(next, clean) − current_psnr`. Increment `step_count`.
  - **STOP (12)**: identity, `reward = 0.0`, terminated.
  - Episode terminates when `step_count ≥ 3` (`_MAX_STEPS`). **The reward at the forced terminal step is still given** — it's computed before the termination check and always returned (Pitfall P4).
  - **Training-only early abort**: in `training_mode`, if after a tool step `current_psnr < initial_psnr`, terminate immediately (the negative reward is still returned). This stops the replay from filling with episodes that dug a deeper hole. It's off for validation (`training_mode=False`).

Tools are loaded once via `load_tools(cfg_name, device)` from `checkpoints/tools/<cfg_name>/toolNN/best.pt`, set to `eval()` with `requires_grad_(False)`.

### 4.3 Reward and the MDP

- `r_t = PSNR_{t+1} − PSNR_t` — a **dense per-step** signal (the per-step dB improvement). STOP gives 0. γ = 0.99.
- Action space: {0..11 tools, 12 = STOP}.
- Horizon: ≤3 tool applications. With ≤3 steps, γ=0.99 discounting is mild but still rewards setups (denoise first so a later deblur pays off). The dense reward is the deliberate fix relative to a prior failed project where a single sparse per-episode signal made learning impossible.

### 4.4 Episode-grouped replay

`agent/replay.py`. The LSTM needs to unroll a full episode from its start, so we cannot replay isolated transitions — we sample whole episodes.

- A **flat ring** over transitions, `capacity = 500,000`, lazily allocated on first push. Images are stored as **uint8** `(cap, 63, 63, 3)` (`(x*255).clip(0,255).astype(uint8)`), retrieved as float32/255. Actions uint8, rewards float32, terminals bool. ~6 GB RAM — 4× lighter than float32. Saturation of unclipped tool outputs at storage time is accepted (it matches the original's implicit uint8 path).
- An episode becomes visible to sampling only after `finalize_episode()` commits its pending steps atomically. `EpisodeRef(start, length 1–3)` lives in a deque. On ring wrap, any committed episode whose slots would be overwritten is evicted whole first — episodes are never partially valid.
- `push_step` is called at *decision time* (before the action), so the stored episode images are `s_0 … s_{T−1}`; the post-final image is never stored. `patch_last_step(reward, terminal)` backfills the actual reward/terminal after `env.step` returns.
- `sample_batch(n_episodes, rng)` returns a dict keyed by episode length, each value a list of materialized COPIES (`images` (T,63,63,3) float, `actions`, `rewards`, `terminals`) — never ring views.

### 4.5 Training schedules

`agent/train.py`. **Everything is derived from `env_step`, never stored** — so a resume from any checkpoint recomputes ε and lr exactly.

| Schedule | Formula | Values (verified) |
|---|---|---|
| Epsilon | `1.0 − 0.9·min(max(s−5000,0), 1e6)/1e6`, floor 0.1 | ε(0)=1.0, ε(1.005M)=0.1, ε(2M)=0.1 |
| LR | `max(1e-4·0.5^(s/1e6), 2.5e-5)` | lr(0)=1e-4, lr(1M)=5e-5, lr(2M)=2.5e-5 |

| Constant | Value |
|---|---|
| Total env steps | 2,000,000 |
| Warmup | 5,000 |
| Train every | 4 episodes |
| Target sync | every 10,000 env steps (≈ 2,500 updates) |
| Replay capacity | 500,000 transitions |
| Batch | 32 episodes |
| γ | 0.99 |
| Loss | Huber on the TD error |
| Optimizer | Adam |

### 4.6 The DQN train step

`dqn_train_step`. Sample 32 episodes, group by length T. For each group of G same-length episodes:

1. Build `imgs (G,T,3,63,63)`, `acts (G,T)`, `rews`, `terms`. Assert terminals are True only at `t == T−1`.
2. Build prev-action one-hots: `prev_oh[:,0]` = zeros; `prev_oh[:,t]` = one-hot(`acts[:,t−1]`) for t ≥ 1 (shift right by one). Only `acts[:,0:T−1]` feed prev-action slots, so a trailing STOP=12 is never used as a prev-action input (asserted).
3. Online `forward_episode_batch` → `q_online (G,T,13)`. Target `forward_episode_batch` over the *same* tensors → `q_target (G,T,13)`.
4. TD target: `target[g,t] = r[g,t] + γ·bootstrap` for `t < T−1`, `target[g,T−1] = r[g,T−1]` (terminal). The bootstrap reads the **next** stored index `t+1` (the temporal shift; Pitfall P3), and is zeroed wherever `terms[g,t]` (handles the early-abort edge).
5. Huber loss on `q_online[taken]` vs the detached target. Scale by `G / 32` so the summed gradient is the mean over all 32 episodes regardless of length-grouping. `backward()` accumulates across groups; one `opt.step()` per train step.

**Double-DQN** (`use_double_dqn`, on in `configs/full.yaml`): the bootstrap becomes `q_target[:, t+1, argmax_a q_online[:, t+1, a]]` instead of `q_target[:, t+1].max()` — the online net selects the action, the target net evaluates it. Vanilla `max` is the `use_double_dqn: false` path, kept as an ablation. See [§12](#12-engineering-decisions-and-notable-fixes) for why Double-DQN is here.

**Gradient clipping** (`grad_clip_norm`): `None` by default = paper-faithful no-clip. When set (5.0 in `configs/full.yaml`), `clip_grad_norm_` covers every parameter including the LSTM; the logged `grad_norm` is always the **pre-clip** value so the metric stays comparable across an intervention. Again, see [§12](#12-engineering-decisions-and-notable-fixes).

### 4.7 The chunked resumable runner

`agent/runner.py::run_training`. Built for supervised Colab training where the VM can die at any time.

- **Chunk** = 25,000 env steps. After each chunk: greedy validation on a fixed 256-episode set (seed 17), one `train_log.jsonl` metrics line, an atomic `status.json` heartbeat, and atomic tmp+rename checkpoints (`agent_net.pt`, `target_net.pt`, `optimizer.pt`, `train_state.json`). The outer loop is `while env_step < total: …` and exits immediately if already done (idempotent).
- **`train_state.json`** stores `env_step`, `wall_seconds`, `updates_done`, the EMA states, and both RNG states — but **never ε or lr** (derived). On resume it reloads net/opt/env_step, then runs a **replay refill**: `min(50_000, env_steps//4)` env steps at the current ε with the optimizer frozen, re-warming the VM-RAM-only buffer without incrementing `env_step`. A session death costs ≈2 minutes of progress.
- **`status.json`** is the supervision heartbeat: `{env_step, epsilon, lr, last_val_psnr, buffer_fill, updates_done, wall_sec_total, last_chunk_sec, session_id, updated_utc}`. Atomic rename syncs over Drive FUSE in 60–90 s; poll `updated_utc` and if it's stale > 5 min the session is dead and the user re-runs the cell (auto-resume). The `jsonl` lags minutes (open-file FUSE) and is for trend only.
- **NaN guard** (revised after the F5 review): `dqn_train_step` runs `opt.step()` internally, so a NaN loss means the in-memory weights are *already* poisoned. The runner therefore **halts** — writes a `nan_halt` status and raises, touching no checkpoint — so the last good rolling checkpoint survives and a re-run resumes from it. (The original guard zeroed grads and continued, which would have overwritten the only clean checkpoint at the next chunk boundary.)
- **Immutable snapshots** every 250,000 env steps copy the four checkpoint files to `snapshots/step_<N>k/` — insurance against a *soft* (non-NaN) divergence silently contaminating the rolling checkpoint.

On an A100 the 2M-step run is ~33 min GPU-active; budgeting 2× overhead gives ~70 min wall. Multi-env vectorization is deliberately avoided (the episode-grouped replay + per-episode LSTM reset make it fragile, and the speedup isn't needed).

---

## 5. Evaluation and baselines

`scripts/evaluate_agent.py` builds a fixed, deterministic, per-class-disjoint episode set from the held-out TEST split (DIV2K 751–800): **500 episodes per class**, seeds **mild 10000.. / moderate 20000.. / severe 30000..**. Because the episodes are fully determined by `(images, seeds, severity)`, the agent and every baseline see byte-identical degraded images per class — every comparison is apples-to-apples. The metric is PSNR gain over the degraded input, reported per class and as the unweighted overall.

The baseline ladder (`agent/train.py::compute_baselines`):

| Code | Baseline | What it is |
|---|---|---|
| B0 | Do nothing | The floor (gain 0 by definition) |
| B1 | Best single tool | Apply each tool once, report the best tool's mean gain |
| B2 | Random 3-chain | Three uniformly-random tools (100 chains/episode), mean ± std |
| B3 | Greedy oracle | At each of ≤3 steps, try all 12 tools and apply the one that most increases `PSNR(·, clean)`; no STOP |
| B4 | Oracle + STOP | B3 but stop when no tool gives a positive gain — the toolchain ceiling |

B3/B4 are **oracles**: they peek at the clean reference (unavailable at deployment) to make per-step-optimal tool choices. They are *informed upper bounds* on what any greedy dispatcher restricted to these 12 frozen tools could achieve — a ceiling to measure the real agent against, not a method we'd ship.

The deployment health check (`scripts/check_tool_health.py`, `reports/tool_health_deployment.json`) is a separate gate run before agent training: it applies 0/1/2 random tools to 500 moderate TRAIN episodes to build realistic mid-chain states, then measures each tool's ΔPSNR distribution on them. The naïve gate "every tool's mean ≥ 0" was revised after it failed the *entire blur family* (means −0.09 to −2.8) — which is correct problem physics, not a defect: deblurring amplifies residual noise, so blur tools are net-negative at random chain positions and the agent's job is to learn to apply them late. The revised gate is per-tool `frac_positive ≥ 0.05 AND p90 > 0` ("helps somewhere") plus a greedy oracle over the same states gaining > +1.0 dB ("a good policy exists"). It passed: `gate_pass: true`, no disabled or defect tools, oracle gain +8.39 dB on these mid-chain states.

---

## 6. Results

All figures from `reports/eval_test/`, `reports/eval_vanilla/`, `reports/eval_monolith/`. 500 episodes/class, seed-matched, PSNR gain in dB over the degraded input.

### 6.1 Headline table

| Method | mild | moderate | severe | overall |
|---|---|---|---|---|
| B0 do nothing | 0.00 | 0.00 | 0.00 | 0.00 |
| B2 random 3-chain | −2.31 ± 5.27 | −1.27 ± 4.71 | −1.02 ± 4.81 | **−1.53** |
| B1 best single tool | 1.51 (tool06) | 2.34 (tool06) | 3.16 (tool07) | **2.34** |
| **Agent (Double-DQN)** | 2.24 | 2.83 | 3.71 | **2.93** |
| Agent (vanilla-DQN) | 2.24 | 2.90 | 3.67 | **2.94** |
| B3 greedy oracle | 2.67 | 3.10 | 3.84 | **3.20** |
| B4 oracle + STOP | 2.73 | 3.13 | 3.86 | **3.24** |
| Monolith CNN (448,195 params) | 3.38 | 3.91 | 4.79 | **4.03** |

Readings:

- **Random button-mashing makes images worse** (−1.53, with huge variance), which quantifies how dangerous the action space is — picking right is the whole game.
- **The agent (+2.93) clearly beats the best single tool (+2.34)** and lands at **+2.93 / +3.24 ≈ 90.4% of the oracle-with-STOP ceiling.** The gap to the oracle is the agent's routing imperfection — the price of having no clean reference at test time.
- **Gains grow with severity** (mild 2.24 → severe 3.71) for every method, because severe images start from a worse PSNR and so have more headroom.
- B1's best single tool is always a **denoise** tool (tool06 on mild/moderate, tool07 on severe) — noise dominates this degradation mix.

### 6.2 Stop behaviour and chain length

The Double-DQN agent (`reports/eval_test/`):

| | mild | moderate | severe | overall |
|---|---|---|---|---|
| stop rate t1 / t2 / t3 | 0.02 / 0.16 / 0.30 | 0.01 / 0.10 / 0.28 | 0.00 / 0.07 / 0.22 | 0.01 / 0.11 / 0.27 |
| mean chain length | 2.31 | 2.48 | 2.64 | **2.48** |

The agent almost never stops at step 1 (an image always needs *something*) and stops more often on mild than severe — exactly the right instinct, since severe images keep rewarding more work. STOP-fraction over all actions (`reports/tool_value/`): mild 17.3%, moderate 13.6%, severe 9.8%.

### 6.3 The Double-DQN ablation

Run to convergence on the identical TEST set, vanilla and Double-DQN essentially **tie**: vanilla +2.94 vs Double +2.93 overall (`reports/eval_vanilla/` vs `reports/eval_test/`). Double-DQN did **not** buy a higher final score. What it bought was *training stability* — see [§12](#12-engineering-decisions-and-notable-fixes). We report the tie rather than overselling the technique.

### 6.4 Why the monolith beats even the oracle

The monolith (+4.03) beats not only the agent (+2.93) but the toolchain oracle ceiling (+3.24). This is not a contradiction. The oracle is the ceiling *conditioned on the 12 frozen tools* — the best any greedy dispatcher restricted to composing those exact weights can do. The monolith is an unconstrained function in a richer hypothesis space (448,195 params, end-to-end), not bound by the tools' fixed weights, their per-band specialization, or the ≤3-step limit, so it can represent restorations outside the span of the toolchain. Beating the toolchain oracle is evidence that the frozen-tool function class is itself the binding constraint. There is also a training-exposure asymmetry that *favours* the monolith and which we disclose: the monolith trained on all three severities, the agent's dispatcher on moderate only (details in [§9](#9-the-monolithic-cnn-baseline)).

The agent's contribution is not peak PSNR — it's modularity, per-image adaptivity, extensibility, and a step-by-step legible decision trace. This matches the original paper's own framing (it claims results only "comparable" to single-degradation baselines on the mixed task) and the broader pixel-RL literature.

---

## 7. Toolbox sizing and tool value

`scripts/analyze_tool_value.py`, `reports/tool_value/tool_value.md`. Two questions: is 12 the right number of tools, and which tools earn their keep? 200 episodes/class (600 pooled), the same seed scheme as the main eval.

### 7.1 Forward selection (the right lens for size)

Build the toolbox greedily from zero, always adding the single tool that most raises the no-STOP greedy-oracle ceiling. Unlike leave-one-out, this is safe to read for sizing — it's the ceiling reachable with the best *k* tools:

| k | tool added | ceiling dB | % of full-12 |
|---|---|---|---|
| 1 | denoise2 | +2.040 | 65.6% |
| 2 | deblur1 | +2.627 | 84.5% |
| 3 | denoise3 | +2.856 | **91.9%** |
| 4 | deblur3 | +2.982 | **95.9%** |
| 5 | dejpeg4 | +3.045 | 97.9% |
| 6 | deblur2 | +3.074 | 98.8% |
| … | … | … | … |
| 12 | dejpeg3 | +3.110 | 100.0% |

**Roughly 4 well-chosen tools (denoise2, deblur1, denoise3, deblur3) capture ~96% of the achievable gain** on this damage mix; 12 is generous insurance for fine routing. This matches the paper's own finding that 12 beat 6 but 18 was flat-to-worse. The full-12 oracle ceiling is +3.110 pooled (mild +2.609, moderate +3.022, severe +3.698).

### 7.2 The leave-one-out caveat

We *also* measured each tool's marginal value by removing it and watching the ceiling drop. The drops are tiny — largest 0.103 dB (denoise2), most under 0.05 — and **you cannot sum them to argue for a 2-tool toolbox.** The tools come in graded families (denoise1/2/3/4); removing denoise3 barely moves the ceiling *only because* its neighbour denoise2 quietly covers for it. The small per-tool drops measure pairwise *substitutability* within a band, not group *removability*; the strong specialists (denoise3/4, own-band +2.24/+2.95 dB) read as "redundant" only because a neighbour substitutes. Summing correlated-feature marginals is the classic feature-importance aggregation fallacy. Forward selection is also greedy, so it's a tight, honest *estimate* of how few tools suffice, not a certified optimum.

---

## 8. Joint fine-tuning (Algorithm 1)

`src/rlrestore/agent/joint_finetune.py`, `scripts/train_joint_finetune.py`. Spec: `docs/plan3/joint_finetune_spec.md`. This is the project's primary "beyond the paper" piece.

### 8.1 Motivation

The 12 tools were trained in isolation on lab-pure single-band damage, then frozen. But in a chain, tool *k* sees the *output of tool k−1* — a partially-cleaned, off-distribution image it never trained on. The specialist matrix shows tools lose ~2.17 dB off-band. RL-Restore's Algorithm 1 closes the loop: let the chain's final error fine-tune the tools on the mid-chain distribution they actually receive. The authors describe it in the paper (reporting ~+0.25 dB) but the public repo only trains tools in isolation — there is no released code for this step.

### 8.2 The algorithm

The paper's Algorithm 1, faithfully. The **agent stays frozen** (the published algorithm updates tools only; unit test T1 asserts agent params are byte-identical before/after a step). The tools are trainable, initialized from `checkpoints/tools/full/`. Per batch of B images (`joint_ft_step`):

1. **Chain selection (no grad).** For each image, the frozen agent greedily picks its ≤3-tool chain from the *current* (drifting) tools (`select_and_apply_chain`). Re-selected every iteration so routing stays valid as the tools change. The agent acts on the **detached** state, so its discrete argmax carries no gradient and its LSTM never enters the autograd graph.
2. **Differentiable replay (with grad).** Re-apply the selected tools in order, in `train()` mode, extending the autograd graph on the image path. The only differentiable path is `degraded → tools → restored`.
3. **Loss** = `MSE(restored, clean)`, averaged over the images that applied ≥1 tool. Images where the agent STOPs at step 0 contribute nothing and are excluded from the mean.
4. **Backprop** accumulates each tool's gradient, summed over every application across the batch.
5. **Per-tool use-count averaging.** Before `opt.step()`, scale tool *t*'s accumulated grad by `1/uses_t` (in code: `scale = 1.0 / ut`, applied for `ut > 1`). A tool used twice as often shouldn't take twice the step. (Unit test T3.)
6. **Optional grad-norm clip** (default 0.5) over the tool params, then `opt.step()` on the tools only.

Optimizer: Adam, lr = 1e-4 (the paper's α), no weight decay. Severities sampled uniform over {mild, moderate, severe} (same reasoning as the monolith — the tools must improve across the full deployment family the frozen agent routes at test). Iteration cap 200,000 (the paper's 2e5) as a safety rail; **early stopping** on the whole-system val gain is the real judge (`MIN_IMPROVEMENT_DB = 0.02`, patience 6 chunks).

`run_joint_finetune` mirrors the agent runner's discipline: chunked (5,000 iters), atomic checkpoints, `status.json` heartbeat, `train_log.jsonl` per chunk, NaN-halt without overwriting the best tools, snapshots, resume. **Originals in `full/` are never overwritten** — fine-tuned tools land in `checkpoints/tools/full_ft/` (unit test T5). The output is a drop-in for `evaluate_agent.py --tools-name full_ft`.

### 8.3 The catastrophic-forgetting monitor

Fine-tuning a specialist to be a good team player in the chain can erode its solo-band sharpness — the specialist matrix diagonal can drop even as end-to-end PSNR rises. Rather than hide this, we instrument it: `own_band_gains()` recomputes each tool's own-band ΔPSNR every eval chunk (a cheap proxy for the Phase-1 matrix diagonal, reusing `ToolPairDataset` + `select_eval_indices`), and we log per-tool weight drift `‖θ_ft − θ_orig‖`. A system improving while individual tools become slightly more generalist is the *expected* outcome of training on the real distribution, and we report it as a finding.

### 8.4 Result so far

On the system validation set, fine-tuning raised the whole-system gain from **+3.18 to +3.61 — +0.43 dB over the frozen-tool baseline**, exceeding the paper's reported ~+0.25 dB for this step.

The held-out **TEST** result (DIV2K 751–800, the byte-identical episodes used for every other number in this document) is larger still. The agent is *unchanged* — `agent_net.pt`, sha256 `824040df…` — and only the toolbox is swapped (`full/` → `full_ft/`, via `evaluate_agent.py --tools-name full_ft`):

| metric | frozen tools | fine-tuned tools | Δ |
|---|---|---|---|
| Agent overall | +2.93 | **+3.67** | **+0.74** |
| — mild / moderate / severe | 2.24 / 2.83 / 3.71 | 2.88 / 3.57 / 4.55 | |
| Best single tool (B1) | +2.34 | +3.09 | +0.75 |
| Greedy-oracle ceiling (B3) | +3.20 | +4.25 | +1.05 |
| Random 3-chain (B2) | −1.53 | +1.30 | +2.83 |

The absolute level matches validation (≈+3.6 dB); the wider margin reflects the frozen tools starting lower on the harder test split (+2.93 vs +3.18), so the same fine-tuned tools clear a bigger gap. Because fine-tuning lifts the *whole* toolbox, every tool-using method rose — including the oracle ceiling, which now sits **above** the +4.03 monolith. The fine-tuned agent (+3.67) narrows the gap to the monolith from 1.10 dB to **0.36 dB**, and the random-chain swing (−1.53 → +1.30) shows the tools learned to compose safely, not just sharpen individually. Raw artifacts: `reports/eval_joint_ft/`.

---

## 9. The monolithic-CNN baseline

`src/rlrestore/baselines/monolith.py`, `scripts/{train_monolith,evaluate_monolith}.py`, `configs/monolith.yaml`. Spec: `docs/plan3/monolith_spec.md`. The headline control: a single end-to-end CNN vs the "12 frozen tools + dispatcher" system.

### 9.1 Architecture

`MonoRestoreCNN`, a DnCNN-style network with depth D and width W:

```
Conv(in → W, 3×3, p1) + ReLU                      # head
(D−2) × [Conv(W → W, 3×3, p1) + BN(W) + ReLU]     # body
Conv(W → in, 3×3, p1)                              # tail
out = x − body(x)                                  # global residual (residual-as-noise, DnCNN)
```

Note the residual sign differs from the tools: the monolith predicts the *noise to subtract* (`x − body(x)`), the DnCNN convention. Output is **unclamped** for loss/PSNR (matching the metric convention); clamp to [0,1] only for display via `restore_for_display`.

At **D=14, W=64** the parameter arithmetic (BN running buffers are not parameters) gives:

```
head = (3·9+1)·64                  =    1,792
body = 12 · [(64·9+1)·64 + 2·64]   =  444,672
tail = (64·9+1)·3                  =    1,731
total                              =  448,195  (verified)
```

### 9.2 The fairness controls

The comparison is designed to be defensible (`monolith_spec.md` decision log):

- **Capacity matched to the toolbox, not the agent head.** 448,195 = **102.8% of the 436,068-param toolbox.** The 88,063-param dispatcher head is routing overhead, not restoration capacity, so the headline match excludes it. (A 253,203-param W=48 variant exists for a capacity sweep.) Receptive field 29×29 covers the 21×21 blur kernel with margin.
- **Severity exposure — UNIFORM over all three classes.** This was resolved in favour of the training panel. The agent's *restoration capacity* is its 12 tools, each trained on its own band, collectively spanning the full grid; the fair control holds restoration-capacity training exposure equal, so the monolith trains on the full severity family too. This if anything **disadvantages the agent**, whose *dispatcher* never saw a severe episode during training yet must route severe images at test — so an agent win would only be stronger. A moderate-only monolith is kept as a labelled ablation.
- Loss Charbonnier (`sqrt((pred−clean)² + ε²)`, ε=1e-3; MSE selectable); Adam lr 2e-4, wd 1e-4, grad-clip 0.5, batch 64, 8,000 steps/epoch, ×0.5 every 8 epochs. Early stopping patience 6 + `MIN_IMPROVEMENT_DB` 0.02 (the tools' methodology verbatim). The loss choice never touches the comparison metric, which stays unclamped PSNR.
- Same data pipeline (`make_training_pair`, the uint8→/255 branch, the 3-scale pyramid, the >50 dB skip), `robust_extras: false` (the input recipe already injects noise+JPEG, keeping train/eval parity).

`evaluate_monolith.py` is a sibling of `evaluate_agent.py` (it does not touch it): it rebuilds the byte-identical TEST episodes via the same `RestoreEnv` + seed scheme, runs a single unclamped forward (no chain), and emits a "B5" row in the same table format.

### 9.3 Result

| Method | mild | moderate | severe | overall |
|---|---|---|---|---|
| B5 single end-to-end CNN (448,195 params) | 3.38 | 3.91 | 4.79 | **4.03** |

The monolith wins on raw PSNR. We report it honestly and position the agent on transparency and modularity, not peak quality (see [§6.4](#64-why-the-monolith-beats-even-the-oracle)).

---

## 10. The web app

`hf_space/`. Spec: `docs/plan3/webapp_spec.md`. A demo for a HuggingFace Docker Space — custom FastAPI + React in a single CPU-only container, *not* Gradio. It imports the `rlrestore` package so the serving math is identical to the evaluated models (nothing is re-implemented).

### 10.1 Architecture

- **Backend** (`hf_space/backend/`, FastAPI). A `ModelRegistry` (`models.py`) loads every checkpoint **once** in the FastAPI lifespan and stashes it on `app.state`. It loads the agent (bare state_dict → `AgentNet`, `weights_only=True`, from `checkpoints/agent_full_v2/agent_net.pt`), the `full` tools and (if present) the `full_ft` tools via `load_tools`, and the monolith (`{"model": sd, "val_psnr"}` → `MonoRestoreCNN(14,64)`, `weights_only=False`). **Any method whose checkpoint is absent is omitted gracefully** — `available_methods` reflects only what loaded, so the app runs with whatever is present (full_ft may arrive later). The agent + `full` toolbox are the required headline path. Posture: `torch.set_grad_enabled(False)`, `set_num_threads(2)`, CPU.
- **Frontend** (`hf_space/frontend/`, Vite + React + TypeScript). A dark, single-page demo whose UX is built around the project's honest thesis: the agent is the "shows-its-work" star (step-by-step restoration with a climbing quality gauge), the monolith is the labelled "strong silent baseline (+4.03 highest PSNR)". Screens: hero, method picker, input (upload / camera capture via native `<input capture>` / example gallery / damage simulator), the step-by-step restoration hero, and a before/after compare with a Q-value explainability panel. Mounted **last** as a StaticFiles SPA with an `index.html` fallback (after the `/api` router), guarded so the backend runs standalone for tests.

### 10.2 The 63×63 policy crop / full-image tools split

`AgentNet`'s `Linear(384, …)` is hard-wired to 63×63 input, but the tools are fully convolutional. So the rollout runs two image tracks per step (`inference.py::run_agent_chain`): the **policy** decides from a 63×63 **center-crop** of the current full image, and the **chosen tool is applied to the full (≤512px) image.** Inputs are capped to 512px long-edge before any model runs (the master latency knob; downscale only, never upscale). This is documented as a modeling note — it's the one place the demo departs from the patch-only training regime.

### 10.3 API

Under `/api`, images as base64 data URLs in JSON:

- `GET /api/health` → `{status, models_loaded, device}`
- `GET /api/methods` → `{methods[], tools[12], severities, stop_action_index: 12, limits}`
- `POST /api/restore` (multipart: file, method ∈ {agent, agent_ft, monolith}, optional reference) → per-step trace `{index, label, action_index, action_name, image, quality, quality_delta, q_values?}` plus `final_image` and a `summary`. The agent rollout loops `forward_step(63×63 crop, prev_oh, hx) → argmax`; a==12 STOPs, else apply the tool to the full image, ≤3 steps. The monolith is presented as a 2-entry `[Degraded, Restored]` chain for a uniform UI.
- `POST /api/simulate` (multipart: clean file, severity, optional seed/then_restore) → degrades a clean upload with `sample_recipe` + `apply_recipe` on the full resized image and returns the recipe params + before/after.

Quality is full-reference PSNR on the simulator path (a clean reference exists) and a NumPy no-reference sharpness/noise proxy (labelled "estimated") for arbitrary uploads. Display images are always clamped to [0,1]; PSNR is computed on the unclamped working image to match how the models were evaluated.

### 10.4 Container

A single multi-stage Docker image (port 7860): `node:20` builds the frontend to `dist/`, then a `python:3.11-slim` runtime installs CPU torch wheels, the `rlrestore` package, the backend, and the checkpoints (~7 MB baked in). Same-origin (frontend + API in one container) so no CORS. The README ships the exact `git push` deploy steps for the user's HF account.

---

## 11. Reproducibility

### 11.1 Environment

Managed with `uv` (`pyproject.toml`, `uv.lock`). `uv sync` creates the venv and installs deps; `uv run pytest` runs the suite. Device selection (`common/device.py`): `RLR_DEVICE` env var > explicit arg > MPS if available > CPU.

| Env var | Effect |
|---|---|
| `RLR_DEVICE` | Force the torch device (`cpu` / `cuda` / `mps`). Wins over everything. |
| `RLR_CACHE_PYRAMID` | `1` pre-caches the 3-scale pyramid per image (faster dataloading, +RAM). |
| `RLR_LOADER_WORKERS` | DataLoader worker count for tool/monolith training (0 = main process). |
| `RLR_PATIENCE` | Early-stopping patience (epochs) for tool training. |

### 11.2 Commands

```bash
# data
uv run python -m rlrestore.data.div2k download        # fetch + verify DIV2K

# tools (Phase 1)
uv run python scripts/train_all_tools.py --config configs/smoke.yaml   # pipeline check
uv run python scripts/train_all_tools.py --config configs/full.yaml    # the real 12 tools
uv run python -m rlrestore.tools.validate --config configs/full.yaml   # specialist matrix

# pre-training gate
uv run python scripts/check_tool_health.py --config configs/full.yaml

# agent (Phase 2)
uv run python scripts/train_agent.py --config configs/smoke.yaml --smoke   # S1–S7 gate
uv run python scripts/train_agent.py --config configs/full.yaml --device cuda
#   vanilla ablation: same with configs/full_vanilla.yaml (use_double_dqn: false)

# monolith baseline
uv run python scripts/train_monolith.py --config configs/monolith.yaml

# joint fine-tuning (Algorithm 1)
uv run python scripts/train_joint_finetune.py \
    --agent-ckpt checkpoints/agent_full_v2/agent_net.pt \
    --in-tools-name full --out-tools-name full_ft

# evaluation (held-out TEST)
uv run python scripts/evaluate_agent.py --tools-name full --per-class 500
uv run python scripts/evaluate_agent.py --tools-name full_ft --per-class 500   # joint-FT system
uv run python scripts/evaluate_monolith.py
uv run python scripts/analyze_tool_value.py                                    # tool value + sizing
```

`configs/`: `smoke.yaml` (fast pipeline check), `full.yaml` (the real tools + agent), `full_vanilla.yaml` (`use_double_dqn: false` ablation), `monolith.yaml`. The smoke agent config asserts the S1–S7 stability gates and exits.

### 11.3 The Colab/Drive workflow

No local GPU, so the heavy training ran on Colab's A100 with checkpoints to Google Drive (`infra_spec.md`). A single idempotent bootstrap cell mounts Drive, unzips the code, `pip install -e .`, links `checkpoints → Drive`, sets `RLR_DEVICE=cuda` + `RLR_CACHE_PYRAMID=1`, runs the smoke gate then the tool-health gate (both fail loudly), then the chunked training loop teeing to a Drive log. Re-running the cell auto-resumes from the last checkpoint (the runner reconstructs ε/lr from `env_step` and refills the replay). Supervision is by polling `status.json` (atomic rename, FUSE-aware) every few minutes. Budgets: ~70 min for the 2M-step agent on A100, ~90 min for the 12 tools, ~30–90 min to early-stop joint fine-tuning. The reported eval numbers carry their checkpoint sha256[:16] and run date in each report's protocol footer.

---

## 12. Engineering decisions and notable fixes

### The uint8 white-patch regression (the central ML lesson)

`make_training_pair` originally assumed float input. When `load_images` started returning uint8 (the RAM-saving change), the function clipped 0–255 values to [0,1] — saturating every clean patch to ~pure white. The Phase-2 agent's first full 2M-step run trained entirely on that degenerate world. **The training metrics looked fine the whole time** (the loss was happy); it surfaced only in held-out *qualitative* evaluation, where the white patches were obvious. Fixed by the dtype branch in `pipeline.py` (commit `0de93e6`, "make_training_pair normalizes uint8 input (CRITICAL)"). The takeaway, which the codebase comments preserve at the fix site: aggregate training metrics are a weak guardrail; out-of-distribution held-out eval and *looking at the outputs* is where silent data bugs die.

### Double-DQN as pre-registered insurance (F1)

The first 5k-step smoke run was machinery-healthy (no NaN, loss finite, training reward rising +0.12 → +0.54, Q magnitudes sane) but the greedy policy collapsed onto a single tool — **tool02 at 96% share → val gain −0.36 dB**. That's the textbook vanilla-DQN overestimation signature: `max_a Q` couples action selection and evaluation in one noisy network and systematically over-rates whichever action got lucky. Double-DQN decouples them (online selects, target evaluates). We enabled it preemptively (`stability_spec` fallback rung F1) in `full.yaml` and `smoke.yaml`, keeping vanilla as a toggle. The honest postscript ([§6.3](#63-the-double-dqn-ablation)): at convergence the two tie (+2.94 vs +2.93), so Double-DQN bought *stability*, not points.

The same smoke run forced three smoke-gate recalibrations, all documented in `train.py::check_smoke_thresholds`: the positive-gain floor (unreachable at 5k steps, high ε) was relaxed to a catastrophic-harm floor; the entropy floor (winner-take-all argmax is canonical early-training) was relaxed to a single-action-only check; and the `q_ratio` lower bound was removed because `Q*(STOP) = 0` exactly by Bellman (STOP gives reward 0 and is terminal, so there's no bootstrap).

### Gradient explosion and the pre-registered clip (F5)

On the production A100 run, `grad_norm_ema` grew **0.56 (25k) → 4.2 (200k) → 7.6 (250k) → 14.5 (400k) → 20 (500k) → 44 (550k)** — doubling roughly every 87k steps — while the Huber-bounded `td_loss_ema` stayed ~1.1 and the Q-values stayed calibrated (5–7, below the +7.22 oracle ceiling). The diagnosis was network-side Jacobian-norm growth in the *unclipped* LSTM under Adam with no weight decay (Huber caps the loss-side gradient, so the growth is on the network side). At trend the EMA would have hit ~10⁶ before 2M steps; the run could not finish unclipped. We intervened at ~550k per the pre-registered F5 rung, setting `grad_clip_norm = 5.0` (default `None` = paper-faithful). A 3-angle review also found the original NaN guard couldn't recover a diverged run — `opt.step()` ran before the check, and the single rolling checkpoint would have been overwritten with poisoned weights — so the guard was changed to **halt without checkpointing** and immutable 250k-step snapshots were added. The learning curve is annotated "grad clip enabled at ~575k" for disclosure. Under Adam, steady-state uniform clipping ≈ gradient normalization, so its real effect is spike insurance, not a regime change.

### The MPS-LSTM CPU fallback

PyTorch's LSTM crashes on Apple MPS with a native `MPSNDArray` slicing assertion (exit 134) for this shape mix — the early "LSTM fine on MPS" assumption was wrong (`infra_spec` §6 correction). Local smoke/test runs therefore use `RLR_DEVICE=cpu` (the nets are tiny, CPU is fine); production is CUDA-only. `evaluate_agent.py` and `train_joint_finetune.py` both auto-fall-back from MPS to CPU. Joint fine-tuning sidesteps this in its hot path anyway, because the agent's LSTM runs in `no_grad` for selection and never enters the gradient graph.

### macOS spawn DataLoader memory

On macOS the DataLoader uses spawn, which re-pickles the (large, uint8) image list per worker. Tool/monolith training expose `RLR_LOADER_WORKERS` (default 0 = main process) so the worker count is tunable to the host; the Colab fork-based workers (8) are where the parallelism actually pays off.

### The .gitignore that swallowed a module

A repo-wide `data/` ignore pattern matched `src/rlrestore/data/mixed_dataset.py` and silently excluded it from version control — code that worked on the authoring machine but would have been *missing* for anyone who cloned. A reproducibility landmine, fixed by anchoring the ignore rule to the repo-root `data/` (commit `7b49165`, "track mixed_dataset.py; anchor .gitignore data/ to repo root").

---

## 13. Where the numbers come from

| Figure(s) | Source |
|---|---|
| Agent +2.93 (2.24/2.83/3.71); stop rates; chain length 2.48; B1 +2.34; B2 −1.53; B3 +3.20; B4 +3.24 | `reports/eval_test/agent_eval_test.{md,json}` |
| Vanilla-DQN +2.94 (2.24/2.90/3.67); chain 2.37 | `reports/eval_vanilla/agent_eval_test.md` |
| Monolith +4.03 (3.38/3.91/4.79); 448,195 params | `reports/eval_monolith/monolith_eval_test.{md,json}` |
| Forward selection (3→91.9%, 4→95.9%); leave-one-out drops; full-oracle +3.110; own-band gains | `reports/tool_value/tool_value.{md,json}` |
| Specialist matrix +0.77 diag / −2.17 off-diag / min −0.44; worst ≈ −14 | `reports/tools_specialist_matrix.md` |
| Tool-health gate pass; per-tool mid-chain ΔPSNR; oracle +8.39 | `reports/tool_health_deployment.json` |
| M4 throughput 44.4 / 14.5 it/s | `reports/benchmark.json` |
| 12-tool table, bands, archs; DQN hyperparams; reward; Algorithm 1; deviations | `docs/paper_config.md` |
| Param counts (88,063 agent / 21,827 small / 50,851 large / 436,068 toolbox / 448,195 monolith / 524,131 tools+agent); ε & lr schedule values | verified programmatically against `agent/net.py`, `tools/models.py`, `baselines/monolith.py`, `agent/train.py` |
| Smoke-collapse −0.36; Double-DQN (F1); grad-explosion 0.56→44 + clip 5.0 (F5); baseline ladder | `docs/plan2/stability_spec.md` |
| Capacity match 102.8%; uniform-severity training; fairness controls | `docs/plan3/monolith_spec.md` + `src/rlrestore/baselines/monolith.py` |
| Algorithm 1 details; `1/uses` averaging; ~+0.25 paper claim; forgetting monitor | `docs/plan3/joint_finetune_spec.md` + `src/rlrestore/agent/joint_finetune.py` |
| Joint-FT validation +3.18 → +3.61 (+0.43 dB) | project notes (validation run) |
| Joint-FT **TEST** +2.93 → +3.67 (+0.74 dB); oracle ceiling +3.20 → +4.25 | `reports/eval_joint_ft/agent_eval_test.{json,md}` |
| Colab/Drive workflow; MPS-LSTM crash | `docs/plan2/infra_spec.md` |
| White-patch fix `0de93e6`; .gitignore fix `7b49165` | git history |
