# RL-Restore — paper & official-code ground truth (extracted 2026-06-11)

Two extraction passes were run (official repo code @ `yuke93/RL-Restore` master `dd1463d`; paper text arXiv:1804.03312 incl. LaTeX source). Facts used in this plan:

**The 12 tools** (paper Table 1 = repo issue #1 table; architectures decoded from shipped TF checkpoints, param counts verified):

| Tool idx | Target | Band | Arch |
|---|---|---|---|
| 0 | Gaussian blur σ | [0, 1.25] | small |
| 1 | Gaussian blur σ | [1.25, 2.5] | small |
| 2 | Gaussian blur σ | [2.5, 3.75] | large |
| 3 | Gaussian blur σ | [3.75, 5] | large |
| 4 | Gaussian noise σ (0–255 scale) | [0, 12.5] | small |
| 5 | Gaussian noise σ | [12.5, 25] | small |
| 6 | Gaussian noise σ | [25, 37.5] | large |
| 7 | Gaussian noise σ | [37.5, 50] | large |
| 8 | JPEG quality Q | [60, 100] | small |
| 9 | JPEG quality Q | [35, 60] | small |
| 10 | JPEG quality Q | [20, 35] | large |
| 11 | JPEG quality Q | [10, 20] | large |

- **small** = conv 9×9 3→32 / ReLU / conv 5×5 32→16 / ReLU / conv 5×5 16→3, plus **global residual** (output = input + conv3). Params = 21,827.
- **large** = conv1 5×5 3→64 / ReLU / conv1_2 1×1 64→32 / ReLU / [two residual blocks: conv 3×3 32→32 / ReLU / conv 3×3 32→32, skip-added] / conv2_2 1×1 32→64 / ReLU / conv3 5×5 64→3, plus **global residual**. Params = 50,851. (Skip sources: block 1 adds conv1_2 pre-ReLU output; block 2 adds block 1's sum.)

**Degradations** (repo `generate_train.m` + paper Sec 4): always composed **blur → noise → JPEG** on float RGB [0,1]; blur = Gaussian, **21×21 kernel**, replicate border, σ ∈ [0,5]; noise = zero-mean Gaussian, σ ∈ [0,50] on 0–255 scale, output clipped to [0,1]; JPEG = real encode/decode round-trip, Q ∈ [10,100]. Severity recipes: each type has a 10-band grid — blur `0:0.5:5`, noise `0:5:50`, JPEG `[100,80,60,50,40,35,30,25,20,15,10]` (non-uniform, descending) — a recipe is a level triple `(k,m,n)`, each 1..10, with severity class defined by `k+m+n−2`: **mild [9,11], moderate [12,17] (training default), severe [18,20]**; sampled value is uniform *within* the chosen band.

**Data** (paper Sec 4): DIV2K; train = images 1–750, test = 751–800, validation = DIV2K valid set (801–900). 63×63 patches; multi-scale pyramid at scales **1, 2/3, 1/3** (code) [paper says "down-scaling by 2, 3, 4" — code wins, discrepancy noted]. Patches with PSNR(degraded, clean) > 50 dB are skipped.

**Tool training** (paper Sec 3.3; no code released): MSE loss, batch 64, 80 epochs = 3.2×10⁵ iters (⇒ 4,000 iters/epoch), SGD LR 0.1 ×0.1 every 20 epochs, "standard setting in VDSR" (⇒ momentum 0.9, weight decay 1e-4, gradient clipping). Robustness trick (Sec 3.1, worth +0.2 dB, Table 5): "add slight Gaussian noises and JPEG compression to **all** tool training data" — magnitudes NOT STATED; **our documented assumption: extra noise σ ∈ [0,5] then JPEG Q ∈ [80,100], applied after the band degradation; config flag `robust_extras`**.

**Our scoped deviations (documented, deliberate):**
1. **20 epochs/tool, not 80** (LR ×0.1 every 5 epochs) — paper-scale is ~4–6h/tool on M4; spec's success bar doesn't need the last 0.1 dB. `configs/full.yaml` value; can be raised later.
2. **Standard RGB HWC float [0,1] everywhere** — the original has a transposed-RGB data-layout bug from MATLAB (harmless but confusing); we do NOT reproduce it.
3. **On-the-fly random patch sampling** instead of pre-extracted HDF5 grids (spec disk rule; original grid stride 96 vs paper's 56 was itself inconsistent).
4. Tool outputs are **not clipped** for loss/PSNR (matches original); clip only when saving images for display.

## Verified equivalences and deviations (from implementation reviews)

- cv2.GaussianBlur(21x21, sigmaX=sigmaY=σ, BORDER_REPLICATE) matches MATLAB
  fspecial('gaussian',21,σ) + imfilter(...,'replicate') to ~7e-18 max abs
  difference (numerically verified).
- cv2 JPEG encoding uses 4:2:0 chroma subsampling: Q=100 is NOT lossless on
  per-pixel chroma noise (≈22 dB on noisy content, >50 dB on smooth content).
  Same behavior class as MATLAB imwrite; fidelity adequate for this project.
- Tool weight init: our PyTorch tools use the default Kaiming-uniform; the
  original TF1 checkpoints used Glorot-uniform. Architecture topology and
  parameter counts are exact (21,827 / 50,851 — pinned by tests); the init
  scheme is a deliberate, documented deviation.
- Images are held in RAM as uint8 and converted to float32/255 per crop —
  identical to the original's uint8-PNG → im2double data path, and 4x lighter
  (750 DIV2K images ≈ 4-6 GB vs 17+ GB as float32 on the 16 GB M4).
- Measured M4 throughput (reports/benchmark.json): small tool 44.4 it/s,
  large tool 14.5 it/s on MPS at batch 64 → full.yaml (20 epochs × 4k iters)
  ≈ 0.5 h/small, 1.5 h/large → ~12 h for all 12 tools.

## For Plan 2 (agent) — extracted, do not re-derive

**Agent network (from official code, paper omits details):**
input 63x63x3 + previous-action one-hot (12-dim; STOP has no slot; zeros at t=0)
conv 9x9x32 /2 -> conv 5x5x24 /2 -> conv 5x5x24 /2 -> conv 5x5x24 /2 (all ReLU, SAME)
-> flatten 384 -> fc 32 (ReLU) -> concat(action one-hot) 44 -> LSTM 50 -> linear 13.
LSTM state persists across the <=3 steps of one episode; reset per episode.

**DQN hyperparameters — code vs paper (code shipped the working model; use code):**

| Item | Code | Paper |
|---|---|---|
| Env steps | 2,000,000 (train every 4 -> 500k updates) | "5e5 iterations" (consistent) |
| LR | 1e-4, x0.5 / 1e6 env steps, floor 2.5e-5 | 2.5e-4 -> 2.5e-5 exponential |
| Target sync | every 10,000 env steps (=2,500 updates) | C = 2,500 iterations (consistent) |
| Epsilon | 1.0 -> 0.1 linear over 1e6 env steps after 5k warmup | only in deleted LaTeX comment (2.5e5 iters — consistent) |
| Replay | 500k transitions, episode-grouped sampling, batch 32 episodes | 5e5, "sequential updates" (DRQN-style) |
| Loss | Huber on TD error; vanilla DQN max | squared error (Eq. 6) |
| Reward | r_t = PSNR_{t+1} - PSNR_t; stop reward 0; gamma 0.99 | identical (Eq. 2) |
| Training-only | early-terminate episode if PSNR < initial PSNR; skip patches with PSNR>50 | not stated |

**Joint fine-tuning (Algorithm 1, paper-only — the unreleased improvement we implement):**
per batch of M=64 toolchains: forward I_1 through the agent-chosen chain; final MSE loss
vs ground truth; backprop through the chain; accumulate each tool's gradient; update each
used tool with its gradient averaged by use-count (alpha = 1e-4, 2e5 iterations). Agent
weights stay FIXED (the published algorithm updates tools only). Reported gain ~+0.25 dB.

**Paper results to sanity-check ordering (Table 3, PSNR/SSIM):**
mild: RL-Restore 28.04/.6498 ~ DnCNN 28.03 ~ VDSR 28.04 > VDSR-s 27.69
moderate: RL-Restore 26.45/.5587 > DnCNN 26.42 > VDSR 26.40 > VDSR-s 25.99
severe: RL-Restore 25.20/.4777 > DnCNN 24.99 > VDSR 24.90 > VDSR-s 24.50
Stop-action ablation: removing STOP costs ~0.15 dB. Stop rate at last step:
mild 60% / moderate 47% / severe 38%.

**Original-code quirks we deliberately do NOT reproduce:** transposed-RGB data layout
(MATLAB artifact); float16 replay storage (we store uint8 images instead, Plan 2);
hardcoded /gpu:0 device pins.

## Phase 1 training outcomes (2026-06-12)

| Tool | Band | Best val PSNR | Best epoch | Stopped at | Note |
|---|---|---|---|---|---|
| 00 | blur [0,1.25] | 32.44 | 8 | 12 (early) | resumed across machines |
| 01 | blur [1.25,2.5] | 28.82 | 1 | 5 (early) | instant convergence |
| 02 | blur [2.5,3.75] | 27.11 | 1 | 5 (early) | instant convergence |
| 03 | blur [3.75,5] | 25.10 | 11 | 15 (early) | hardest blur kept learning |
| 04 | noise [0,12.5] | 33.52 | 20 | 24 (early, post-cap-raise) | plateau proven w/ bonus rounds |
| 05 | noise [12.5,25] | 29.14 | 10 | 14 (early) | |
| 06 | noise [25,37.5] | 27.50 | 8 | 12 (early) | |
| 07 | noise [37.5,50] | 26.31 | 13 | 17 (early) | |
| 08 | jpeg [60,100] | 33.37 | 20* | 24 (early, post-cap-raise) | *micro-creep best; plateau proven |
| 09 | jpeg [35,60] | 31.26 | 20* | 24 (early, post-cap-raise) | flat from epoch 6 |
| 10 | jpeg [20,35] | 29.74 | 1 | 5 (early) | |
| 11 | jpeg [10,20] | 28.37 | 19 | 23 (early, post-cap-raise) | |

Training: Colab A100 (free-priority, ~90 min, pyramid cache + 8 fork workers)
for the bulk; M4 finished/proved plateaus. Early stopping: patience 4 with
0.02 dB minimum improvement (micro-creep does not reset patience); epoch cap
40 = safety rail only.

Specialist matrix (eval: pure band damage, pairs with base PSNR > 50 dB
excluded — the too-clean rule): diagonal mean +0.77 dB, min -0.44 dB (tool04,
see robustness-vs-purity finding in companion.md), off-diagonal mean
-2.17 dB. Gate result reported as-is; deployment-distribution health check is
Plan 2's mandatory first task.
