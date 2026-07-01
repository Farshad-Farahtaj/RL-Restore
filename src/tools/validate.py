"""M2 gate: the 12x12 specialist matrix.

matrix[i, j] = mean PSNR gain of tool i applied to images degraded with band j
(bands = the 12 tool bands). Gate: strong positive diagonal; off-type entries
near zero or negative => "why no single tool suffices".

Pairs whose damage is already negligible (base PSNR > 50 dB - the project's
standard too-clean rule) are excluded: the mildest bands include near-zero
degradations whose baselines (60-90+ dB) make any modification score as a
huge loss, drowning the signal. First observed as a -13 dB "failure" of a
tool that was provably near-identity elsewhere.

CLI: uv run python -m rlrestore.tools.validate --config configs/full.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from rlrestore.common.config import load_config
from rlrestore.common.device import get_device
from rlrestore.common.metrics import psnr
from rlrestore.data.div2k import image_paths
from rlrestore.tools.dataset import ToolPairDataset
from rlrestore.tools.models import build_tool
from rlrestore.tools.specs import TOOL_SPECS
from rlrestore.tools.train import load_images

MAX_EVAL_BASE_PSNR = 50.0  # same too-clean rule as pipeline.MAX_CLEAN_PSNR
_OVERSAMPLE = 4


def select_eval_indices(base: np.ndarray, pairs_per_band: int) -> np.ndarray:
    """Indices of pairs with real damage (base <= 50 dB), capped at pairs_per_band.

    Falls back to everything if fewer than a quarter survive (degenerate inputs,
    e.g. tiny synthetic test images) so callers never divide by zero.
    """
    valid = np.flatnonzero(base <= MAX_EVAL_BASE_PSNR)
    if len(valid) < max(1, pairs_per_band // 4):
        return np.arange(len(base))
    return valid[:pairs_per_band]


@torch.no_grad()
def specialist_matrix(models, images, pairs_per_band, seed, device):
    models = [m.to(device).eval() for m in models]
    mat = np.zeros((12, 12), dtype=np.float64)
    for j, band_spec in enumerate(TOOL_SPECS):
        ds = ToolPairDataset(images, band_spec, seed=seed + 20_000 + j,
                             length=pairs_per_band * _OVERSAMPLE, robust_extras=False)
        pairs = [ds[k] for k in range(len(ds))]
        clean_np = torch.stack([c for _, c in pairs]).numpy().transpose(0, 2, 3, 1)
        degraded_np = torch.stack([d for d, _ in pairs]).numpy().transpose(0, 2, 3, 1)
        base_all = np.array([psnr(degraded_np[k], clean_np[k]) for k in range(len(pairs))])
        keep = select_eval_indices(base_all, pairs_per_band)
        if len(keep) < len(pairs):
            print(f"band {j:02d}: evaluating {len(keep)}/{len(pairs)} pairs"
                  f" ({len(pairs) - int((base_all <= MAX_EVAL_BASE_PSNR).sum())}"
                  " too clean to score)")
        clean_np = clean_np[keep]
        base = base_all[keep]
        degraded_dev = torch.stack([pairs[k][0] for k in keep]).to(device)
        for i, model in enumerate(models):
            restored = model(degraded_dev).cpu().numpy().transpose(0, 2, 3, 1)
            gains = [psnr(restored[k], clean_np[k]) - base[k] for k in range(len(keep))]
            mat[i, j] = float(np.mean(gains))
    return mat


def render_report(mat: np.ndarray, out_dir) -> tuple[Path, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [f"tool{i:02d}" for i in range(12)]

    lines = ["# Tool specialist matrix (ΔPSNR dB)", "",
             "| tool \\ band | " + " | ".join(names) + " |",
             "|" + "---|" * 13]
    for i, row in enumerate(mat):
        lines.append(f"| {names[i]} | " + " | ".join(f"{v:+.2f}" for v in row) + " |")
    diag = np.diag(mat)
    lines += ["", f"Diagonal mean: {diag.mean():+.2f} dB; min: {diag.min():+.2f} dB",
              f"Off-diagonal mean: {(mat.sum() - diag.sum()) / 132:+.2f} dB"]
    md = out_dir / "tools_specialist_matrix.md"
    md.write_text("\n".join(lines))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-np.abs(mat).max(), vmax=np.abs(mat).max())
    ax.set_xticks(range(12), names, rotation=90)
    ax.set_yticks(range(12), names)
    ax.set_xlabel("degradation band")
    ax.set_ylabel("tool applied")
    fig.colorbar(im, label="ΔPSNR (dB)")
    fig.tight_layout()
    png = out_dir / "tools_specialist_matrix.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return md, png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--pairs-per-band", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device)

    images = load_images(
        image_paths(cfg.data.root, "test"), cfg.data.max_train_images
    )  # 0 = all; reuses the smoke image cap
    models = []
    for spec in TOOL_SPECS:
        ck_path = Path("checkpoints/tools") / cfg.name / f"tool{spec.index:02d}" / "best.pt"
        model = build_tool(spec)
        model.load_state_dict(torch.load(ck_path, map_location="cpu", weights_only=False)["model"])
        models.append(model)

    mat = specialist_matrix(models, images, args.pairs_per_band, cfg.seed, device)
    md, png = render_report(mat, "reports")
    print(f"wrote {md} and {png}")
    diag = np.diag(mat)
    print(f"M2 GATE: diagonal min {diag.min():+.2f} dB "
          f"({'PASS' if diag.min() > 0 else 'FAIL'} - every tool must help its own band)")


if __name__ == "__main__":
    main()
