"""Gate figures for the RL-Restore course project.

Produces two figures:
  Figure A (reports/degradation_ramps.png): 3×5 grid showing single-damage ramps
    across the four specialist-band midpoints for BLUR, NOISE, and JPEG.
  Figure B (reports/degradation_gallery.png): 3×4 grid showing fixed mixed-damage
    recipes (MILD / MODERATE / SEVERE) on three hero crops.

Usage:
    uv run python scripts/gallery.py [--root data/DIV2K]
                                     [--out reports/degradation_gallery.png]
                                     [--ramps-out reports/degradation_ramps.png]
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rlrestore.common.metrics import psnr  # noqa: E402
from rlrestore.data.degradations import (  # noqa: E402
    gaussian_blur,
    gaussian_noise,
    jpeg_compress,
)
from rlrestore.data.pipeline import apply_recipe  # noqa: E402
from rlrestore.data.recipes import RecipeParams  # noqa: E402

# ---------------------------------------------------------------------------
# Hero crop definitions: (filename, crop_x, crop_y)
# BLUR row  – 0022.png: steel bridge lattice / fine hard edges
# NOISE row – 0106.png: parliament facade + smooth Danube water
# JPEG row  – 0070.png: white stucco wall + roof tiles (x aligned to 8-px for DCT)
# ---------------------------------------------------------------------------
HEROES = [
    ("0022.png", 1152, 256),  # bridge lattice
    ("0106.png", 800, 600),   # parliament + water
    ("0070.png", 1600, 600),  # stucco wall + roof tiles (1600 = 200*8)
]

# Zoom-inset 64×64 source regions (relative to the 256×256 crop, (x, y))
# Chosen per row for maximum visual information:
#   BLUR  – upper-left steel lattice for sharpest cross-member edges (edge strength 88.6)
#   NOISE – pure smooth water at bottom (lowest edge=43.4, noise grain most visible)
#   JPEG  – flat white stucco wall, aligned to 8-px DCT grid (mean≈144, edge=35.5)
INSET_REGIONS = [
    (0, 0),     # BLUR row: steel lattice top-left (edge strength 88.6)
    (0, 160),   # NOISE row: pure smooth water surface (edge strength 43.4)
    (160, 128), # JPEG row: flat white stucco (mean≈242, std≈9), 8-px aligned (160=20×8, 128=16×8)
]
INSET_SIZE = 64

# Single-damage ramp: (kind, symbol, values list)
RAMPS = [
    ("blur",  "σ",  [0.6, 1.9, 3.1, 4.4]),
    ("noise", "σ",  [6,   19,  31,  44]),
    ("jpeg",  "Q",  [80,  47,  27,  15]),
]
RAMP_LABELS = ["BLUR", "NOISE", "JPEG"]

# Fixed mixed-damage recipes
RECIPE_MILD     = RecipeParams((3, 3, 5), 1.25, 12.5, 37.5)   # sum 9
RECIPE_MODERATE = RecipeParams((5, 5, 5), 2.25, 22.5, 37.5)   # sum 13
RECIPE_SEVERE   = RecipeParams((7, 7, 8), 3.25, 32.5, 22.5)   # sum 20
GALLERY_RECIPES = [RECIPE_MILD, RECIPE_MODERATE, RECIPE_SEVERE]
GALLERY_LABELS  = [
    "MILD (sum 9)",
    "MODERATE (sum 13)",
    "SEVERE (sum 20)",
]

GALLERY_ROW_LABELS = ["BRIDGE", "PARLIAMENT", "HOUSE"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_hero(root: pathlib.Path, filename: str, cx: int, cy: int) -> np.ndarray:
    """Load a 256×256 float32 crop from a DIV2K HR image.

    Args:
        root: path to the DIV2K root directory.
        filename: image filename inside DIV2K_train_HR/.
        cx: left column of the crop.
        cy: top row of the crop.

    Returns:
        float32 RGB array of shape (256, 256, 3) in [0, 1].
    """
    path = root / "DIV2K_train_HR" / filename
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    crop = rgb[cy : cy + 256, cx : cx + 256].copy()
    return crop.astype(np.float32) / 255.0


def add_badge(ax: plt.Axes, text: str) -> None:
    """Add a lower-left badge with white text on a semi-transparent black box."""
    ax.text(
        0.03,
        0.04,
        text,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color="white",
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "black",
            "alpha": 0.55,
            "edgecolor": "none",
        },
    )


def add_zoom_inset(
    ax: plt.Axes,
    img: np.ndarray,
    ix: int,
    iy: int,
    size: int = INSET_SIZE,
) -> None:
    """Overlay a zoom inset from (ix, iy)→(ix+size, iy+size) in the image."""
    region = img[iy : iy + size, ix : ix + size]

    inset_ax = ax.inset_axes([0.58, 0.58, 0.40, 0.40])
    inset_ax.imshow(region, interpolation="nearest")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#FFD700")
        spine.set_linewidth(1.5)

    # Yellow rectangle on main image marking the source region
    rect = mpatches.Rectangle(
        (ix, iy),
        size,
        size,
        linewidth=1.5,
        edgecolor="#FFD700",
        facecolor="none",
    )
    ax.add_patch(rect)


# ---------------------------------------------------------------------------
# Figure A: single-damage ramps
# ---------------------------------------------------------------------------

def make_ramps(
    heroes: list[np.ndarray],
    out_path: pathlib.Path,
) -> None:
    """Produce Figure A — 3 rows × 5 cols degradation ramp figure."""
    rng = np.random.default_rng(7)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9), squeeze=False)
    fig.suptitle(
        "Single-damage ramps — the four severity bands each specialist tool covers",
        fontsize=12,
        fontweight="bold",
    )

    col_headers = ["CLEAN", "BAND 1 (mildest)", "BAND 2", "BAND 3", "BAND 4 (worst)"]

    for r, ((kind, sym, values), row_label, clean) in enumerate(
        zip(RAMPS, RAMP_LABELS, heroes)
    ):
        ix, iy = INSET_REGIONS[r]

        # Column 0: clean reference
        ax0 = axes[r, 0]
        ax0.imshow(clean)
        add_zoom_inset(ax0, clean, ix, iy)
        add_badge(ax0, "clean")
        ax0.axis("off")
        if r == 0:
            ax0.set_title(col_headers[0], fontsize=10, fontweight="bold", pad=4)

        # Columns 1–4: single damage at each band midpoint
        for c, v in enumerate(values, start=1):
            if kind == "blur":
                damaged = gaussian_blur(clean, v)
                badge_label = f"σ = {v}"
            elif kind == "noise":
                damaged = gaussian_noise(clean, v, rng)
                badge_label = f"σ = {v}"
            else:
                damaged = jpeg_compress(clean, float(v))
                badge_label = f"Q = {v}"

            db = psnr(clean, damaged)
            badge_text = f"{badge_label} · {db:.1f} dB"

            ax = axes[r, c]
            ax.imshow(np.clip(damaged, 0.0, 1.0))
            add_zoom_inset(ax, np.clip(damaged, 0.0, 1.0), ix, iy)
            add_badge(ax, badge_text)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_headers[c], fontsize=10, fontweight="bold", pad=4)

        # Row label at left margin (centred within the 0.06 left band).
        # subplot area: bottom=0.02, top=0.90 → each row spans (0.90-0.02)/3=0.293.
        row_y = 0.90 - (r + 0.5) * (0.90 - 0.02) / 3.0
        fig.text(
            0.02,
            row_y,
            row_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.90, bottom=0.02,
        wspace=0.05, hspace=0.12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure B: mixed-damage gallery
# ---------------------------------------------------------------------------

def make_gallery(
    heroes: list[np.ndarray],
    out_path: pathlib.Path,
) -> None:
    """Produce Figure B — 3 rows × 4 cols mixed-damage gallery figure."""
    rng = np.random.default_rng(11)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5), squeeze=False)
    fig.suptitle(
        "Mixed damage at fixed recipes — mild vs moderate vs severe"
        " are now true class extremes/center",
        fontsize=11,
        fontweight="bold",
    )

    col_headers = ["CLEAN"] + GALLERY_LABELS

    for r, (clean, row_label) in enumerate(zip(heroes, GALLERY_ROW_LABELS)):
        ix, iy = INSET_REGIONS[r]

        # Column 0: clean reference
        ax0 = axes[r, 0]
        ax0.imshow(clean)
        add_zoom_inset(ax0, clean, ix, iy)
        add_badge(ax0, "clean")
        ax0.axis("off")
        if r == 0:
            ax0.set_title(col_headers[0], fontsize=11, fontweight="bold", pad=4)

        # Columns 1–3: fixed recipes
        for c, recipe in enumerate(GALLERY_RECIPES, start=1):
            damaged = apply_recipe(clean, recipe, rng)
            db = psnr(clean, np.clip(damaged, 0.0, 1.0))
            badge_text = (
                f"σb {recipe.blur_sigma:.1f} · "
                f"σn {recipe.noise_sigma:.0f} · "
                f"Q {recipe.jpeg_quality:.0f}\n"
                f"{db:.1f} dB"
            )

            ax = axes[r, c]
            ax.imshow(np.clip(damaged, 0.0, 1.0))
            add_zoom_inset(ax, np.clip(damaged, 0.0, 1.0), ix, iy)
            add_badge(ax, badge_text)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_headers[c], fontsize=11, fontweight="bold", pad=4)

        # Row label at left margin (centred within the 0.06 left band).
        # subplot area: bottom=0.02, top=0.90 → each row spans (0.90-0.02)/3=0.293.
        row_y = 0.90 - (r + 0.5) * (0.90 - 0.02) / 3.0
        fig.text(
            0.02,
            row_y,
            row_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=11,
            fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.06, right=0.99, top=0.90, bottom=0.02,
        wspace=0.05, hspace=0.12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Produce gate figures for the RL-Restore course project."
    )
    ap.add_argument("--root", default="data/DIV2K", help="DIV2K dataset root directory")
    ap.add_argument(
        "--out",
        default="reports/degradation_gallery.png",
        help="Output path for Figure B (mixed-damage gallery)",
    )
    ap.add_argument(
        "--ramps-out",
        default="reports/degradation_ramps.png",
        help="Output path for Figure A (single-damage ramps)",
    )
    args = ap.parse_args()

    root = pathlib.Path(args.root)

    heroes = [
        load_hero(root, filename, cx, cy)
        for filename, cx, cy in HEROES
    ]

    make_ramps(heroes, pathlib.Path(args.ramps_out))
    make_gallery(heroes, pathlib.Path(args.out))


if __name__ == "__main__":
    main()
