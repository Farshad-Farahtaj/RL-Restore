"""Smoke for the monolith TEST-eval core (evaluate_monolith_on_class + render).

Tiny per_class=2 run with fake tools (pair generation only) and an untrained
MonoRestoreCNN; asserts the per-class gain dict, the assembled JSON (gains per
class + overall + protocol), and that the B5 markdown row renders.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from scripts.evaluate_monolith import (
    SEVERITIES,
    evaluate_monolith_on_class,
    render_markdown,
)
from torch import nn

from rlrestore.baselines.monolith import MonoRestoreCNN, count_parameters

_DEVICE = torch.device("cpu")
_N_TOOLS = 12


class _Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


def _fake_tools() -> list[nn.Module]:
    return [_Identity() for _ in range(_N_TOOLS)]


def _make_images(n: int = 4, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    base = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    return [
        np.clip(base + 0.1 * rng.standard_normal(base.shape), 0, 1).astype(np.float32)
        for _ in range(n)
    ]


def test_evaluate_monolith_on_class_smoke():
    torch.manual_seed(0)
    model = MonoRestoreCNN(depth=4, width=8).eval()  # untrained is fine for a smoke
    seeds = [10_000, 10_001]
    result = evaluate_monolith_on_class(
        model=model,
        tools=_fake_tools(),
        images=_make_images(),
        device=_DEVICE,
        severity="mild",
        seeds=seeds,
    )
    assert set(result) == {"psnr_gain_db", "gain_std_db", "n_episodes"}
    assert result["n_episodes"] == 2
    assert np.isfinite(result["psnr_gain_db"])
    json.dumps(result)  # must be JSON-able


def test_full_report_assembly_and_markdown(tmp_path):
    torch.manual_seed(0)
    model = MonoRestoreCNN(depth=4, width=8).eval()
    n_params = count_parameters(model)

    per_class = {}
    for sev in SEVERITIES:
        per_class[sev] = evaluate_monolith_on_class(
            model=model,
            tools=_fake_tools(),
            images=_make_images(),
            device=_DEVICE,
            severity=sev,
            seeds=[10, 11],
        )
    overall = float(np.mean([per_class[c]["psnr_gain_db"] for c in SEVERITIES]))
    protocol = {
        "n_per_class": 2,
        "seeds_scheme": "mild 10000.., moderate 20000.., severe 30000..",
        "checkpoint": "fake.pt",
        "checkpoint_sha256": "deadbeefdeadbeef",
        "param_count": n_params,
        "test_split": "DIV2K 751-800",
        "image_dtype": "uint8 (training parity)",
        "device": "cpu",
        "date": "2026-06-13T00:00:00+00:00",
    }
    result = {
        "per_class": per_class,
        "overall": {"monolith_psnr_gain_db": overall},
        "protocol": protocol,
    }

    # JSON has gain keys per class + overall + protocol
    for sev in SEVERITIES:
        assert "psnr_gain_db" in result["per_class"][sev]
    assert "monolith_psnr_gain_db" in result["overall"]
    assert result["protocol"]["param_count"] == n_params
    json.dumps(result)

    md = render_markdown(per_class, overall, protocol)
    assert "B5: single end-to-end CNN" in md
    assert f"{n_params:,} params" in md
    assert "| Method | mild | moderate | severe | overall |" in md
    assert "## Protocol" in md
    # The B5 row carries 3 class cells + overall (4 numeric cells -> 6 pipes).
    b5_line = [ln for ln in md.splitlines() if ln.startswith("| B5:")][0]
    assert b5_line.count("|") == 6
