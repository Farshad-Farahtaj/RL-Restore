import numpy as np
import torch

from rlrestore.tools.specs import TOOL_SPECS
from rlrestore.tools.validate import render_report, specialist_matrix


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _images(n=2):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 96.0
    base = np.stack([yy, xx, (yy + xx) / 2], axis=-1)
    return [
        np.clip(base + 0.1 * rng.standard_normal(base.shape), 0, 1).astype(np.float32)
        for _ in range(n)
    ]


def test_identity_tools_give_zero_gain_matrix():
    models = [_Identity() for _ in TOOL_SPECS]
    mat = specialist_matrix(models, _images(), pairs_per_band=3, seed=0,
                            device=torch.device("cpu"))
    assert mat.shape == (12, 12)
    assert np.allclose(mat, 0.0, atol=1e-5)


def test_render_report_writes_files(tmp_path):
    mat = np.random.default_rng(0).normal(0, 1, (12, 12))
    md, png = render_report(mat, tmp_path)
    assert md.exists() and png.exists()
    text = md.read_text()
    assert "tool00" in text and "ΔPSNR" in text


def test_select_eval_indices_drops_too_clean_pairs():
    from rlrestore.tools.validate import select_eval_indices

    base = np.array([22.0, 95.0, 30.0, 88.0, 41.0, 47.0, 60.0, 33.0])
    keep = select_eval_indices(base, pairs_per_band=4)
    assert list(keep) == [0, 2, 4, 5]  # 95/88/60 dB pairs excluded, capped at 4


def test_select_eval_indices_falls_back_when_all_clean():
    from rlrestore.tools.validate import select_eval_indices

    base = np.full(8, 90.0)
    keep = select_eval_indices(base, pairs_per_band=4)
    assert len(keep) == 8  # degenerate input: keep everything rather than nan
