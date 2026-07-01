from pathlib import Path

import cv2
import numpy as np

from rlrestore.data.div2k import SPLITS, URLS, image_paths, verify


def _fake_div2k(root: Path, indices: list[int]) -> None:
    (root / "DIV2K_train_HR").mkdir(parents=True, exist_ok=True)
    (root / "DIV2K_valid_HR").mkdir(parents=True, exist_ok=True)
    img = (np.random.default_rng(0).random((80, 80, 3)) * 255).astype("uint8")
    for i in indices:
        sub = "DIV2K_valid_HR" if i > 800 else "DIV2K_train_HR"
        cv2.imwrite(str(root / sub / f"{i:04d}.png"), img)


def test_urls_and_splits():
    assert "DIV2K_train_HR.zip" in URLS["train"]
    assert "DIV2K_valid_HR.zip" in URLS["valid"]
    assert SPLITS == {"train": (1, 750), "test": (751, 800), "val": (801, 900)}


def test_image_paths_respects_split(tmp_path):
    _fake_div2k(tmp_path, [1, 2, 750, 751, 800, 801, 900])
    train = image_paths(tmp_path, "train")
    test = image_paths(tmp_path, "test")
    val = image_paths(tmp_path, "val")
    assert [p.stem for p in train] == ["0001", "0002", "0750"]
    assert [p.stem for p in test] == ["0751", "0800"]
    assert [p.stem for p in val] == ["0801", "0900"]


def test_verify_fails_on_missing(tmp_path):
    _fake_div2k(tmp_path, [1])
    ok, msg = verify(tmp_path)
    assert not ok and "train" in msg


def test_verify_passes_on_complete(tmp_path):
    _fake_div2k(tmp_path, list(range(1, 901)))
    ok, msg = verify(tmp_path)
    assert ok, msg


def test_image_paths_ignores_non_numeric_pngs(tmp_path):
    _fake_div2k(tmp_path, [1, 2])
    (tmp_path / "DIV2K_train_HR" / "README.png").write_bytes(b"not an image")
    assert [p.stem for p in image_paths(tmp_path, "train")] == ["0001", "0002"]
