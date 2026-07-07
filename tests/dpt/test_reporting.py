from pathlib import Path

import numpy as np
from PIL import Image

from precisionai.agrieval.dpt import print_result, run_dpt_eval, run_dpt_image_eval


def test_print_result_tile_wiring_unsupervised(synthetic_tiles, capsys):
    result = run_dpt_eval(tiles=synthetic_tiles)
    print_result(result)
    out = capsys.readouterr().out

    assert "n_entries: 6" in out
    assert "── global_metrics ──" in out
    assert "effective_rank" in out
    assert "pca_explained_variance" in out
    assert "pairwise cosine" in out
    assert "centroid cosine" in out
    assert "── per_class ──" not in out
    assert "── knn_confusion ──" not in out
    assert "Warnings:" not in out


def test_print_result_tile_wiring_labeled(labeled_dataset, classes_path, capsys):
    tiles, masks_dir, crop_name = labeled_dataset
    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)
    print_result(result)
    out = capsys.readouterr().out

    assert "── per_class ──" in out
    assert crop_name in out
    assert "── knn_confusion ──" in out
    assert "purity@5" in out


def test_print_result_image_wiring(synthetic_tiles, capsys):
    result = run_dpt_image_eval(images=synthetic_tiles)
    print_result(result)
    out = capsys.readouterr().out

    assert "n_entries: 6" in out
    assert "── global_metrics ──" in out


def test_print_result_with_warnings(labeled_dataset, classes_path, capsys):
    tiles, masks_dir, _ = labeled_dataset
    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path, max_patches=10)
    print_result(result)
    out = capsys.readouterr().out

    assert "Warnings:" in out
    assert "Subsampled 10 of" in out


def test_print_result_per_class_without_effective_rank(tmp_path: Path, classes: list, classes_path: Path, capsys):
    """A class with fewer than 2 patches gets no effective_rank/pca_explained_variance line."""
    bg_color = next(c for _, c, i in classes if i == 0)
    crop_name, crop_color, _ = next((n, c, i) for n, c, i in classes if i == 1)

    # 1x2 grid: one patch background, one patch crop -> the crop class has exactly 1 patch.
    mask = np.zeros((2, 4, 3), dtype=np.uint8)
    mask[:, :2] = bg_color
    mask[:, 2:] = crop_color

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    Image.fromarray(mask).save(masks_dir / "only.png")

    rng = np.random.default_rng(0)
    tiles = {"only": rng.normal(size=(3, 1, 2)).astype(np.float32).tolist()}

    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path, k_values=[1])
    print_result(result)
    out = capsys.readouterr().out

    assert f"{crop_name:<20} n_patches=1\n" in out
