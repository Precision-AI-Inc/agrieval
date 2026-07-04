from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from precisionai.agrieval.dpt.services.evaluate import run_dpt_eval

# ---------------------------------------------------------------------------
# unsupervised-only path
# ---------------------------------------------------------------------------


def test_unsupervised_result_shape(synthetic_tiles):
    result = run_dpt_eval(tiles=synthetic_tiles)
    assert result["n_tiles"] == len(synthetic_tiles)
    assert result["embed_dim"] == 6
    assert result["grid_height"] == 4
    assert result["grid_width"] == 5
    assert result["n_patches"] == len(synthetic_tiles) * 4 * 5
    assert sorted(result["tile_ids"]) == sorted(synthetic_tiles)
    assert result["classes"] is None
    assert result["per_class"] is None
    assert result["knn_confusion"] is None


def test_unsupervised_global_metrics_present(synthetic_tiles):
    result = run_dpt_eval(tiles=synthetic_tiles)
    gm = result["global_metrics"]
    for key in (
        "effective_rank",
        "pca_explained_variance",
        "pairwise_similarity_stats",
        "centroid_similarity_stats",
        "uniformity",
        "mean_patch_smoothness",
        "mean_outlier_fraction",
    ):
        assert key in gm


def test_per_tile_diagnostics_present(synthetic_tiles):
    result = run_dpt_eval(tiles=synthetic_tiles)
    for tile_id in synthetic_tiles:
        diag = result["per_tile"][tile_id]
        assert set(diag) == {"patch_norm_stats", "patch_smoothness", "outlier_fraction"}


def test_no_warnings_below_max_patches(synthetic_tiles):
    result = run_dpt_eval(tiles=synthetic_tiles, max_patches=10_000)
    assert result["warnings"] is None


# ---------------------------------------------------------------------------
# label-aware path
# ---------------------------------------------------------------------------


def test_labeled_result_includes_classes(labeled_dataset, classes_path):
    tiles, masks_dir, crop_name = labeled_dataset
    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)
    assert result["classes"] is not None
    assert "background" in result["classes"]
    assert crop_name in result["classes"]


def test_labeled_result_has_per_class_and_confusion(labeled_dataset, classes_path):
    tiles, masks_dir, _ = labeled_dataset
    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)
    assert result["per_class"] is not None
    for metrics in result["per_class"].values():
        assert "n_patches" in metrics
    assert result["knn_confusion"] is not None
    assert "confusion" in result["knn_confusion"]
    assert "purity" in result["knn_confusion"]


def test_missing_masks_dir_raises(labeled_dataset, classes_path, tmp_path):
    tiles, _, _ = labeled_dataset
    with pytest.raises(FileNotFoundError, match="masks_dir"):
        run_dpt_eval(tiles=tiles, masks_dir=tmp_path / "does_not_exist", classes_path=classes_path)


def test_missing_tile_mask_raises(labeled_dataset, classes_path):
    tiles, masks_dir, _ = labeled_dataset
    extra_tiles = dict(tiles)
    extra_tiles["tile_without_mask"] = tiles[next(iter(tiles))]
    with pytest.raises(FileNotFoundError, match="tile_without_mask"):
        run_dpt_eval(tiles=extra_tiles, masks_dir=masks_dir, classes_path=classes_path)


# ---------------------------------------------------------------------------
# max_patches subsampling
# ---------------------------------------------------------------------------


def test_subsampling_reports_warning(labeled_dataset, classes_path):
    tiles, masks_dir, _ = labeled_dataset
    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path, max_patches=5)
    assert result["warnings"] is not None
    assert any("Subsampled" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def _save_exact_mask(path: Path, class_ids: np.ndarray, classes: list) -> None:
    """Save a colour-coded mask at exactly the given class-ID grid resolution.

    Using a mask sized identically to the target patch grid makes the
    nearest-neighbor downsample an identity mapping, so callers can place
    specific classes at specific patches deterministically.
    """
    color_by_id = {int(cid): tuple(color) for _, color, cid in classes}
    h, w = class_ids.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in color_by_id.items():
        rgb[class_ids == cid] = color
    Image.fromarray(rgb).save(path)


def test_single_tile_evaluation():
    rng = np.random.default_rng(0)
    tiles = {"only": rng.normal(size=(4, 3, 3)).astype(np.float32).tolist()}
    result = run_dpt_eval(tiles=tiles)
    assert result["n_tiles"] == 1
    assert result["n_patches"] == 9


def test_single_total_patch_does_not_crash():
    # 1 tile, 1x1 grid → exactly one patch token in the whole request. Exercises
    # every geometry function's n<2 guard path without raising.
    tiles = {"only": [[[1.0]], [[2.0]], [[3.0]]]}  # shape (C=3, H=1, W=1)
    result = run_dpt_eval(tiles=tiles)
    assert result["n_patches"] == 1
    assert result["global_metrics"]["effective_rank"]["effective_rank"] is None
    assert result["global_metrics"]["pca_explained_variance"]["pc1"] is None


def test_all_identical_patches_collapse_effective_rank():
    # Constant raw features → centered covariance is exactly zero → every
    # eigenvalue is filtered out → effective rank collapses. Floating-point
    # noise may leak one near-zero eigenvalue, in which case the participation
    # ratio reports 1.0 — either way far below the embedding dimension.
    embed_dim = 5
    tile = np.ones((embed_dim, 4, 4), dtype=np.float32)
    tiles = {"a": tile.tolist(), "b": tile.tolist()}
    result = run_dpt_eval(tiles=tiles)
    er = result["global_metrics"]["effective_rank"]["effective_rank"]
    assert er is not None
    assert er <= 1.0 + 1e-6
    assert result["global_metrics"]["mean_patch_smoothness"] == pytest.approx(1.0)


def test_single_class_mask_all_background(tmp_path, classes):
    grid_h, grid_w, embed_dim = 4, 5, 6
    rng = np.random.default_rng(2)
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()

    tiles = {}
    for i in range(3):
        tile_id = f"tile_{i}"
        tiles[tile_id] = rng.normal(size=(embed_dim, grid_h, grid_w)).astype(np.float32).tolist()
        _save_exact_mask(masks_dir / f"{tile_id}.png", np.zeros((grid_h, grid_w), dtype=np.int64), classes)

    classes_path = Path(__file__).parent.parent / "data" / "class_map.json"
    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)

    assert result["classes"] == ["background"]
    assert result["per_class"]["background"]["n_patches"] == 3 * grid_h * grid_w
    # Only one class exists, so every patch's neighbors trivially share its class.
    purity = result["knn_confusion"]["purity"]
    for stats in purity.values():
        assert stats["mean"] == pytest.approx(1.0)


def test_rare_single_patch_class_reports_n_patches_only(tmp_path, classes):
    grid_h, grid_w, embed_dim = 4, 5, 6
    rng = np.random.default_rng(3)
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()

    class_ids = np.zeros((grid_h, grid_w), dtype=np.int64)
    class_ids[0, 0] = 2  # exactly one patch of a rare third class
    tile_id = "tile_0"
    tiles = {tile_id: rng.normal(size=(embed_dim, grid_h, grid_w)).astype(np.float32).tolist()}
    _save_exact_mask(masks_dir / f"{tile_id}.png", class_ids, classes)

    classes_path = Path(__file__).parent.parent / "data" / "class_map.json"
    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)

    rare_name = classes[2][0]
    assert rare_name in result["per_class"]
    rare_metrics = result["per_class"][rare_name]
    assert rare_metrics["n_patches"] == 1
    assert "effective_rank" not in rare_metrics
    assert "pca_explained_variance" not in rare_metrics


def test_unknown_mask_color_raises(tmp_path, classes):
    grid_h, grid_w, embed_dim = 2, 2, 4
    rng = np.random.default_rng(4)
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()

    bad_rgb = np.full((grid_h, grid_w, 3), 123, dtype=np.uint8)  # not in class_map.json
    Image.fromarray(bad_rgb).save(masks_dir / "tile_0.png")

    tiles = {"tile_0": rng.normal(size=(embed_dim, grid_h, grid_w)).astype(np.float32).tolist()}
    classes_path = Path(__file__).parent.parent / "data" / "class_map.json"
    with pytest.raises(ValueError, match="unknown color"):
        run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)


def test_k_value_larger_than_pool_size_clamps_without_crash(labeled_dataset, classes_path):
    tiles, masks_dir, _ = labeled_dataset
    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path, k_values=[10_000])
    # top_k_neighbors clamps k to the available pool size internally.
    assert len(result["knn_confusion"]["purity"]) == 1


def test_integer_tile_values_accepted():
    tiles = {"a": [[[1, 2], [3, 4]]], "b": [[[5, 6], [7, 8]]]}
    result = run_dpt_eval(tiles=tiles)
    assert result["n_tiles"] == 2
