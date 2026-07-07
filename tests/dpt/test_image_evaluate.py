"""Tests for the whole-image wiring (run_dpt_image_eval / load_images).

The image wiring shares its entire computation with the tiles wiring
(run_dpt_eval) — see docs/DENSE_PATCH_TOKENS.md. These tests focus on the
wrapper's key-relabeling correctness and parity with the tiles wiring,
reusing the same synthetic_tiles/labeled_dataset fixtures as test_evaluate.py
since the underlying array shapes are identical.
"""

import numpy as np
import pytest

from precisionai.agrieval.dpt.services.evaluate import load_images, run_dpt_eval, run_dpt_image_eval

# ---------------------------------------------------------------------------
# parity with run_dpt_eval
# ---------------------------------------------------------------------------


def test_result_matches_run_dpt_eval_modulo_key_names(synthetic_tiles):
    tiles_result = run_dpt_eval(tiles=synthetic_tiles)
    images_result = run_dpt_image_eval(images=synthetic_tiles)

    assert images_result["n_images"] == tiles_result["n_tiles"]
    assert images_result["embed_dim"] == tiles_result["embed_dim"]
    assert images_result["grid_height"] == tiles_result["grid_height"]
    assert images_result["grid_width"] == tiles_result["grid_width"]
    assert images_result["n_patches"] == tiles_result["n_patches"]
    assert images_result["image_ids"] == tiles_result["tile_ids"]
    assert images_result["global_metrics"] == tiles_result["global_metrics"]
    assert images_result["per_image"] == tiles_result["per_tile"]
    assert images_result["classes"] == tiles_result["classes"]


def test_labeled_result_matches_run_dpt_eval(labeled_dataset, classes_path):
    images, masks_dir, crop_name = labeled_dataset
    tiles_result = run_dpt_eval(tiles=images, masks_dir=masks_dir, classes_path=classes_path)
    images_result = run_dpt_image_eval(images=images, masks_dir=masks_dir, classes_path=classes_path)

    assert images_result["classes"] == tiles_result["classes"]
    assert crop_name in images_result["classes"]
    assert images_result["per_class"] == tiles_result["per_class"]
    assert images_result["knn_confusion"] == tiles_result["knn_confusion"]


# ---------------------------------------------------------------------------
# response shape
# ---------------------------------------------------------------------------


def test_response_uses_image_vocabulary(synthetic_tiles):
    result = run_dpt_image_eval(images=synthetic_tiles)
    assert "n_images" in result
    assert "image_ids" in result
    assert "per_image" in result
    assert "n_tiles" not in result
    assert "tile_ids" not in result
    assert "per_tile" not in result


def test_unsupervised_result_has_no_classes(synthetic_tiles):
    result = run_dpt_image_eval(images=synthetic_tiles)
    assert result["classes"] is None
    assert result["per_class"] is None
    assert result["knn_confusion"] is None


def test_mismatched_shapes_rejected(synthetic_tiles):
    bad = dict(synthetic_tiles)
    bad["extra"] = np.zeros((99, 1, 1)).tolist()
    with pytest.raises(ValueError, match=r"inhomogeneous shape"):
        run_dpt_image_eval(images=bad)


# ---------------------------------------------------------------------------
# load_images — batched .npz archive (features/filenames)
# ---------------------------------------------------------------------------


def _write_images_batch(path, filenames, *, shape=(4, 6, 8), seed=0):
    rng = np.random.default_rng(seed)
    features = np.stack([rng.normal(size=shape).astype(np.float32) for _ in filenames])
    np.savez(path, features=features, filenames=np.array(filenames))
    return path


def test_load_images_round_trip(tmp_path):
    archive = _write_images_batch(tmp_path / "images.npz", ["field_001.png", "field_002.png"])
    images = load_images(archive)
    assert sorted(images) == ["field_001", "field_002"]
    assert all(arr.shape == (4, 6, 8) for arr in images.values())
    assert all(arr.dtype == np.float32 for arr in images.values())


def test_load_images_missing_required_key_raises(tmp_path):
    archive = tmp_path / "images.npz"
    np.savez(archive, features=np.zeros((1, 4, 6, 8), dtype=np.float32))  # no filenames
    with pytest.raises(ValueError, match=r"missing required key\(s\).*filenames"):
        load_images(archive)


def test_load_images_non_finite_raises(tmp_path):
    archive = tmp_path / "images.npz"
    features = np.zeros((1, 4, 6, 8), dtype=np.float32)
    features[0, 0, 0, 0] = np.nan
    np.savez(archive, features=features, filenames=np.array(["a"]))
    with pytest.raises(ValueError, match="non-finite"):
        load_images(archive)


def test_load_images_not_4d_raises(tmp_path):
    archive = tmp_path / "images.npz"
    np.savez(archive, features=np.zeros((4, 6, 8), dtype=np.float32), filenames=np.array(["a"]))
    with pytest.raises(ValueError, match="4-dimensional"):
        load_images(archive)


def test_load_images_mismatched_filenames_length_raises(tmp_path):
    archive = tmp_path / "images.npz"
    np.savez(
        archive,
        features=np.zeros((2, 4, 6, 8), dtype=np.float32),
        filenames=np.array(["only_one"]),
    )
    with pytest.raises(ValueError, match="filenames"):
        load_images(archive)


def test_load_images_duplicate_stem_raises(tmp_path):
    archive = _write_images_batch(tmp_path / "images.npz", ["field.png", "field.jpg"])
    with pytest.raises(ValueError, match="Duplicate image ID"):
        load_images(archive)


def test_load_images_empty_axis_raises(tmp_path):
    archive = tmp_path / "images.npz"
    np.savez(archive, features=np.zeros((1, 4, 0, 8), dtype=np.float32), filenames=np.array(["a"]))
    with pytest.raises(ValueError, match="empty along any axis"):
        load_images(archive)


def test_load_images_missing_path_error_says_images_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="images_path"):
        load_images(tmp_path / "does_not_exist.npz")


def test_loaded_images_flow_through_run_dpt_image_eval(tmp_path):
    archive = _write_images_batch(tmp_path / "images.npz", ["field_001.png"], seed=2)

    result = run_dpt_image_eval(images=load_images(archive))
    assert result["n_images"] == 1
    assert result["embed_dim"] == 4
    assert result["grid_height"] == 6
    assert result["grid_width"] == 8
