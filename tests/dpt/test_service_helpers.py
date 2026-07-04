from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from precisionai.agrieval.dpt.services.evaluate import (
    _downsample_mask_to_grid,
    _find_mask_file,
    _flatten_patches,
    _stack_tiles,
)

# ---------------------------------------------------------------------------
# _stack_tiles
# ---------------------------------------------------------------------------


def test_stack_tiles_orders_by_id():
    tiles = {
        "b": [[[3.0, 4.0]]],
        "a": [[[1.0, 2.0]]],
    }
    tile_ids, arr = _stack_tiles(tiles)
    assert tile_ids == ["a", "b"]
    assert arr.shape == (2, 1, 1, 2)
    np.testing.assert_array_equal(arr[0, 0, 0], [1.0, 2.0])
    np.testing.assert_array_equal(arr[1, 0, 0], [3.0, 4.0])


def test_stack_tiles_dtype_is_float32():
    _, arr = _stack_tiles({"a": [[[1, 2]]]})
    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# _flatten_patches
# ---------------------------------------------------------------------------


def test_flatten_patches_known_values():
    # One tile, C=2, H=2, W=2. Channel 0 is 1..4, channel 1 is 5..8 (row-major).
    tiles_arr = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=np.float32)
    flat = _flatten_patches(tiles_arr)
    expected = np.array([[1, 5], [2, 6], [3, 7], [4, 8]], dtype=np.float32)
    np.testing.assert_array_equal(flat, expected)


def test_flatten_patches_shape():
    tiles_arr = np.zeros((3, 4, 5, 6), dtype=np.float32)
    flat = _flatten_patches(tiles_arr)
    assert flat.shape == (3 * 5 * 6, 4)


# ---------------------------------------------------------------------------
# _downsample_mask_to_grid
# ---------------------------------------------------------------------------


def test_downsample_mask_to_grid_quadrants():
    # 4x4 mask split into four distinct-valued quadrants, downsampled to 2x2.
    mask = np.zeros((4, 4), dtype=np.int64)
    mask[0:2, 0:2] = 1
    mask[0:2, 2:4] = 2
    mask[2:4, 0:2] = 3
    mask[2:4, 2:4] = 4
    grid = _downsample_mask_to_grid(mask, 2, 2)
    assert grid.shape == (2, 2)
    np.testing.assert_array_equal(grid, [[1, 2], [3, 4]])


def test_downsample_mask_to_grid_majority_vote():
    # 3x3 mask → 1x1 grid: class 2 covers 5 pixels vs class 1's 4, so majority
    # vote must pick 2 even though the top-left (nearest-neighbor sample point)
    # is class 1.
    mask = np.array([[1, 1, 2], [1, 2, 2], [1, 2, 2]], dtype=np.int64)
    grid = _downsample_mask_to_grid(mask, 1, 1)
    assert grid.shape == (1, 1)
    assert grid[0, 0] == 2


def test_downsample_mask_to_grid_upsampling_falls_back_to_nearest():
    # Mask smaller than the grid: cells with no covering pixels are filled by
    # nearest-neighbor sampling.
    mask = np.array([[7]], dtype=np.int64)
    grid = _downsample_mask_to_grid(mask, 3, 3)
    assert grid.shape == (3, 3)
    assert np.all(grid == 7)


def test_downsample_mask_to_grid_preserves_class_ids_no_blending():
    # Majority vote must never invent a class ID between two real ones.
    mask = np.zeros((10, 10), dtype=np.int64)
    mask[:, 5:] = 9
    grid = _downsample_mask_to_grid(mask, 4, 4)
    assert set(np.unique(grid).tolist()) <= {0, 9}


# ---------------------------------------------------------------------------
# _find_mask_file
# ---------------------------------------------------------------------------


def test_find_mask_file_recursive(tmp_path: Path):
    sub = tmp_path / "nested" / "deeper"
    sub.mkdir(parents=True)
    target = sub / "tile_01.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(target)
    found = _find_mask_file(tmp_path, "tile_01")
    assert found == target


def test_find_mask_file_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="tile_missing"):
        _find_mask_file(tmp_path, "tile_missing")


def test_find_mask_file_deterministic_with_multiple_extensions(tmp_path: Path):
    img = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
    img.save(tmp_path / "tile_01.jpg")
    img.save(tmp_path / "tile_01.png")
    found = _find_mask_file(tmp_path, "tile_01")
    # sorted() picks .jpg before .png deterministically
    assert found.suffix == ".jpg"


def test_find_mask_file_rejects_path_separators(tmp_path: Path):
    with pytest.raises(ValueError, match="path separators"):
        _find_mask_file(tmp_path, "nested/tile_01")
    with pytest.raises(ValueError, match="path separators"):
        _find_mask_file(tmp_path, "nested\\tile_01")


def test_find_mask_file_rejects_empty_id(tmp_path: Path):
    with pytest.raises(ValueError, match="non-empty"):
        _find_mask_file(tmp_path, "")


def test_find_mask_file_glob_metacharacters_matched_literally(tmp_path: Path):
    # A tile ID containing glob wildcards must match its own file, not act as
    # a pattern that could match (or fail on) unrelated files.
    img = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
    img.save(tmp_path / "tile[1].png")
    img.save(tmp_path / "tile1.png")
    found = _find_mask_file(tmp_path, "tile[1]")
    assert found.name == "tile[1].png"
