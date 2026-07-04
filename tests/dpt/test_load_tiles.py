from pathlib import Path

import numpy as np
import pytest

from precisionai.agrieval.dpt import run_dpt_eval
from precisionai.agrieval.dpt.services.evaluate import _validate_tile_arrays, load_tiles

_SHAPE = (3, 2, 4)  # (C, H, W)


def _write_npy_dir(root: Path, n: int = 3, shape=_SHAPE) -> Path:
    rng = np.random.default_rng(0)
    root.mkdir(exist_ok=True)
    for i in range(n):
        np.save(root / f"tile_{i:02d}.npy", rng.normal(size=shape).astype(np.float32))
    return root


# ---------------------------------------------------------------------------
# .npy directory layout
# ---------------------------------------------------------------------------


def test_load_npy_directory(tmp_path):
    tiles_dir = _write_npy_dir(tmp_path / "tiles")
    tiles = load_tiles(tiles_dir)
    assert sorted(tiles) == ["tile_00", "tile_01", "tile_02"]
    for arr in tiles.values():
        assert arr.shape == _SHAPE
        assert arr.dtype == np.float32


def test_load_npy_directory_recursive(tmp_path):
    nested = tmp_path / "tiles" / "batch_a"
    _write_npy_dir(nested.parent, n=1)
    nested.mkdir()
    np.save(nested / "tile_99.npy", np.zeros(_SHAPE, dtype=np.float32))
    tiles = load_tiles(tmp_path / "tiles")
    assert "tile_99" in tiles
    assert "tile_00" in tiles


def test_empty_directory_raises(tmp_path):
    (tmp_path / "tiles").mkdir()
    with pytest.raises(ValueError, match=r"No \.npy tile files"):
        load_tiles(tmp_path / "tiles")


def test_duplicate_stems_across_subdirs_raise(tmp_path):
    tiles_dir = _write_npy_dir(tmp_path / "tiles", n=1)
    sub = tiles_dir / "dup"
    sub.mkdir()
    np.save(sub / "tile_00.npy", np.zeros(_SHAPE, dtype=np.float32))
    with pytest.raises(ValueError, match="Duplicate tile ID"):
        load_tiles(tiles_dir)


# ---------------------------------------------------------------------------
# .npz archive layout
# ---------------------------------------------------------------------------


def test_load_npz_archive(tmp_path):
    rng = np.random.default_rng(1)
    archive = tmp_path / "tiles.npz"
    np.savez(archive, a=rng.normal(size=_SHAPE), b=rng.normal(size=_SHAPE))
    tiles = load_tiles(archive)
    assert sorted(tiles) == ["a", "b"]
    assert all(arr.dtype == np.float32 for arr in tiles.values())


def test_empty_npz_archive_raises(tmp_path):
    archive = tmp_path / "empty.npz"
    np.savez(archive)  # no arrays at all
    with pytest.raises(ValueError, match="contains no arrays"):
        load_tiles(archive)


def test_npz_object_array_rejected(tmp_path):
    archive = tmp_path / "tiles.npz"
    obj = np.empty((1,), dtype=object)
    obj[0] = {"not": "an array"}
    np.savez(archive, a=obj)
    # allow_pickle=False refuses to deserialise object arrays.
    with pytest.raises(ValueError, match=r"allow_pickle|pickle"):
        load_tiles(archive)


# ---------------------------------------------------------------------------
# path errors
# ---------------------------------------------------------------------------


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="tiles_path"):
        load_tiles(tmp_path / "nope")


def test_non_npz_file_raises(tmp_path):
    stray = tmp_path / "tiles.json"
    stray.write_text("{}")
    with pytest.raises(FileNotFoundError, match="tiles_path"):
        load_tiles(stray)


# ---------------------------------------------------------------------------
# array validation
# ---------------------------------------------------------------------------


def test_mismatched_shapes_across_files_raise(tmp_path):
    tiles_dir = _write_npy_dir(tmp_path / "tiles", n=1)
    np.save(tiles_dir / "odd.npy", np.zeros((3, 2, 5), dtype=np.float32))
    with pytest.raises(ValueError, match=r"same \(C, H, W\) shape"):
        load_tiles(tiles_dir)


def test_non_3d_file_raises(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    np.save(tiles_dir / "flat.npy", np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="3-dimensional"):
        load_tiles(tiles_dir)


def test_non_finite_file_raises(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    bad = np.zeros(_SHAPE, dtype=np.float32)
    bad[0, 0, 0] = np.nan
    np.save(tiles_dir / "bad.npy", bad)
    with pytest.raises(ValueError, match="non-finite"):
        load_tiles(tiles_dir)


def test_float64_input_converted_to_float32(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    np.save(tiles_dir / "wide.npy", np.ones(_SHAPE, dtype=np.float64))
    tiles = load_tiles(tiles_dir)
    assert tiles["wide"].dtype == np.float32


def test_validate_tile_arrays_empty_dict_raises():
    with pytest.raises(ValueError, match="At least 1 tile"):
        _validate_tile_arrays({})


def test_validate_tile_arrays_string_dtype_raises():
    arr = np.array([[["a"]]])
    with pytest.raises(ValueError, match="float32"):
        _validate_tile_arrays({"a": arr})


def test_validate_tile_arrays_empty_axis_raises():
    with pytest.raises(ValueError, match="empty along any axis"):
        _validate_tile_arrays({"a": np.zeros((3, 0, 4), dtype=np.float32)})


def test_validate_tile_arrays_bad_id_raises():
    with pytest.raises(ValueError, match="path separators"):
        _validate_tile_arrays({"a/b": np.zeros(_SHAPE, dtype=np.float32)})


# ---------------------------------------------------------------------------
# end-to-end with run_dpt_eval
# ---------------------------------------------------------------------------


def test_loaded_tiles_flow_through_run_dpt_eval(tmp_path):
    tiles_dir = _write_npy_dir(tmp_path / "tiles")
    result = run_dpt_eval(tiles=load_tiles(tiles_dir))
    assert result["n_tiles"] == 3
    assert result["embed_dim"] == _SHAPE[0]
    assert result["grid_height"] == _SHAPE[1]
    assert result["grid_width"] == _SHAPE[2]
