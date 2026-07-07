import json
from pathlib import Path

import numpy as np
import pytest

from precisionai.agrieval.dpt import TilePlacement, run_dpt_eval
from precisionai.agrieval.dpt.services.evaluate import load_tile_placement, load_tiles

_SHAPE = (3, 2, 4)  # (P, H, W)
_TILE_META = json.dumps({"model": "test", "embed_dim": _SHAPE[0], "tile": "798x532"})


def _make_batch_arrays(
    image_tiles: dict[str, list[tuple[int, int]]],
    *,
    shape: tuple[int, int, int] = _SHAPE,
    meta: str | None = _TILE_META,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build feature_maps/tile_image_id/tile_index/tile_y0/tile_x0/filenames/meta.

    ``image_tiles`` maps each image filename to a list of ``(y0, x0)`` pixel
    offsets — list order becomes each tile's ``tile_index``.
    """
    rng = np.random.default_rng(seed)
    filenames = list(image_tiles)
    feature_maps, tile_image_id, tile_index, tile_y0, tile_x0 = [], [], [], [], []
    for img_idx, fname in enumerate(filenames):
        for idx, (y0, x0) in enumerate(image_tiles[fname]):
            feature_maps.append(rng.normal(size=shape).astype(np.float32))
            tile_image_id.append(img_idx)
            tile_index.append(idx)
            tile_y0.append(y0)
            tile_x0.append(x0)
    arrays: dict[str, np.ndarray] = {
        "feature_maps": np.stack(feature_maps).astype(np.float32),
        "tile_image_id": np.array(tile_image_id, dtype=np.int64),
        "tile_index": np.array(tile_index, dtype=np.int64),
        "tile_y0": np.array(tile_y0, dtype=np.int64),
        "tile_x0": np.array(tile_x0, dtype=np.int64),
        "filenames": np.array(filenames),
    }
    if meta is not None:
        arrays["meta"] = np.array(meta)
    return arrays


def _savez(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    """Write ``arrays`` to ``path`` as an ``.npz`` archive.

    Isolates the one spot where unpacking a ``dict[str, np.ndarray]`` into
    ``np.savez``'s ``**kwds: ArrayLike`` parameter trips a pyright false
    positive against the unrelated ``allow_pickle: bool`` parameter — pyright
    cannot statically rule out ``"allow_pickle"`` being a dict key.
    """
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType]
    return path


def _write_batch_npz(path: Path, image_tiles: dict[str, list[tuple[int, int]]], **kwargs) -> Path:
    return _savez(path, _make_batch_arrays(image_tiles, **kwargs))


# ---------------------------------------------------------------------------
# basic round-trip
# ---------------------------------------------------------------------------


def test_load_single_tile_per_image(tmp_path):
    archive = _write_batch_npz(tmp_path / "tiles.npz", {"a.png": [(0, 0)], "b.png": [(0, 0)], "c.png": [(0, 0)]})
    tiles = load_tiles(archive)
    assert sorted(tiles) == ["a_tile_00", "b_tile_00", "c_tile_00"]
    for arr in tiles.values():
        assert arr.shape == _SHAPE
        assert arr.dtype == np.float32


def test_load_multiple_tiles_per_image(tmp_path):
    archive = _write_batch_npz(tmp_path / "tiles.npz", {"field.png": [(0, 0), (0, 534), (8, 0), (8, 534)]})
    tiles = load_tiles(archive)
    assert sorted(tiles) == ["field_tile_00", "field_tile_01", "field_tile_02", "field_tile_03"]


def test_load_variable_tiles_per_image(tmp_path):
    archive = _write_batch_npz(
        tmp_path / "tiles.npz", {"small.png": [(0, 0), (0, 100)], "big.png": [(0, 0), (0, 100), (100, 0)]}
    )
    tiles = load_tiles(archive)
    assert len(tiles) == 2 + 3
    assert "small_tile_01" in tiles
    assert "big_tile_02" in tiles


def test_load_tile_placement_matches_load_tiles_keys(tmp_path):
    archive = _write_batch_npz(tmp_path / "tiles.npz", {"field.png": [(0, 0), (0, 534), (8, 0), (8, 534)]})
    tiles = load_tiles(archive)
    placement = load_tile_placement(archive)
    assert set(placement) == set(tiles)
    assert placement["field_tile_03"] == TilePlacement(image_stem="field", y0=8, x0=534, height=532, width=798)


# ---------------------------------------------------------------------------
# path errors
# ---------------------------------------------------------------------------


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="tiles_path"):
        load_tiles(tmp_path / "nope.npz")


def test_non_npz_file_raises(tmp_path):
    stray = tmp_path / "tiles.json"
    stray.write_text("{}")
    with pytest.raises(FileNotFoundError, match="tiles_path"):
        load_tiles(stray)


def test_directory_no_longer_supported_raises(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="tiles_path"):
        load_tiles(tiles_dir)


# ---------------------------------------------------------------------------
# schema validation
# ---------------------------------------------------------------------------


def test_missing_required_key_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]})
    del arrays["tile_x0"]
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match=r"missing required key\(s\).*tile_x0"):
        load_tiles(archive)


def test_feature_maps_not_4d_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]})
    arrays["feature_maps"] = arrays["feature_maps"][:, 0]  # drop one axis -> 3-D
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match="4-dimensional"):
        load_tiles(archive)


def test_feature_maps_empty_axis_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]}, shape=(3, 0, 4))
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match="empty along any axis"):
        load_tiles(archive)


def test_mismatched_companion_length_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0), (0, 10)]})
    arrays["tile_y0"] = arrays["tile_y0"][:1]
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match="tile_y0"):
        load_tiles(archive)


def test_tile_image_id_out_of_range_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]})
    arrays["tile_image_id"] = np.array([5], dtype=np.int64)
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match="out of range"):
        load_tiles(archive)


def test_duplicate_tile_position_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0), (0, 10)]})
    arrays["tile_index"] = np.array([0, 0], dtype=np.int64)  # both tiles now index 0
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match="Duplicate tile position"):
        load_tiles(archive)


def test_non_finite_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]})
    arrays["feature_maps"][0, 0, 0, 0] = np.nan
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match="non-finite"):
        load_tiles(archive)


def test_negative_tile_offset_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]})
    arrays["tile_y0"] = np.array([-5], dtype=np.int64)
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match="must not contain negative values"):
        load_tiles(archive)


def test_negative_tile_index_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]})
    arrays["tile_index"] = np.array([-1], dtype=np.int64)
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match="must not contain negative values"):
        load_tiles(archive)


def test_duplicate_filename_stems_raise(tmp_path):
    # "a.png" and "a.jpg" share the stem "a" — their tiles would collide in
    # the synthesized tile IDs and silently overwrite each other.
    arrays = _make_batch_arrays({"a.png": [(0, 0)], "a.jpg": [(0, 0)]})
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match="Duplicate image stem 'a'"):
        load_tiles(archive)


def test_uppercase_npz_suffix_accepted(tmp_path):
    # np.savez itself appends ".npz" to non-lowercase suffixes, so write
    # normally and rename to get a genuine .NPZ file on disk.
    written = _write_batch_npz(tmp_path / "tiles.npz", {"a.png": [(0, 0)]})
    archive = written.rename(tmp_path / "tiles.NPZ")
    tiles = load_tiles(archive)
    assert list(tiles) == ["a_tile_00"]


def test_float64_input_converted_to_float32(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]})
    arrays["feature_maps"] = arrays["feature_maps"].astype(np.float64)
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    tiles = load_tiles(archive)
    assert next(iter(tiles.values())).dtype == np.float32


def test_pickled_object_feature_maps_rejected(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]})
    obj = np.empty((1,), dtype=object)
    obj[0] = {"not": "an array"}
    arrays["feature_maps"] = obj
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    # allow_pickle=False refuses to deserialise object arrays.
    with pytest.raises(ValueError, match=r"allow_pickle|pickle"):
        load_tiles(archive)


# ---------------------------------------------------------------------------
# load_tile_placement — meta.tile parsing
# ---------------------------------------------------------------------------


def test_load_tile_placement_missing_meta_raises(tmp_path):
    arrays = _make_batch_arrays({"a.png": [(0, 0)]}, meta=None)
    archive = tmp_path / "tiles.npz"
    _savez(archive, arrays)
    with pytest.raises(ValueError, match=r"missing required key\(s\).*meta"):
        load_tile_placement(archive)


def test_load_tile_placement_malformed_meta_json_raises(tmp_path):
    archive = _write_batch_npz(tmp_path / "tiles.npz", {"a.png": [(0, 0)]}, meta="not json")
    with pytest.raises(ValueError, match="malformed 'meta' JSON"):
        load_tile_placement(archive)


def test_load_tile_placement_missing_tile_field_raises(tmp_path):
    archive = _write_batch_npz(tmp_path / "tiles.npz", {"a.png": [(0, 0)]}, meta=json.dumps({"model": "test"}))
    with pytest.raises(ValueError, match=r"meta\.tile must be a"):
        load_tile_placement(archive)


def test_load_tile_placement_malformed_tile_field_raises(tmp_path):
    archive = _write_batch_npz(tmp_path / "tiles.npz", {"a.png": [(0, 0)]}, meta=json.dumps({"tile": "not-a-size"}))
    with pytest.raises(ValueError, match=r"meta\.tile must be a"):
        load_tile_placement(archive)


def test_load_tile_placement_non_numeric_tile_field_raises(tmp_path):
    archive = _write_batch_npz(tmp_path / "tiles.npz", {"a.png": [(0, 0)]}, meta=json.dumps({"tile": "abcxdef"}))
    with pytest.raises(ValueError, match=r"meta\.tile must be a"):
        load_tile_placement(archive)


# ---------------------------------------------------------------------------
# end-to-end with run_dpt_eval
# ---------------------------------------------------------------------------


def test_loaded_tiles_flow_through_run_dpt_eval(tmp_path):
    archive = _write_batch_npz(tmp_path / "tiles.npz", {"a.png": [(0, 0)], "b.png": [(0, 0)], "c.png": [(0, 0)]})
    result = run_dpt_eval(tiles=load_tiles(archive))
    assert result["n_tiles"] == 3
    assert result["embed_dim"] == _SHAPE[0]
    assert result["grid_height"] == _SHAPE[1]
    assert result["grid_width"] == _SHAPE[2]
