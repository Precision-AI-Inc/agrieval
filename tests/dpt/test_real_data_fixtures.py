"""Tests using real (non-synthetic) tile fixtures under tests/data/dpt_tiles/.

These exercise the full tiles_path -> load_tiles/load_tile_placement -> run_dpt_eval
path against genuine feature values and real ground-truth masks, as a check
beyond the synthetic-data tests elsewhere in this package.
"""

from pathlib import Path

import numpy as np
import pytest

from precisionai.agrieval.dpt import load_tile_placement, load_tiles, run_dpt_eval

_EXPECTED_TILES_PER_IMAGE = 4
_EXPECTED_GRID = (8, 12)
_EXPECTED_EMBED_DIM = 384


def test_each_fixture_loads_with_expected_shape(real_dpt_tile_archives: list[Path]) -> None:
    assert len(real_dpt_tile_archives) == 4
    for path in real_dpt_tile_archives:
        tiles = load_tiles(path)
        assert len(tiles) == _EXPECTED_TILES_PER_IMAGE
        for arr in tiles.values():
            assert arr.shape == (_EXPECTED_EMBED_DIM, *_EXPECTED_GRID)
            assert arr.dtype == np.float32
            assert np.all(np.isfinite(arr))


def test_each_fixture_placement_matches_its_own_image_stem(real_dpt_tile_archives: list[Path]) -> None:
    for path in real_dpt_tile_archives:
        image_stem = path.stem
        placement = load_tile_placement(path)
        assert len(placement) == _EXPECTED_TILES_PER_IMAGE
        assert all(p.image_stem == image_stem for p in placement.values())
        # meta.tile ("798x532") parsed correctly for every tile.
        assert all((p.height, p.width) == (532, 798) for p in placement.values())


def test_unsupervised_eval_runs_on_each_fixture(real_dpt_tile_archives: list[Path]) -> None:
    for path in real_dpt_tile_archives:
        result = run_dpt_eval(tiles=load_tiles(path), k_values=[3])
        assert result["n_tiles"] == _EXPECTED_TILES_PER_IMAGE
        assert result["embed_dim"] == _EXPECTED_EMBED_DIM
        assert (result["grid_height"], result["grid_width"]) == _EXPECTED_GRID
        assert result["classes"] is None


def test_labeled_eval_runs_on_each_fixture(real_dpt_tile_archives: list[Path], real_masks_dir: Path, classes_path):
    for path in real_dpt_tile_archives:
        tiles = load_tiles(path)
        placement = load_tile_placement(path)
        result = run_dpt_eval(
            tiles=tiles, tile_placement=placement, masks_dir=real_masks_dir, classes_path=classes_path, k_values=[3]
        )
        assert result["n_patches"] == _EXPECTED_TILES_PER_IMAGE * _EXPECTED_GRID[0] * _EXPECTED_GRID[1]
        assert "background" in result["classes"]
        if len(result["classes"]) >= 2:
            assert result["warnings"] is None
            assert result["separation"]["calinski_harabasz"] is not None
        else:
            # An all-background image can't support any separation metric:
            # each is skipped with an explanatory warning instead of failing.
            assert all(v is None for v in result["separation"].values())
            assert all("skipped" in w for w in result["warnings"])


def test_combined_multi_image_real_corpus(real_dpt_tile_archives: list[Path], real_masks_dir: Path, classes_path):
    """Merge all 4 real per-image archives into one multi-image evaluation, like a real batch job would."""
    tiles: dict[str, np.ndarray] = {}
    placement = {}
    for path in real_dpt_tile_archives:
        tiles.update(load_tiles(path))
        placement.update(load_tile_placement(path))

    result = run_dpt_eval(
        tiles=tiles, tile_placement=placement, masks_dir=real_masks_dir, classes_path=classes_path, k_values=[5]
    )
    assert result["n_tiles"] == len(real_dpt_tile_archives) * _EXPECTED_TILES_PER_IMAGE
    assert "background" in result["classes"]
    # Every real image contributes patches with a real, positive effective rank.
    assert result["global_metrics"]["effective_rank"]["effective_rank"] > 0
    # The combined corpus spans background + foreground classes, so every
    # separation metric is computable on real features.
    assert all(v is not None for v in result["separation"].values())


def test_missing_fixture_reported_clearly(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="tiles_path"):
        load_tiles(tmp_path / "not_a_real_fixture.npz")
