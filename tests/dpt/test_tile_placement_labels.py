"""Correctness test for crop-aware label alignment (run_dpt_eval with tile_placement).

Proves the whole-image-mask crop step actually crops per tile's exact pixel
rectangle rather than downsampling the whole mask identically for every tile
— the regression this guards against would make every tile of an image
report the same (whole-mask-majority) label regardless of its own region's
content. Also proves overlapping tiles (as real producers emit — see
../sample-patch-tokens) get independent, correct labels, which a
grid-partition approach (assuming tiles never overlap) could not guarantee.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from precisionai.agrieval.dpt import TilePlacement, run_dpt_eval

_TILE_IDS = ["field_tile_00", "field_tile_01", "field_tile_02", "field_tile_03"]
_PLACEMENT = {
    "field_tile_00": TilePlacement("field", y0=0, x0=0, height=16, width=16),
    "field_tile_01": TilePlacement("field", y0=0, x0=16, height=16, width=16),
    "field_tile_02": TilePlacement("field", y0=16, x0=0, height=16, width=16),
    "field_tile_03": TilePlacement("field", y0=16, x0=16, height=16, width=16),
}


def test_quadrant_crop_isolates_label_to_its_own_tile(tmp_path: Path, classes: list, classes_path: Path) -> None:
    bg_color = next(c for _, c, i in classes if i == 0)
    crop_name, crop_color, _ = next((n, c, i) for n, c, i in classes if i == 1)

    # Whole 32x32 mask: three quadrants background, bottom-right quadrant
    # (y0=16, x0=16) is entirely the crop colour. Background dominates the
    # mask overall (75%), so a bug that downsamples the whole mask once per
    # tile (skipping the per-tile crop) would assign every tile the same
    # "background" majority and never surface the crop colour at all.
    mask = np.zeros((32, 32, 3), dtype=np.uint8)
    mask[:16, :] = bg_color
    mask[16:, :16] = bg_color
    mask[16:, 16:] = crop_color

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    Image.fromarray(mask).save(masks_dir / "field.png")

    rng = np.random.default_rng(0)
    tiles = {tid: rng.normal(size=(3, 1, 1)).astype(np.float32).tolist() for tid in _TILE_IDS}

    result = run_dpt_eval(
        tiles=tiles,
        masks_dir=masks_dir,
        classes_path=classes_path,
        tile_placement=_PLACEMENT,
        k_values=[1],
    )

    assert set(result["classes"]) == {"background", crop_name}
    assert result["per_class"]["background"]["n_patches"] == 3
    assert result["per_class"][crop_name]["n_patches"] == 1


def test_overlapping_tiles_get_independent_labels(tmp_path: Path, classes: list, classes_path: Path) -> None:
    """Real producers emit overlapping tiles (see ../sample-patch-tokens) — prove
    two tiles whose pixel rectangles overlap still get independently-correct labels,
    which a non-overlapping grid-partition approach could not guarantee.
    """
    bg_color = next(c for _, c, i in classes if i == 0)
    crop_name, crop_color, _ = next((n, c, i) for n, c, i in classes if i == 1)

    # 32x32 mask split left/right: cols 0-16 background, cols 16-32 crop.
    mask = np.zeros((32, 32, 3), dtype=np.uint8)
    mask[:, :16] = bg_color
    mask[:, 16:] = crop_color

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    Image.fromarray(mask).save(masks_dir / "field.png")

    # Two tiles, both full-height, width 20, overlapping in cols 12-20 (8px).
    # Tile 00 spans cols 0-20 (16 bg cols + 4 crop cols) -> background majority.
    # Tile 01 spans cols 12-32 (4 bg cols + 16 crop cols) -> crop majority.
    placement = {
        "field_tile_00": TilePlacement("field", y0=0, x0=0, height=32, width=20),
        "field_tile_01": TilePlacement("field", y0=0, x0=12, height=32, width=20),
    }
    rng = np.random.default_rng(0)
    tiles = {tid: rng.normal(size=(3, 1, 1)).astype(np.float32).tolist() for tid in placement}

    result = run_dpt_eval(
        tiles=tiles,
        masks_dir=masks_dir,
        classes_path=classes_path,
        tile_placement=placement,
        k_values=[1],
    )

    assert set(result["classes"]) == {"background", crop_name}
    assert result["per_class"]["background"]["n_patches"] == 1
    assert result["per_class"][crop_name]["n_patches"] == 1


def test_without_placement_whole_mask_downsample_gives_different_result(
    tmp_path: Path, classes: list, classes_path: Path
) -> None:
    """Control: matching each tile directly to the whole mask (old, tile-ID-matched path)
    gives every tile the same whole-mask-majority label — the behavior the crop step exists to avoid.
    """
    bg_color = next(c for _, c, i in classes if i == 0)
    _, crop_color, _ = next((n, c, i) for n, c, i in classes if i == 1)

    mask = np.zeros((32, 32, 3), dtype=np.uint8)
    mask[:16, :] = bg_color
    mask[16:, :16] = bg_color
    mask[16:, 16:] = crop_color

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    # Every tile ID matches the SAME whole-image mask directly (no crop) —
    # this is what would happen if tile_placement were ignored.
    for tile_id in _TILE_IDS:
        Image.fromarray(mask).save(masks_dir / f"{tile_id}.png")

    rng = np.random.default_rng(0)
    tiles = {tid: rng.normal(size=(3, 1, 1)).astype(np.float32).tolist() for tid in _TILE_IDS}

    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path, k_values=[1])

    # Every tile independently downsamples the SAME whole 32x32 mask to its
    # own 1x1 grid, so every tile lands on the mask's own overall majority
    # (background, 75% coverage) — all 4 patches get the same label.
    assert result["classes"] == ["background"]
    assert result["per_class"]["background"]["n_patches"] == 4


def test_missing_masks_dir_raises_with_tile_placement(tmp_path: Path, classes_path: Path) -> None:
    rng = np.random.default_rng(0)
    tiles = {tid: rng.normal(size=(3, 1, 1)).astype(np.float32).tolist() for tid in _TILE_IDS}

    with pytest.raises(FileNotFoundError, match="masks_dir does not exist"):
        run_dpt_eval(
            tiles=tiles,
            masks_dir=tmp_path / "does_not_exist",
            classes_path=classes_path,
            tile_placement=_PLACEMENT,
        )


def test_tile_missing_from_placement_raises_clearly(tmp_path: Path, classes: list, classes_path: Path) -> None:
    bg_color = next(c for _, c, i in classes if i == 0)
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    Image.fromarray(np.full((32, 32, 3), bg_color, dtype=np.uint8)).save(masks_dir / "field.png")

    rng = np.random.default_rng(0)
    tiles = {tid: rng.normal(size=(3, 1, 1)).astype(np.float32).tolist() for tid in _TILE_IDS}
    incomplete = {tid: _PLACEMENT[tid] for tid in _TILE_IDS[:-1]}  # drop the last tile's entry

    with pytest.raises(ValueError, match=r"tile_placement is missing entries.*field_tile_03"):
        run_dpt_eval(
            tiles=tiles, masks_dir=masks_dir, classes_path=classes_path, tile_placement=incomplete, k_values=[1]
        )


def test_rectangle_entirely_outside_mask_raises_clearly(tmp_path: Path, classes: list, classes_path: Path) -> None:
    bg_color = next(c for _, c, i in classes if i == 0)
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    Image.fromarray(np.full((20, 20, 3), bg_color, dtype=np.uint8)).save(masks_dir / "field.png")

    placement = {
        "field_tile_00": TilePlacement("field", y0=100, x0=0, height=10, width=10),  # below the 20x20 mask
        "field_tile_01": TilePlacement("field", y0=0, x0=0, height=20, width=20),
    }
    rng = np.random.default_rng(0)
    tiles = {tid: rng.normal(size=(3, 1, 1)).astype(np.float32).tolist() for tid in placement}

    with pytest.raises(ValueError, match=r"field_tile_00.*lies entirely outside"):
        run_dpt_eval(
            tiles=tiles, masks_dir=masks_dir, classes_path=classes_path, tile_placement=placement, k_values=[1]
        )


def test_crop_clipped_to_mask_bounds(tmp_path: Path, classes: list, classes_path: Path) -> None:
    """A tile rectangle extending past the mask's edge is clipped, not an error."""
    bg_color = next(c for _, c, i in classes if i == 0)

    mask = np.full((20, 20, 3), bg_color, dtype=np.uint8)
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    Image.fromarray(mask).save(masks_dir / "field.png")

    # field_tile_00's rectangle (y0=10, x0=10, height=32, width=32) extends well
    # past the 20x20 mask; field_tile_01 is fully in-bounds. A second tile keeps
    # kNN's "at least 2 patches" requirement satisfied.
    placement = {
        "field_tile_00": TilePlacement("field", y0=10, x0=10, height=32, width=32),
        "field_tile_01": TilePlacement("field", y0=0, x0=0, height=20, width=20),
    }
    rng = np.random.default_rng(0)
    tiles = {tid: rng.normal(size=(3, 1, 1)).astype(np.float32).tolist() for tid in placement}

    result = run_dpt_eval(
        tiles=tiles, masks_dir=masks_dir, classes_path=classes_path, tile_placement=placement, k_values=[1]
    )
    assert result["classes"] == ["background"]
    assert result["per_class"]["background"]["n_patches"] == 2
