# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the split-free separation metrics block in the dpt evaluation wirings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import precisionai.agrieval.emb.metrics.separation as separation_module
from precisionai.agrieval.dpt.services.evaluate import run_dpt_eval, run_dpt_image_eval

_SEPARATION_KEYS = {"silhouette", "calinski_harabasz", "ari", "nmi", "pc1_auroc"}

_GRID_H = 4
_GRID_W = 5
_EMBED_DIM = 6


def _split_tiles_and_masks(
    tmp_path: Path,
    class_a: tuple[str, list[int], int],
    class_b: tuple[str, list[int], int],
    *,
    n_tiles: int = 4,
) -> tuple[dict[str, list], Path]:
    """Build tiles split left/right between two class prototypes plus matching masks."""
    rng = np.random.default_rng(3)
    proto_a = rng.normal(size=_EMBED_DIM).astype(np.float32)
    proto_b = proto_a + 8.0

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir(exist_ok=True)
    tiles: dict[str, list] = {}

    mask_h, mask_w = 20, 25
    half_col = _GRID_W // 2
    for i in range(n_tiles):
        tile_id = f"tile_{i:02d}"

        mask_rgb = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
        mask_rgb[:, : mask_w // 2] = class_a[1]
        mask_rgb[:, mask_w // 2 :] = class_b[1]
        Image.fromarray(mask_rgb).save(masks_dir / f"{tile_id}.png")

        tile = np.empty((_EMBED_DIM, _GRID_H, _GRID_W), dtype=np.float32)
        for col in range(_GRID_W):
            proto = proto_a if col < half_col else proto_b
            noise = rng.normal(scale=0.05, size=(_GRID_H, _EMBED_DIM)).astype(np.float32)
            tile[:, :, col] = (proto + noise).T
        tiles[tile_id] = tile.tolist()

    return tiles, masks_dir


class TestSeparationBlock:
    def test_absent_without_ground_truth(self, synthetic_tiles: dict) -> None:
        result = run_dpt_eval(tiles=synthetic_tiles)
        assert result["separation"] is None

    def test_present_with_ground_truth(self, labeled_dataset: tuple, classes_path: Path) -> None:
        tiles, masks_dir, _ = labeled_dataset
        result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)
        assert set(result["separation"]) == _SEPARATION_KEYS

    def test_well_separated_classes_score_high(self, labeled_dataset: tuple, classes_path: Path) -> None:
        tiles, masks_dir, _ = labeled_dataset
        sep = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)["separation"]
        assert sep["silhouette"] > 0.2
        assert sep["calinski_harabasz"] > 100.0
        assert sep["ari"] > 0.99
        assert sep["nmi"] > 0.99
        assert sep["pc1_auroc"] > 0.99

    def test_image_wiring_matches_tile_wiring(self, labeled_dataset: tuple, classes_path: Path) -> None:
        tiles, masks_dir, _ = labeled_dataset
        tile_result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)
        image_result = run_dpt_image_eval(images=tiles, masks_dir=masks_dir, classes_path=classes_path)
        assert image_result["separation"] == tile_result["separation"]

    def test_values_are_json_safe(self, labeled_dataset: tuple, classes_path: Path) -> None:
        tiles, masks_dir, _ = labeled_dataset
        result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)
        json.dumps(result["separation"])


class TestSeparationDegenerateCases:
    def test_single_class_skips_all_with_warnings(self, tmp_path: Path, classes: list, classes_path: Path) -> None:
        bg_color = next(c for _, c, i in classes if i == 0)
        rng = np.random.default_rng(4)

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        tiles: dict[str, list] = {}
        for i in range(3):
            tile_id = f"tile_{i:02d}"
            mask_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            mask_rgb[:, :] = bg_color
            Image.fromarray(mask_rgb).save(masks_dir / f"{tile_id}.png")
            tiles[tile_id] = rng.normal(size=(_EMBED_DIM, _GRID_H, _GRID_W)).astype(np.float32).tolist()

        result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)

        assert result["classes"] == ["background"]
        assert all(v is None for v in result["separation"].values())
        warnings = "\n".join(result["warnings"])
        assert "calinski_harabasz skipped" in warnings
        assert "silhouette/ari/nmi skipped" in warnings
        assert "pc1_auroc skipped" in warnings

    def test_no_background_class_skips_pc1_auroc_only(self, tmp_path: Path) -> None:
        class_a = ("soil", [10, 10, 10], 0)
        class_b = ("crop", [0, 255, 0], 1)
        classes_path = tmp_path / "class_map.json"
        classes_path.write_text(
            json.dumps(
                {"classes": [{"id": cid, "name": name, "color": color} for name, color, cid in (class_a, class_b)]}
            )
        )
        tiles, masks_dir = _split_tiles_and_masks(tmp_path, class_a, class_b)

        result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)

        sep = result["separation"]
        assert sep["pc1_auroc"] is None
        assert sep["silhouette"] is not None
        assert sep["calinski_harabasz"] is not None
        assert sep["ari"] is not None
        assert sep["nmi"] is not None
        assert any("no class named 'background'" in w for w in result["warnings"])

    def test_missing_sklearn_skips_ari_nmi_only(
        self, labeled_dataset: tuple, classes_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(separation_module, "_SKLEARN_AVAILABLE", False)
        tiles, masks_dir, _ = labeled_dataset
        result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)

        sep = result["separation"]
        assert sep["ari"] is None
        assert sep["nmi"] is None
        assert sep["silhouette"] is not None
        assert sep["calinski_harabasz"] is not None
        assert sep["pc1_auroc"] is not None
        assert any("scikit-learn is not installed" in w for w in result["warnings"])
