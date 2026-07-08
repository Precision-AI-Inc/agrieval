from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from precisionai.agrieval.seg.services.evaluate import load_classes

DATA_DIR = Path(__file__).parent.parent / "data"

_GRID_H = 4
_GRID_W = 5
_EMBED_DIM = 6
_N_TILES = 6


@pytest.fixture(scope="module")
def classes_path() -> Path:
    return DATA_DIR / "class_map.json"


@pytest.fixture(scope="module")
def classes(classes_path: Path) -> list:
    return load_classes(classes_path)


@pytest.fixture(scope="module")
def real_masks_dir() -> Path:
    """Ground-truth masks for the real dpt tile fixtures, keyed by image filename stem."""
    return DATA_DIR / "masks"


@pytest.fixture(scope="module")
def real_dpt_tile_archives() -> list[Path]:
    """Real (non-synthetic) tile-batch archives, one per source image.

    Each archive's ``feature_maps`` are genuine backbone output, average-pooled
    down to an 8x12 grid to keep fixtures small (~580 KB each) — embedding dim
    and tile pixel geometry are untouched.
    """
    return sorted((DATA_DIR / "dpt_tiles").rglob("*.npz"))


def _make_tile(rng: np.random.Generator) -> list:
    """Random (P, H, W) tile as nested lists."""
    return rng.normal(size=(_EMBED_DIM, _GRID_H, _GRID_W)).astype(np.float32).tolist()


@pytest.fixture
def synthetic_tiles() -> dict[str, list]:
    rng = np.random.default_rng(0)
    return {f"tile_{i:02d}": _make_tile(rng) for i in range(_N_TILES)}


@pytest.fixture
def labeled_dataset(tmp_path: Path, classes: list) -> tuple[dict[str, list], Path, str]:
    """Class-correlated tiles plus matching ground-truth masks on disk.

    Each tile is split left/right between two well-separated class
    prototypes, and each corresponding mask is colored the same way at a
    different (larger) resolution — forcing the mask-to-patch-grid
    downsampling path to be exercised.
    """
    rng = np.random.default_rng(1)
    bg_color = next(c for _, c, i in classes if i == 0)
    crop_name, crop_color, _ = next((n, c, i) for n, c, i in classes if i == 1)

    bg_proto = rng.normal(size=_EMBED_DIM).astype(np.float32)
    crop_proto = bg_proto + 8.0

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    tiles: dict[str, list] = {}

    mask_h, mask_w = 20, 25
    half_col = _GRID_W // 2
    for i in range(_N_TILES):
        tile_id = f"tile_{i:02d}"

        mask_rgb = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
        mask_rgb[:, : mask_w // 2] = bg_color
        mask_rgb[:, mask_w // 2 :] = crop_color
        Image.fromarray(mask_rgb).save(masks_dir / f"{tile_id}.png")

        tile = np.empty((_EMBED_DIM, _GRID_H, _GRID_W), dtype=np.float32)
        for col in range(_GRID_W):
            proto = bg_proto if col < half_col else crop_proto
            noise = rng.normal(scale=0.05, size=(_GRID_H, _EMBED_DIM)).astype(np.float32)
            tile[:, :, col] = (proto + noise).T
        tiles[tile_id] = tile.tolist()

    return tiles, masks_dir, crop_name
