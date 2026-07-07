# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Run the dense patch token evaluation service directly — no HTTP server required.

Generates small synthetic tiles (random feature maps; this example does not
depend on a real vision backbone) and evaluates them. By default one tile is
generated per ground-truth mask shipped in ``tests/data/masks/`` so the
label-aware metrics (``classes``, ``per_class``, ``knn_confusion``,
``separation``) are exercised out of the box — since the tiles are random
noise, expect near-chance values (kNN purity at chance, silhouette/ARI/NMI
near 0, PC1-AUROC near 0.5); this only demonstrates the wiring, not a real
model.

Usage (from repo root)::

    python examples/dpt/example.py
    python examples/dpt/example.py --no-labels
    python examples/dpt/example.py --masks path/to/masks --classes path/to/class_map.json
    python examples/dpt/example.py --grid-height 8 --grid-width 12 --embed-dim 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from precisionai.agrieval.dpt import print_result, run_dpt_eval

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_MASKS = _REPO_ROOT / "tests" / "data" / "masks"
_DEFAULT_CLASSES = _REPO_ROOT / "tests" / "data" / "class_map.json"
_N_UNLABELED_TILES = 10
_SEED = 0


def _display_path(path: Path) -> Path:
    """Show a path relative to the repo root when possible, never an absolute local path."""
    try:
        return path.relative_to(_REPO_ROOT)
    except ValueError:
        return path


def _synthetic_tiles(tile_ids: list[str], *, embed_dim: int, grid_h: int, grid_w: int) -> dict[str, list]:
    rng = np.random.default_rng(_SEED)
    return {tid: rng.normal(size=(embed_dim, grid_h, grid_w)).astype(np.float32).tolist() for tid in tile_ids}


def _tile_ids_from_masks(masks_dir: Path) -> list[str]:
    return sorted({p.stem for p in masks_dir.rglob("*") if p.suffix in {".png", ".PNG", ".jpg", ".JPG"}})


def main() -> None:
    """Generate synthetic tiles, run dense patch token evaluation, and print a summary.

    Recognised arguments
    --------------------
    --masks : str
        Directory of ground-truth colour-coded masks, one per tile (filename
        stem becomes the tile ID). Defaults to ``tests/data/masks``.
    --classes : str
        Path to the AgriBench class-definition JSON. Defaults to ``tests/data/class_map.json``.
    --no-labels
        Skip ground truth entirely and generate a handful of unlabeled tiles instead.
    --grid-height / --grid-width / --embed-dim
        Synthetic tile dimensions.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate dense patch token tiles (synthetic data for demonstration).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--masks", type=Path, default=_DEFAULT_MASKS, metavar="DIR")
    parser.add_argument("--classes", type=Path, default=_DEFAULT_CLASSES, metavar="FILE")
    parser.add_argument("--no-labels", action="store_true", help="Skip ground truth; generate unlabeled tiles.")
    parser.add_argument("--grid-height", type=int, default=8, metavar="H")
    parser.add_argument("--grid-width", type=int, default=12, metavar="W")
    parser.add_argument("--embed-dim", type=int, default=32, metavar="P")
    args = parser.parse_args()

    if args.no_labels:
        tile_ids = [f"tile_{i:02d}" for i in range(_N_UNLABELED_TILES)]
        masks_dir, classes_path = None, None
    else:
        tile_ids = _tile_ids_from_masks(args.masks)
        masks_dir, classes_path = args.masks, args.classes

    tiles = _synthetic_tiles(tile_ids, embed_dim=args.embed_dim, grid_h=args.grid_height, grid_w=args.grid_width)

    print(
        f"Tiles       : {len(tiles)} synthetic tile(s), grid {args.grid_height}x{args.grid_width}, P={args.embed_dim}"
    )
    if masks_dir is not None and classes_path is not None:
        print(f"Ground truth: {_display_path(masks_dir)}")
        print(f"Classes     : {_display_path(classes_path)}")
    print()

    result = run_dpt_eval(tiles=tiles, masks_dir=masks_dir, classes_path=classes_path)
    print_result(result)


if __name__ == "__main__":
    main()
