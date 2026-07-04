# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Dense patch token evaluation route.

- ``POST /dense-patch-tokens/evaluate`` — evaluate tile feature-map quality,
  optionally against per-tile ground-truth masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from precisionai.agrieval.dpt.api.config import get_dataset_root
from precisionai.agrieval.dpt.schemas.evaluate import DptEvalRequest, DptEvalResponse
from precisionai.agrieval.dpt.services.evaluate import load_tiles, run_dpt_eval

router = APIRouter(prefix="/dense-patch-tokens", tags=["evaluate"])


def _resolve_dataset_root(dataset_root: str | None) -> str:
    """Use the configured server default when a request omits ``dataset_root``."""
    return dataset_root if dataset_root is not None else get_dataset_root()


@router.post("/evaluate", response_model=DptEvalResponse)
def evaluate(request: DptEvalRequest) -> DptEvalResponse:
    """Evaluate dense patch token tile quality, with optional label-aware metrics.

    Supply tiles either inline (``tiles``: dict of ``tile_id -> [C, H, W]``
    feature maps) or from disk (``tiles_path``: a directory of ``.npy`` files
    or a ``.npz`` archive, resolved against ``dataset_root``) — exactly one
    of the two. Always returns unsupervised geometry and per-tile
    diagnostics. When ``masks_dir`` and ``classes_path`` are also supplied
    (resolved against ``dataset_root``, same convention as
    ``/v1/segmentation/evaluate``), the response also includes ``classes``,
    ``per_class``, and ``knn_confusion``.

    All other fields use server defaults.
    """
    if request.tiles_path is not None:
        tiles_path = Path(_resolve_dataset_root(request.dataset_root)) / request.tiles_path
        try:
            tiles: dict[str, Any] = load_tiles(tiles_path)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        tiles = request.tiles or {}

    masks_dir: Path | None = None
    classes_path: Path | None = None
    if request.masks_dir is not None and request.classes_path is not None:
        root = Path(_resolve_dataset_root(request.dataset_root))
        masks_dir = root / request.masks_dir
        classes_path = root / request.classes_path
        if not masks_dir.is_dir():
            raise HTTPException(
                status_code=400, detail=f"masks_dir does not exist or is not a directory: '{masks_dir}'"
            )
        if not classes_path.is_file():
            raise HTTPException(
                status_code=400, detail=f"classes_path does not exist or is not a file: '{classes_path}'"
            )

    try:
        result = run_dpt_eval(
            tiles=tiles,
            k_values=request.k_values,
            sample_pairs=request.sample_pairs,
            max_patches=request.max_patches,
            masks_dir=masks_dir,
            classes_path=classes_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DptEvalResponse(**result)
