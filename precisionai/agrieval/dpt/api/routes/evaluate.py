# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Dense patch token evaluation routes.

- ``POST /dense-patch-tokens/evaluate/tiles`` — evaluate many tile feature
  maps (e.g. crops of one or more larger images), optionally against
  per-tile ground-truth masks.
- ``POST /dense-patch-tokens/evaluate/image`` — evaluate one feature map per
  whole (untiled) image, optionally against per-image ground-truth masks.
  Identical rules and metrics to the tiles wiring; only the field naming and
  response shape differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from precisionai.agrieval.dpt.api.config import get_dataset_root
from precisionai.agrieval.dpt.schemas.evaluate import (
    DptEvalRequest,
    DptEvalResponse,
    DptImageEvalRequest,
    DptImageEvalResponse,
)
from precisionai.agrieval.dpt.services.evaluate import (
    TilePlacement,
    load_images,
    load_tile_placement,
    load_tiles,
    run_dpt_eval,
    run_dpt_image_eval,
)

router = APIRouter(prefix="/dense-patch-tokens", tags=["evaluate"])


def _resolve_dataset_root(dataset_root: str | None) -> str:
    """Use the configured server default when a request omits ``dataset_root``."""
    return dataset_root if dataset_root is not None else get_dataset_root()


def _resolve_label_paths(
    masks_dir: str | None, classes_path: str | None, dataset_root: str | None
) -> tuple[Path | None, Path | None]:
    """Resolve masks_dir/classes_path against dataset_root and validate they exist.

    Returns ``(None, None)`` when either field is omitted — label-aware
    metrics are simply skipped in that case (the schema already guarantees
    both-or-neither are set).
    """
    if masks_dir is None or classes_path is None:
        return None, None
    root = Path(_resolve_dataset_root(dataset_root))
    resolved_masks_dir = root / masks_dir
    resolved_classes_path = root / classes_path
    if not resolved_masks_dir.is_dir():
        raise HTTPException(
            status_code=400, detail=f"masks_dir does not exist or is not a directory: '{resolved_masks_dir}'"
        )
    if not resolved_classes_path.is_file():
        raise HTTPException(
            status_code=400, detail=f"classes_path does not exist or is not a file: '{resolved_classes_path}'"
        )
    return resolved_masks_dir, resolved_classes_path


@router.post("/evaluate/tiles", response_model=DptEvalResponse)
def evaluate_tiles(request: DptEvalRequest) -> DptEvalResponse:
    """Evaluate dense patch token tile quality, with optional label-aware metrics.

    Supply tiles either inline (``tiles``: dict of ``tile_id -> [P, H, W]``
    feature maps) or from disk (``tiles_path``: a batched ``.npz`` archive,
    resolved against ``dataset_root`` — see :func:`load_tiles`) — exactly
    one of the two. Always returns unsupervised geometry and per-tile
    diagnostics. When ``masks_dir`` and ``classes_path`` are also supplied
    (resolved against ``dataset_root``, same convention as
    ``/v1/segmentation/evaluate``), the response also includes ``classes``,
    ``per_class``, and ``knn_confusion``. When tiles come from ``tiles_path``,
    ground truth is one mask per *source image* (cropped to each tile's
    exact pixel rectangle); when tiles are inline, ground truth is one mask
    per tile (matched by tile ID).

    Use ``POST /v1/dense-patch-tokens/evaluate/image`` instead when each
    entry represents one whole (untiled) image rather than an arbitrary
    tile crop.

    All other fields use server defaults.
    """
    tile_placement: dict[str, TilePlacement] | None = None
    if request.tiles_path is not None:
        tiles_path = Path(_resolve_dataset_root(request.dataset_root)) / request.tiles_path
        try:
            tiles: dict[str, Any] = load_tiles(tiles_path)
            # Placement (and therefore meta.tile) is only needed to crop
            # ground-truth masks — unsupervised runs work without it.
            if request.masks_dir is not None:
                tile_placement = load_tile_placement(tiles_path)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        tiles = request.tiles or {}

    masks_dir, classes_path = _resolve_label_paths(request.masks_dir, request.classes_path, request.dataset_root)

    try:
        result = run_dpt_eval(
            tiles=tiles,
            k_values=request.k_values,
            sample_pairs=request.sample_pairs,
            max_patches=request.max_patches,
            masks_dir=masks_dir,
            classes_path=classes_path,
            tile_placement=tile_placement,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DptEvalResponse(**result)


@router.post("/evaluate/image", response_model=DptImageEvalResponse)
def evaluate_image(request: DptImageEvalRequest) -> DptImageEvalResponse:
    """Evaluate dense patch token quality for whole-image feature maps, with optional label-aware metrics.

    Identical rules and metrics to ``POST /v1/dense-patch-tokens/evaluate/tiles``
    — every entry must share the same ``[P, H, W]`` shape — for callers whose
    entries each represent one whole (untiled) image rather than an
    arbitrary tile crop. Supply images either inline (``images``: dict of
    ``image_id -> [P, H, W]`` feature maps) or from disk (``images_path``: a
    batched ``.npz`` archive, resolved against ``dataset_root`` — see
    :func:`load_images`) — exactly one of the two. When ``masks_dir`` and
    ``classes_path`` are also supplied, the ground-truth mask for each image
    is downsampled to that image's own patch grid.

    All other fields use server defaults.
    """
    if request.images_path is not None:
        images_path = Path(_resolve_dataset_root(request.dataset_root)) / request.images_path
        try:
            images: dict[str, Any] = load_images(images_path)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        images = request.images or {}

    masks_dir, classes_path = _resolve_label_paths(request.masks_dir, request.classes_path, request.dataset_root)

    try:
        result = run_dpt_image_eval(
            images=images,
            k_values=request.k_values,
            sample_pairs=request.sample_pairs,
            max_patches=request.max_patches,
            masks_dir=masks_dir,
            classes_path=classes_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DptImageEvalResponse(**result)
