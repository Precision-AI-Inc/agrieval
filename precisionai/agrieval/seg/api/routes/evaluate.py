# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Segmentation evaluation route.

- ``POST /segmentation/evaluate`` — evaluate predicted masks against ground truth
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from precisionai.agrieval.seg.api.config import get_dataset_root
from precisionai.agrieval.seg.schemas.evaluate import (
    SegEvalRequest,
    SegEvalResponse,
)
from precisionai.agrieval.seg.services.evaluate import run_seg_eval

router = APIRouter(prefix="/segmentation", tags=["evaluate"])


def _resolve_dataset_root(dataset_root: str | None) -> str:
    """Use the configured server default when a request omits ``dataset_root``."""
    return dataset_root if dataset_root is not None else get_dataset_root()


@router.post("/evaluate", response_model=SegEvalResponse)
def evaluate(request: SegEvalRequest) -> SegEvalResponse:
    """Evaluate predicted segmentation masks against ground-truth masks.

    Resolves all path arguments against ``dataset_root`` (or the
    ``PAI_DATASET_ROOT`` environment variable when ``dataset_root`` is
    omitted).  Absolute paths are used as-is regardless of ``dataset_root``.

    Returns the full dataset-level and per-image confusion-matrix-derived KPIs.
    Raises HTTP 400 for any input validation error (unknown colours, size
    mismatch, missing ground-truth mask, non-contiguous class IDs, missing or
    unreadable paths, or invalid image files).
    """
    root = Path(_resolve_dataset_root(request.dataset_root))
    pred_dir = root / request.pred_dir
    masks_dir = root / request.masks_dir
    classes_path = root / request.classes_path
    output_dir = (root / request.output_dir) if request.output_dir is not None else None

    if not pred_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"pred_dir does not exist or is not a directory: '{pred_dir}'")
    if not masks_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"masks_dir does not exist or is not a directory: '{masks_dir}'")
    if not classes_path.is_file():
        raise HTTPException(status_code=400, detail=f"classes_path does not exist or is not a file: '{classes_path}'")

    try:
        dataset_summary, image_summary = run_seg_eval(
            pred_dir=pred_dir,
            masks_dir=masks_dir,
            classes_path=classes_path,
            output_dir=output_dir,
            output_summary_name=request.output_summary_name,
            image_summary_name=request.image_summary_name,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"I/O error: {exc}") from exc

    return SegEvalResponse(
        **dataset_summary,
        image_summary=image_summary,
    )
