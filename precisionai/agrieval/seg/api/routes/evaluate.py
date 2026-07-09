# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Segmentation evaluation route.

- ``POST /segmentation/evaluate`` — evaluate predicted masks against ground truth
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from precisionai.agrieval.api.paths import resolve_under_root
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
    omitted).  The resolved path must stay within ``dataset_root`` — an
    absolute path or a ``..`` segment that would escape it is rejected with
    HTTP 400, regardless of how ``dataset_root`` itself was supplied.

    Returns the full dataset-level and per-image confusion-matrix-derived KPIs.
    Raises HTTP 400 for any input validation error (unknown colors, size
    mismatch, missing ground-truth mask, non-contiguous class IDs, missing or
    unreadable paths, or invalid image files).
    """
    root = Path(_resolve_dataset_root(request.dataset_root))
    try:
        pred_dir = resolve_under_root(root, request.pred_dir, field_name="pred_dir")
        masks_dir = resolve_under_root(root, request.masks_dir, field_name="masks_dir")
        classes_path = resolve_under_root(root, request.classes_path, field_name="classes_path")
        output_dir = (
            resolve_under_root(root, request.output_dir, field_name="output_dir")
            if request.output_dir is not None
            else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
            show_progress=False,
            num_workers=request.num_workers,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"I/O error: {exc}") from exc

    return SegEvalResponse(
        **dataset_summary,
        image_summary=image_summary,
    )
