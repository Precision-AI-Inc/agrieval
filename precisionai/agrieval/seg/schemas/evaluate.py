# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Request and response schemas for the segmentation evaluation endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SegEvalRequest(BaseModel):
    """Request body for ``POST /v1/segmentation/evaluate``.

    Parameters
    ----------
    pred_dir : str
        Path to the directory of predicted color-coded masks.  Relative paths
        are resolved against ``dataset_root``; absolute paths are used as-is.
    masks_dir : str
        Path to the directory of ground-truth color-coded masks.
    classes_path : str
        Path to the AgriStress class-definition JSON file.
    output_dir : str | None
        Directory to write ``output_summary.json`` and ``image_summary.json``.
        ``None`` (the default) skips file output.
    output_summary_name : str
        Filename for the dataset-level summary.  Default: ``output_summary.json``.
    image_summary_name : str
        Filename for the per-image summary.  Default: ``image_summary.json``.
    num_workers : int | None
        Number of worker threads for mask-pair processing.  ``None`` selects
        an automatic default.
    dataset_root : str | None
        Base directory prepended to relative ``pred_dir``, ``masks_dir``,
        ``classes_path``, and ``output_dir`` values.  Falls back to the
        ``PAI_DATASET_ROOT`` environment variable, then ``"dataset"``.
    """

    pred_dir: str = Field(description="Directory of predicted color-coded masks.")
    masks_dir: str = Field(description="Directory of ground-truth color-coded masks.")
    classes_path: str = Field(description="Path to the AgriStress class-definition JSON.")
    output_dir: str | None = Field(
        default=None, description="Output directory for JSON results. None skips file output."
    )
    output_summary_name: str = Field(
        default="output_summary.json", description="Filename for the dataset-level summary."
    )
    image_summary_name: str = Field(default="image_summary.json", description="Filename for the per-image summary.")
    num_workers: int | None = Field(
        default=None,
        ge=1,
        description="Number of worker threads. None selects an automatic default, up to 4.",
    )
    dataset_root: str | None = Field(default=None, description="Base path prepended to relative directory arguments.")


class ClassMetrics(BaseModel):
    """Per-class segmentation metrics.

    Parameters
    ----------
    iou : float | None
        Intersection over Union.  ``null`` when the class is absent from both
        prediction and ground truth.
    dice : float | None
        Dice coefficient / F1 score.  ``null`` on the same condition as ``iou``.
    accuracy : float | None
        Per-class recall.  ``null`` when the class is absent from ground truth.
    """

    iou: float | None
    dice: float | None
    accuracy: float | None


class SummaryMetrics(BaseModel):
    """Dataset- or image-level aggregate metrics.

    Parameters
    ----------
    mIoU : float | None
        Macro-average IoU over valid classes.
    mAcc : float | None
        Macro-average per-class accuracy over valid classes.
    FWIoU : float | None
        Frequency-weighted IoU.
    """

    mIoU: float | None
    mAcc: float | None
    FWIoU: float | None


class ImageResult(BaseModel):
    """Per-image evaluation result.

    Parameters
    ----------
    classes : dict[str, ClassMetrics]
        Per-class metrics keyed by class name.
    summary : SummaryMetrics
        Aggregate metrics for this image.
    """

    classes: dict[str, ClassMetrics]
    summary: SummaryMetrics


class SegEvalResponse(BaseModel):
    """Response body for ``POST /v1/segmentation/evaluate``.

    Parameters
    ----------
    n_images : int
        Number of image pairs evaluated.
    classes : dict[str, ClassMetrics]
        Dataset-level per-class metrics keyed by class name.
    summary : SummaryMetrics
        Dataset-level aggregate metrics computed from the accumulated confusion
        matrix — not a mean of per-image metrics.
    image_summary : dict[str, ImageResult]
        Per-image results.  Keys are predicted mask paths relative to
        ``pred_dir``, using forward slashes on all platforms.
    """

    n_images: int
    classes: dict[str, ClassMetrics]
    summary: SummaryMetrics
    image_summary: dict[str, ImageResult]
