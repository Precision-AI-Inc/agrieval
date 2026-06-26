# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Embedding evaluation routes.

Three named endpoints cover the full agricultural retrieval taxonomy:

- ``POST /evaluate/image2image`` — Image→Image (full field image vs. full field image)
- ``POST /evaluate/plant2image`` — Plant→Image (instance crop vs. full field image)
- ``POST /evaluate/plant2plant`` — Plant→Plant (instance crop vs. instance crop)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException

from precisionai.agrieval.emb.api.config import get_dataset_root
from precisionai.agrieval.emb.schemas.evaluate import (
    EmbeddingEvaluateRequest,
    EmbeddingEvaluateResponse,
    Plant2ImageRequest,
    Plant2PlantRequest,
)
from precisionai.agrieval.emb.services.evaluate import (
    run_image2image_eval,
    run_plant2image_eval,
    run_plant2plant_eval,
)

router = APIRouter(prefix="/embeddings", tags=["evaluate"])


def _resolve_dataset_root(dataset_root: str | None) -> str | None:
    """Use the configured server default when a request omits dataset_root."""
    return dataset_root if dataset_root is not None else get_dataset_root()


def _run_or_400(func: Callable[..., dict[str, Any]], /, **kwargs: Any) -> EmbeddingEvaluateResponse:
    """Convert service-layer ValueError exceptions into 400 responses."""
    try:
        result = func(**kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EmbeddingEvaluateResponse(**result)


@router.post("/evaluate/image2image", response_model=EmbeddingEvaluateResponse)
def evaluate_image2image(request: EmbeddingEvaluateRequest) -> EmbeddingEvaluateResponse:
    """Evaluate embedding quality for the Image→Image retrieval scenario.

    Given a complete agricultural field image, retrieve visually and
    agronomically similar full field images.  Similarity can be coarse
    (crop type, soil, growth stage) or fine (disease symptoms, weed pressure).

    Pass a dict of ``image_path → embedding`` vectors.  When ``metadata`` is
    omitted the L1 class label is inferred from the folder name of each path
    (e.g. ``images/A1/img.png`` → ``A``).  Supply ``metadata`` to enable
    explicit-positive groups, graded nDCG (0-3 by L2/L1 hierarchy and
    ``plants`` overlap), and per-attribute KPIs.

    **Only ``embeddings`` is required** — all other fields use server defaults.

    Returns a fixed JSON report with:
    - **global_metrics** — pairwise similarity stats, intra/inter class gap,
      KNN label purity, effective rank, centroid similarity, duplicate counts.
    - **per_class** — the same metrics broken down per crop class.
    """
    return _run_or_400(
        run_image2image_eval,
        image_embeddings=request.embeddings,
        k_values=request.k_values,
        dataset_root=_resolve_dataset_root(request.dataset_root),
        sample_pairs=request.sample_pairs,
        metadata=request.metadata,
    )


@router.post("/evaluate/plant2image", response_model=EmbeddingEvaluateResponse)
def evaluate_plant2image(request: Plant2ImageRequest) -> EmbeddingEvaluateResponse:
    """Evaluate embedding quality for the Plant→Image retrieval scenario.

    Pass a mixed corpus of full-field image embeddings and per-plant instance
    crop embeddings together with an ``instance_to_image`` mapping that
    declares which instances were cropped from which parent image.

    The mapping is the explicit-positive ground truth: when an instance is the
    query its parent full image is the grade-3 positive; when a full image is
    the query all its instances are grade-3 positives.  Items that share the
    same class label (inferred from the parent's folder name) are grade-2
    positives across groups.

    Returns the same fixed JSON report as ``POST /v1/embeddings/evaluate``
    with metadata-aware KPIs enabled (``knn_metadata_precision``,
    ``knn_metadata_ndcg``, ``alignment``, etc.).
    """
    return _run_or_400(
        run_plant2image_eval,
        embeddings=request.embeddings,
        instance_to_image=request.instance_to_image,
        k_values=request.k_values,
        dataset_root=_resolve_dataset_root(request.dataset_root),
        sample_pairs=request.sample_pairs,
    )


@router.post("/evaluate/plant2plant", response_model=EmbeddingEvaluateResponse)
def evaluate_plant2plant(request: Plant2PlantRequest) -> EmbeddingEvaluateResponse:
    """Evaluate embedding quality for the Plant→Plant retrieval scenario.

    Pass per-plant instance crop embeddings with an ``instance_labels`` mapping
    that assigns each instance its crop or weed class.  All instances sharing
    the same class label are treated as mutual explicit positives.

    Returns the same fixed JSON report as ``POST /v1/embeddings/evaluate``
    with metadata-aware KPIs enabled.
    """
    return _run_or_400(
        run_plant2plant_eval,
        embeddings=request.embeddings,
        instance_labels=request.instance_labels,
        k_values=request.k_values,
        sample_pairs=request.sample_pairs,
    )
