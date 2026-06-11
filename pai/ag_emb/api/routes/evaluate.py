# ======================================================================
#  CONFIDENTIAL — © Precision AI 2025. All Rights Reserved.
#
#  This source code and any accompanying documentation contain
#  confidential and proprietary information of Precision AI.
#
#  Unauthorized reproduction, disclosure, modification, or distribution
#  of this material is strictly prohibited and will be prosecuted to the
#  fullest extent of the law.
# ======================================================================

"""Embedding evaluation route.

Ingests image-path→embedding dicts and returns a fixed JSON report broken
down by crop class.
"""

from __future__ import annotations

from fastapi import APIRouter

from pai.ag_emb.api.config import get_dataset_root
from pai.ag_emb.schemas.evaluate import (
    EmbeddingEvaluateRequest,
    EmbeddingEvaluateResponse,
)
from pai.ag_emb.services.evaluate import run_evaluation

router = APIRouter(prefix="/embeddings", tags=["evaluate"])


@router.post("/evaluate", response_model=EmbeddingEvaluateResponse)
def evaluate_embeddings(request: EmbeddingEvaluateRequest) -> EmbeddingEvaluateResponse:
    """Evaluate embedding quality for a single model.

    Pass a dict of ``image_path → embedding`` vectors.  The crop class is
    inferred from each path's folder name using the convention
    ``{root}/{crop}_[{camera}]/img/{image}`` (e.g.
    ``dataset/corn_[HB-25000SBC]/img/220622-img.png``).

    **Only ``embeddings`` is required** — all other fields use server defaults.

    Returns a fixed JSON report with:
    - **global_metrics** — pairwise similarity stats, intra/inter class gap,
      KNN label purity, effective rank, centroid similarity, duplicate counts.
    - **per_class** — the same metrics broken down per crop class.
    """
    dataset_root = request.dataset_root if request.dataset_root is not None else get_dataset_root()
    result = run_evaluation(
        image_embeddings=request.embeddings,
        k_values=request.k_values,
        dataset_root=dataset_root,
        sample_pairs=request.sample_pairs,
    )
    return EmbeddingEvaluateResponse(**result)
