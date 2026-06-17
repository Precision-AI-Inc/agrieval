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

"""Request and response schemas for the embedding evaluation endpoint."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Acceptable deviation from unit norm for L2-normalised inputs.
# float32 round-trip through JSON can introduce ~1e-5 error; 1e-3 is generous.
_NORM_TOLERANCE = 1e-3


class MetadataGroup(BaseModel):
    """One group of semantically similar images sharing a class label and optional attributes.

    Parameters
    ----------
    images : list[str]
        Exact path keys of the images in this group.  Each entry must match
        a key in the ``embeddings`` dict exactly — not a basename, the full path.
    class_name : str
        Class label shared by all images in this group (e.g. ``"barley"``).
        Required — used as the label for all label-aware metrics.
    attributes : dict[str, str]
        Optional key-value attributes shared by all images in this group.
        Any string keys are accepted, e.g. ``{"growth_stage": "medium",
        "camera": "anafi", "sunlight": "bright"}``.  Used by the graded
        relevance function: two items score grade 2 when they share the same
        ``class_name`` and ALL of the query's attribute values match.
    """

    images: list[str] = Field(
        description="Exact path keys (must match keys used in `embeddings`) belonging to this group."
    )
    class_name: str = Field(description="Class label for this group. Required.")
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Optional key-value attributes (e.g. growth_stage, camera, sunlight).",
    )


class EmbeddingEvaluateRequest(BaseModel):
    """Evaluate embedding quality for a single model.

    **Required**

    ``embeddings``
        The only required field — a dict mapping each image path to its
        embedding vector.  Paths must follow the dataset convention
        ``{root}/{crop}_[{camera}]/img/{image}`` (e.g.
        ``dataset/corn_[HB-25000SBC]/img/220622-img.png``).  The crop class
        is extracted automatically from the folder name.

        Each vector must be a **flat, L2-normalised float32 array** of length
        ``embedding_dim``.  All vectors must share the same length.

    **Optional (server defaults apply when omitted)**

    ``metadata``
        Explicit similarity groups.  When supplied, class labels are taken
        from ``class_name`` here instead of from the path hierarchy.  Each
        key is an arbitrary group ID; each value is a
        :class:`MetadataGroup` listing the exact embedding path keys (not
        basenames) in that group together with their shared ``class_name``
        and optional ``attributes`` (e.g. ``growth_stage``, ``camera``,
        ``sunlight``).
        Images that belong to the same group are treated as explicit
        positives (relevance grade 3) in the metadata-aware nDCG metric.
    ``dataset_root``
        Root prefix to strip before extracting the crop label (ignored when
        ``metadata`` is supplied).
        Defaults to ``dataset/`` (override at startup with ``--dataset-root``).
    ``k_values``
        K cutoffs for nearest-neighbour metrics. Default: ``[5, 10, 20]``.
    ``sample_pairs``
        Max random pairs for global pairwise stats. Default: ``1 000 000``.
    """

    embeddings: dict[str, list[float]] = Field(
        description=(
            "Required. Map of image_path → flat L2-normalised float32 embedding. "
            "Paths follow {root}/{crop}_[{camera}]/img/{image}. "
            "All vectors must share the same length."
        )
    )
    metadata: dict[str, MetadataGroup] | None = Field(
        default=None,
        description=(
            "Optional similarity groups. Keys are arbitrary group IDs; values describe "
            "images in that group with their class_name and optional attributes dict "
            "(e.g. growth_stage, camera, sunlight). "
            "When provided, enables metadata-aware graded nDCG and uses class_name as the "
            "class label for all other metrics."
        ),
    )
    dataset_root: str | None = Field(
        default=None,
        description="Dataset root prefix for crop-label extraction. Defaults to dataset/.",
    )
    k_values: list[int] = Field(
        default=[5, 10, 20],
        description="K cutoffs for nearest-neighbour metrics.",
    )
    sample_pairs: int | None = Field(
        default=1_000_000,
        description="Max random pairs for global pairwise stats. None = exact (slow for large N).",
    )

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        """Validate embedding dict: ≥2 items, uniform dim, non-empty, L2-normalised."""
        if len(v) < 2:
            raise ValueError("At least 2 embeddings are required.")
        dims = {len(vec) for vec in v.values()}
        if len(dims) > 1:
            raise ValueError(f"All embeddings must have the same dimension. Found: {sorted(dims)}")
        dim = next(iter(dims))
        if dim == 0:
            raise ValueError("Embedding vectors must not be empty.")
        # Verify L2 normalisation — compute norm of each vector and check ≈ 1.
        for path, vec in v.items():
            norm = math.sqrt(sum(x * x for x in vec))
            if abs(norm - 1.0) > _NORM_TOLERANCE:
                raise ValueError(
                    f"Embedding for '{path}' is not L2-normalised "
                    f"(‖v‖₂ = {norm:.6f}, expected 1.0 ± {_NORM_TOLERANCE})."
                )
        return v

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, v: list[int]) -> list[int]:
        """Validate that all K values are positive integers."""
        if any(k <= 0 for k in v):
            raise ValueError("All K values must be positive.")
        return v


class EmbeddingEvaluateResponse(BaseModel):
    """Fixed JSON report returned by POST /v1/embeddings/evaluate."""

    n_items: int
    embedding_dim: int
    classes: list[str]
    k_values: list[int]
    item_paths: list[str]
    item_labels: list[str]
    knn_confusion: dict[str, Any]
    global_metrics: dict[str, Any]
    per_class: dict[str, dict[str, Any]]
    group_analysis: dict[str, Any] | None = None
