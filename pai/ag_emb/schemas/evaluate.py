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

from pydantic import BaseModel, Field, field_validator, model_validator

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
        # Verify every component is finite, then check L2 normalisation.
        for path, vec in v.items():
            norm = math.sqrt(sum(x * x for x in vec))
            if math.isnan(norm) or math.isinf(norm):
                raise ValueError(
                    f"Embedding for '{path}' contains non-finite values (NaN or inf). "
                    f"All vector components must be finite floats."
                )
            if abs(norm - 1.0) > _NORM_TOLERANCE:
                raise ValueError(
                    f"Embedding for '{path}' is not L2-normalised "
                    f"(‖v‖₂ = {norm:.6f}, expected 1.0 ± {_NORM_TOLERANCE})."
                )
        return v

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, v: list[int]) -> list[int]:
        """Validate that k_values is non-empty and all values are positive integers."""
        if not v:
            raise ValueError("k_values must contain at least one value.")
        if any(k <= 0 for k in v):
            raise ValueError("All K values must be positive.")
        return v

    @model_validator(mode="after")
    def validate_metadata_wiring(self) -> EmbeddingEvaluateRequest:
        """Validate that every metadata image path exists in embeddings and is unique across groups.

        Raises ``ValueError`` if any metadata group references a path not present in
        the ``embeddings`` dict, or if the same path appears in more than one group.
        Both errors would otherwise produce silently incorrect retrieval metrics:
        missing paths inflate ``explicit_positive_ids`` with unreachable entries, and
        duplicate paths cause last-writer-wins label assignment.
        """
        if self.metadata is None:
            return self
        path_set = set(self.embeddings)
        seen: dict[str, str] = {}
        for group_id, group in self.metadata.items():
            for img in group.images:
                if img not in path_set:
                    raise ValueError(
                        f"Metadata group '{group_id}' references image '{img}' which is not found in embeddings."
                    )
                if img in seen:
                    raise ValueError(
                        f"Image '{img}' appears in multiple metadata groups: "
                        f"'{seen[img]}' and '{group_id}'. "
                        f"Each image must belong to exactly one group."
                    )
                seen[img] = group_id
        return self


class Plant2ImageRequest(BaseModel):
    """Evaluate embedding quality for the Plant→Image retrieval scenario.

    Accepts a mixed corpus of full-field image embeddings and per-plant
    instance crop embeddings.  The ``instance_to_image`` mapping declares which
    instances were cropped from which parent full image, defining the
    explicit-positive ground truth used by all retrieval KPIs.

    Instance IDs should follow the convention
    ``{original_image_name}-{ID}{ext}`` (e.g. ``field_001-0.png``), where
    *ID* is a zero-based counter or a short UUID suffix.

    **Required**

    ``embeddings``
        Mapping of path → L2-normalised float32 embedding.  Must include both
        parent full-image paths **and** all instance crop paths listed in
        ``instance_to_image``.

    ``instance_to_image``
        Mapping from parent full-image path to a list of its instance crop
        paths.  Every path referenced here must be an exact key in
        ``embeddings``.  Each instance path must appear in exactly one parent
        group.

    **Optional**

    ``dataset_root``
        Root prefix for extracting the crop class label from parent image
        paths.  Defaults to the longest common directory prefix.
    ``k_values``
        K cutoffs for nearest-neighbour metrics. Default: ``[5, 10, 20]``.
    ``sample_pairs``
        Max random pairs for global pairwise stats. Default: ``1 000 000``.
    """

    embeddings: dict[str, list[float]] = Field(
        description=(
            "Required. Map of image_path → flat L2-normalised float32 embedding. "
            "Must include both parent full-image paths and instance crop paths. "
            "All vectors must share the same length."
        )
    )
    instance_to_image: dict[str, list[str]] = Field(
        description=(
            "Mapping from parent full-image path → list of instance crop paths cropped from it. "
            "Every path must be an exact key in embeddings. "
            "Each instance must belong to exactly one parent."
        )
    )
    dataset_root: str | None = Field(
        default=None,
        description="Dataset root prefix for crop-label extraction from parent image paths. Defaults to common prefix.",
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
        """Validate that k_values is non-empty and all values are positive integers."""
        if not v:
            raise ValueError("k_values must contain at least one value.")
        if any(k <= 0 for k in v):
            raise ValueError("All K values must be positive.")
        return v

    @model_validator(mode="after")
    def validate_wiring_consistency(self) -> Plant2ImageRequest:
        """Validate that all instance_to_image paths exist in embeddings and are non-overlapping."""
        path_set = set(self.embeddings)
        seen_instances: dict[str, str] = {}
        for parent_path, instance_paths in self.instance_to_image.items():
            if parent_path not in path_set:
                raise ValueError(f"Parent image '{parent_path}' from instance_to_image not found in embeddings.")
            for inst in instance_paths:
                if inst not in path_set:
                    raise ValueError(f"Instance '{inst}' of parent '{parent_path}' not found in embeddings.")
                if inst in seen_instances:
                    raise ValueError(
                        f"Instance '{inst}' appears in multiple parent groups: "
                        f"'{seen_instances[inst]}' and '{parent_path}'."
                    )
                seen_instances[inst] = parent_path
        return self


class Plant2PlantRequest(BaseModel):
    """Evaluate embedding quality for the Plant→Plant retrieval scenario.

    Accepts per-plant instance crop embeddings labelled by crop or weed class.
    Every instance must have a class label in ``instance_labels``.  All
    instances sharing the same class label are treated as mutual explicit
    positives (grade 3).

    Instance IDs should follow the convention
    ``{original_image_name}-{ID}{ext}`` (e.g. ``field_001-0.png``), where
    *ID* is a zero-based counter or a short UUID suffix.

    **Required**

    ``embeddings``
        Mapping of instance_path → L2-normalised float32 embedding.

    ``instance_labels``
        Mapping from each instance path to its crop/weed class label
        (e.g. ``{"field_001-0.png": "corn"}``).  Every key must be an exact
        key in ``embeddings`` and every embedding key must have a label.

    **Optional**

    ``k_values``
        K cutoffs for nearest-neighbour metrics. Default: ``[5, 10, 20]``.
    ``sample_pairs``
        Max random pairs for global pairwise stats. Default: ``1 000 000``.
    """

    embeddings: dict[str, list[float]] = Field(
        description=(
            "Required. Map of instance_path → flat L2-normalised float32 embedding. "
            "Every key must appear in instance_labels. All vectors must share the same length."
        )
    )
    instance_labels: dict[str, str] = Field(
        description=(
            "Mapping from instance path to its crop/weed class label. "
            "Every key must be present in embeddings, and every embedding must have a label."
        )
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
        """Validate that k_values is non-empty and all values are positive integers."""
        if not v:
            raise ValueError("k_values must contain at least one value.")
        if any(k <= 0 for k in v):
            raise ValueError("All K values must be positive.")
        return v

    @model_validator(mode="after")
    def validate_wiring_consistency(self) -> Plant2PlantRequest:
        """Validate label/embedding consistency."""
        path_set = set(self.embeddings)
        label_set = set(self.instance_labels)

        missing_labels = label_set - path_set
        if missing_labels:
            raise ValueError(f"instance_labels keys not found in embeddings: {sorted(missing_labels)[:5]}")
        unlabeled = path_set - label_set
        if unlabeled:
            raise ValueError(f"Embeddings without a label in instance_labels: {sorted(unlabeled)[:5]}")

        return self


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
