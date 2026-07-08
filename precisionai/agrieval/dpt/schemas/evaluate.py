# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Request and response schemas for the dense patch token evaluation endpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


def _validate_dense_entries(v: dict[str, list], *, kind: str) -> dict[str, list]:
    """Shared body for the ``tiles``/``images`` field validators.

    Both wirings share the exact same ``[P, H, W]`` contract — every entry
    must be non-empty, safely-IDed, rectangular, finite, and the whole batch
    must share one ``(P, H, W)`` shape, since every image (whole or tiled)
    is processed at the backbone's fixed native input resolution. Only the
    wording of error messages (via ``kind``) differs between the two
    wirings.

    Parameters
    ----------
    v : dict[str, list]
        Map of entry ID to a nested ``[P, H, W]`` list.
    kind : str
        Either ``"tile"`` or ``"image"`` — used to word error messages
        appropriately for the calling wiring.

    Returns
    -------
    dict[str, list]
        The same mapping, unchanged (validation only).
    """
    if len(v) < 1:
        raise ValueError(f"At least 1 {kind} is required.")

    shapes: set[tuple[int, ...]] = set()
    for entry_id, entry in v.items():
        if not entry_id or "/" in entry_id or "\\" in entry_id or "\x00" in entry_id:
            raise ValueError(f"Invalid {kind} ID {entry_id!r}: must be non-empty and must not contain path separators.")
        try:
            # float32 matches the evaluation dtype, so values that only
            # overflow at float32 precision are rejected by the finiteness
            # check below rather than silently becoming inf during metric
            # computation. Pydantic's type layer guarantees exactly three
            # nesting levels, so a successful conversion is always 3-D.
            with np.errstate(over="ignore"):
                arr = np.asarray(entry, dtype=np.float32)
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"{kind.capitalize()} '{entry_id}' is malformed: nested lists must form a rectangular [P, H, W] array."
            ) from err
        if 0 in arr.shape:
            raise ValueError(
                f"{kind.capitalize()} '{entry_id}' must not be empty along any axis. Got shape {arr.shape}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{kind.capitalize()} '{entry_id}' contains non-finite values (NaN or inf).")
        shapes.add(arr.shape)

    if len(shapes) > 1:
        raise ValueError(f"All {kind}s must share the same (P, H, W) shape. Found: {sorted(shapes)}")
    return v


class DptEvalRequest(BaseModel):
    """Evaluate dense patch token tile quality for a single model.

    **Tile source (exactly one required)**

    ``tiles``
        Inline tile arrays — a dict mapping each tile ID to its feature map,
        shaped ``[P, H, W]`` (patch-first: embedding dimension, patch-grid
        height, patch-grid width). All tiles must share the same ``P``, ``H``,
        and ``W`` — a single evaluation run is always against one
        backbone/tiling configuration. Suitable for small/demo payloads;
        prefer ``tiles_path`` for realistically sized feature maps, which are
        far smaller and faster as binary files than as JSON text.
    ``tiles_path``
        Path to a batched ``.npz`` archive on disk (resolved against
        ``dataset_root``) — see :func:`precisionai.agrieval.dpt.services.evaluate.load_tiles`
        for the exact schema (``feature_maps``/``tile_image_id``/``tile_index``/
        ``tile_y0``/``tile_x0``/``filenames``). Arrays are loaded with
        ``allow_pickle=False``.

    **Optional (server defaults apply when omitted)**

    ``masks_dir``
        Directory of ground-truth color-coded masks. When tiles come from
        ``tiles_path``, one mask per *source image* (filename stem must
        match each tile's source-image stem), cropped to each tile's exact
        pixel rectangle (``tile_y0``/``tile_x0`` + pixel size from
        ``meta.tile``) before scoring. When tiles are inline, one mask per
        tile (filename stem must match the tile ID). Same format as
        ``precisionai.agrieval.seg``. Enables label-aware metrics
        (``classes``, ``per_class``, ``knn_confusion``, ``separation``).
        Requires ``classes_path`` to also be set.
    ``classes_path``
        Path to an AgriBench ``class_map.json`` (same format as ``seg``).
        Requires ``masks_dir`` to also be set.
    ``dataset_root``
        Root prefix used to resolve relative ``masks_dir``/``classes_path``.
        Defaults to ``dataset/`` (override at startup with ``--dataset-root``).
    ``k_values``
        K cutoffs for nearest-neighbor label metrics. Default: ``[5, 10, 20]``.
    ``sample_pairs``
        Max random pairs for global pairwise similarity stats. Default: ``1 000 000``.
    ``max_patches``
        Max patches used for O(N²) label-aware kNN computation; larger
        corpora are randomly subsampled (seeded) and the drop is reported in
        ``warnings``. Default: ``20 000``.
    """

    tiles: dict[str, list[list[list[float]]]] | None = Field(
        default=None,
        description=(
            "Inline tile arrays: map of tile_id → patch-first feature map [P, H, W]. "
            "All tiles must share the same P, H, and W. Mutually exclusive with tiles_path."
        ),
    )
    tiles_path: str | None = Field(
        default=None,
        description=(
            "Path to a batched .npz archive on disk "
            "(feature_maps/tile_image_id/tile_index/tile_y0/tile_x0/filenames). "
            "Resolved against dataset_root. Mutually exclusive with tiles."
        ),
    )
    masks_dir: str | None = Field(
        default=None,
        description="Directory of ground-truth color-coded masks, one per tile (filename stem == tile_id).",
    )
    classes_path: str | None = Field(
        default=None,
        description="Path to an AgriBench class_map.json, same format as seg.",
    )
    dataset_root: str | None = Field(
        default=None,
        description="Root prefix for resolving relative masks_dir/classes_path. Defaults to dataset/.",
    )
    k_values: list[int] = Field(
        default=[5, 10, 20],
        description="K cutoffs for nearest-neighbor label metrics.",
    )
    sample_pairs: int | None = Field(
        default=1_000_000,
        description="Max random pairs for global pairwise stats. None = exact (slow for large N).",
        ge=0,
    )
    max_patches: int = Field(
        default=20_000,
        description="Max patches used for O(N²) label-aware kNN computation; larger corpora are subsampled.",
        ge=2,
    )

    @field_validator("tiles")
    @classmethod
    def validate_tiles(cls, v: dict[str, list[list[list[float]]]] | None) -> dict[str, list[list[list[float]]]] | None:
        """Validate tile dict: ≥1 tile, safe IDs, rectangular 3-D shape, uniform (P, H, W), all finite."""
        if v is None:
            return v
        return _validate_dense_entries(v, kind="tile")

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
    def validate_tiles_source(self) -> DptEvalRequest:
        """Validate that exactly one tile source — tiles or tiles_path — is provided."""
        if (self.tiles is None) == (self.tiles_path is None):
            raise ValueError("Provide exactly one of 'tiles' (inline arrays) or 'tiles_path' (batched .npz archive).")
        return self

    @model_validator(mode="after")
    def validate_label_wiring(self) -> DptEvalRequest:
        """Validate that masks_dir and classes_path are both set or both omitted."""
        has_masks = self.masks_dir is not None
        has_classes = self.classes_path is not None
        if has_masks != has_classes:
            raise ValueError(
                "masks_dir and classes_path must both be set to enable label-aware metrics, or both omitted."
            )
        return self


class DptEvalResponse(BaseModel):
    """Fixed JSON report returned by POST /v1/dense-patch-tokens/evaluate/tiles."""

    n_tiles: int
    embed_dim: int
    grid_height: int
    grid_width: int
    n_patches: int
    tile_ids: list[str]
    k_values: list[int]
    global_metrics: dict[str, Any]
    per_tile: dict[str, dict[str, Any]]
    classes: list[str] | None = None
    per_class: dict[str, dict[str, Any]] | None = None
    knn_confusion: dict[str, Any] | None = None
    separation: dict[str, Any] | None = None
    warnings: list[str] | None = None


class DptImageEvalRequest(BaseModel):
    """Evaluate dense patch token quality for a batch of whole-image feature maps.

    Structurally identical to :class:`DptEvalRequest` — every entry must
    share the same ``[P, H, W]`` shape, since every image (whether tiled or
    evaluated whole) is processed at the backbone's fixed native input
    resolution before dense features are extracted, so a "whole image" and
    a "tile" share the exact same fixed-grid contract. Use this wiring when
    each entry represents one full (untiled) image rather than an arbitrary
    tile crop — the difference from
    :class:`DptEvalRequest` is field naming and response shape (``images``
    instead of ``tiles``, ``per_image`` instead of ``per_tile``, etc.) to
    match how results are keyed and interpreted, not the underlying rules.

    **Image source (exactly one required)**

    ``images``
        Inline whole-image arrays — a dict mapping each image ID to its
        feature map, shaped ``[P, H, W]``. All images must share the same
        ``P``, ``H``, and ``W``. Suitable for small/demo payloads; prefer
        ``images_path`` for realistically sized feature maps.
    ``images_path``
        Path to a batched ``.npz`` archive on disk (resolved against
        ``dataset_root``) — see :func:`precisionai.agrieval.dpt.services.evaluate.load_images`
        for the exact schema (``features``/``filenames``). Arrays are loaded
        with ``allow_pickle=False``.

    **Optional (server defaults apply when omitted)**

    ``masks_dir``
        Directory of ground-truth color-coded masks, one per image — the
        filename stem must match the image ID. Same format as
        ``precisionai.agrieval.seg``. Enables label-aware metrics
        (``classes``, ``per_class``, ``knn_confusion``, ``separation``).
        Requires ``classes_path`` to also be set.
    ``classes_path``
        Path to an AgriBench ``class_map.json`` (same format as ``seg``).
        Requires ``masks_dir`` to also be set.
    ``dataset_root``
        Root prefix used to resolve relative ``masks_dir``/``classes_path``.
        Defaults to ``dataset/`` (override at startup with ``--dataset-root``).
    ``k_values``
        K cutoffs for nearest-neighbor label metrics. Default: ``[5, 10, 20]``.
    ``sample_pairs``
        Max random pairs for global pairwise similarity stats. Default: ``1 000 000``.
    ``max_patches``
        Max patches used for O(N²) label-aware kNN computation; larger
        corpora are randomly subsampled (seeded) and the drop is reported in
        ``warnings``. Default: ``20 000``.
    """

    images: dict[str, list[list[list[float]]]] | None = Field(
        default=None,
        description=(
            "Inline whole-image arrays: map of image_id → patch-first feature map [P, H, W]. "
            "All images must share the same P, H, and W. Mutually exclusive with images_path."
        ),
    )
    images_path: str | None = Field(
        default=None,
        description=(
            "Path to a batched .npz archive on disk (features/filenames). "
            "Resolved against dataset_root. Mutually exclusive with images."
        ),
    )
    masks_dir: str | None = Field(
        default=None,
        description="Directory of ground-truth color-coded masks, one per image (filename stem == image_id).",
    )
    classes_path: str | None = Field(
        default=None,
        description="Path to an AgriBench class_map.json, same format as seg.",
    )
    dataset_root: str | None = Field(
        default=None,
        description="Root prefix for resolving relative masks_dir/classes_path. Defaults to dataset/.",
    )
    k_values: list[int] = Field(
        default=[5, 10, 20],
        description="K cutoffs for nearest-neighbor label metrics.",
    )
    sample_pairs: int | None = Field(
        default=1_000_000,
        description="Max random pairs for global pairwise stats. None = exact (slow for large N).",
        ge=0,
    )
    max_patches: int = Field(
        default=20_000,
        description="Max patches used for O(N²) label-aware kNN computation; larger corpora are subsampled.",
        ge=2,
    )

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: dict[str, list[list[list[float]]]] | None) -> dict[str, list[list[list[float]]]] | None:
        """Validate image dict: ≥1 image, safe IDs, rectangular 3-D shape, uniform (P, H, W), all finite."""
        if v is None:
            return v
        return _validate_dense_entries(v, kind="image")

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
    def validate_images_source(self) -> DptImageEvalRequest:
        """Validate that exactly one image source — images or images_path — is provided."""
        if (self.images is None) == (self.images_path is None):
            raise ValueError("Provide exactly one of 'images' (inline arrays) or 'images_path' (batched .npz archive).")
        return self

    @model_validator(mode="after")
    def validate_label_wiring(self) -> DptImageEvalRequest:
        """Validate that masks_dir and classes_path are both set or both omitted."""
        has_masks = self.masks_dir is not None
        has_classes = self.classes_path is not None
        if has_masks != has_classes:
            raise ValueError(
                "masks_dir and classes_path must both be set to enable label-aware metrics, or both omitted."
            )
        return self


class DptImageEvalResponse(BaseModel):
    """Fixed JSON report returned by POST /v1/dense-patch-tokens/evaluate/image."""

    n_images: int
    embed_dim: int
    grid_height: int
    grid_width: int
    n_patches: int
    image_ids: list[str]
    k_values: list[int]
    global_metrics: dict[str, Any]
    per_image: dict[str, dict[str, Any]]
    classes: list[str] | None = None
    per_class: dict[str, dict[str, Any]] | None = None
    knn_confusion: dict[str, Any] | None = None
    separation: dict[str, Any] | None = None
    warnings: list[str] | None = None
