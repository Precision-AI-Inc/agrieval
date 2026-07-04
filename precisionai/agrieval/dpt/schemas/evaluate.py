# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Request and response schemas for the dense patch token evaluation endpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class DptEvalRequest(BaseModel):
    """Evaluate dense patch token tile quality for a single model.

    **Tile source (exactly one required)**

    ``tiles``
        Inline tile arrays — a dict mapping each tile ID to its feature map,
        shaped ``[C, H, W]`` (channels-first: embedding dimension, patch-grid
        height, patch-grid width). All tiles must share the same ``C``, ``H``,
        and ``W`` — a single evaluation run is always against one
        backbone/tiling configuration. Suitable for small/demo payloads;
        prefer ``tiles_path`` for realistically sized feature maps, which are
        far smaller and faster as binary files than as JSON text.
    ``tiles_path``
        Path to tiles on disk (resolved against ``dataset_root``): either a
        directory containing one ``.npy`` file per tile (file stem = tile ID,
        searched recursively) or a single ``.npz`` archive whose member names
        are the tile IDs. Arrays are loaded with ``allow_pickle=False`` and
        must satisfy the same shape/finiteness rules as inline ``tiles``.

    **Optional (server defaults apply when omitted)**

    ``masks_dir``
        Directory of ground-truth colour-coded masks, one per tile — the
        filename stem must match the tile ID. Same format as
        ``precisionai.agrieval.seg``. Enables label-aware metrics
        (``classes``, ``per_class``, ``knn_confusion``). Requires
        ``classes_path`` to also be set.
    ``classes_path``
        Path to an AgriBench ``class_map.json`` (same format as ``seg``).
        Requires ``masks_dir`` to also be set.
    ``dataset_root``
        Root prefix used to resolve relative ``masks_dir``/``classes_path``.
        Defaults to ``dataset/`` (override at startup with ``--dataset-root``).
    ``k_values``
        K cutoffs for nearest-neighbour label metrics. Default: ``[5, 10, 20]``.
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
            "Inline tile arrays: map of tile_id → channels-first feature map [C, H, W]. "
            "All tiles must share the same C, H, and W. Mutually exclusive with tiles_path."
        ),
    )
    tiles_path: str | None = Field(
        default=None,
        description=(
            "Path to tiles on disk: a directory of .npy files (stem = tile ID) or a single .npz archive "
            "(member name = tile ID). Resolved against dataset_root. Mutually exclusive with tiles."
        ),
    )
    masks_dir: str | None = Field(
        default=None,
        description="Directory of ground-truth colour-coded masks, one per tile (filename stem == tile_id).",
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
        description="K cutoffs for nearest-neighbour label metrics.",
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
        """Validate tile dict: ≥1 tile, safe IDs, rectangular 3-D shape, uniform (C, H, W), all finite."""
        if v is None:
            return v
        if len(v) < 1:
            raise ValueError("At least 1 tile is required.")

        shapes: set[tuple[int, ...]] = set()
        for tile_id, tile in v.items():
            if not tile_id or "/" in tile_id or "\\" in tile_id or "\x00" in tile_id:
                raise ValueError(
                    f"Invalid tile ID {tile_id!r}: must be non-empty and must not contain path separators."
                )
            try:
                # float32 matches the evaluation dtype, so values that only
                # overflow at float32 precision are rejected by the finiteness
                # check below rather than silently becoming inf during metric
                # computation. Pydantic's type layer guarantees exactly three
                # nesting levels, so a successful conversion is always 3-D.
                with np.errstate(over="ignore"):
                    arr = np.asarray(tile, dtype=np.float32)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Tile '{tile_id}' is malformed: nested lists must form a rectangular [C, H, W] array."
                ) from err
            if 0 in arr.shape:
                raise ValueError(f"Tile '{tile_id}' must not be empty along any axis. Got shape {arr.shape}.")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Tile '{tile_id}' contains non-finite values (NaN or inf).")
            shapes.add(arr.shape)

        if len(shapes) > 1:
            raise ValueError(f"All tiles must share the same (C, H, W) shape. Found: {sorted(shapes)}")
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
    def validate_tiles_source(self) -> DptEvalRequest:
        """Validate that exactly one tile source — tiles or tiles_path — is provided."""
        if (self.tiles is None) == (self.tiles_path is None):
            raise ValueError(
                "Provide exactly one of 'tiles' (inline arrays) or 'tiles_path' (.npy directory or .npz archive)."
            )
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
    """Fixed JSON report returned by POST /v1/dense-patch-tokens/evaluate."""

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
    warnings: list[str] | None = None
