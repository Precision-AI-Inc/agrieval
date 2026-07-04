# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Service layer: orchestrates embedding metrics into the fixed analysis JSON.

Label extraction and metadata wiring live in
:mod:`~precisionai.agrieval.emb.services.labels`, the KNN metric bundles and
result assembly in :mod:`~precisionai.agrieval.emb.services.knn_metrics`, and
JSON conversion in :mod:`~precisionai.agrieval.emb.services.serialization`.
This module owns input validation, advisory warnings, and the three
evaluation entry points.
"""

from __future__ import annotations

import gc
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from precisionai.agrieval.emb.metrics import knn_confusion_matrix, top_k_neighbors
from precisionai.agrieval.emb.schemas.evaluate import MetadataGroup
from precisionai.agrieval.emb.services.knn_metrics import (
    _assemble_global_metrics,
    _build_per_class_metrics,
    _compute_group_analysis,
    _compute_label_knn_metrics,
    _compute_metadata_knn_metrics,
    _compute_neighbor_diagnostics,
)
from precisionai.agrieval.emb.services.labels import (
    _labels_from_metadata,
    extract_labels,
    plant2image_to_metadata,
    plant2plant_to_metadata,
)
from precisionai.agrieval.emb.services.serialization import _jsonify

# Supported embedding path extensions — case-sensitive exact match.
_SUPPORTED_EXTENSIONS = frozenset({".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"})

# Advisory thresholds for local runs. Users can override these for larger machines.
_WARN_ITEMS_ENV = "PAI_EMB_WARN_ITEMS"
_WARN_COMPONENTS_ENV = "PAI_EMB_WARN_COMPONENTS"
_WARN_METADATA_CELLS_ENV = "PAI_EMB_WARN_METADATA_CELLS"
_WARN_EXACT_PAIRS_ENV = "PAI_EMB_WARN_EXACT_PAIRS"
_DEFAULT_WARN_ITEMS = 10_000
_DEFAULT_WARN_COMPONENTS = 20_000_000
_DEFAULT_WARN_METADATA_CELLS = 25_000_000
_DEFAULT_WARN_EXACT_PAIRS = 10_000_000


# ---------------------------------------------------------------------------
# Validation and advisory warnings
# ---------------------------------------------------------------------------


def _validate_embeddings(paths: list[str], vectors: list[list[float]]) -> None:
    """Raise ``ValueError`` if the embeddings are not usable for evaluation.

    Checks minimum count, consistent dimension, non-empty vectors, and that
    every path ends with a supported extension.  Extensions are matched
    case-sensitively — ``.JPG`` and ``.jpg`` are both valid but ``.Jpg`` is not.

    Supported extensions: ``.jpg``, ``.JPG``, ``.jpeg``, ``.JPEG``, ``.png``, ``.PNG``.

    Parameters
    ----------
    paths : list[str]
        Image path keys from the embeddings dict.
    vectors : list[list[float]]
        Embedding vectors corresponding to ``paths``.
    """
    if len(paths) < 2:
        raise ValueError("At least 2 embeddings are required.")
    dims = {len(v) for v in vectors}
    if next(iter(dims)) == 0:
        raise ValueError("Embedding vectors must not be empty.")
    if len(dims) > 1:
        raise ValueError(f"All embeddings must have the same dimension. Found: {sorted(dims)}")
    bad = [p for p in paths if Path(p).suffix not in _SUPPORTED_EXTENSIONS]
    if bad:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file extension(s) in embedding paths (supported: {supported}): {bad}")


def _warn_threshold(env_name: str, default: int) -> int:
    """Read an integer advisory threshold from the environment."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, 0)


def _build_evaluation_warnings(
    *,
    n_items: int,
    embedding_dim: int,
    metadata: dict[str, MetadataGroup] | None,
    sample_pairs: int | None,
) -> list[str]:
    """Build non-blocking warnings for expensive local evaluations."""
    messages: list[str] = []
    item_threshold = _warn_threshold(_WARN_ITEMS_ENV, _DEFAULT_WARN_ITEMS)
    component_threshold = _warn_threshold(_WARN_COMPONENTS_ENV, _DEFAULT_WARN_COMPONENTS)
    metadata_cell_threshold = _warn_threshold(_WARN_METADATA_CELLS_ENV, _DEFAULT_WARN_METADATA_CELLS)
    exact_pair_threshold = _warn_threshold(_WARN_EXACT_PAIRS_ENV, _DEFAULT_WARN_EXACT_PAIRS)

    if item_threshold and n_items >= item_threshold:
        messages.append(
            f"Large evaluation: {n_items} embeddings were provided. Runtime scales roughly with the number of items."
        )

    total_components = n_items * embedding_dim
    if component_threshold and total_components >= component_threshold:
        approx_mb = total_components * np.dtype(np.float32).itemsize / (1024 * 1024)
        messages.append(
            f"Large embedding matrix: {n_items} x {embedding_dim} contains {total_components:,} values "
            f"(about {approx_mb:.1f} MiB as float32)."
        )

    if metadata is not None:
        metadata_cells = n_items * n_items
        if metadata_cell_threshold and metadata_cells >= metadata_cell_threshold:
            approx_mb = metadata_cells * 2 / (1024 * 1024)
            messages.append(
                f"Metadata-aware metrics build dense {n_items} x {n_items} relevance matrices "
                f"(at least about {approx_mb:.1f} MiB before temporary arrays)."
            )

    if sample_pairs is None:
        exact_pairs = n_items * (n_items - 1) // 2
        if exact_pair_threshold and exact_pairs >= exact_pair_threshold:
            messages.append(
                f"Exact pairwise statistics will evaluate {exact_pairs:,} pairs; set sample_pairs to bound this work."
            )

    return messages


# ---------------------------------------------------------------------------
# Evaluation entry points
# ---------------------------------------------------------------------------


def run_image2image_eval(
    image_embeddings: dict[str, list[float]],
    k_values: list[int],
    dataset_root: str | None,
    sample_pairs: int | None,
    metadata: dict[str, MetadataGroup] | None = None,
) -> dict:
    """Compute the full fixed-schema embedding evaluation.

    Parameters
    ----------
    image_embeddings : dict[str, list[float]]
        Mapping of image path to embedding vector.
    k_values : list[int]
        K cutoffs for nearest-neighbour metrics.
    dataset_root : str | None
        Optional explicit dataset root for label extraction (ignored when
        ``metadata`` is provided).
    sample_pairs : int | None
        Pair-sampling budget for global similarity stats.
    metadata : dict[str, MetadataGroup] | None
        Optional similarity groups.  When supplied, class labels are taken
        from ``MetadataGroup.l1_cluster`` and additional metadata-aware
        metrics (``knn_metadata_ndcg``, ``knn_attribute_ndcg``) are computed.

    Returns
    -------
    dict
        Fixed-schema result ready for JSON serialisation.
    """
    paths = list(image_embeddings.keys())
    vectors = list(image_embeddings.values())
    _validate_embeddings(paths, vectors)

    path_labels = extract_labels(paths, dataset_root)
    if metadata is not None:
        labels_list = _labels_from_metadata(paths, metadata, path_labels)
    else:
        labels_list = path_labels
    labels_arr = np.array(labels_list)
    embeddings = np.array(vectors, dtype=np.float32)
    n, d = embeddings.shape
    unique_classes = sorted(set(labels_list))
    advisory_warnings = _build_evaluation_warnings(
        n_items=n,
        embedding_dim=d,
        metadata=metadata,
        sample_pairs=sample_pairs,
    )
    for message in advisory_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    # Stage 1: Build the k-NN graph (all subsequent stages depend on this).
    neighbors = top_k_neighbors(embeddings, ks=k_values)
    effective_ks = sorted(neighbors.keys())

    # Stage 2: Run independent metric stages in parallel.  NumPy releases the
    # GIL during most operations, so threading gives real concurrency here.
    n_workers = min(4, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        label_future = pool.submit(_compute_label_knn_metrics, neighbors, labels_arr)
        diag_future = pool.submit(_compute_neighbor_diagnostics, neighbors, n)
        confusion_future = pool.submit(knn_confusion_matrix, neighbors, labels_arr)
        meta_future = (
            pool.submit(_compute_metadata_knn_metrics, neighbors, paths, metadata) if metadata is not None else None
        )
        group_future = (
            pool.submit(_compute_group_analysis, embeddings, paths, metadata) if metadata is not None else None
        )

        label_metrics = label_future.result()
        neighbor_diags = diag_future.result()
        confusion_raw = confusion_future.result()
        meta_metrics: dict | None = meta_future.result() if meta_future is not None else None
        group_analysis: dict | None = group_future.result() if group_future is not None else None

    # Free the large neighbor index/score arrays as soon as they are no longer needed.
    del neighbors
    gc.collect()

    # Stage 3: Assemble outputs that depend on stage-2 results.
    global_metrics = _assemble_global_metrics(
        embeddings, labels_arr, sample_pairs, neighbor_diags, meta_metrics, label_metrics
    )
    per_class = _build_per_class_metrics(
        unique_classes, embeddings, labels_arr, sample_pairs, label_metrics, meta_metrics
    )

    result: dict = {
        "n_items": n,
        "embedding_dim": d,
        "classes": unique_classes,
        "k_values": effective_ks,
        "item_paths": paths,
        "item_labels": labels_list,
        "knn_confusion": _jsonify(confusion_raw),
        "global_metrics": _jsonify(global_metrics),
        "per_class": {cls: _jsonify(m) for cls, m in per_class.items()},
    }
    if advisory_warnings:
        result["warnings"] = advisory_warnings
    if group_analysis is not None:
        result["group_analysis"] = _jsonify(group_analysis)
    return result


def run_plant2image_eval(
    embeddings: dict[str, list[float]],
    instance_to_image: dict[str, list[str]],
    k_values: list[int],
    dataset_root: str | None,
    sample_pairs: int | None,
) -> dict:
    """Compute the full evaluation for the Plant→Image retrieval scenario.

    Converts ``instance_to_image`` into :class:`MetadataGroup` objects (one
    group per parent image) then delegates to :func:`run_image2image_eval`.  Each
    parent full-image and all its instance crops form a group of mutual
    explicit positives.

    Parameters
    ----------
    embeddings : dict[str, list[float]]
        Mapping of image/instance path → L2-normalised embedding vector.
        Must include both parent full-image paths and all instance crop paths
        referenced in ``instance_to_image``.
    instance_to_image : dict[str, list[str]]
        Parent full-image path → list of instance crop paths.
    k_values : list[int]
        K cutoffs for nearest-neighbour metrics.
    dataset_root : str | None
        Optional dataset root for class label extraction from parent paths.
    sample_pairs : int | None
        Pair-sampling budget for global similarity stats.

    Returns
    -------
    dict
        Fixed-schema result ready for JSON serialisation.
    """
    metadata = plant2image_to_metadata(instance_to_image, dataset_root)
    return run_image2image_eval(
        image_embeddings=embeddings,
        k_values=k_values,
        dataset_root=dataset_root,
        sample_pairs=sample_pairs,
        metadata=metadata,
    )


def run_plant2plant_eval(
    embeddings: dict[str, list[float]],
    instance_labels: dict[str, str],
    k_values: list[int],
    sample_pairs: int | None,
) -> dict:
    """Compute the full evaluation for the Plant→Plant retrieval scenario.

    Converts ``instance_labels`` into :class:`MetadataGroup` objects (one per
    class label) then delegates to :func:`run_image2image_eval`.

    Parameters
    ----------
    embeddings : dict[str, list[float]]
        Mapping of instance path → L2-normalised embedding vector.
    instance_labels : dict[str, str]
        Mapping of instance path → crop/weed class label.
    k_values : list[int]
        K cutoffs for nearest-neighbour metrics.
    sample_pairs : int | None
        Pair-sampling budget for global similarity stats.

    Returns
    -------
    dict
        Fixed-schema result ready for JSON serialisation.
    """
    metadata = plant2plant_to_metadata(instance_labels)
    return run_image2image_eval(
        image_embeddings=embeddings,
        k_values=k_values,
        dataset_root=None,
        sample_pairs=sample_pairs,
        metadata=metadata,
    )
