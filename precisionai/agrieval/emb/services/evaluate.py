# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Service layer: orchestrates embedding metrics into the fixed analysis JSON."""

from __future__ import annotations

import gc
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.cluster import HDBSCAN as _HDBSCAN  # type: ignore[import]
    from sklearn.metrics import silhouette_score as _silhouette_score  # type: ignore[import]

    _SKLEARN_AVAILABLE = True
except ImportError:
    _HDBSCAN = None  # type: ignore[assignment]
    _silhouette_score = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

from precisionai.agrieval.emb.metrics import (
    ImageItem,
    alignment,
    centroid_similarity_stats,
    effective_rank,
    hubness_at_k,
    intra_inter_similarity_gap,
    knn_confusion_matrix,
    knn_label_mrr_at_k,
    knn_label_ndcg_at_k,
    knn_label_purity_at_k,
    knn_label_r_precision,
    knn_map_at_k,
    knn_metadata_map_at_k,
    knn_metadata_mrr_at_k,
    knn_metadata_ndcg_at_k,
    knn_metadata_precision_at_k,
    knn_metadata_r_precision,
    knn_per_attribute_ndcg_at_k,
    knn_radius_at_k,
    mean_top_k_similarity,
    outlier_score_at_k,
    pairwise_similarity_stats,
    top_k_neighbors,
    uniformity,
)
from precisionai.agrieval.emb.metrics.label_aware import _build_grade_matrix
from precisionai.agrieval.emb.schemas.evaluate import MetadataGroup

# Supported embedding path extensions — case-sensitive exact match.
_SUPPORTED_EXTENSIONS = frozenset({".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"})

# Matches an L2 cluster folder name: one or more letters followed by one or more digits.
_L2_FOLDER_RE = re.compile(r"^([A-Za-z]+)\d+$")

# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------


def _parse_crop(folder_name: str) -> str:
    """Extract the L1 cluster label or crop name from a folder name.

    Two naming conventions are supported:

    * **L1/L2 dataset layout** — folder names like ``A1``, ``G7`` (letters + digits):
      the leading letters are returned as the L1 cluster label.
    * **Legacy crop_camera layout** — folder names like ``corn_HB-25000SBC``:
      everything before the first ``_`` is returned as the class name.

    Examples
    --------
    ``A1``               →  ``A``
    ``G7``               →  ``G``
    ``corn_HB-25000SBC`` →  ``corn``
    ``corn_nikon_d610``  →  ``corn``
    ``corn``             →  ``corn``   (plain folder, returned as-is)
    """
    m = _L2_FOLDER_RE.match(folder_name)
    if m:
        return m.group(1)
    if "_" in folder_name:
        return folder_name.split("_", maxsplit=1)[0]
    return folder_name


def extract_labels(paths: list[str], dataset_root: str | None = None) -> list[str]:
    """Infer crop class from image paths.

    The class is extracted from the first path component after the dataset
    root.  Two folder naming conventions are supported:

    * ``{root}/{crop}_{[camera]}/img/{image}``  — real dataset layout, e.g.
      ``dummy/corn_[HB-25000SBC]/img/220622-img.png``
    * ``{root}/{crop}/{camera}/{image}``  — plain layout without camera tags.

    If ``dataset_root`` is not given the longest common directory prefix of
    all paths is used as the root automatically.

    Parameters
    ----------
    paths : list[str]
        Image paths as they appear in the embeddings dictionary.
    dataset_root : str | None
        Explicit root to strip before label extraction.

    Returns
    -------
    list[str]
        Crop class label per path (same order as ``paths``).
    """
    if not paths:
        return []

    posix_paths = [Path(p.replace("\\", "/")).as_posix() for p in paths]

    if dataset_root is not None:
        root_prefix = Path(dataset_root.replace("\\", "/")).as_posix().rstrip("/") + "/"
    else:
        raw = os.path.commonprefix(posix_paths)
        # Trim to the last directory separator so we don't clip mid-word
        root_prefix = raw[: raw.rfind("/") + 1] if "/" in raw else ""

    labels: list[str] = []
    for posix_path in posix_paths:
        remainder = posix_path[len(root_prefix) :] if posix_path.startswith(root_prefix) else posix_path
        parts = Path(remainder).parts
        folder = parts[0] if parts else "unknown"
        labels.append(_parse_crop(folder))

    return labels


def _labels_from_metadata(
    paths: list[str],
    metadata: dict[str, MetadataGroup],
    fallback_labels: list[str],
) -> list[str]:
    """Return per-path class labels sourced from metadata, falling back to path-extracted labels.

    Matching is performed by exact path key comparison.  The ``images`` entries
    in each group must exactly match the keys used in the embeddings dictionary.

    Parameters
    ----------
    paths : list[str]
        Image paths as they appear in the embeddings dictionary.
    metadata : dict[str, MetadataGroup]
        Metadata groups keyed by arbitrary group ID.
    fallback_labels : list[str]
        Labels to use when a path has no matching metadata group.

    Returns
    -------
    list[str]
        One label per path, in the same order as ``paths``.
    """
    path_to_class: dict[str, str] = {}
    for group in metadata.values():
        for img in group.images:
            path_to_class[img] = group.l1_cluster

    return [path_to_class.get(p, fallback) for p, fallback in zip(paths, fallback_labels, strict=True)]


def _normalise_attributes(raw: dict[str, Any]) -> dict[str, str]:
    """Convert a MetadataGroup attributes dict to a flat ``dict[str, str]``.

    List values are sorted and joined with ``","`` for stable canonical form.
    Null values are excluded so they do not participate in attribute matching.

    Parameters
    ----------
    raw : dict[str, Any]
        Attributes as stored in a :class:`MetadataGroup` (may contain lists or ``None``).

    Returns
    -------
    dict[str, str]
        Normalised attributes suitable for :class:`~precisionai.agrieval.emb.metrics.ImageItem`.
    """
    result: dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, list):
            result[k] = ",".join(sorted(str(x) for x in v))
        else:
            result[k] = str(v)
    return result


def build_image_items(
    paths: list[str],
    metadata: dict[str, MetadataGroup],
) -> list[ImageItem]:
    """Build :class:`~precisionai.agrieval.emb.metrics.ImageItem` objects from embedding paths and metadata.

    Each item's ``image_id`` is the exact embedding path key.
    The ``explicit_positive_ids`` are the exact paths of all other
    images in the same metadata group.  Items with no matching group receive
    empty ``explicit_positive_ids``, ``None`` for ``class_name``, and an
    empty ``attributes`` dict.

    Parameters
    ----------
    paths : list[str]
        Image paths as they appear in the embeddings dictionary.
    metadata : dict[str, MetadataGroup]
        Metadata groups keyed by arbitrary group ID.

    Returns
    -------
    list[ImageItem]
        One :class:`~precisionai.agrieval.emb.metrics.ImageItem` per path, in the same order.
    """
    # exact path → (l1_cluster, normalised str attributes, frozenset of all member paths)
    path_to_group: dict[str, tuple[str, dict[str, str], frozenset[str]]] = {}
    for group in metadata.values():
        member_ids = frozenset(group.images)
        norm_attrs = _normalise_attributes(group.attributes)
        for img in group.images:
            path_to_group[img] = (group.l1_cluster, norm_attrs, member_ids)

    items: list[ImageItem] = []
    for path in paths:
        if path in path_to_group:
            l1_cluster, norm_attrs, member_ids = path_to_group[path]
            items.append(
                ImageItem(
                    image_id=path,
                    explicit_positive_ids=member_ids - {path},
                    class_name=l1_cluster,
                    attributes=norm_attrs,
                )
            )
        else:
            items.append(ImageItem(image_id=path))
    return items


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _jsonify(obj: Any) -> Any:  # noqa: PLR0911
    """Recursively convert a metrics result to JSON-serialisable types.

    numpy arrays are dropped because they are per-item visualisation artefacts
    that do not belong in a summary JSON.  numpy scalars are converted to
    Python native types.  ``NaN`` and ``inf`` become ``None``.

    Parameters
    ----------
    obj : Any
        Arbitrary metrics result object — dict, list, numpy array/scalar, or
        Python scalar.

    Returns
    -------
    object
        JSON-serialisable equivalent of ``obj``.  ``None`` is returned for
        numpy arrays (sentinel value; callers omit these keys).
    """
    if isinstance(obj, np.ndarray):
        return None  # sentinel — callers filter this key out
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                continue  # drop per_item / hub_counts etc.
            serialised = _jsonify(v)
            out[str(k)] = serialised
        return out
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return obj


# ---------------------------------------------------------------------------
# Core analysis helpers
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


def _slice_knn_stats(per_item_by_k: dict[int, np.ndarray], indices: np.ndarray) -> dict:
    """Summarise per-item KNN scores for a subset of items.

    Parameters
    ----------
    per_item_by_k : dict[int, np.ndarray]
        Mapping of k → per-item score array (full dataset).
    indices : np.ndarray
        Row indices of the subset to summarise.

    Returns
    -------
    dict
        Keyed by str(k), each value has mean/std/p05/p50/p95.
    """
    out: dict = {}
    for k, per_item in per_item_by_k.items():
        vals = per_item[indices].astype(np.float64)
        p05, p50, p95 = np.percentile(vals, [5, 50, 95]).tolist()
        out[str(k)] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p05": p05,
            "p50": p50,
            "p95": p95,
        }
    return out


def _slice_flat_stats(per_item: np.ndarray, indices: np.ndarray) -> dict:
    """Summarise a flat per-item score array for a subset of items.

    Parameters
    ----------
    per_item : np.ndarray
        Full-dataset per-item scores (no K nesting).
    indices : np.ndarray
        Row indices of the subset to summarise.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``p05``, ``p50``, ``p95``.
    """
    vals = per_item[indices].astype(np.float64)
    p05, p50, p95 = np.percentile(vals, [5, 50, 95]).tolist()
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "p05": p05, "p50": p50, "p95": p95}


def _build_per_class_metrics(
    unique_classes: list[str],
    embeddings: np.ndarray,
    labels_arr: np.ndarray,
    sample_pairs: int | None,
    label_metrics: dict,
    meta_metrics: dict | None = None,
) -> dict:
    """Compute per-class embedding metrics.

    Parameters
    ----------
    unique_classes : list[str]
        Sorted list of class labels.
    embeddings : np.ndarray
        Full embedding matrix, shape ``(n, d)``.
    labels_arr : np.ndarray
        Label per row, aligned with ``embeddings``.
    sample_pairs : int | None
        Pair-sampling budget (capped at 100 000 per class).
    label_metrics : dict
        Bundle returned by ``_compute_label_knn_metrics``.
    meta_metrics : dict | None
        Bundle returned by ``_compute_metadata_knn_metrics``.  When provided,
        metadata KPIs replace the label-based KPIs in per-class output.

    Returns
    -------
    dict
        Keyed by class label, each value is the class metrics dict.
    """
    per_class: dict = {}
    for cls in unique_classes:
        cls_mask = labels_arr == cls
        cls_indices = np.where(cls_mask)[0]
        cls_embeddings = embeddings[cls_mask]
        cls_n = int(np.sum(cls_mask))
        cls_metrics: dict = {"n_items": cls_n}
        if cls_n >= 2:
            cls_sample = min(sample_pairs, 100_000) if sample_pairs is not None else None
            cls_metrics["pairwise_similarity_stats"] = pairwise_similarity_stats(
                cls_embeddings, sample_pairs=cls_sample
            )
            cls_metrics["centroid_similarity_stats"] = centroid_similarity_stats(cls_embeddings)
            cls_metrics["effective_rank"] = effective_rank(cls_embeddings)
        if meta_metrics is not None:
            cls_metrics["knn_metadata_precision"] = _slice_knn_stats(meta_metrics["precision_per_item"], cls_indices)
            cls_metrics["knn_metadata_ndcg"] = _slice_knn_stats(meta_metrics["ndcg_per_item"], cls_indices)
            cls_metrics["knn_metadata_map"] = _slice_knn_stats(meta_metrics["map_per_item"], cls_indices)
            cls_metrics["knn_metadata_mrr"] = _slice_knn_stats(meta_metrics["mrr_per_item"], cls_indices)
            cls_metrics["knn_metadata_r_precision"] = _slice_flat_stats(meta_metrics["r_prec_per_item"], cls_indices)
        else:
            cls_metrics["knn_label_purity"] = _slice_knn_stats(label_metrics["purity_per_item"], cls_indices)
            cls_metrics["knn_label_ndcg"] = _slice_knn_stats(label_metrics["ndcg_per_item"], cls_indices)
            cls_metrics["knn_map"] = _slice_knn_stats(label_metrics["map_per_item"], cls_indices)
            cls_metrics["knn_label_mrr"] = _slice_knn_stats(label_metrics["mrr_per_item"], cls_indices)
            cls_metrics["knn_label_r_precision"] = _slice_flat_stats(label_metrics["r_prec_per_item"], cls_indices)
        per_class[cls] = cls_metrics
    return per_class


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _compute_metadata_knn_metrics(
    neighbors: dict[int, np.ndarray],
    paths: list[str],
    metadata: dict[str, MetadataGroup],
) -> dict:
    """Compute all metadata-aware KNN metrics and return them as a bundle.

    Parameters
    ----------
    neighbors : dict[int, np.ndarray]
        Precomputed top-K neighbour index arrays keyed by K.
    paths : list[str]
        Embedding path keys in the same index order as ``neighbors``.
    metadata : dict[str, MetadataGroup]
        Similarity groups from the evaluation request.

    Returns
    -------
    dict
        Keys: ndcg, precision, map, mrr, r_prec (raw + per_item),
        attribute_ndcg_raw, positive_pairs.
    """
    items = build_image_items(paths, metadata)

    # Precompute the grade matrix and positive mask once and reuse across all
    # metric functions, avoiding O(N^2 x |K|) redundant relevance_grade calls.
    grade_matrix = _build_grade_matrix(items)
    positive_mask = grade_matrix == 3

    ndcg_raw = knn_metadata_ndcg_at_k(neighbors, items, _grade_matrix=grade_matrix)
    precision_raw = knn_metadata_precision_at_k(neighbors, items, _positive_mask=positive_mask)
    map_raw = knn_metadata_map_at_k(neighbors, items, _positive_mask=positive_mask)
    mrr_raw = knn_metadata_mrr_at_k(neighbors, items, _positive_mask=positive_mask)
    r_prec_raw = knn_metadata_r_precision(neighbors, items, _positive_mask=positive_mask)
    return {
        "ndcg_raw": ndcg_raw,
        "ndcg_per_item": {k: s["per_item"] for k, s in ndcg_raw.items()},
        "precision_raw": precision_raw,
        "precision_per_item": {k: s["per_item"] for k, s in precision_raw.items()},
        "map_raw": map_raw,
        "map_per_item": {k: s["per_item"] for k, s in map_raw.items()},
        "mrr_raw": mrr_raw,
        "mrr_per_item": {k: s["per_item"] for k, s in mrr_raw.items()},
        "r_prec_raw": r_prec_raw,
        "r_prec_per_item": r_prec_raw["per_item"],
        "attribute_ndcg_raw": knn_per_attribute_ndcg_at_k(neighbors, items),
        "positive_pairs": _build_positive_pairs(items),
    }


def _compute_label_knn_metrics(
    neighbors: dict[int, np.ndarray],
    labels_arr: np.ndarray,
) -> dict:
    """Run all label-aware KNN metrics and return them as a bundle.

    Parameters
    ----------
    neighbors : dict[int, np.ndarray]
        Precomputed top-K neighbour index arrays keyed by K.
    labels_arr : np.ndarray
        Class label for every item in the same index order as ``neighbors``.

    Returns
    -------
    dict
        Keys: purity, ndcg, map, mrr, r_prec (raw + per_item).
    """
    purity_raw = knn_label_purity_at_k(neighbors, labels_arr)
    ndcg_raw = knn_label_ndcg_at_k(neighbors, labels_arr)
    map_raw = knn_map_at_k(neighbors, labels_arr)
    mrr_raw = knn_label_mrr_at_k(neighbors, labels_arr)
    r_prec_raw = knn_label_r_precision(neighbors, labels_arr)
    return {
        "purity_raw": purity_raw,
        "purity_per_item": {k: s["per_item"] for k, s in purity_raw.items()},
        "ndcg_raw": ndcg_raw,
        "ndcg_per_item": {k: s["per_item"] for k, s in ndcg_raw.items()},
        "map_raw": map_raw,
        "map_per_item": {k: s["per_item"] for k, s in map_raw.items()},
        "mrr_raw": mrr_raw,
        "mrr_per_item": {k: s["per_item"] for k, s in mrr_raw.items()},
        "r_prec_raw": r_prec_raw,
        "r_prec_per_item": r_prec_raw["per_item"],
    }


def _build_positive_pairs(items: list[ImageItem]) -> list[tuple[int, int]]:
    """Build a deduplicated list of (i, j) index pairs for explicit positives.

    Parameters
    ----------
    items : list[ImageItem]
        Items in embedding-row order.

    Returns
    -------
    list[tuple[int, int]]
        Unique pairs where both members are in the same metadata group.
    """
    id_to_idx = {item.image_id: i for i, item in enumerate(items)}
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    for item in items:
        qi = id_to_idx.get(item.image_id)
        if qi is None:
            continue
        for pos_id in item.explicit_positive_ids:
            pi = id_to_idx.get(pos_id)
            if pi is not None:
                key = (min(qi, pi), max(qi, pi))
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
    return pairs


def _compute_neighbor_diagnostics(
    neighbors: dict[int, np.ndarray],
    n: int,
) -> dict:
    """Surface pre-existing neighbour diagnostics as a named bundle.

    Parameters
    ----------
    neighbors : dict[int, np.ndarray]
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    n : int
        Total number of items (needed for hubness count array).

    Returns
    -------
    dict
        Keys: ``hubness``, ``knn_radius``, ``mean_top_k_sim``, ``outlier_score``.
    """
    return {
        "hubness": hubness_at_k(neighbors, n_items=n),
        "knn_radius": knn_radius_at_k(neighbors),
        "mean_top_k_sim": mean_top_k_similarity(neighbors),
        "outlier_score": outlier_score_at_k(neighbors),
    }


def _assemble_global_metrics(
    embeddings: np.ndarray,
    labels_arr: np.ndarray,
    sample_pairs: int | None,
    neighbor_diags: dict,
    meta_metrics: dict | None,
    label_metrics: dict,
) -> dict:
    """Compute geometry stats and assemble the global_metrics dict.

    Parameters
    ----------
    embeddings : np.ndarray
        Full embedding matrix.
    labels_arr : np.ndarray
        Class label per row.
    sample_pairs : int | None
        Pair-sampling budget for pairwise/intra-inter stats.
    neighbor_diags : dict
        Bundle from ``_compute_neighbor_diagnostics``.
    meta_metrics : dict | None
        Bundle from ``_compute_metadata_knn_metrics``, or ``None``.
    label_metrics : dict
        Bundle from ``_compute_label_knn_metrics``.

    Returns
    -------
    dict
        Complete global metrics dict ready for ``_jsonify``.
    """
    pairwise_s = pairwise_similarity_stats(embeddings, sample_pairs=sample_pairs)
    intra_inter = intra_inter_similarity_gap(embeddings, labels_arr, sample_pairs=sample_pairs)
    eff_rank = effective_rank(embeddings)
    centroid_s = centroid_similarity_stats(embeddings)
    gm: dict = {
        "pairwise_similarity_stats": pairwise_s,
        "intra_inter_similarity_gap": intra_inter,
        "effective_rank": eff_rank,
        "centroid_similarity_stats": centroid_s,
        "uniformity": uniformity(embeddings),
        "hubness": neighbor_diags["hubness"],
        "knn_radius": neighbor_diags["knn_radius"],
        "mean_top_k_sim": neighbor_diags["mean_top_k_sim"],
        "outlier_score": neighbor_diags["outlier_score"],
    }
    if meta_metrics is not None:
        gm["alignment"] = alignment(embeddings, meta_metrics["positive_pairs"])
        gm["knn_metadata_precision"] = meta_metrics["precision_raw"]
        gm["knn_metadata_ndcg"] = meta_metrics["ndcg_raw"]
        gm["knn_metadata_map"] = meta_metrics["map_raw"]
        gm["knn_metadata_mrr"] = meta_metrics["mrr_raw"]
        gm["knn_metadata_r_precision"] = meta_metrics["r_prec_raw"]
        if meta_metrics["attribute_ndcg_raw"]:
            gm["knn_attribute_ndcg"] = meta_metrics["attribute_ndcg_raw"]
    else:
        gm["knn_label_purity"] = label_metrics["purity_raw"]
        gm["knn_label_ndcg"] = label_metrics["ndcg_raw"]
        gm["knn_map"] = label_metrics["map_raw"]
        gm["knn_label_mrr"] = label_metrics["mrr_raw"]
        gm["knn_label_r_precision"] = label_metrics["r_prec_raw"]
    return gm


def _compute_group_analysis(
    embeddings: np.ndarray,
    paths: list[str],
    metadata: dict[str, MetadataGroup],
) -> dict:
    """Analyse the coherence of each declared metadata group using HDBSCAN.

    For each group, compute pairwise intra-group cosine statistics and
    attempt to detect natural sub-clusters.  Groups with ``cluster_count > 1``
    and ``silhouette_score > 0.25`` are flagged with ``suggested_split: true``.

    Parameters
    ----------
    embeddings : np.ndarray
        Full embedding matrix, shape ``(n, d)``.
    paths : list[str]
        Embedding path keys in the same index order as ``embeddings``.
    metadata : dict[str, MetadataGroup]
        Declared similarity groups.

    Returns
    -------
    dict
        Keyed by group name.  Each value: ``n_images``, ``mean_intra_cosine``,
        ``std_intra_cosine``, ``cluster_count``, ``noise_count``,
        ``silhouette_score``, ``suggested_split``.
    """
    path_to_idx = {p: i for i, p in enumerate(paths)}
    result: dict = {}
    for group_key, group in metadata.items():
        idxs = [path_to_idx[img] for img in group.images if img in path_to_idx]
        n = len(idxs)
        if n < 2:
            result[group_key] = {
                "n_images": n,
                "mean_intra_cosine": None,
                "std_intra_cosine": None,
                "cluster_count": None,
                "noise_count": 0,
                "silhouette_score": None,
                "suggested_split": False,
            }
            continue
        g_emb = embeddings[np.array(idxs)]
        sims = np.clip(g_emb @ g_emb.T, -1.0, 1.0)
        cos_dist = np.maximum(1.0 - sims, 0.0).astype(np.float64)
        np.fill_diagonal(cos_dist, 0.0)
        triu = np.triu_indices(n, k=1)
        mean_intra = float(np.mean(sims[triu]))
        std_intra = float(np.std(sims[triu]))
        cluster_count, noise_count, sil_score = 1, 0, None
        if _SKLEARN_AVAILABLE and n >= 3:
            min_cs = max(2, n // 3)
            clusterer = _HDBSCAN(
                min_cluster_size=min_cs,
                metric="precomputed",
                cluster_selection_method="leaf",
                copy=True,  # type: ignore[arg-type]
            )  # type: ignore[call]
            labels = clusterer.fit_predict(cos_dist)
            valid = labels[labels >= 0]
            cluster_count = len(set(valid.tolist())) if len(valid) > 0 else 0
            noise_count = int(np.sum(labels == -1))
            if cluster_count >= 2:
                non_noise = np.where(labels >= 0)[0]
                if len(non_noise) >= 2 and _silhouette_score is not None:
                    sil_score = float(
                        _silhouette_score(
                            cos_dist[np.ix_(non_noise, non_noise)], labels[non_noise], metric="precomputed"
                        )
                    )
        result[group_key] = {
            "n_images": n,
            "mean_intra_cosine": mean_intra,
            "std_intra_cosine": std_intra,
            "cluster_count": cluster_count,
            "noise_count": noise_count,
            "silhouette_score": sil_score,
            "suggested_split": cluster_count > 1 and (sil_score is None or sil_score > 0.25),
        }
    return result


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
    if group_analysis is not None:
        result["group_analysis"] = _jsonify(group_analysis)
    return result


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_dataset_metadata(dataset_root: str) -> dict[str, MetadataGroup]:
    """Load all cluster metadata from the dataset directory.

    Reads every ``metadata/{L2}/metadata.json`` file under ``dataset_root``
    and returns a dict keyed by L2 cluster identifier (e.g. ``"A1"``).
    Each JSON file is parsed as a :class:`MetadataGroup` — see that class for
    the required fields.  Files using the legacy ``class_name`` field instead
    of ``l1_cluster`` / ``l2_cluster`` are accepted transparently.

    Image paths in the returned groups are stored exactly as written in the
    JSON files (relative to ``dataset_root``).  Ensure the embedding dict
    keys use the same relative convention, or prepend the dataset root before
    passing to the evaluation functions.

    Parameters
    ----------
    dataset_root : str
        Root directory of the dataset (e.g. ``"dataset"``).

    Returns
    -------
    dict[str, MetadataGroup]
        Keyed by L2 cluster (e.g. ``"A1"``).  Each value is a
        :class:`MetadataGroup` with ``l1_cluster``, ``l2_cluster``,
        ``images``, and ``attributes`` populated from the JSON file.

    Raises
    ------
    FileNotFoundError
        If the ``metadata/`` subdirectory does not exist under ``dataset_root``.
    """
    root = Path(dataset_root.replace("\\", "/"))
    metadata_dir = root / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Dataset metadata directory not found: {metadata_dir}")

    groups: dict[str, MetadataGroup] = {}
    for meta_file in sorted(metadata_dir.glob("*/metadata.json")):
        with meta_file.open(encoding="utf-8") as fh:
            data = json.load(fh)
        group = MetadataGroup(**data)
        groups[group.l2_cluster] = group
    return groups


# ---------------------------------------------------------------------------
# Plant wiring adapters
# ---------------------------------------------------------------------------


def plant2image_to_metadata(
    instance_to_image: dict[str, list[str]],
    dataset_root: str | None = None,
) -> dict[str, MetadataGroup]:
    """Convert a Plant2Image wiring map to :class:`MetadataGroup` form.

    Each parent full-image and all its instance crops form one group whose
    members are mutual explicit positives.  The crop class is extracted from
    the parent image path and used as ``l1_cluster``; the parent path is used
    as the unique ``l2_cluster`` identifier.

    Parameters
    ----------
    instance_to_image : dict[str, list[str]]
        Mapping from parent full-image path to a list of its instance crop
        paths.
    dataset_root : str | None
        Optional dataset root prefix for extracting the crop class label from
        parent image paths.  Defaults to the longest common directory prefix.

    Returns
    -------
    dict[str, MetadataGroup]
        One :class:`MetadataGroup` per parent image, keyed by the parent path.
        Each group's ``images`` list contains the parent path followed by all
        its instance paths.
    """
    parent_paths = list(instance_to_image.keys())
    if not parent_paths:
        return {}
    parent_labels = extract_labels(parent_paths, dataset_root)
    return {
        parent_path: MetadataGroup(
            images=[parent_path, *instance_paths],
            l1_cluster=class_name,
            l2_cluster=parent_path,
        )
        for (parent_path, instance_paths), class_name in zip(instance_to_image.items(), parent_labels, strict=True)
    }


def plant2plant_to_metadata(
    instance_labels: dict[str, str],
) -> dict[str, MetadataGroup]:
    """Convert a Plant2Plant wiring map to :class:`MetadataGroup` form.

    All instances sharing the same class label are grouped into one
    :class:`MetadataGroup`, making them mutual explicit positives.

    Parameters
    ----------
    instance_labels : dict[str, str]
        Mapping from instance path to its crop/weed class label.

    Returns
    -------
    dict[str, MetadataGroup]
        :class:`MetadataGroup` objects ready for :func:`run_image2image_eval`.
    """
    class_to_instances: dict[str, list[str]] = {}
    for inst, cls in instance_labels.items():
        class_to_instances.setdefault(cls, []).append(inst)
    return {
        cls: MetadataGroup(images=instances, l1_cluster=cls, l2_cluster=cls)
        for cls, instances in class_to_instances.items()
    }


# ---------------------------------------------------------------------------
# Plant wiring entry points
# ---------------------------------------------------------------------------


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
