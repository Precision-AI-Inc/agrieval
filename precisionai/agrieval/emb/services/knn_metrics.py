# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""KNN metric bundles, per-class breakdowns, and group analysis for embedding evaluation."""

from __future__ import annotations

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
    uniformity,
)
from precisionai.agrieval.emb.metrics.label_aware import _build_grade_matrix
from precisionai.agrieval.emb.schemas.evaluate import MetadataGroup
from precisionai.agrieval.emb.services.labels import build_image_items

# ---------------------------------------------------------------------------
# Per-item stat summarization
# ---------------------------------------------------------------------------


def _slice_knn_stats(per_item_by_k: dict[int, np.ndarray], indices: np.ndarray) -> dict:
    """Summarize per-item KNN scores for a subset of items.

    Parameters
    ----------
    per_item_by_k : dict[int, np.ndarray]
        Mapping of k → per-item score array (full dataset).
    indices : np.ndarray
        Row indices of the subset to summarize.

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
    """Summarize a flat per-item score array for a subset of items.

    Parameters
    ----------
    per_item : np.ndarray
        Full-dataset per-item scores (no K nesting).
    indices : np.ndarray
        Row indices of the subset to summarize.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``p05``, ``p50``, ``p95``.
    """
    vals = per_item[indices].astype(np.float64)
    p05, p50, p95 = np.percentile(vals, [5, 50, 95]).tolist()
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "p05": p05, "p50": p50, "p95": p95}


# ---------------------------------------------------------------------------
# KNN metric bundles
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
        Precomputed top-K neighbor index arrays keyed by K.
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
        Precomputed top-K neighbor index arrays keyed by K.
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
    """Surface pre-existing neighbor diagnostics as a named bundle.

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


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


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
# Group analysis
# ---------------------------------------------------------------------------


def _compute_group_analysis(
    embeddings: np.ndarray,
    paths: list[str],
    metadata: dict[str, MetadataGroup],
) -> dict:
    """Analyze the coherence of each declared metadata group using HDBSCAN.

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
