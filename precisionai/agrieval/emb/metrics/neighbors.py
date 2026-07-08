# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Nearest-neighbor diagnostics derived from top_k_neighbors output."""

from __future__ import annotations

import numpy as np

from precisionai.agrieval.emb.metrics._utils import _neighbor_stats


def gini_coefficient(values: object) -> float | None:
    """Gini coefficient of a non-negative array.

    Parameters
    ----------
    values : array-like
        Non-negative values (e.g., hub counts).

    Returns
    -------
    float | None
        Gini coefficient in ``[0, 1]``, or ``None`` if the input is empty or
        all zeros.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[arr >= 0]
    if len(arr) == 0 or np.sum(arr) == 0:
        return None
    arr = np.sort(arr)
    n = len(arr)
    indices = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(indices, arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))


def _skewness(values: np.ndarray) -> float | None:
    """Sample skewness using the standard unbiased formula (ddof=1).

    Parameters
    ----------
    values : np.ndarray
        1-D array of numeric values.

    Returns
    -------
    float | None
        Skewness of ``values``, or ``None`` if fewer than 3 elements are
        present.  Returns ``0.0`` when the standard deviation is zero.
    """
    n = len(values)
    if n < 3:
        return None
    std = float(np.std(values, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(((values - np.mean(values)) / std) ** 3))


def mean_top_k_similarity(neighbors_by_k: dict) -> dict:
    """Mean cosine similarity to each item's top-K neighborhood.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.

    Returns
    -------
    dict
        Keyed by K. Each value: ``per_item``, ``mean``, ``std``, ``p05``,
        ``p50``, ``p95``.
    """
    result = {}
    for k, data in neighbors_by_k.items():
        per_item = data["scores"].mean(axis=1).astype(np.float32)
        result[k] = _neighbor_stats(per_item)
    return result


def knn_radius_at_k(neighbors_by_k: dict) -> dict:
    """Cosine similarity to the k-th nearest neighbor (local density proxy).

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.

    Returns
    -------
    dict
        Keyed by K. Each value: ``per_item``, ``mean``, ``std``, ``p05``,
        ``p50``, ``p95``.
    """
    result = {}
    for k, data in neighbors_by_k.items():
        per_item = data["scores"][:, k - 1].astype(np.float32)
        result[k] = _neighbor_stats(per_item)
    return result


def outlier_score_at_k(
    neighbors_by_k: dict,
    *,
    method: str = "one_minus_mean_top_k",
    top_n: int = 20,
) -> dict:
    """Per-item outlier score based on nearest-neighbor similarity.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    method : str
        ``"one_minus_mean_top_k"`` or ``"one_minus_kth_similarity"``.
    top_n : int
        Number of top outliers to return per K.

    Returns
    -------
    dict
        Keyed by K. Each value: ``per_item``, ``mean``, ``std``, ``p95``,
        ``p99``, ``top_outliers``.
    """
    if method not in ("one_minus_mean_top_k", "one_minus_kth_similarity"):
        raise ValueError(f"Unknown method {method!r}. Use 'one_minus_mean_top_k' or 'one_minus_kth_similarity'.")

    result = {}
    for k, data in neighbors_by_k.items():
        scores = data["scores"]
        if method == "one_minus_mean_top_k":
            base = scores.mean(axis=1)
        else:
            base = scores[:, k - 1]

        per_item = (1.0 - base).astype(np.float32)
        top_idx = np.argsort(-per_item)[:top_n]

        p95, p99 = np.percentile(per_item, [95, 99]).tolist()
        result[k] = {
            "per_item": per_item,
            "mean": float(np.mean(per_item)),
            "std": float(np.std(per_item)),
            "p95": p95,
            "p99": p99,
            "top_outliers": [{"index": int(i), "score": float(per_item[i])} for i in top_idx],
        }
    return result


def hubness_at_k(
    neighbors_by_k: dict,
    *,
    n_items: int,
    top_n: int = 20,
) -> dict:
    """Measure how often each vector appears in others' nearest-neighbor lists.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    n_items : int
        Total number of items (needed to initialize the hub-count array).
    top_n : int
        Number of top hubs to return per K.

    Returns
    -------
    dict
        Keyed by K. Each value: ``hub_counts``, ``mean``, ``std``, ``max``,
        ``p95``, ``p99``, ``gini``, ``skewness``, ``anti_hub_count``,
        ``top_hubs``.
    """
    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        hub_counts = np.bincount(indices.ravel(), minlength=n_items).astype(np.int64)

        p95, p99 = np.percentile(hub_counts, [95, 99]).tolist()
        top_idx = np.argsort(-hub_counts)[:top_n]

        result[k] = {
            "hub_counts": hub_counts,
            "mean": float(np.mean(hub_counts)),
            "std": float(np.std(hub_counts)),
            "max": int(np.max(hub_counts)),
            "p95": p95,
            "p99": p99,
            "gini": gini_coefficient(hub_counts),
            "skewness": _skewness(hub_counts.astype(np.float64)),
            "anti_hub_count": int(np.sum(hub_counts == 0)),
            "top_hubs": [{"index": int(i), "count": int(hub_counts[i])} for i in top_idx if hub_counts[i] > 0],
        }
    return result
