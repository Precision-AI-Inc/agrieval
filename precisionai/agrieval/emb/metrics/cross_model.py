# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Cross-model comparison metrics for two embedding spaces over the same items."""

from __future__ import annotations

import numpy as np

from precisionai.agrieval.emb.metrics._utils import _neighbor_stats, _prepare_embeddings, _rankdata

try:
    from scipy.stats import pearsonr, spearmanr  # type: ignore[import]

    _SCIPY_AVAILABLE = True
except ImportError:
    pearsonr = None  # type: ignore[assignment]
    spearmanr = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False


def _validate_cross_model(emb_a: np.ndarray, emb_b: np.ndarray) -> None:
    """Assert that two embedding matrices have the same number of rows.

    Parameters
    ----------
    emb_a : np.ndarray
        First embedding matrix, shape ``[N, D1]``.
    emb_b : np.ndarray
        Second embedding matrix, shape ``[N, D2]``.

    Raises
    ------
    ValueError
        If the row counts of ``emb_a`` and ``emb_b`` differ.
    """
    if emb_a.shape[0] != emb_b.shape[0]:
        raise ValueError(
            "embeddings_a and embeddings_b must have the same number of items. "
            f"Got {emb_a.shape[0]} and {emb_b.shape[0]}."
        )


def pairwise_similarity_correlation(
    embeddings_a: object,
    embeddings_b: object,
    *,
    normalize: bool = True,
    sample_pairs: int = 1_000_000,
    random_seed: int = 42,
) -> dict:
    """Pearson and Spearman correlation of pairwise similarity scores across two models.

    Parameters
    ----------
    embeddings_a : array-like
        Shape ``[N, D1]``.
    embeddings_b : array-like
        Shape ``[N, D2]``. N must match; D may differ.
    normalize : bool
        L2-normalize rows of each matrix independently before computing similarity.
    sample_pairs : int
        Number of random (i, j) pairs to evaluate.
    random_seed : int
        Seed for reproducible sampling.

    Returns
    -------
    dict
        Keys: ``sample_pairs``, ``pearson``, ``spearman``.
    """
    emb_a = _prepare_embeddings(embeddings_a, normalize, "embeddings_a")
    emb_b = _prepare_embeddings(embeddings_b, normalize, "embeddings_b")
    _validate_cross_model(emb_a, emb_b)

    n = emb_a.shape[0]
    rng = np.random.default_rng(random_seed)
    pairs_i = rng.integers(0, n, size=sample_pairs)
    pairs_j = rng.integers(0, n, size=sample_pairs)
    mask = pairs_i != pairs_j
    pairs_i, pairs_j = pairs_i[mask], pairs_j[mask]

    sims_a = np.einsum("ij,ij->i", emb_a[pairs_i], emb_a[pairs_j]).astype(np.float64)
    sims_b = np.einsum("ij,ij->i", emb_b[pairs_i], emb_b[pairs_j]).astype(np.float64)

    if pearsonr is not None and spearmanr is not None:
        pearson = float(pearsonr(sims_a, sims_b)[0])  # type: ignore[arg-type]
        spearman = float(spearmanr(sims_a, sims_b)[0])  # type: ignore[arg-type]
    else:
        pearson = float(np.corrcoef(sims_a, sims_b)[0, 1])
        spearman = float(np.corrcoef(_rankdata(sims_a), _rankdata(sims_b))[0, 1])

    return {
        "sample_pairs": len(sims_a),
        "pearson": pearson,
        "spearman": spearman,
    }


def _knn_set_metric(
    neighbors_a_by_k: dict,
    neighbors_b_by_k: dict,
    *,
    metric: str,
) -> dict:
    """Compute per-item set similarity between two models' K-NN index lists.

    Parameters
    ----------
    neighbors_a_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` for
        model A.
    neighbors_b_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` for
        model B.
    metric : str
        ``"overlap"`` — intersection size divided by K.
        ``"jaccard"`` — intersection size divided by union size.

    Returns
    -------
    dict
        Keyed by K (only Ks present in both inputs).  Each value is a stats
        dict with keys ``per_item``, ``mean``, ``std``, ``p05``, ``p50``,
        ``p95``.
    """
    shared_ks = sorted(set(neighbors_a_by_k) & set(neighbors_b_by_k))
    result = {}

    for k in shared_ks:
        idx_a = neighbors_a_by_k[k]["indices"]
        idx_b = neighbors_b_by_k[k]["indices"]
        n = idx_a.shape[0]
        per_item = np.empty(n, dtype=np.float32)

        combined = np.sort(np.concatenate([idx_a, idx_b], axis=1), axis=1)
        intersections = (combined[:, 1:] == combined[:, :-1]).sum(axis=1).astype(np.float32)
        if metric == "overlap":
            per_item[:] = intersections / k
        else:
            unions = (2 * k - intersections).astype(np.float32)
            per_item[:] = np.where(unions > 0, intersections / unions, 0.0)

        result[k] = _neighbor_stats(per_item)

    return result


def knn_overlap_at_k(
    neighbors_a_by_k: dict,
    neighbors_b_by_k: dict,
) -> dict:
    """Mean fraction of shared neighbors between two models' top-K lists.

    Parameters
    ----------
    neighbors_a_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` for model A.
    neighbors_b_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` for model B.

    Returns
    -------
    dict
        Keyed by K. Each value: ``per_item``, ``mean``, ``std``, ``p05``,
        ``p50``, ``p95``.
    """
    return _knn_set_metric(neighbors_a_by_k, neighbors_b_by_k, metric="overlap")


def knn_jaccard_at_k(
    neighbors_a_by_k: dict,
    neighbors_b_by_k: dict,
) -> dict:
    """Jaccard similarity of two models' top-K neighbor sets.

    Parameters
    ----------
    neighbors_a_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` for model A.
    neighbors_b_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` for model B.

    Returns
    -------
    dict
        Keyed by K. Each value: ``per_item``, ``mean``, ``std``, ``p05``,
        ``p50``, ``p95``.
    """
    return _knn_set_metric(neighbors_a_by_k, neighbors_b_by_k, metric="jaccard")


def per_item_neighbor_disagreement(
    neighbors_a_by_k: dict,
    neighbors_b_by_k: dict,
    *,
    metric: str = "jaccard",
    top_n: int = 20,
) -> dict:
    """Per-item disagreement scalar for visualization (UMAP / t-SNE coloring).

    Parameters
    ----------
    neighbors_a_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` for model A.
    neighbors_b_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` for model B.
    metric : str
        ``"jaccard"`` (default) or ``"overlap"``.
    top_n : int
        Number of highest-disagreement items to return per K.

    Returns
    -------
    dict
        Keyed by K. Each value: ``per_item``, ``mean``, ``p95``,
        ``top_disagreements``.
    """
    if metric not in ("jaccard", "overlap"):
        raise ValueError(f"Unknown metric {metric!r}. Use 'jaccard' or 'overlap'.")

    agreement = _knn_set_metric(neighbors_a_by_k, neighbors_b_by_k, metric=metric)
    result = {}
    for k, stats in agreement.items():
        agreement_scores = stats["per_item"]
        per_item = (1.0 - agreement_scores).astype(np.float32)
        top_idx = np.argsort(-per_item)[:top_n]

        result[k] = {
            "per_item": per_item,
            "mean": float(np.mean(per_item)),
            "p95": float(np.percentile(per_item, 95)),
            "top_disagreements": [{"index": int(i), "score": float(per_item[i])} for i in top_idx],
        }
    return result
