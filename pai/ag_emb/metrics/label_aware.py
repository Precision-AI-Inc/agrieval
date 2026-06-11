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

"""Optional label-aware metrics for supervised embedding evaluation."""

from __future__ import annotations

import numpy as np

from pai.ag_emb.metrics._utils import _prepare_embeddings
from pai.ag_emb.metrics.ranking import average_precision_at_k, ndcg_at_k


def intra_inter_similarity_gap(
    embeddings: object,
    labels: object,
    *,
    normalize: bool = True,
    sample_pairs: int | None = 1_000_000,
    random_seed: int = 42,
) -> dict:
    """Mean intra-class vs inter-class cosine similarity gap.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    labels : array-like
        Shape ``[N]``. Integer or string class labels.
    normalize : bool
        L2-normalize rows before computing similarity.
    sample_pairs : int | None
        ``None`` evaluates all pairs; an integer samples that many.
    random_seed : int
        Seed for reproducible sampling.

    Returns
    -------
    dict
        Keys: ``mean_intra_class_similarity``, ``mean_inter_class_similarity``,
        ``gap``, ``num_intra_pairs``, ``num_inter_pairs``.
    """
    emb = _prepare_embeddings(embeddings, normalize)
    lbl = np.asarray(labels)
    n = emb.shape[0]

    if n != len(lbl):
        raise ValueError(f"embeddings and labels must have the same length. " f"Got {n} and {len(lbl)}.")

    rng = np.random.default_rng(random_seed)

    if sample_pairs is None:
        i_idx, j_idx = np.triu_indices(n, k=1)
    else:
        i_idx = rng.integers(0, n, size=sample_pairs)
        j_idx = rng.integers(0, n, size=sample_pairs)
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]

    sims = np.einsum("ij,ij->i", emb[i_idx], emb[j_idx]).astype(np.float32)
    same_label = lbl[i_idx] == lbl[j_idx]

    intra = sims[same_label]
    inter = sims[~same_label]

    mean_intra = float(np.mean(intra)) if len(intra) > 0 else None
    mean_inter = float(np.mean(inter)) if len(inter) > 0 else None
    gap = (mean_intra - mean_inter) if (mean_intra is not None and mean_inter is not None) else None

    return {
        "mean_intra_class_similarity": mean_intra,
        "mean_inter_class_similarity": mean_inter,
        "gap": gap,
        "num_intra_pairs": len(intra),
        "num_inter_pairs": len(inter),
    }


def knn_label_ndcg_at_k(
    neighbors_by_k: dict,
    labels: object,
) -> dict:
    """NDCG at K using same-class membership as binary relevance.

    For each item the ranked neighbor list is scored with relevance=1 for
    same-class neighbors and 0 otherwise.  The ideal DCG is computed from
    the full corpus (all same-class items excluding self).

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    labels : array-like
        Shape ``[N]``. Integer or string class labels.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    lbl = np.asarray(labels)
    n = lbl.shape[0]
    result = {}

    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        per_item = np.empty(n, dtype=np.float64)

        for i in range(n):
            ranked_relevances = [1 if lbl[idx] == lbl[i] else 0 for idx in indices[i]]
            n_relevant = int(np.sum(lbl == lbl[i])) - 1  # exclude self
            # ideal: top positions filled with relevant items, rest zero
            ideal_relevances = [1] * n_relevant + [0] * max(0, n - 1 - n_relevant)
            score = ndcg_at_k(ranked_relevances, ideal_relevances, k)
            per_item[i] = score if score is not None else 0.0

        result[k] = {
            "mean": float(np.mean(per_item)),
            "std": float(np.std(per_item)),
            "per_item": per_item,
        }
    return result


def knn_map_at_k(
    neighbors_by_k: dict,
    labels: object,
) -> dict:
    """Mean Average Precision at K using same-class membership as relevance.

    For each item, AP@K is the average of precision values computed at each
    rank position where a same-class neighbor appears in the top-K list.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    labels : array-like
        Shape ``[N]``. Integer or string class labels.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    lbl = np.asarray(labels)
    n = lbl.shape[0]
    result = {}

    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        per_item = np.empty(n, dtype=np.float64)

        for i in range(n):
            neighbor_ids = list(map(int, indices[i]))
            relevant_ids = {j for j in range(n) if j != i and lbl[j] == lbl[i]}
            ap = average_precision_at_k(neighbor_ids, relevant_ids, k)
            per_item[i] = ap if ap is not None else 0.0

        result[k] = {
            "mean": float(np.mean(per_item)),
            "std": float(np.std(per_item)),
            "per_item": per_item,
        }
    return result


def knn_confusion_matrix(
    neighbors_by_k: dict,
    labels: object,
) -> dict:
    """KNN confusion matrix: fraction of k-NN neighbors belonging to each class.

    For each class (row), computes the fraction of its members' k nearest
    neighbors that belong to each class (column).  The diagonal equals mean
    KNN purity.  Each row sums to 1.0.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    labels : array-like
        Shape ``[N]``. Integer or string class labels.

    Returns
    -------
    dict
        Keyed by K. Each value: ``{true_class: {neighbor_class: float}}``.
    """
    lbl = np.asarray(labels)
    unique_classes = sorted(set(lbl.tolist()))
    cls_to_idx = {c: i for i, c in enumerate(unique_classes)}
    n_cls = len(unique_classes)
    n = len(lbl)
    label_indices = np.array([cls_to_idx[lb] for lb in lbl], dtype=np.int32)

    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        matrix = np.zeros((n_cls, n_cls), dtype=np.float64)
        row_counts = np.zeros(n_cls, dtype=np.int64)

        for i in range(n):
            ti = label_indices[i]
            row_counts[ti] += 1
            for nb in indices[i]:
                matrix[ti, label_indices[int(nb)]] += 1.0

        for ci in range(n_cls):
            total = k * int(row_counts[ci])
            if total > 0:
                matrix[ci] /= total

        result[k] = {
            c1: {c2: float(matrix[i, j]) for j, c2 in enumerate(unique_classes)} for i, c1 in enumerate(unique_classes)
        }
    return result


def knn_label_purity_at_k(
    neighbors_by_k: dict,
    labels: object,
) -> dict:
    """Fraction of each item's K nearest neighbors that share its label.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    labels : array-like
        Shape ``[N]``. Integer or string class labels.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    lbl = np.asarray(labels)
    result = {}

    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        n = indices.shape[0]
        per_item = np.empty(n, dtype=np.float32)

        for i in range(n):
            neighbor_labels = lbl[indices[i]]
            per_item[i] = float(np.sum(neighbor_labels == lbl[i])) / k

        result[k] = {
            "mean": float(np.mean(per_item)),
            "std": float(np.std(per_item)),
            "per_item": per_item,
        }
    return result
