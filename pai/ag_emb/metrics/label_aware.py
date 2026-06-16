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

from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm

from pai.ag_emb.metrics._utils import _prepare_embeddings
from pai.ag_emb.metrics.ranking import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    reciprocal_rank,
)


@dataclass
class ImageItem:
    """Metadata for a single image used in retrieval evaluation.

    Parameters
    ----------
    image_id : str
        Lowercase filename (basename) of the image, used as the canonical ID.
    explicit_positive_ids : frozenset[str]
        Lowercase basenames of images explicitly marked as similar to this
        item (e.g. all other members of the same metadata group).
    class_name : str | None
        Class label from metadata, if available.
    attributes : dict[str, str]
        Optional key-value attributes shared by this item's metadata group,
        e.g. ``{"growth_stage": "medium", "camera": "anafi"}``.
    """

    image_id: str
    explicit_positive_ids: frozenset[str] = field(default_factory=frozenset)
    class_name: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)


def relevance_grade(query: ImageItem, candidate: ImageItem) -> int:
    """Compute a graded relevance score between a query item and a candidate.

    Parameters
    ----------
    query : ImageItem
        The query image.
    candidate : ImageItem
        The candidate image being ranked.

    Returns
    -------
    int
        Relevance grade:

        * ``0`` — self, or no semantic relationship.
        * ``1`` — same class, but attributes absent or not fully matching.
        * ``2`` — same class and all of the query's attributes match the candidate.
        * ``3`` — explicit positive (same metadata group).
    """
    if candidate.image_id == query.image_id:
        return 0
    if candidate.image_id in query.explicit_positive_ids:
        return 3
    if query.class_name and candidate.class_name and query.class_name == candidate.class_name:
        if query.attributes and all(candidate.attributes.get(k) == v for k, v in query.attributes.items()):
            return 2
        return 1
    return 0


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
        raise ValueError(f"embeddings and labels must have the same length. Got {n} and {len(lbl)}.")

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

        for i in tqdm(range(n), desc=f"nDCG@{k}", leave=False, unit="item"):
            ranked_relevances = [1 if lbl[idx] == lbl[i] else 0 for idx in indices[i]]
            n_relevant = int(np.sum(lbl == lbl[i])) - 1  # exclude self
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

        for i in tqdm(range(n), desc=f"MAP@{k}", leave=False, unit="item"):
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

        for i in tqdm(range(n), desc=f"confusion@{k}", leave=False, unit="item"):
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

        for i in tqdm(range(n), desc=f"purity@{k}", leave=False, unit="item"):
            neighbor_labels = lbl[indices[i]]
            per_item[i] = float(np.sum(neighbor_labels == lbl[i])) / k

        result[k] = {
            "mean": float(np.mean(per_item)),
            "std": float(np.std(per_item)),
            "per_item": per_item,
        }
    return result


def knn_label_mrr_at_k(
    neighbors_by_k: dict,
    labels_arr: np.ndarray,
) -> dict:
    """Mean Reciprocal Rank at K using class labels as ground truth.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    labels_arr : np.ndarray
        Class label per item, aligned with the embedding matrix.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    n = len(labels_arr)
    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]
        per_item = np.empty(n, dtype=np.float64)
        for i in range(n):
            ranked_labels = [labels_arr[int(idx)] for idx in indices[i][:k]]
            rr = 0.0
            for rank, lbl in enumerate(ranked_labels, start=1):
                if lbl == labels_arr[i]:
                    rr = 1.0 / rank
                    break
            per_item[i] = rr
        result[k] = {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}
    return result


def knn_label_r_precision(
    neighbors_by_k: dict,
    labels_arr: np.ndarray,
) -> dict:
    """R-Precision using class labels: R equals the number of same-class items.

    Uses the largest available K as the candidate pool.  When the number of
    same-class items exceeds the pool size the metric is conservative
    (denominator remains R).

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    labels_arr : np.ndarray
        Class label per item, aligned with the embedding matrix.

    Returns
    -------
    dict
        Flat stats (not nested by K): ``mean``, ``std``, ``per_item``.
    """
    n = len(labels_arr)
    max_k = max(neighbors_by_k.keys())
    indices = neighbors_by_k[max_k]["indices"]
    per_item = np.empty(n, dtype=np.float64)
    for i in range(n):
        r = int(np.sum(labels_arr == labels_arr[i])) - 1
        if r <= 0:
            per_item[i] = 0.0
            continue
        ranked = [labels_arr[int(idx)] for idx in indices[i]]
        hits = sum(1 for lbl in ranked[:r] if lbl == labels_arr[i])
        per_item[i] = hits / float(r)
    return {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}


def knn_per_attribute_ndcg_at_k(
    neighbors_by_k: dict,
    items: list[ImageItem],
) -> dict:
    """NDCG at K computed independently for each attribute key in the metadata.

    For each unique attribute key found across all items, this function builds
    an attribute-value label array and computes a binary nDCG score (same
    attribute value = relevant) using the existing label-based nDCG function.

    The result is keyed by attribute name, making the output automatically
    expand when more attributes are added to the metadata.  Adding
    ``growth_stage`` yields one metric; adding ``camera`` yields a second;
    and so on.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order as the
        embedding matrix.

    Returns
    -------
    dict
        ``{attribute_key: {k: {mean, std, per_item}}}``
        Items whose attribute value is missing (empty string sentinel) are
        still included in the neighbor computation but will never share a
        label with any real item, so their per-item score will be 0.
    """
    all_keys: list[str] = sorted({k for item in items for k in item.attributes})
    if not all_keys:
        return {}

    result: dict = {}
    for key in all_keys:
        labels = np.array([item.attributes.get(key, "") for item in items])
        result[key] = knn_label_ndcg_at_k(neighbors_by_k, labels)
    return result


def knn_metadata_precision_at_k(
    neighbors_by_k: dict,
    items: list[ImageItem],
) -> dict:
    """Precision at K using explicit-positive metadata groups as ground truth.

    For each query item the relevant set is its ``explicit_positive_ids``
    frozenset.  Items with no explicit positives contribute ``0.0`` to the mean.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order as the
        embedding matrix passed to ``top_k_neighbors``.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]
        per_item = np.empty(n, dtype=np.float64)
        for i in range(n):
            ranked_ids = [items[int(idx)].image_id for idx in indices[i]]
            score = precision_at_k(ranked_ids, set(items[i].explicit_positive_ids), k)
            per_item[i] = score if score is not None else 0.0
        result[k] = {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}
    return result


def knn_metadata_map_at_k(
    neighbors_by_k: dict,
    items: list[ImageItem],
) -> dict:
    """MAP at K using explicit-positive metadata groups as ground truth.

    For each query item the relevant set is its ``explicit_positive_ids``
    frozenset.  Items with no explicit positives contribute ``0.0`` to the mean.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order as the
        embedding matrix passed to ``top_k_neighbors``.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]
        per_item = np.empty(n, dtype=np.float64)
        for i in range(n):
            ranked_ids = [items[int(idx)].image_id for idx in indices[i]]
            score = average_precision_at_k(ranked_ids, set(items[i].explicit_positive_ids), k)
            per_item[i] = score if score is not None else 0.0
        result[k] = {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}
    return result


def knn_metadata_mrr_at_k(
    neighbors_by_k: dict,
    items: list[ImageItem],
) -> dict:
    """Mean Reciprocal Rank at K using explicit-positive metadata as ground truth.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]
        per_item = np.empty(n, dtype=np.float64)
        for i in range(n):
            ranked_ids = [items[int(idx)].image_id for idx in indices[i][:k]]
            score = reciprocal_rank(ranked_ids, set(items[i].explicit_positive_ids))
            per_item[i] = score if score is not None else 0.0
        result[k] = {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}
    return result


def knn_metadata_ndcg_at_k(
    neighbors_by_k: dict,
    items: list[ImageItem],
) -> dict:
    """NDCG at K using graded relevance derived from image metadata.

    Relevance is graded 0-3 by :func:`relevance_grade`:

    * ``3`` — explicit positive (same metadata group)
    * ``2`` — same crop and same growth stage
    * ``1`` — same crop, different or unknown growth stage
    * ``0`` — no relationship

    The ideal DCG is computed against all other items in the corpus.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order as the
        embedding matrix passed to ``top_k_neighbors``.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    result = {}

    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        per_item = np.empty(n, dtype=np.float64)

        for i in tqdm(range(n), desc=f"metadata_nDCG@{k}", leave=False, unit="item"):
            query = items[i]
            ranked_relevances = [relevance_grade(query, items[int(idx)]) for idx in indices[i]]
            all_relevances = [relevance_grade(query, items[j]) for j in range(n) if j != i]
            score = ndcg_at_k(ranked_relevances, all_relevances, k)
            per_item[i] = score if score is not None else 0.0

        result[k] = {
            "mean": float(np.mean(per_item)),
            "std": float(np.std(per_item)),
            "per_item": per_item,
        }
    return result


def knn_metadata_r_precision(
    neighbors_by_k: dict,
    items: list[ImageItem],
) -> dict:
    """R-Precision using explicit-positive metadata groups as ground truth.

    R equals the number of explicit positives for each query item.  Uses
    the largest available K as the candidate pool.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~pai.ag_emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order.

    Returns
    -------
    dict
        Flat stats (not nested by K): ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    max_k = max(neighbors_by_k.keys())
    indices = neighbors_by_k[max_k]["indices"]
    per_item = np.empty(n, dtype=np.float64)
    for i in range(n):
        ranked_ids = [items[int(idx)].image_id for idx in indices[i]]
        score = r_precision(ranked_ids, set(items[i].explicit_positive_ids))
        per_item[i] = score if score is not None else 0.0
    return {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}
