# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Optional label-aware metrics for supervised embedding evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from precisionai.agrieval.emb.metrics._utils import _prepare_embeddings


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

        * ``0`` — self, or no semantic relationship (different L1 class).
        * ``1`` — different L1 class but at least one ``plants`` value overlaps.
        * ``2`` — same L1 class (same coarse crop/weed category).
        * ``3`` — explicit positive (same L2 metadata group).
    """
    if candidate.image_id == query.image_id:
        return 0
    if candidate.image_id in query.explicit_positive_ids:
        return 3
    if query.class_name and candidate.class_name and query.class_name == candidate.class_name:
        return 2
    q_ci = query.attributes.get("plants", "")
    c_ci = candidate.attributes.get("plants", "")
    if q_ci and c_ci:
        q_set = set(q_ci.split(",")) - {""}
        c_set = set(c_ci.split(",")) - {""}
        if q_set & c_set:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Precomputation helpers (private)
# ---------------------------------------------------------------------------


def _build_explicit_positive_mask(items: list[ImageItem]) -> np.ndarray:
    """Build a boolean mask where ``mask[i, j]`` is True when j is an explicit positive of i.

    Parameters
    ----------
    items : list[ImageItem]
        Items in embedding-row order.

    Returns
    -------
    np.ndarray
        Shape ``[N, N]`` bool array.
    """
    n = len(items)
    id_to_idx: dict[str, int] = {item.image_id: i for i, item in enumerate(items)}
    mask = np.zeros((n, n), dtype=bool)
    for i, item in enumerate(items):
        for pos_id in item.explicit_positive_ids:
            j = id_to_idx.get(pos_id)
            if j is not None:
                mask[i, j] = True
    return mask


def _encode_class_names(items: list[ImageItem]) -> tuple[list[int], np.ndarray]:
    """Encode class names as dense integers and mark items with a class label."""
    class_name_to_int: dict[str, int] = {}
    class_ints: list[int] = []
    for item in items:
        cn = item.class_name or ""
        if cn not in class_name_to_int:
            class_name_to_int[cn] = len(class_name_to_int)
        class_ints.append(class_name_to_int[cn])
    has_class = np.array([bool(item.class_name) for item in items])
    return class_ints, has_class


def _build_class_instance_sets(items: list[ImageItem]) -> list[set[str]]:
    """Collect per-item ``plants`` values as sets."""
    ci_sets: list[set[str]] = []
    for item in items:
        ci_val = item.attributes.get("plants", "")
        ci_sets.append(set(ci_val.split(",")) - {""} if ci_val else set())
    return ci_sets


def _assign_class_instance_overlap_grades(
    grades: np.ndarray,
    class_ints: list[int],
    has_class: np.ndarray,
    ci_sets: list[set[str]],
) -> None:
    """Assign grade 1 where items differ by L1 class but share class instances.

    Uses an inverted index over plant values to avoid the O(N²) nested loop —
    only item pairs that share at least one plant value are ever visited.
    """
    class_int_arr = np.array(class_ints, dtype=np.int32)
    plant_to_items: dict[str, list[int]] = defaultdict(list)
    for i, ci_set in enumerate(ci_sets):
        for plant in ci_set:
            plant_to_items[plant].append(i)

    for items_list in plant_to_items.values():
        if len(items_list) < 2:
            continue
        arr = np.array(items_list, dtype=np.intp)
        ci = class_int_arr[arr]
        ii, jj = np.where(ci[:, None] != ci[None, :])
        grades[arr[ii], arr[jj]] = 1


def _build_grade_matrix(items: list[ImageItem]) -> np.ndarray:
    """Precompute the full ``[N, N]`` graded relevance matrix for all item pairs.

    Grades follow :func:`relevance_grade`:

    * ``3`` — explicit positive (same L2 metadata group)
    * ``2`` — same L1 class
    * ``1`` — different L1 class but overlapping ``plants`` attribute values
    * ``0`` — no relationship or self

    Building the matrix once and reusing it across all K values avoids
    O(N^2 x |K|) redundant Python calls to :func:`relevance_grade`.

    Parameters
    ----------
    items : list[ImageItem]
        Items in embedding-row order.

    Returns
    -------
    np.ndarray
        Shape ``[N, N]`` int8 array with the diagonal zeroed.
    """
    n = len(items)
    grades = np.zeros((n, n), dtype=np.int8)

    class_ints, has_class = _encode_class_names(items)
    class_int_arr = np.array(class_ints, dtype=np.int32)
    ci_sets = _build_class_instance_sets(items)
    _assign_class_instance_overlap_grades(grades, class_ints, has_class, ci_sets)

    # Grade 2: same L1 class (both must have non-empty class names).
    both_have_class = has_class[:, None] & has_class[None, :]
    same_class = (class_int_arr[:, None] == class_int_arr[None, :]) & both_have_class
    grades[same_class] = 2

    # Grade 3: explicit positives (overrides grades 1 and 2).
    id_to_idx: dict[str, int] = {item.image_id: i for i, item in enumerate(items)}
    for i, item in enumerate(items):
        for pos_id in item.explicit_positive_ids:
            j = id_to_idx.get(pos_id)
            if j is not None:
                grades[i, j] = 3

    np.fill_diagonal(grades, 0)
    return grades


# ---------------------------------------------------------------------------
# Global similarity gap
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Label-based KNN metrics (vectorised)
# ---------------------------------------------------------------------------


def knn_label_purity_at_k(
    neighbors_by_k: dict,
    labels: object,
) -> dict:
    """Fraction of each item's K nearest neighbors that share its label.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
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
        neighbor_labels = lbl[indices]  # [N, k]
        per_item = (neighbor_labels == lbl[:, None]).mean(axis=1).astype(np.float32)
        result[k] = {
            "mean": float(np.mean(per_item)),
            "std": float(np.std(per_item)),
            "per_item": per_item,
        }
    return result


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
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    labels : array-like
        Shape ``[N]``. Integer or string class labels.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    lbl = np.asarray(labels)
    _, inverse, counts = np.unique(lbl, return_inverse=True, return_counts=True)
    n_rel = counts[inverse] - 1  # same-class count minus self, shape [N]

    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        relevances = (lbl[indices] == lbl[:, None]).astype(np.float64)  # [N, k] binary

        # Discount positions: 1/log2(2), 1/log2(3), ..., 1/log2(k+1)
        discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))  # [k]
        dcg = (relevances * discounts).sum(axis=1)  # [N]

        # IDCG: cumulative discount sum up to min(n_rel, k) positions
        cumsum = np.cumsum(discounts)  # [k]
        ideal_at = np.minimum(n_rel, k)  # [N]
        # When ideal_at == 0 the where-mask selects 0.0, so the clip to 0 is safe.
        idcg = np.where(ideal_at > 0, cumsum[np.maximum(ideal_at - 1, 0)], 0.0)  # [N]

        per_item = np.divide(dcg, idcg, out=np.zeros(len(lbl), dtype=np.float64), where=idcg > 0)
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
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    labels : array-like
        Shape ``[N]``. Integer or string class labels.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    lbl = np.asarray(labels)
    _, inverse, counts = np.unique(lbl, return_inverse=True, return_counts=True)
    n_rel = (counts[inverse] - 1).astype(np.float64)  # [N]

    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        relevances = (lbl[indices] == lbl[:, None]).astype(np.float64)  # [N, k]

        ranks = np.arange(1, k + 1, dtype=np.float64)  # [k]
        cumhits = np.cumsum(relevances, axis=1)  # [N, k]
        sum_prec = ((cumhits / ranks) * relevances).sum(axis=1)  # [N]

        per_item = np.divide(sum_prec, n_rel, out=np.zeros_like(sum_prec), where=n_rel > 0)
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
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
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
    label_indices = np.array([cls_to_idx[lb] for lb in lbl], dtype=np.int32)  # [N]

    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]

        # Expand queries and flatten neighbors for a single vectorised accumulation.
        query_cls = np.repeat(label_indices, k)  # [N*k]
        neighbor_cls = label_indices[indices].ravel()  # [N*k]
        matrix = np.zeros((n_cls, n_cls), dtype=np.float64)
        np.add.at(matrix, (query_cls, neighbor_cls), 1.0)

        row_counts = np.bincount(label_indices, minlength=n_cls).astype(np.float64)
        row_totals = k * row_counts  # total votes from each class
        matrix /= np.where(row_totals > 0, row_totals, 1.0)[:, None]

        result[k] = {
            c1: {c2: float(matrix[i, j]) for j, c2 in enumerate(unique_classes)} for i, c1 in enumerate(unique_classes)
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
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    labels_arr : np.ndarray
        Class label per item, aligned with the embedding matrix.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        neighbor_labels = labels_arr[indices[:, :k]]  # [N, k]
        hits = neighbor_labels == labels_arr[:, None]  # [N, k] bool

        # argmax returns the position of the first True; returns 0 for all-False rows
        # — guarded by has_hit below.
        first_hit_pos = np.argmax(hits, axis=1)  # [N]
        has_hit = hits.any(axis=1)  # [N]
        per_item = np.where(has_hit, 1.0 / (first_hit_pos + 1), 0.0)

        result[k] = {
            "mean": float(np.mean(per_item)),
            "std": float(np.std(per_item)),
            "per_item": per_item,
        }
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
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    labels_arr : np.ndarray
        Class label per item, aligned with the embedding matrix.

    Returns
    -------
    dict
        Flat stats (not nested by K): ``mean``, ``std``, ``per_item``.
    """
    max_k = max(neighbors_by_k.keys())
    indices = neighbors_by_k[max_k]["indices"]  # [N, max_k]

    _, inverse, counts = np.unique(labels_arr, return_inverse=True, return_counts=True)
    n_rel = (counts[inverse] - 1).astype(np.int64)  # [N] same-class count excluding self

    neighbor_labels = labels_arr[indices]  # [N, max_k]
    relevances = (neighbor_labels == labels_arr[:, None]).astype(np.float64)

    # Mask to only count hits within the first n_rel[i] positions.
    positions = np.arange(max_k)  # [max_k]
    r_mask = positions[None, :] < n_rel[:, None]  # [N, max_k]
    hits = (relevances * r_mask).sum(axis=1)  # [N]

    per_item = np.where(n_rel > 0, hits / np.maximum(n_rel, 1).astype(np.float64), 0.0)
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
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
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


# ---------------------------------------------------------------------------
# Metadata-based KNN metrics (vectorised with optional precomputed matrix)
# ---------------------------------------------------------------------------


def knn_metadata_precision_at_k(
    neighbors_by_k: dict,
    items: list[ImageItem],
    *,
    _positive_mask: np.ndarray | None = None,
) -> dict:
    """Precision at K using explicit-positive metadata groups as ground truth.

    For each query item the relevant set is its ``explicit_positive_ids``
    frozenset.  Items with no explicit positives contribute ``0.0`` to the mean.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order as the
        embedding matrix passed to ``top_k_neighbors``.
    _positive_mask : np.ndarray | None
        Precomputed ``[N, N]`` bool mask from :func:`_build_explicit_positive_mask`.
        Computed internally when not provided.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    if _positive_mask is None:
        _positive_mask = _build_explicit_positive_mask(items)

    row_idx = np.arange(n)[:, None]
    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        hits = _positive_mask[row_idx, indices].sum(axis=1).astype(np.float64)
        per_item = hits / k
        result[k] = {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}
    return result


def knn_metadata_map_at_k(
    neighbors_by_k: dict,
    items: list[ImageItem],
    *,
    _positive_mask: np.ndarray | None = None,
) -> dict:
    """MAP at K using explicit-positive metadata groups as ground truth.

    For each query item the relevant set is its ``explicit_positive_ids``
    frozenset.  Items with no explicit positives contribute ``0.0`` to the mean.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order as the
        embedding matrix passed to ``top_k_neighbors``.
    _positive_mask : np.ndarray | None
        Precomputed ``[N, N]`` bool mask from :func:`_build_explicit_positive_mask`.
        Computed internally when not provided.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    if _positive_mask is None:
        _positive_mask = _build_explicit_positive_mask(items)

    n_positives = _positive_mask.sum(axis=1).astype(np.float64)  # [N]
    row_idx = np.arange(n)[:, None]
    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        is_pos = _positive_mask[row_idx, indices].astype(np.float64)  # [N, k]
        ranks = np.arange(1, k + 1, dtype=np.float64)  # [k]
        sum_prec = ((np.cumsum(is_pos, axis=1) / ranks) * is_pos).sum(axis=1)  # [N]
        per_item = np.divide(sum_prec, n_positives, out=np.zeros_like(sum_prec), where=n_positives > 0)
        result[k] = {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}
    return result


def knn_metadata_mrr_at_k(
    neighbors_by_k: dict,
    items: list[ImageItem],
    *,
    _positive_mask: np.ndarray | None = None,
) -> dict:
    """Mean Reciprocal Rank at K using explicit-positive metadata as ground truth.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order.
    _positive_mask : np.ndarray | None
        Precomputed ``[N, N]`` bool mask from :func:`_build_explicit_positive_mask`.
        Computed internally when not provided.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    if _positive_mask is None:
        _positive_mask = _build_explicit_positive_mask(items)

    row_idx = np.arange(n)[:, None]
    result = {}
    for k, data in neighbors_by_k.items():
        indices = data["indices"][:, :k]  # [N, k]
        is_pos = _positive_mask[row_idx, indices]  # [N, k] bool
        first_hit = np.argmax(is_pos, axis=1)  # [N]; 0 when no hit — guarded below
        has_hit = is_pos.any(axis=1)
        per_item = np.where(has_hit, 1.0 / (first_hit + 1), 0.0)
        result[k] = {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}
    return result


def knn_metadata_ndcg_at_k(
    neighbors_by_k: dict,
    items: list[ImageItem],
    *,
    _grade_matrix: np.ndarray | None = None,
) -> dict:
    """NDCG at K using graded relevance derived from image metadata.

    Relevance is graded 0-3 by :func:`relevance_grade`:

    * ``3`` — explicit positive (same L2 metadata group)
    * ``2`` — same L1 class
    * ``1`` — different L1 class but overlapping ``plants`` attribute values
    * ``0`` — no relationship

    The ideal DCG is computed against all other items in the corpus.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order as the
        embedding matrix passed to ``top_k_neighbors``.
    _grade_matrix : np.ndarray | None
        Precomputed ``[N, N]`` int8 grade matrix from :func:`_build_grade_matrix`.
        Built internally when not provided.

    Returns
    -------
    dict
        Keyed by K. Each value: ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    if _grade_matrix is None:
        _grade_matrix = _build_grade_matrix(items)

    # Precompute the sorted-descending grade matrix once for IDCG reuse across k.
    sorted_grades = np.sort(_grade_matrix, axis=1)[:, ::-1]  # [N, N] descending

    result = {}
    row_idx = np.arange(n)[:, None]
    for k, data in neighbors_by_k.items():
        indices = data["indices"]  # [N, k]
        ranked_grades = _grade_matrix[row_idx, indices].astype(np.float64)  # [N, k]

        discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))  # [k]
        dcg = ((np.power(2.0, ranked_grades) - 1.0) * discounts).sum(axis=1)  # [N]

        ideal_grades = sorted_grades[:, :k].astype(np.float64)  # [N, k]
        idcg = ((np.power(2.0, ideal_grades) - 1.0) * discounts).sum(axis=1)  # [N]

        per_item = np.divide(dcg, idcg, out=np.zeros(n, dtype=np.float64), where=idcg > 0)
        result[k] = {
            "mean": float(np.mean(per_item)),
            "std": float(np.std(per_item)),
            "per_item": per_item,
        }
    return result


def knn_metadata_r_precision(
    neighbors_by_k: dict,
    items: list[ImageItem],
    *,
    _positive_mask: np.ndarray | None = None,
) -> dict:
    """R-Precision using explicit-positive metadata groups as ground truth.

    R equals the number of explicit positives for each query item.  Uses
    the largest available K as the candidate pool.

    Parameters
    ----------
    neighbors_by_k : dict
        Output of :func:`~precisionai.agrieval.emb.metrics.similarity.top_k_neighbors`.
    items : list[ImageItem]
        One :class:`ImageItem` per embedding row, in the same order.
    _positive_mask : np.ndarray | None
        Precomputed ``[N, N]`` bool mask from :func:`_build_explicit_positive_mask`.
        Computed internally when not provided.

    Returns
    -------
    dict
        Flat stats (not nested by K): ``mean``, ``std``, ``per_item``.
    """
    n = len(items)
    if _positive_mask is None:
        _positive_mask = _build_explicit_positive_mask(items)

    n_positives = _positive_mask.sum(axis=1).astype(np.int64)  # [N]
    max_k = max(neighbors_by_k.keys())
    indices = neighbors_by_k[max_k]["indices"]  # [N, max_k]
    row_idx = np.arange(n)[:, None]

    is_pos = _positive_mask[row_idx, indices].astype(np.float64)  # [N, max_k]
    positions = np.arange(max_k)  # [max_k]
    r_mask = positions[None, :] < n_positives[:, None]  # [N, max_k]
    hits = (is_pos * r_mask).sum(axis=1)  # [N]

    per_item = np.where(n_positives > 0, hits / np.maximum(n_positives, 1).astype(np.float64), 0.0)
    return {"mean": float(np.mean(per_item)), "std": float(np.std(per_item)), "per_item": per_item}
