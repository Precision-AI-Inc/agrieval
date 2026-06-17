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

"""Ranking and retrieval metrics for embedding evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def normalize_ks(requested_ks: Sequence[int], candidate_pool_size: int) -> list[int]:
    """Clamp and deduplicate K values against the available candidate pool.

    Parameters
    ----------
    requested_ks : Sequence[int]
        Requested K values. All must be positive.
    candidate_pool_size : int
        Number of candidates available (excluding the query itself).

    Returns
    -------
    list[int]
        Sorted, deduplicated K values clamped to ``candidate_pool_size``.

    Raises
    ------
    ValueError
        If ``candidate_pool_size`` is less than 1, or any K value is not positive.
    """
    if candidate_pool_size < 1:
        raise ValueError("At least one valid candidate is required for retrieval evaluation.")
    normalized: list[int] = []
    seen: set[int] = set()
    for value in requested_ks:
        if value <= 0:
            raise ValueError(f"All K values must be positive. Received: {value}")
        clamped = min(value, candidate_pool_size)
        if clamped not in seen:
            normalized.append(clamped)
            seen.add(clamped)
    return sorted(normalized)


def safe_mean(values: Iterable[float | None]) -> float | None:
    """Return the mean of non-None values, or None if all values are None.

    Parameters
    ----------
    values : Iterable[float | None]
        Values to average, with None entries ignored.

    Returns
    -------
    float | None
        Mean of the non-None values, or None if no valid values exist.
    """
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def precision_at_k(ranked_ids: Sequence[T], relevant_ids: set[T], k: int) -> float | None:
    """Fraction of the top-K retrieved items that are relevant.

    Parameters
    ----------
    ranked_ids : Sequence[T]
        Retrieved item IDs in ranked order (most similar first).
    relevant_ids : set[T]
        Ground-truth relevant item IDs for the query.
    k : int
        Cutoff rank.

    Returns
    -------
    float | None
        Precision@K, or None if ``relevant_ids`` is empty.
    """
    if not relevant_ids:
        return None
    return sum(1 for candidate_id in ranked_ids[:k] if candidate_id in relevant_ids) / float(k)


def recall_at_k(ranked_ids: Sequence[T], relevant_ids: set[T], k: int) -> float | None:
    """Fraction of all relevant items that appear in the top-K results.

    Parameters
    ----------
    ranked_ids : Sequence[T]
        Retrieved item IDs in ranked order (most similar first).
    relevant_ids : set[T]
        Ground-truth relevant item IDs for the query.
    k : int
        Cutoff rank.

    Returns
    -------
    float | None
        Recall@K, or None if ``relevant_ids`` is empty.
    """
    if not relevant_ids:
        return None
    return sum(1 for candidate_id in ranked_ids[:k] if candidate_id in relevant_ids) / float(len(relevant_ids))


def average_precision_at_k(
    ranked_ids: Sequence[T],
    relevant_ids: set[T],
    k: int,
) -> float | None:
    """Average precision over the top-K results.

    Parameters
    ----------
    ranked_ids : Sequence[T]
        Retrieved item IDs in ranked order (most similar first).
    relevant_ids : set[T]
        Ground-truth relevant item IDs for the query.
    k : int
        Cutoff rank.

    Returns
    -------
    float | None
        AP@K, or None if ``relevant_ids`` is empty. Returns 0.0 if no hits in top-K.
    """
    if not relevant_ids:
        return None
    hits = 0
    precisions: list[float] = []
    for rank, candidate_id in enumerate(ranked_ids[:k], start=1):
        if candidate_id in relevant_ids:
            hits += 1
            precisions.append(hits / rank)
    if not precisions:
        return 0.0
    return float(sum(precisions) / len(relevant_ids))


def r_precision(ranked_ids: Sequence[T], relevant_ids: set[T]) -> float | None:
    """Precision at R where R equals the number of relevant items.

    Parameters
    ----------
    ranked_ids : Sequence[T]
        Retrieved item IDs in ranked order (most similar first).
    relevant_ids : set[T]
        Ground-truth relevant item IDs for the query.

    Returns
    -------
    float | None
        R-Precision, or None if ``relevant_ids`` is empty.
        If fewer than R candidates are available the available prefix is used
        and the denominator remains R (conservative estimate).
    """
    r = len(relevant_ids)
    if r == 0:
        return None
    hits = sum(1 for cid in ranked_ids[:r] if cid in relevant_ids)
    return hits / float(r)


def reciprocal_rank(ranked_ids: Sequence[T], relevant_ids: set[T]) -> float | None:
    """Reciprocal rank of the first relevant result.

    Parameters
    ----------
    ranked_ids : Sequence[T]
        Retrieved item IDs in ranked order (most similar first).
    relevant_ids : set[T]
        Ground-truth relevant item IDs for the query.

    Returns
    -------
    float | None
        ``1 / rank`` of the first hit, ``0.0`` if no hit is found in
        ``ranked_ids``, or ``None`` if ``relevant_ids`` is empty.
    """
    if not relevant_ids:
        return None
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """Discounted Cumulative Gain at rank K.

    Parameters
    ----------
    relevances : Sequence[int]
        Graded relevance scores in ranked order (higher is more relevant).
    k : int
        Cutoff rank.

    Returns
    -------
    float
        DCG@K score.
    """
    total = 0.0
    for rank, relevance in enumerate(relevances[:k], start=1):
        total += (2**relevance - 1) / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_relevances: Sequence[int], ideal_relevances: Sequence[int], k: int) -> float | None:
    """Compute normalized discounted cumulative gain at rank K.

    Parameters
    ----------
    ranked_relevances : Sequence[int]
        Graded relevance scores in the order returned by the model.
    ideal_relevances : Sequence[int]
        All available relevance scores for the query (used to compute ideal DCG).
    k : int
        Cutoff rank.

    Returns
    -------
    float | None
        NDCG@K, or None if ``ideal_relevances`` is empty or ideal DCG is zero.
    """
    if not ideal_relevances:
        return None
    ideal = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if ideal == 0:
        return None
    return dcg_at_k(ranked_relevances, k) / ideal


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row of an embedding matrix.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(N, D)`` embedding matrix.

    Returns
    -------
    np.ndarray
        Row-normalized embeddings of the same shape. Zero vectors are handled
        by clipping norms to ``1e-12`` to avoid division by zero.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms
