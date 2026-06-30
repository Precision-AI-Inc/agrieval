# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Core cosine similarity primitives and pairwise statistics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from precisionai.agrieval.emb.metrics._utils import (
    _auto_batch_size,
    _get_pair_similarities,
    _percentile_stats,
    _prepare_embeddings,
)
from precisionai.agrieval.emb.metrics.ranking import normalize_ks


def cosine_similarity_matrix(
    embeddings: object,
    *,
    normalize: bool = True,
    exclude_self: bool = True,
) -> np.ndarray:
    """Compute the full pairwise cosine similarity matrix.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``. Suitable for small/medium N only.
    normalize : bool
        L2-normalize rows before computing similarity.
    exclude_self : bool
        Set diagonal to ``NaN`` when True.

    Returns
    -------
    np.ndarray
        Shape ``[N, N]`` float32 cosine similarity matrix.
    """
    emb = _prepare_embeddings(embeddings, normalize)
    sim = (emb @ emb.T).astype(np.float32)
    if exclude_self:
        np.fill_diagonal(sim, np.nan)
    return sim


def top_k_neighbors(
    embeddings: object,
    ks: Sequence[int] = (5, 10, 50, 100),
    *,
    normalize: bool = True,
    exclude_self: bool = True,
    batch_size: int | None = None,
) -> dict:
    """For each vector return its top-K nearest neighbors by cosine similarity.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    ks : sequence of int
        Requested K values. Values larger than the candidate pool are clamped.
    normalize : bool
        L2-normalize rows before computing similarity.
    exclude_self : bool
        Exclude each item from its own neighbor list.
    batch_size : int | None
        Rows per batch. Defaults to an auto-tuned value targeting ~1 GB memory.

    Returns
    -------
    dict
        Keys are effective (clamped) K values. Each value is
        ``{"indices": ndarray [N, k], "scores": ndarray [N, k]}``.
    """
    emb = _prepare_embeddings(embeddings, normalize)
    n = emb.shape[0]

    pool_size = (n - 1) if exclude_self else n
    if pool_size < 1:
        raise ValueError("Need at least 2 embeddings when exclude_self=True, else at least 1.")

    effective_ks = normalize_ks(list(ks), pool_size)
    max_k = effective_ks[-1]

    if batch_size is None:
        batch_size = _auto_batch_size(n)

    all_indices = np.empty((n, max_k), dtype=np.int64)
    all_scores = np.empty((n, max_k), dtype=np.float32)

    with tqdm(total=n, desc="k-NN graph", leave=False, unit="item") as pbar:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            b = end - start
            sims = (emb[start:end] @ emb.T).astype(np.float32)  # [B, N]

            if exclude_self:
                sims[np.arange(b), np.arange(start, end)] = -np.inf

            b_range = np.arange(b)[:, None]
            if max_k >= sims.shape[1]:
                sorted_idx = np.argsort(-sims, axis=1)
                sorted_scores = np.take_along_axis(sims, sorted_idx, axis=1)
            else:
                part_idx = np.argpartition(sims, -max_k, axis=1)[:, -max_k:]
                part_scores = sims[b_range, part_idx]
                order = np.argsort(-part_scores, axis=1)
                sorted_idx = part_idx[b_range, order]
                sorted_scores = part_scores[b_range, order]

            all_indices[start:end] = sorted_idx[:, :max_k]
            all_scores[start:end] = sorted_scores[:, :max_k]
            pbar.update(b)

    return {
        k: {
            "indices": all_indices[:, :k].copy(),
            "scores": all_scores[:, :k].copy(),
        }
        for k in effective_ks
    }


def pairwise_similarity_stats(
    embeddings: object,
    *,
    normalize: bool = True,
    sample_pairs: int | None = None,
    random_seed: int = 42,
    exclude_self: bool = True,
) -> dict:
    """Summarize the global distribution of pairwise cosine similarities.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    normalize : bool
        L2-normalize rows before computing similarity.
    sample_pairs : int | None
        ``None`` computes all unique non-self pairs (suitable for small N).
        An integer randomly samples that many pairs.
    random_seed : int
        Seed for reproducible sampling.
    exclude_self : bool
        Exclude self-pairs (i == j).

    Returns
    -------
    dict
        Keys: ``count``, ``mean``, ``std``, ``min``, ``max``, ``p01``-``p99``.
    """
    emb = _prepare_embeddings(embeddings, normalize)
    sims, _ = _get_pair_similarities(emb, sample_pairs, random_seed)
    return _percentile_stats(sims)


def similarity_threshold_counts(
    embeddings: object,
    thresholds: Sequence[float] = (0.80, 0.85, 0.90, 0.95, 0.98, 0.99),
    *,
    normalize: bool = True,
    sample_pairs: int | None = None,
    random_seed: int = 42,
    exclude_self: bool = True,
) -> list[dict]:
    """Count pairs whose cosine similarity exceeds each threshold.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    thresholds : sequence of float
        Cosine similarity cutoffs to count against.
    normalize : bool
        L2-normalize rows before computing similarity.
    sample_pairs : int | None
        ``None`` for exact counts over all unique pairs. An integer samples
        that many pairs and scales the fraction to the full dataset.
    random_seed : int
        Seed for reproducible sampling.
    exclude_self : bool
        Exclude self-pairs.

    Returns
    -------
    list[dict]
        One entry per threshold with keys ``threshold``, ``pair_count``,
        ``pair_fraction``, ``estimated_total_pairs``.
    """
    emb = _prepare_embeddings(embeddings, normalize)
    sims, total_unique = _get_pair_similarities(emb, sample_pairs, random_seed)
    n_evaluated = len(sims)

    result = []
    for t in sorted(thresholds):
        count = int(np.sum(sims >= t))
        fraction = count / n_evaluated if n_evaluated > 0 else 0.0
        result.append(
            {
                "threshold": float(t),
                "pair_count": count,
                "pair_fraction": float(fraction),
                "estimated_total_pairs": total_unique,
            }
        )
    return result
