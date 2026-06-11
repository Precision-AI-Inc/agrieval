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

"""Shared internal helpers for the metrics package. Not part of the public API."""

from __future__ import annotations

import itertools

import numpy as np

from pai.ag_emb.metrics.ranking import l2_normalize


def _validate_embeddings(embeddings: object, name: str = "embeddings") -> np.ndarray:
    """Convert ``embeddings`` to a validated float32 numpy array.

    Parameters
    ----------
    embeddings : array-like
        Input to convert.  Must be 2-D, non-empty, and contain only finite
        values.
    name : str
        Variable name used in error messages.

    Returns
    -------
    np.ndarray
        Shape ``[N, D]`` float32 array.

    Raises
    ------
    ValueError
        If the result is not 2-D, is empty, or contains NaN / inf.
    """
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf values.")
    return arr


def _auto_batch_size(n: int, target_bytes: int = 1 << 30) -> int:
    """Compute a row-batch size so that one batch x N similarity matrix fits in memory.

    The heuristic targets ``target_bytes`` of float32 storage for a
    ``[batch, N]`` similarity matrix, i.e. ``batch = target_bytes / (N * 4)``.

    Parameters
    ----------
    n : int
        Total number of items (columns of the similarity matrix).
    target_bytes : int
        Memory budget in bytes.  Defaults to 1 GiB.

    Returns
    -------
    int
        Batch size, always at least 1.
    """
    return max(1, target_bytes // max(1, n * 4))


def _percentile_stats(values: np.ndarray) -> dict:
    """Build a full descriptive-statistics dict for a 1-D array.

    Parameters
    ----------
    values : np.ndarray
        1-D array of numeric values.

    Returns
    -------
    dict
        Keys: ``count``, ``mean``, ``std``, ``min``, ``max``, ``p01``,
        ``p05``, ``p25``, ``p50``, ``p75``, ``p95``, ``p99``.  All scalar
        values are ``None`` when ``values`` is empty.
    """
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
        }
    pcts = np.percentile(values, [1, 5, 25, 50, 75, 95, 99]).tolist()
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": pcts[0],
        "p05": pcts[1],
        "p25": pcts[2],
        "p50": pcts[3],
        "p75": pcts[4],
        "p95": pcts[5],
        "p99": pcts[6],
    }


def _neighbor_stats(per_item: np.ndarray) -> dict:
    """Build a compact per-item statistics dict for nearest-neighbor metrics.

    Parameters
    ----------
    per_item : np.ndarray
        1-D array of per-item scalar values (e.g., mean neighbor similarity).

    Returns
    -------
    dict
        Keys: ``per_item``, ``mean``, ``std``, ``p05``, ``p50``, ``p95``.
        Scalar statistics are ``None`` when the array is empty.
    """
    if len(per_item) == 0:
        return {
            "per_item": per_item,
            "mean": None,
            "std": None,
            "p05": None,
            "p50": None,
            "p95": None,
        }
    pcts = np.percentile(per_item, [5, 50, 95]).tolist()
    return {
        "per_item": per_item,
        "mean": float(np.mean(per_item)),
        "std": float(np.std(per_item)),
        "p05": pcts[0],
        "p50": pcts[1],
        "p95": pcts[2],
    }


def _prepare_embeddings(embeddings: object, normalize: bool, name: str = "embeddings") -> np.ndarray:
    """Validate and optionally L2-normalize an embedding array.

    Parameters
    ----------
    embeddings : array-like
        Input embeddings to prepare.
    normalize : bool
        When ``True``, L2-normalize each row before returning.
    name : str
        Variable name used in validation error messages.

    Returns
    -------
    np.ndarray
        Shape ``[N, D]`` float32 array, optionally row-normalized.
    """
    emb = _validate_embeddings(embeddings, name)
    return l2_normalize(emb) if normalize else emb


def _get_pair_similarities(
    embeddings: np.ndarray,
    sample_pairs: int | None,
    random_seed: int,
) -> tuple[np.ndarray, int]:
    """Compute pairwise cosine similarities, either exhaustively or by sampling.

    ``embeddings`` must already be L2-normalized by the caller.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``[N, D]`` row-normalized float32 embedding matrix.
    sample_pairs : int | None
        ``None`` computes all ``N*(N-1)/2`` unique upper-triangle pairs.
        An integer randomly samples that many (i, j) pairs with ``i != j``.
    random_seed : int
        Seed for the random number generator used when sampling.

    Returns
    -------
    tuple[np.ndarray, int]
        A 2-tuple of:

        * **sims** — float32 array of sampled (or all) pairwise similarities.
        * **total_unique** — exact total number of unique pairs in the full
          dataset, regardless of sampling.
    """
    n = len(embeddings)
    total_unique = n * (n - 1) // 2

    if sample_pairs is None:
        sim_matrix = embeddings @ embeddings.T
        i_idx, j_idx = np.triu_indices(n, k=1)
        sims = sim_matrix[i_idx, j_idx]
    else:
        rng = np.random.default_rng(random_seed)
        pairs_i = rng.integers(0, n, size=sample_pairs)
        pairs_j = rng.integers(0, n, size=sample_pairs)
        mask = pairs_i != pairs_j
        pairs_i, pairs_j = pairs_i[mask], pairs_j[mask]
        sims = np.einsum("ij,ij->i", embeddings[pairs_i], embeddings[pairs_j])

    return sims.astype(np.float32), total_unique


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Assign average ranks to a 1-D array, handling ties correctly.

    Tied values receive the mean of the ranks they would occupy, matching the
    behaviour of ``scipy.stats.rankdata`` with ``method='average'``.  Used as
    a fallback for Spearman correlation when scipy is not installed.

    Parameters
    ----------
    arr : np.ndarray
        1-D array of values to rank.

    Returns
    -------
    np.ndarray
        Float64 array of average ranks (1-based) with the same length as
        ``arr``.
    """
    n = len(arr)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranked_vals = arr[order]
    changes = np.concatenate([[0], np.where(ranked_vals[1:] != ranked_vals[:-1])[0] + 1, [n]])
    for s, e in itertools.pairwise(changes):
        ranks[order[s:e]] = (s + e - 1) / 2.0 + 1.0
    return ranks
