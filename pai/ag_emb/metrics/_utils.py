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

import numpy as np

from pai.ag_emb.metrics.ranking import l2_normalize


def _validate_embeddings(embeddings: object, name: str = "embeddings") -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf values.")
    return arr


def _auto_batch_size(n: int, target_bytes: int = 1 << 30) -> int:
    """Rows per batch that keep a batch @ full-matrix product under target_bytes."""
    return max(1, target_bytes // max(1, n * 4))


def _percentile_stats(values: np.ndarray) -> dict:
    """Full descriptive stats dict (count, mean, std, min, max, p01–p99)."""
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None, "std": None, "min": None, "max": None,
            "p01": None, "p05": None, "p25": None, "p50": None,
            "p75": None, "p95": None, "p99": None,
        }
    pcts = np.percentile(values, [1, 5, 25, 50, 75, 95, 99]).tolist()
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p01": pcts[0], "p05": pcts[1], "p25": pcts[2], "p50": pcts[3],
        "p75": pcts[4], "p95": pcts[5], "p99": pcts[6],
    }


def _neighbor_stats(per_item: np.ndarray) -> dict:
    """Stats dict used by mean_top_k_similarity and knn_radius_at_k."""
    if len(per_item) == 0:
        return {
            "per_item": per_item,
            "mean": None, "std": None, "p05": None, "p50": None, "p95": None,
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
    emb = _validate_embeddings(embeddings, name)
    return l2_normalize(emb) if normalize else emb


def _get_pair_similarities(
    embeddings: np.ndarray,
    sample_pairs: int | None,
    random_seed: int,
) -> tuple[np.ndarray, int]:
    """Return (sampled_or_all_similarities, total_unique_pairs_in_dataset).

    embeddings must already be normalized by the caller.
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
    """Vectorized average-tie ranking (Spearman-safe)."""
    n = len(arr)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranked_vals = arr[order]
    changes = np.concatenate([[0], np.where(ranked_vals[1:] != ranked_vals[:-1])[0] + 1, [n]])
    for s, e in zip(changes[:-1], changes[1:]):
        ranks[order[s:e]] = (s + e - 1) / 2.0 + 1.0
    return ranks
