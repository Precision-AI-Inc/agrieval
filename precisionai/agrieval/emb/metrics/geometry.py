# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Embedding-space geometry and health metrics."""

from __future__ import annotations

import numpy as np

from precisionai.agrieval.emb.metrics._utils import _prepare_embeddings

try:
    from sklearn.decomposition import PCA  # type: ignore[import]

    _PCA_AVAILABLE = True
except ImportError:
    PCA = None  # type: ignore[assignment]
    _PCA_AVAILABLE = False
from precisionai.agrieval.emb.metrics.similarity import pairwise_similarity_stats


def centroid_similarity_stats(
    embeddings: object,
    *,
    normalize: bool = True,
) -> dict:
    """Cosine similarity of each embedding to the dataset centroid.

    High mean cosine to centroid indicates anisotropy or embedding collapse.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    normalize : bool
        L2-normalize rows before computing similarity.

    Returns
    -------
    dict
        Keys: ``mean_cosine_to_centroid``, ``std_cosine_to_centroid``,
        ``p05``, ``p50``, ``p95``, ``centroid_norm``.
    """
    emb = _prepare_embeddings(embeddings, normalize)
    centroid = emb.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))

    if centroid_norm < 1e-12:
        return {
            "mean_cosine_to_centroid": None,
            "std_cosine_to_centroid": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "centroid_norm": 0.0,
        }

    centroid_unit = centroid / centroid_norm
    sims = (emb @ centroid_unit).astype(np.float32)
    p05, p50, p95 = np.percentile(sims, [5, 50, 95]).tolist()

    return {
        "mean_cosine_to_centroid": float(np.mean(sims)),
        "std_cosine_to_centroid": float(np.std(sims)),
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "centroid_norm": centroid_norm,
    }


def pca_explained_variance(
    embeddings: object,
    *,
    normalize: bool = True,
    n_components: int = 100,
) -> dict:
    """Fraction of variance explained by the leading principal components.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    normalize : bool
        L2-normalize rows before PCA.
    n_components : int
        Maximum number of components to compute.

    Returns
    -------
    dict
        Keys: ``embedding_dim``, ``n_components``, ``explained_variance_ratio``
        (list), ``pc1``, ``top_5``, ``top_10``, ``top_50``, ``top_100``.
    """
    emb = _prepare_embeddings(embeddings, normalize)
    n, d = emb.shape

    _empty = {
        "embedding_dim": d,
        "n_components": 0,
        "explained_variance_ratio": [],
        "pc1": None,
        "top_5": None,
        "top_10": None,
        "top_50": None,
        "top_100": None,
    }
    if n < 2:
        return _empty

    actual = min(n - 1, d, n_components)

    if PCA is not None:
        pca = PCA(n_components=actual)
        pca.fit(emb.astype(np.float64))
        evr = pca.explained_variance_ratio_.tolist()
    else:
        centered = emb.astype(np.float64) - emb.mean(axis=0)
        _, s_all, _ = np.linalg.svd(centered, full_matrices=False)
        total = float(np.sum(s_all**2))
        if total == 0.0:
            return _empty
        evr = (s_all[:actual] ** 2 / total).tolist()

    cumsum = np.cumsum(evr)

    def _top(k: int) -> float | None:
        idx = min(k, len(cumsum)) - 1
        return float(cumsum[idx]) if idx >= 0 else None

    return {
        "embedding_dim": d,
        "n_components": actual,
        "explained_variance_ratio": [float(x) for x in evr],
        "pc1": float(evr[0]) if evr else None,
        "top_5": _top(5),
        "top_10": _top(10),
        "top_50": _top(50),
        "top_100": _top(100),
    }


def effective_rank(
    embeddings: object,
    *,
    normalize: bool = True,
) -> dict:
    """Estimate the effective dimensionality of the embedding space.

    Uses the participation ratio: ``(Σλ)² / Σλ²`` over covariance eigenvalues.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    normalize : bool
        L2-normalize rows before computation.

    Returns
    -------
    dict
        Keys: ``embedding_dim``, ``effective_rank``, ``effective_rank_ratio``.
    """
    emb = _prepare_embeddings(embeddings, normalize)
    n, d = emb.shape

    if n < 2:
        return {"embedding_dim": d, "effective_rank": None, "effective_rank_ratio": None}

    centered = emb.astype(np.float64) - emb.mean(axis=0)
    cov = (centered.T @ centered) / (n - 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0 or np.sum(eigenvalues**2) == 0:
        return {"embedding_dim": d, "effective_rank": 0.0, "effective_rank_ratio": 0.0}

    eff = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2))
    return {
        "embedding_dim": d,
        "effective_rank": eff,
        "effective_rank_ratio": eff / d,
    }


def uniformity(embeddings: object, *, t: float = 2.0) -> float:
    """Wang & Isola (2020) uniformity of the embedding distribution.

    Measures how evenly embeddings are spread on the unit hypersphere.
    Lower (more negative) values indicate better uniformity.

    For L2-normalized vectors ``||u - v||² = 2(1 - cosine_sim(u, v))``.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``. Rows are L2-normalized before computation.
    t : float
        Temperature parameter (default ``2.0`` as in the original paper).

    Returns
    -------
    float
        ``log E[exp(-t * ||u - v||²)]`` over all unique pairs.
    """
    emb = _prepare_embeddings(embeddings, normalize=True)
    n = emb.shape[0]
    sim = (emb @ emb.T).clip(-1.0, 1.0)
    sq_dist = 2.0 * (1.0 - sim)
    i_idx, j_idx = np.triu_indices(n, k=1)
    return float(np.log(np.mean(np.exp(-t * sq_dist[i_idx, j_idx]))))


def alignment(
    embeddings: object,
    positive_pairs: list[tuple[int, int]],
    *,
    alpha: float = 2.0,
) -> float | None:
    """Wang & Isola (2020) alignment of explicit positive pairs.

    Measures how close embeddings of declared-similar items are.
    Lower values indicate better alignment.

    For L2-normalized vectors ``||u - v||^alpha`` with ``alpha=2`` simplifies
    to ``2(1 - cosine_sim(u, v))``.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``. Rows are L2-normalized before computation.
    positive_pairs : list[tuple[int, int]]
        Index pairs ``(i, j)`` where both items are explicit positives.
    alpha : float
        Exponent (default ``2.0``).

    Returns
    -------
    float | None
        Mean ``||u - v||^alpha`` over all positive pairs, or ``None`` if
        ``positive_pairs`` is empty.
    """
    if not positive_pairs:
        return None
    emb = _prepare_embeddings(embeddings, normalize=True)
    pairs = np.array(positive_pairs, dtype=np.intp)
    sq_dists = 2.0 * np.maximum(0.0, 1.0 - np.einsum("ij,ij->i", emb[pairs[:, 0]], emb[pairs[:, 1]]))
    dists = sq_dists if alpha == 2.0 else sq_dists ** (alpha / 2.0)
    return float(np.mean(dists))


def anisotropy_summary(
    embeddings: object,
    *,
    normalize: bool = True,
    sample_pairs: int | None = 1_000_000,
) -> dict:
    """Combine all anisotropy-related metrics into a single summary dict.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    normalize : bool
        L2-normalize rows before computation.
    sample_pairs : int | None
        Pair sampling budget for pairwise similarity stats.

    Returns
    -------
    dict
        Keys: ``pairwise_similarity_stats``, ``centroid_similarity_stats``,
        ``pca_explained_variance``, ``effective_rank``.
    """
    return {
        "pairwise_similarity_stats": pairwise_similarity_stats(
            embeddings, normalize=normalize, sample_pairs=sample_pairs
        ),
        "centroid_similarity_stats": centroid_similarity_stats(embeddings, normalize=normalize),
        "pca_explained_variance": pca_explained_variance(embeddings, normalize=normalize),
        "effective_rank": effective_rank(embeddings, normalize=normalize),
    }
