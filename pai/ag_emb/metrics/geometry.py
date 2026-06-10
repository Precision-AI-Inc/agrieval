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

"""P2 — Embedding-space geometry and health metrics."""

from __future__ import annotations

import numpy as np

from pai.ag_emb.metrics._utils import _prepare_embeddings, _validate_embeddings
from pai.ag_emb.metrics.similarity import pairwise_similarity_stats


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
            "p05": None, "p50": None, "p95": None,
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
        "embedding_dim": d, "n_components": 0,
        "explained_variance_ratio": [],
        "pc1": None, "top_5": None, "top_10": None,
        "top_50": None, "top_100": None,
    }
    if n < 2:
        return _empty

    actual = min(n - 1, d, n_components)

    try:
        from sklearn.decomposition import PCA  # type: ignore[import]
        pca = PCA(n_components=actual)
        pca.fit(emb.astype(np.float64))
        evr = pca.explained_variance_ratio_.tolist()
    except ImportError:
        centered = (emb.astype(np.float64) - emb.mean(axis=0))
        _, s_all, _ = np.linalg.svd(centered, full_matrices=False)
        total = float(np.sum(s_all ** 2))
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

    if len(eigenvalues) == 0 or np.sum(eigenvalues ** 2) == 0:
        return {"embedding_dim": d, "effective_rank": 0.0, "effective_rank_ratio": 0.0}

    eff = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2))
    return {
        "embedding_dim": d,
        "effective_rank": eff,
        "effective_rank_ratio": eff / d,
    }


def anisotropy_summary(
    embeddings: object,
    *,
    normalize: bool = True,
    sample_pairs: int | None = 1_000_000,
) -> dict:
    """Convenience wrapper combining all anisotropy-related metrics.

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
        "centroid_similarity_stats": centroid_similarity_stats(
            embeddings, normalize=normalize
        ),
        "pca_explained_variance": pca_explained_variance(
            embeddings, normalize=normalize
        ),
        "effective_rank": effective_rank(embeddings, normalize=normalize),
    }
