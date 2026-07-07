# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Split-free label separation metrics.

Quantify how well ground-truth classes separate in an embedding space
without any train/eval split: cluster geometry (silhouette,
Calinski-Harabasz), cluster recoverability (adjusted Rand index and
normalized mutual information against a k-means clustering), and linear
separability of a designated foreground/background partition along the
first principal component (PC1-AUROC). All are non-parametric or
closed-form — no held-out fit anywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from precisionai.agrieval.emb.metrics._utils import _prepare_embeddings, _rankdata

try:
    from sklearn.cluster import KMeans  # type: ignore[import]
    from sklearn.metrics import (  # type: ignore[import]
        adjusted_rand_score,
        normalized_mutual_info_score,
    )
    from threadpoolctl import threadpool_limits  # type: ignore[import]

    _SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = None  # type: ignore[assignment]
    adjusted_rand_score = None  # type: ignore[assignment]
    normalized_mutual_info_score = None  # type: ignore[assignment]
    threadpool_limits = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

_SILHOUETTE_CHUNK_ROWS = 1024


def _prepare_labels(labels: object, n: int) -> np.ndarray:
    """Convert ``labels`` to a validated 1-D array matching ``n`` feature rows.

    Parameters
    ----------
    labels : array-like
        Per-row labels of any comparable dtype (ints or strings).
    n : int
        Number of feature rows the labels must align with.

    Returns
    -------
    np.ndarray
        Shape ``[n]`` label array.

    Raises
    ------
    ValueError
        If ``labels`` is not 1-D of length ``n``.
    """
    arr = np.asarray(labels)
    if arr.shape != (n,):
        raise ValueError(f"labels must be a 1D array of length {n} to match features; got shape {arr.shape}.")
    return arr


def calinski_harabasz(features: object, labels: object) -> float:
    """Calinski-Harabasz index of the ground-truth class partition.

    Ratio of between-class to within-class dispersion, scaled by
    ``(N - k) / (k - 1)`` — higher means classes form tighter, better
    separated clusters. Computed on raw (uncentered, unnormalized) features
    with Euclidean geometry, matching
    ``sklearn.metrics.calinski_harabasz_score``. Closed-form and O(N·D), so
    it runs on the full corpus without subsampling.

    Parameters
    ----------
    features : array-like
        Shape ``[N, D]``. Used raw — no L2 normalization.
    labels : array-like
        Shape ``[N]``. Ground-truth class per row.

    Returns
    -------
    float
        The Calinski-Harabasz index.

    Raises
    ------
    ValueError
        If fewer than 2 distinct labels are present, or every row is its
        own class (``k == N``, leaving no within-class dispersion).
    """
    feats = _prepare_embeddings(features, normalize=False, name="features").astype(np.float64)
    labs = _prepare_labels(labels, feats.shape[0])
    n = feats.shape[0]

    unique, inverse = np.unique(labs, return_inverse=True)
    k = len(unique)
    if k < 2:
        raise ValueError(f"calinski_harabasz requires at least 2 distinct labels; got {k}.")
    if k >= n:
        raise ValueError(f"calinski_harabasz requires fewer classes than samples; got {k} classes for {n} samples.")

    counts = np.bincount(inverse, minlength=k).astype(np.float64)
    # Per-dimension bincount beats np.add.at scatter by orders of magnitude
    # on corpora with millions of rows.
    sums = np.empty((k, feats.shape[1]), dtype=np.float64)
    for j in range(feats.shape[1]):
        sums[:, j] = np.bincount(inverse, weights=feats[:, j], minlength=k)
    centroids = sums / counts[:, None]

    overall = feats.mean(axis=0)
    between = float(np.sum(counts * np.sum((centroids - overall) ** 2, axis=1)))
    within = float(np.sum((feats - centroids[inverse]) ** 2))
    if within == 0.0:
        return float("inf") if between > 0.0 else 0.0
    return float((between / within) * ((n - k) / (k - 1)))


def silhouette_cosine(features: object, labels: object) -> float:
    """Mean silhouette coefficient of the class partition under cosine distance.

    For each sample, compares its mean cosine distance to its own class
    (``a``) against its mean distance to the nearest other class (``b``),
    scoring ``(b - a) / max(a, b)``; samples in singleton classes score
    ``0``, matching ``sklearn.metrics.silhouette_score``. Time is O(N²·D)
    — pass an already-subsampled set for large corpora (memory stays
    bounded: the distance matrix is processed in row chunks, never
    materialized whole).

    Parameters
    ----------
    features : array-like
        Shape ``[N, D]``. Rows are L2-normalized before computing cosine
        distances.
    labels : array-like
        Shape ``[N]``. Ground-truth class per row.

    Returns
    -------
    float
        Mean silhouette coefficient in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If the number of distinct labels is not in ``[2, N - 1]``.
    """
    unit = _prepare_embeddings(features, normalize=True, name="features").astype(np.float64)
    labs = _prepare_labels(labels, unit.shape[0])
    n = unit.shape[0]

    unique, inverse = np.unique(labs, return_inverse=True)
    k = len(unique)
    if not 2 <= k <= n - 1:
        raise ValueError(f"silhouette_cosine requires 2 to n-1 distinct labels; got {k} labels for {n} samples.")

    counts = np.bincount(inverse, minlength=k).astype(np.float64)
    onehot = np.zeros((n, k), dtype=np.float64)
    onehot[np.arange(n), inverse] = 1.0

    scores = np.empty(n, dtype=np.float64)
    for start in range(0, n, _SILHOUETTE_CHUNK_ROWS):
        stop = min(start + _SILHOUETTE_CHUNK_ROWS, n)
        dists = 1.0 - (unit[start:stop] @ unit.T).clip(-1.0, 1.0)
        class_sums = dists @ onehot
        rows = np.arange(stop - start)
        own = inverse[start:stop]

        own_count = counts[own]
        # d(i, i) == 0 sits inside the own-class sum, so dividing by
        # (count - 1) excludes self exactly.
        a = np.divide(class_sums[rows, own], own_count - 1.0, out=np.zeros(stop - start), where=own_count > 1.0)

        mean_other = class_sums / counts[None, :]
        mean_other[rows, own] = np.inf
        b = mean_other.min(axis=1)

        s = np.divide(b - a, np.maximum(a, b), out=np.zeros(stop - start), where=np.maximum(a, b) > 0.0)
        s[own_count == 1.0] = 0.0
        scores[start:stop] = s

    return float(scores.mean())


def kmeans_label_agreement(
    features: object,
    labels: object,
    *,
    random_state: int = 0,
    n_init: int = 10,
) -> dict[str, Any]:
    """Agreement between a k-means clustering and the ground-truth classes.

    Fits k-means (``k`` = number of distinct labels) on L2-normalized
    features and scores the resulting cluster assignment against the true
    labels with the adjusted Rand index and normalized mutual information —
    measuring whether the classes are recoverable as unsupervised clusters.
    K-means runs single-threaded with a fixed ``random_state``: parallel
    Lloyd reductions sum in a nondeterministic order, which flips boundary
    points run-to-run and perturbs ARI/NMI at the 1e-3 level.

    Parameters
    ----------
    features : array-like
        Shape ``[N, D]``. Rows are L2-normalized before clustering.
    labels : array-like
        Shape ``[N]``. Ground-truth class per row.
    random_state : int
        Seed for the k-means initialisation.
    n_init : int
        Number of k-means++ restarts.

    Returns
    -------
    dict
        Keys: ``ari``, ``nmi``, ``n_clusters``.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    ValueError
        If fewer than 2 distinct labels are present.
    """
    if (
        not _SKLEARN_AVAILABLE
        or KMeans is None
        or adjusted_rand_score is None
        or normalized_mutual_info_score is None
        or threadpool_limits is None
    ):
        raise ImportError("scikit-learn is required: pip install scikit-learn") from None

    unit = _prepare_embeddings(features, normalize=True, name="features").astype(np.float64)
    labs = _prepare_labels(labels, unit.shape[0])

    k = len(np.unique(labs))
    if k < 2:
        raise ValueError(f"kmeans_label_agreement requires at least 2 distinct labels; got {k}.")

    kmeans_params: dict[str, Any] = {"n_clusters": k, "n_init": n_init, "random_state": random_state}
    with threadpool_limits(1), np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        km = KMeans(**kmeans_params).fit(unit)

    return {
        "ari": float(adjusted_rand_score(labs, km.labels_)),
        "nmi": float(normalized_mutual_info_score(labs, km.labels_)),
        "n_clusters": k,
    }


def pc1_auroc(features: object, is_foreground: object) -> float:
    """Linear separability of a binary partition along the first principal component.

    Projects raw centered features onto the top principal direction of the
    full corpus and scores how well that single scalar ranks foreground
    above background, as a threshold-free AUROC (tie-aware Mann-Whitney
    form). The direction's sign is arbitrary, so the score is reported
    sign-free as ``max(auc, 1 - auc)`` — ``0.5`` means PC1 carries no
    foreground/background signal, ``1.0`` means perfect separation.

    Parameters
    ----------
    features : array-like
        Shape ``[N, D]``. Used raw — centered internally, no L2
        normalization.
    is_foreground : array-like
        Shape ``[N]`` boolean. ``True`` marks foreground rows.

    Returns
    -------
    float
        Sign-free AUROC in ``[0.5, 1.0]``.

    Raises
    ------
    ValueError
        If ``is_foreground`` is not 1-D of matching length, or does not
        contain both foreground and background rows.
    """
    feats = _prepare_embeddings(features, normalize=False, name="features").astype(np.float64)
    fg = np.asarray(is_foreground, dtype=bool)
    if fg.shape != (feats.shape[0],):
        raise ValueError(f"is_foreground must be a 1D boolean array of length {feats.shape[0]}; got shape {fg.shape}.")
    n_pos = int(fg.sum())
    n_neg = int((~fg).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("is_foreground must contain both foreground (True) and background (False) rows.")

    centered = feats - feats.mean(axis=0)
    # Top eigenvector of the D x D covariance: one BLAS matmul plus a small
    # eigh, far cheaper than a tall-matrix SVD for N >> D.
    _, eigvecs = np.linalg.eigh(centered.T @ centered)
    score = centered @ eigvecs[:, -1]

    ranks = _rankdata(score)
    auc = (float(ranks[fg].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))
