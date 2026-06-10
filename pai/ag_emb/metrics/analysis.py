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

"""Main entry points: analyze_embedding_space and compare_embedding_spaces."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pai.ag_emb.metrics._utils import _prepare_embeddings, _validate_embeddings
from pai.ag_emb.metrics.cross_model import (
    knn_jaccard_at_k,
    knn_overlap_at_k,
    per_item_neighbor_disagreement,
    pairwise_similarity_correlation,
)
from pai.ag_emb.metrics.duplicates import (
    duplicate_groups_at_threshold,
    duplicate_pairs_at_threshold,
)
from pai.ag_emb.metrics.geometry import (
    centroid_similarity_stats,
    effective_rank,
    pca_explained_variance,
)
from pai.ag_emb.metrics.label_aware import intra_inter_similarity_gap, knn_label_purity_at_k
from pai.ag_emb.metrics.neighbors import (
    hubness_at_k,
    knn_radius_at_k,
    mean_top_k_similarity,
    outlier_score_at_k,
)
from pai.ag_emb.metrics.similarity import (
    pairwise_similarity_stats,
    similarity_threshold_counts,
    top_k_neighbors,
)


def analyze_embedding_space(
    embeddings: object,
    *,
    ks: Sequence[int] = (5, 10, 50, 100),
    thresholds: Sequence[float] = (0.80, 0.85, 0.90, 0.95, 0.98, 0.99),
    normalize: bool = True,
    sample_pairs: int | None = 1_000_000,
    include_neighbors: bool = True,
    include_duplicate_pairs: bool = False,
    include_geometry: bool = True,
    labels: object = None,
) -> dict:
    """Full unsupervised analysis of a single embedding space.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    ks : sequence of int
        K values for nearest-neighbor diagnostics.
    thresholds : sequence of float
        Similarity thresholds for pair counting and duplicate detection.
    normalize : bool
        L2-normalize rows before all computations.
    sample_pairs : int | None
        Pair-sampling budget for global stats. ``None`` computes all pairs.
    include_neighbors : bool
        Compute mean-top-k, knn-radius, outlier score, and hubness.
    include_duplicate_pairs : bool
        Detect and group near-duplicate pairs (expensive for large N).
    include_geometry : bool
        Compute centroid similarity, PCA explained variance, effective rank.
    labels : array-like | None
        Optional class labels for supervised metrics.

    Returns
    -------
    dict
        Nested result with keys ``n_items``, ``embedding_dim``, ``ks``,
        ``thresholds``, ``pairwise_similarity_stats``,
        ``similarity_threshold_counts``, and optional ``nearest_neighbors``,
        ``geometry``, ``duplicates``, ``label_aware``.
    """
    emb = _validate_embeddings(embeddings)
    n, d = emb.shape

    result: dict = {
        "n_items": n,
        "embedding_dim": d,
        "ks": list(ks),
        "thresholds": list(thresholds),
        "pairwise_similarity_stats": pairwise_similarity_stats(
            emb, normalize=normalize, sample_pairs=sample_pairs
        ),
        "similarity_threshold_counts": similarity_threshold_counts(
            emb, thresholds, normalize=normalize, sample_pairs=sample_pairs
        ),
    }

    if include_neighbors:
        neighbors = top_k_neighbors(emb, ks, normalize=normalize)
        result["nearest_neighbors"] = {
            "mean_top_k_similarity": mean_top_k_similarity(neighbors),
            "knn_radius_at_k": knn_radius_at_k(neighbors),
            "outlier_score_at_k": outlier_score_at_k(neighbors),
            "hubness_at_k": hubness_at_k(neighbors, n_items=n),
        }
    else:
        neighbors = None

    if include_geometry:
        result["geometry"] = {
            "centroid_similarity_stats": centroid_similarity_stats(emb, normalize=normalize),
            "pca_explained_variance": pca_explained_variance(emb, normalize=normalize),
            "effective_rank": effective_rank(emb, normalize=normalize),
        }

    if include_duplicate_pairs:
        dup_pairs = duplicate_pairs_at_threshold(emb, thresholds, normalize=normalize)
        result["duplicates"] = {
            "duplicate_pairs_at_threshold": dup_pairs,
            "duplicate_groups_at_threshold": duplicate_groups_at_threshold(
                dup_pairs, n_items=n
            ),
        }

    if labels is not None:
        label_result: dict = {
            "intra_inter_similarity_gap": intra_inter_similarity_gap(
                emb, labels, normalize=normalize, sample_pairs=sample_pairs
            ),
        }
        if neighbors is not None:
            label_result["knn_label_purity_at_k"] = knn_label_purity_at_k(
                neighbors, labels
            )
        result["label_aware"] = label_result

    return result


def compare_embedding_spaces(
    embeddings_a: object,
    embeddings_b: object,
    *,
    ks: Sequence[int] = (5, 10, 50, 100),
    normalize: bool = True,
    sample_pairs: int = 1_000_000,
    random_seed: int = 42,
) -> dict:
    """Compare two embedding spaces that encode the same N items.

    Parameters
    ----------
    embeddings_a : array-like
        Shape ``[N, D1]``.
    embeddings_b : array-like
        Shape ``[N, D2]``. N must match; D may differ.
    ks : sequence of int
        K values for nearest-neighbor agreement metrics.
    normalize : bool
        L2-normalize rows of each matrix independently.
    sample_pairs : int
        Number of random pairs for similarity correlation.
    random_seed : int
        Seed for reproducible sampling.

    Returns
    -------
    dict
        Keys: ``n_items``, ``embedding_dim_a``, ``embedding_dim_b``,
        ``pairwise_similarity_correlation``, ``neighbor_agreement``.

    Raises
    ------
    ValueError
        If ``embeddings_a`` and ``embeddings_b`` have different N.
    """
    emb_a = _validate_embeddings(embeddings_a, "embeddings_a")
    emb_b = _validate_embeddings(embeddings_b, "embeddings_b")

    if emb_a.shape[0] != emb_b.shape[0]:
        raise ValueError(
            "embeddings_a and embeddings_b must have the same number of items. "
            f"Got {emb_a.shape[0]} and {emb_b.shape[0]}."
        )

    n = emb_a.shape[0]

    neighbors_a = top_k_neighbors(emb_a, ks, normalize=normalize)
    neighbors_b = top_k_neighbors(emb_b, ks, normalize=normalize)

    return {
        "n_items": n,
        "embedding_dim_a": int(emb_a.shape[1]),
        "embedding_dim_b": int(emb_b.shape[1]),
        "pairwise_similarity_correlation": pairwise_similarity_correlation(
            emb_a, emb_b,
            normalize=normalize,
            sample_pairs=sample_pairs,
            random_seed=random_seed,
        ),
        "neighbor_agreement": {
            "knn_overlap_at_k": knn_overlap_at_k(neighbors_a, neighbors_b),
            "knn_jaccard_at_k": knn_jaccard_at_k(neighbors_a, neighbors_b),
            "per_item_neighbor_disagreement": per_item_neighbor_disagreement(
                neighbors_a, neighbors_b
            ),
        },
    }
