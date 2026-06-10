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

# --- Ranking / retrieval (original) ---
from pai.ag_emb.metrics.ranking import (
    average_precision_at_k,
    dcg_at_k,
    l2_normalize,
    ndcg_at_k,
    normalize_ks,
    precision_at_k,
    recall_at_k,
    safe_mean,
)

# --- P0: Core similarity primitives ---
from pai.ag_emb.metrics.similarity import (
    cosine_similarity_matrix,
    pairwise_similarity_stats,
    similarity_threshold_counts,
    top_k_neighbors,
)

# --- P1: Nearest-neighbor diagnostics ---
from pai.ag_emb.metrics.neighbors import (
    gini_coefficient,
    hubness_at_k,
    knn_radius_at_k,
    mean_top_k_similarity,
    outlier_score_at_k,
)

# --- P2: Geometry / embedding-space health ---
from pai.ag_emb.metrics.geometry import (
    anisotropy_summary,
    centroid_similarity_stats,
    effective_rank,
    pca_explained_variance,
)

# --- P3: Cross-model comparison ---
from pai.ag_emb.metrics.cross_model import (
    knn_jaccard_at_k,
    knn_overlap_at_k,
    pairwise_similarity_correlation,
    per_item_neighbor_disagreement,
)

# --- P4: Duplicate detection ---
from pai.ag_emb.metrics.duplicates import (
    duplicate_groups_at_threshold,
    duplicate_pairs_at_threshold,
)

# --- Optional: label-aware ---
from pai.ag_emb.metrics.label_aware import (
    intra_inter_similarity_gap,
    knn_confusion_matrix,
    knn_label_ndcg_at_k,
    knn_label_purity_at_k,
    knn_map_at_k,
)

# --- Main entry points ---
from pai.ag_emb.metrics.analysis import analyze_embedding_space, compare_embedding_spaces

__all__ = [
    # ranking
    "average_precision_at_k",
    "dcg_at_k",
    "l2_normalize",
    "ndcg_at_k",
    "normalize_ks",
    "precision_at_k",
    "recall_at_k",
    "safe_mean",
    # similarity
    "cosine_similarity_matrix",
    "pairwise_similarity_stats",
    "similarity_threshold_counts",
    "top_k_neighbors",
    # neighbors
    "gini_coefficient",
    "hubness_at_k",
    "knn_radius_at_k",
    "mean_top_k_similarity",
    "outlier_score_at_k",
    # geometry
    "anisotropy_summary",
    "centroid_similarity_stats",
    "effective_rank",
    "pca_explained_variance",
    # cross-model
    "knn_jaccard_at_k",
    "knn_overlap_at_k",
    "pairwise_similarity_correlation",
    "per_item_neighbor_disagreement",
    # duplicates
    "duplicate_groups_at_threshold",
    "duplicate_pairs_at_threshold",
    # label-aware
    "intra_inter_similarity_gap",
    "knn_confusion_matrix",
    "knn_label_ndcg_at_k",
    "knn_label_purity_at_k",
    "knn_map_at_k",
    # analysis
    "analyze_embedding_space",
    "compare_embedding_spaces",
]
