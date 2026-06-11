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

from pai.ag_emb.metrics.analysis import analyze_embedding_space, compare_embedding_spaces
from pai.ag_emb.metrics.cross_model import (
    knn_jaccard_at_k,
    knn_overlap_at_k,
    pairwise_similarity_correlation,
    per_item_neighbor_disagreement,
)
from pai.ag_emb.metrics.duplicates import (
    duplicate_groups_at_threshold,
    duplicate_pairs_at_threshold,
)
from pai.ag_emb.metrics.geometry import (
    anisotropy_summary,
    centroid_similarity_stats,
    effective_rank,
    pca_explained_variance,
)
from pai.ag_emb.metrics.label_aware import (
    intra_inter_similarity_gap,
    knn_confusion_matrix,
    knn_label_ndcg_at_k,
    knn_label_purity_at_k,
    knn_map_at_k,
)
from pai.ag_emb.metrics.neighbors import (
    gini_coefficient,
    hubness_at_k,
    knn_radius_at_k,
    mean_top_k_similarity,
    outlier_score_at_k,
)
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
from pai.ag_emb.metrics.similarity import (
    cosine_similarity_matrix,
    pairwise_similarity_stats,
    similarity_threshold_counts,
    top_k_neighbors,
)

__all__ = [
    "analyze_embedding_space",
    "anisotropy_summary",
    "average_precision_at_k",
    "centroid_similarity_stats",
    "compare_embedding_spaces",
    "cosine_similarity_matrix",
    "dcg_at_k",
    "duplicate_groups_at_threshold",
    "duplicate_pairs_at_threshold",
    "effective_rank",
    "gini_coefficient",
    "hubness_at_k",
    "intra_inter_similarity_gap",
    "knn_confusion_matrix",
    "knn_jaccard_at_k",
    "knn_label_ndcg_at_k",
    "knn_label_purity_at_k",
    "knn_map_at_k",
    "knn_overlap_at_k",
    "knn_radius_at_k",
    "l2_normalize",
    "mean_top_k_similarity",
    "ndcg_at_k",
    "normalize_ks",
    "outlier_score_at_k",
    "pairwise_similarity_correlation",
    "pairwise_similarity_stats",
    "pca_explained_variance",
    "per_item_neighbor_disagreement",
    "precision_at_k",
    "recall_at_k",
    "safe_mean",
    "similarity_threshold_counts",
    "top_k_neighbors",
]
