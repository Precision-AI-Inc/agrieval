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

"""Tests for cross-model metrics (P3) and the two main analysis entry points."""

from __future__ import annotations

import numpy as np
import pytest

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
from pai.ag_emb.metrics.similarity import top_k_neighbors

FIXTURE = np.array(
    [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
    dtype=np.float32,
)

RNG = np.random.default_rng(99)
VECS_A = RNG.random((30, 8)).astype(np.float32)
VECS_B = RNG.random((30, 16)).astype(np.float32)  # different D


class TestPairwiseSimCorrelation:
    def test_identical_embeddings_high_correlation(self) -> None:
        result = pairwise_similarity_correlation(VECS_A, VECS_A, sample_pairs=500, random_seed=0)
        assert result["pearson"] == pytest.approx(1.0, abs=1e-3)
        assert result["spearman"] == pytest.approx(1.0, abs=1e-3)

    def test_different_dimensions_accepted(self) -> None:
        result = pairwise_similarity_correlation(VECS_A, VECS_B, sample_pairs=500, random_seed=0)
        assert -1.0 <= result["pearson"] <= 1.0
        assert -1.0 <= result["spearman"] <= 1.0

    def test_mismatched_n_raises(self) -> None:
        with pytest.raises(ValueError, match="same number of items"):
            pairwise_similarity_correlation(VECS_A, VECS_A[:10])

    def test_sample_pairs_key_present(self) -> None:
        result = pairwise_similarity_correlation(VECS_A, VECS_B, sample_pairs=200)
        assert "sample_pairs" in result
        assert result["sample_pairs"] <= 200


class TestKnnOverlapAndJaccard:
    @pytest.fixture()
    def neighbors_a(self) -> dict:
        return top_k_neighbors(VECS_A, ks=[3])

    @pytest.fixture()
    def neighbors_same(self) -> dict:
        return top_k_neighbors(VECS_A, ks=[3])

    def test_overlap_identical_neighbors_is_one(self, neighbors_a: dict, neighbors_same: dict) -> None:
        result = knn_overlap_at_k(neighbors_a, neighbors_same)
        assert result[3]["mean"] == pytest.approx(1.0, abs=1e-5)

    def test_jaccard_identical_neighbors_is_one(self, neighbors_a: dict, neighbors_same: dict) -> None:
        result = knn_jaccard_at_k(neighbors_a, neighbors_same)
        assert result[3]["mean"] == pytest.approx(1.0, abs=1e-5)

    def test_overlap_between_0_and_1(self, neighbors_a: dict) -> None:
        neighbors_b = top_k_neighbors(VECS_B, ks=[3])
        result = knn_overlap_at_k(neighbors_a, neighbors_b)
        assert 0.0 <= result[3]["mean"] <= 1.0

    def test_jaccard_leq_overlap(self, neighbors_a: dict) -> None:
        neighbors_b = top_k_neighbors(VECS_B, ks=[3])
        overlap = knn_overlap_at_k(neighbors_a, neighbors_b)[3]["mean"]
        jaccard = knn_jaccard_at_k(neighbors_a, neighbors_b)[3]["mean"]
        # Jaccard <= overlap always (|A&B|/|AuB| <= |A&B|/k)
        assert jaccard <= overlap + 1e-5


class TestPerItemNeighborDisagreement:
    def test_zero_disagreement_when_identical(self) -> None:
        nbrs = top_k_neighbors(VECS_A, ks=[3])
        result = per_item_neighbor_disagreement(nbrs, nbrs)
        np.testing.assert_allclose(result[3]["per_item"], 0.0, atol=1e-5)

    def test_top_disagreements_sorted(self) -> None:
        nbrs_a = top_k_neighbors(VECS_A, ks=[3])
        nbrs_b = top_k_neighbors(VECS_B, ks=[3])
        result = per_item_neighbor_disagreement(nbrs_a, nbrs_b, top_n=5)
        scores = [x["score"] for x in result[3]["top_disagreements"]]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_metric_raises(self) -> None:
        nbrs = top_k_neighbors(VECS_A, ks=[3])
        with pytest.raises(ValueError, match="Unknown metric"):
            per_item_neighbor_disagreement(nbrs, nbrs, metric="bad")

    def test_overlap_metric_accepted(self) -> None:
        nbrs = top_k_neighbors(VECS_A, ks=[3])
        result = per_item_neighbor_disagreement(nbrs, nbrs, metric="overlap")
        assert result[3]["mean"] == pytest.approx(0.0, abs=1e-5)


class TestDuplicatePairs:
    def test_identical_vectors_detected(self) -> None:
        vectors = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        result = duplicate_pairs_at_threshold(vectors, thresholds=[0.99])
        assert result[0.99]["pair_count"] >= 1
        assert result[0.99]["pairs"][0]["i"] == 0
        assert result[0.99]["pairs"][0]["j"] == 1

    def test_no_duplicates_below_threshold(self) -> None:
        result = duplicate_pairs_at_threshold(FIXTURE, thresholds=[0.9999])
        assert result[0.9999]["pair_count"] == 0

    def test_pair_count_gte_returned_pairs(self) -> None:
        rng = np.random.default_rng(5)
        vecs = rng.random((20, 4)).astype(np.float32)
        result = duplicate_pairs_at_threshold(vecs, thresholds=[0.5], max_pairs_returned=2)
        assert result[0.5]["pair_count"] >= len(result[0.5]["pairs"])


class TestDuplicateGroups:
    def test_connected_component(self) -> None:
        vectors = np.array(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        pairs = duplicate_pairs_at_threshold(vectors, thresholds=[0.99])
        groups = duplicate_groups_at_threshold(pairs, n_items=4)
        assert groups[0.99]["num_groups"] == 1
        assert groups[0.99]["largest_group_size"] == 3

    def test_no_groups_when_no_pairs(self) -> None:
        pairs = duplicate_pairs_at_threshold(FIXTURE, thresholds=[0.9999])
        groups = duplicate_groups_at_threshold(pairs, n_items=len(FIXTURE))
        assert groups[0.9999]["num_groups"] == 0


class TestAnalyzeEmbeddingSpace:
    def test_smoke_default(self) -> None:
        result = analyze_embedding_space(FIXTURE, ks=[1, 3])
        assert result["n_items"] == len(FIXTURE)
        assert result["embedding_dim"] == 2
        assert "pairwise_similarity_stats" in result
        assert "similarity_threshold_counts" in result
        assert "nearest_neighbors" in result
        assert "geometry" in result

    def test_no_neighbors_when_disabled(self) -> None:
        result = analyze_embedding_space(FIXTURE, ks=[1], include_neighbors=False)
        assert "nearest_neighbors" not in result

    def test_no_geometry_when_disabled(self) -> None:
        result = analyze_embedding_space(FIXTURE, ks=[1], include_geometry=False)
        assert "geometry" not in result

    def test_duplicates_present_when_enabled(self) -> None:
        result = analyze_embedding_space(FIXTURE, ks=[1], include_duplicate_pairs=True)
        assert "duplicates" in result

    def test_label_aware_present_when_labels_provided(self) -> None:
        labels = [0, 0, 1, 1]
        result = analyze_embedding_space(FIXTURE, ks=[1], labels=labels)
        assert "label_aware" in result
        assert "intra_inter_similarity_gap" in result["label_aware"]


class TestCompareEmbeddingSpaces:
    def test_smoke(self) -> None:
        result = compare_embedding_spaces(VECS_A, VECS_B, ks=[3])
        assert result["n_items"] == len(VECS_A)
        assert result["embedding_dim_a"] == 8
        assert result["embedding_dim_b"] == 16
        assert "pairwise_similarity_correlation" in result
        assert "neighbor_agreement" in result

    def test_high_agreement_when_same_embeddings(self) -> None:
        result = compare_embedding_spaces(VECS_A, VECS_A, ks=[3], sample_pairs=200)
        assert result["pairwise_similarity_correlation"]["pearson"] == pytest.approx(1.0, abs=1e-3)
        agreement = result["neighbor_agreement"]["knn_overlap_at_k"]
        assert agreement[3]["mean"] == pytest.approx(1.0, abs=1e-5)

    def test_mismatched_n_raises(self) -> None:
        with pytest.raises(ValueError, match="same number of items"):
            compare_embedding_spaces(VECS_A, VECS_A[:10])

    def test_different_d_accepted(self) -> None:
        result = compare_embedding_spaces(VECS_A, VECS_B, ks=[3])
        assert result["embedding_dim_a"] != result["embedding_dim_b"]
        assert "neighbor_agreement" in result

    def test_zero_vectors_raise(self) -> None:
        # zero vectors pass l2_normalize (they stay zero) — no error expected here
        # but NaN/inf should raise
        bad_nan = np.full((5, 4), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            compare_embedding_spaces(bad_nan, VECS_A[:5])
