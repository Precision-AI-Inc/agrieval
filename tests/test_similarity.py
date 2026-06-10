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

"""Tests for pai.ag_emb.metrics.similarity (P0 primitives)."""

from __future__ import annotations

import numpy as np
import pytest

from pai.ag_emb.metrics.similarity import (
    cosine_similarity_matrix,
    pairwise_similarity_stats,
    similarity_threshold_counts,
    top_k_neighbors,
)

# Deterministic fixture from the spec
FIXTURE = np.array(
    [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
    dtype=np.float32,
)


class TestCosineSimMatrix:
    def test_shape(self) -> None:
        out = cosine_similarity_matrix(FIXTURE)
        assert out.shape == (4, 4)

    def test_diagonal_is_nan_when_exclude_self(self) -> None:
        out = cosine_similarity_matrix(FIXTURE, exclude_self=True)
        assert np.all(np.isnan(np.diag(out)))

    def test_diagonal_present_when_exclude_self_false(self) -> None:
        out = cosine_similarity_matrix(FIXTURE, exclude_self=False)
        diag = np.diag(out)
        np.testing.assert_allclose(diag, np.ones(4), atol=1e-5)

    def test_symmetry(self) -> None:
        out = cosine_similarity_matrix(FIXTURE, exclude_self=False)
        np.testing.assert_allclose(out, out.T, atol=1e-5)

    def test_known_values(self) -> None:
        # [1,0] and [0,1] are orthogonal
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        out = cosine_similarity_matrix(vectors, exclude_self=False)
        assert out[0, 1] == pytest.approx(0.0, abs=1e-5)

    def test_raises_on_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            cosine_similarity_matrix(np.array([1.0, 2.0]))

    def test_raises_on_nan(self) -> None:
        bad = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            cosine_similarity_matrix(bad)


class TestTopKNeighbors:
    def test_item0_nearest_neighbor_is_item1(self) -> None:
        result = top_k_neighbors(FIXTURE, ks=[1])
        assert result[1]["indices"][0, 0] == 1

    def test_item2_nearest_neighbor_is_item3(self) -> None:
        result = top_k_neighbors(FIXTURE, ks=[1])
        assert result[1]["indices"][2, 0] == 3

    def test_query_item_never_in_its_own_list(self) -> None:
        result = top_k_neighbors(FIXTURE, ks=[3], exclude_self=True)
        for i in range(len(FIXTURE)):
            assert i not in result[3]["indices"][i].tolist()

    def test_shape(self) -> None:
        n = len(FIXTURE)
        result = top_k_neighbors(FIXTURE, ks=[2])
        assert result[2]["indices"].shape == (n, 2)
        assert result[2]["scores"].shape == (n, 2)

    def test_scores_descending(self) -> None:
        result = top_k_neighbors(FIXTURE, ks=[3])
        scores = result[3]["scores"]
        for i in range(len(FIXTURE)):
            row = scores[i].tolist()
            assert row == sorted(row, reverse=True)

    def test_k_clamped_to_pool_size(self) -> None:
        result = top_k_neighbors(FIXTURE, ks=[100], exclude_self=True)
        # pool_size = 4 - 1 = 3; k should be clamped to 3
        assert 3 in result
        assert 100 not in result

    def test_multiple_ks(self) -> None:
        result = top_k_neighbors(FIXTURE, ks=[1, 2, 3])
        assert set(result.keys()) == {1, 2, 3}

    def test_exclude_self_false(self) -> None:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = top_k_neighbors(vectors, ks=[2], exclude_self=False)
        # With exclude_self=False, each item's top-2 includes itself
        for i in range(2):
            assert i in result[2]["indices"][i].tolist()

    def test_single_neighbor_allowed(self) -> None:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = top_k_neighbors(vectors, ks=[1], exclude_self=True)
        assert result[1]["indices"].shape == (2, 1)

    def test_raises_when_n1_exclude_self(self) -> None:
        with pytest.raises(ValueError):
            top_k_neighbors(np.array([[1.0, 0.0]]), ks=[1], exclude_self=True)

    def test_batching_matches_single_pass(self) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.random((20, 8)).astype(np.float32)
        ref = top_k_neighbors(vectors, ks=[5])
        batched = top_k_neighbors(vectors, ks=[5], batch_size=3)
        np.testing.assert_array_equal(ref[5]["indices"], batched[5]["indices"])


class TestPairwiseSimStats:
    def test_excludes_self_pairs(self) -> None:
        # Fixture items 0&1 and 2&3 are very similar; stats must not include
        # self-pairs (which would always be 1.0 and inflate the mean)
        stats = pairwise_similarity_stats(FIXTURE)
        assert stats["max"] < 1.0 + 1e-5
        # exact self-similarities would all be 1.0; mean should not be 1.0
        assert stats["mean"] < 0.99

    def test_returns_expected_keys(self) -> None:
        stats = pairwise_similarity_stats(FIXTURE)
        for key in ("count", "mean", "std", "min", "max", "p01", "p05", "p25", "p50", "p75", "p95", "p99"):
            assert key in stats

    def test_sampled_returns_stats(self) -> None:
        rng_vecs = np.random.default_rng(7).random((50, 16)).astype(np.float32)
        stats = pairwise_similarity_stats(rng_vecs, sample_pairs=200)
        assert stats["mean"] is not None
        assert 0 < stats["count"] <= 200

    def test_single_pair(self) -> None:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        stats = pairwise_similarity_stats(vectors)
        assert stats["count"] == 1
        assert stats["mean"] == pytest.approx(0.0, abs=1e-5)


class TestSimThresholdCounts:
    def test_higher_threshold_fewer_pairs(self) -> None:
        rng_vecs = np.random.default_rng(1).random((30, 8)).astype(np.float32)
        counts = similarity_threshold_counts(rng_vecs, thresholds=[0.5, 0.9])
        assert counts[0]["pair_count"] >= counts[1]["pair_count"]

    def test_unique_unordered_pairs_only(self) -> None:
        # 4 items → 6 unique pairs; no pair should be counted twice
        counts = similarity_threshold_counts(FIXTURE, thresholds=[0.0])
        assert counts[0]["pair_count"] == 6

    def test_estimated_total_pairs_formula(self) -> None:
        n = len(FIXTURE)
        expected_total = n * (n - 1) // 2
        counts = similarity_threshold_counts(FIXTURE, thresholds=[0.0])
        assert counts[0]["estimated_total_pairs"] == expected_total

    def test_fraction_between_zero_and_one(self) -> None:
        counts = similarity_threshold_counts(FIXTURE, thresholds=[0.5])
        assert 0.0 <= counts[0]["pair_fraction"] <= 1.0

    def test_sampled_mode(self) -> None:
        rng_vecs = np.random.default_rng(2).random((50, 8)).astype(np.float32)
        counts = similarity_threshold_counts(rng_vecs, thresholds=[0.5], sample_pairs=100)
        assert isinstance(counts[0]["pair_count"], int)
