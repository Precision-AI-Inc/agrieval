# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for precisionai.agrieval.emb.metrics.geometry (P2 metrics)."""

from __future__ import annotations

import numpy as np
import pytest

from precisionai.agrieval.emb.metrics.geometry import (
    alignment,
    anisotropy_summary,
    centroid_similarity_stats,
    effective_rank,
    pca_explained_variance,
    uniformity,
)

FIXTURE = np.array(
    [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
    dtype=np.float32,
)


class TestCentroidSimilarityStats:
    def test_expected_keys(self) -> None:
        result = centroid_similarity_stats(FIXTURE)
        for key in ("mean_cosine_to_centroid", "std_cosine_to_centroid", "p05", "p50", "p95", "centroid_norm"):
            assert key in result

    def test_identical_vectors_high_centroid_sim(self) -> None:
        vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (10, 1))
        result = centroid_similarity_stats(vectors)
        assert result["mean_cosine_to_centroid"] == pytest.approx(1.0, abs=1e-5)

    def test_zero_centroid_returns_none(self) -> None:
        # Perfectly balanced: centroid is the zero vector
        vectors = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        result = centroid_similarity_stats(vectors)
        assert result["mean_cosine_to_centroid"] is None
        assert result["centroid_norm"] == pytest.approx(0.0, abs=1e-5)


class TestPcaExplainedVariance:
    def test_expected_keys(self) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.random((20, 8)).astype(np.float32)
        result = pca_explained_variance(vectors)
        for key in ("embedding_dim", "n_components", "explained_variance_ratio", "pc1", "top_5"):
            assert key in result

    def test_evr_sums_to_at_most_one(self) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.random((20, 8)).astype(np.float32)
        result = pca_explained_variance(vectors)
        total = sum(result["explained_variance_ratio"])
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_n_lt_2_returns_empty(self) -> None:
        result = pca_explained_variance(np.array([[1.0, 2.0]], dtype=np.float32))
        assert result["n_components"] == 0
        assert result["explained_variance_ratio"] == []

    def test_embedding_dim_preserved(self) -> None:
        rng = np.random.default_rng(1)
        vectors = rng.random((10, 16)).astype(np.float32)
        result = pca_explained_variance(vectors)
        assert result["embedding_dim"] == 16

    def test_n_components_clamped(self) -> None:
        vectors = np.random.default_rng(2).random((5, 4)).astype(np.float32)
        result = pca_explained_variance(vectors, n_components=100)
        assert result["n_components"] <= 4


class TestEffectiveRank:
    def test_identical_vectors_low_effective_rank(self) -> None:
        vectors = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (10, 1))
        result = effective_rank(vectors)
        assert result["effective_rank"] == pytest.approx(0.0, abs=1e-4)

    def test_orthogonal_vectors_high_effective_rank(self) -> None:
        # N orthogonal unit vectors in D dimensions → full rank
        vectors = np.eye(8, dtype=np.float32)
        result = effective_rank(vectors)
        assert result["effective_rank"] is not None
        # Should be close to D (or n-1 since rank is min(N-1, D))
        assert result["effective_rank"] > 4.0

    def test_effective_rank_ratio_between_0_and_1(self) -> None:
        rng = np.random.default_rng(3)
        vectors = rng.random((20, 16)).astype(np.float32)
        result = effective_rank(vectors)
        ratio = result["effective_rank_ratio"]
        assert ratio is not None
        assert 0.0 <= ratio <= 1.0 + 1e-5

    def test_n_lt_2_returns_none(self) -> None:
        result = effective_rank(np.array([[1.0, 2.0]], dtype=np.float32))
        assert result["effective_rank"] is None

    def test_embedding_dim_key_present(self) -> None:
        result = effective_rank(FIXTURE)
        assert result["embedding_dim"] == 2


class TestAnisotropySummary:
    def test_all_keys_present(self) -> None:
        rng = np.random.default_rng(4)
        vectors = rng.random((20, 8)).astype(np.float32)
        result = anisotropy_summary(vectors)
        for key in (
            "pairwise_similarity_stats",
            "centroid_similarity_stats",
            "pca_explained_variance",
            "effective_rank",
        ):
            assert key in result


# ---------------------------------------------------------------------------
# uniformity
# ---------------------------------------------------------------------------


class TestUniformity:
    def test_returns_float(self) -> None:
        result = uniformity(FIXTURE)
        assert isinstance(result, float)

    def test_non_positive(self) -> None:
        # log(mean(exp(-t * ||u-v||²))) with t > 0 produces values in (-∞, 0]
        result = uniformity(FIXTURE)
        assert result <= 0.0

    def test_collapsed_worse_than_spread(self) -> None:
        # Identical vectors → all pairwise distances are 0 → uniformity = log(1) = 0
        # Orthogonal vectors → large distances → uniformity much less than 0
        collapsed = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1))
        spread = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
        assert uniformity(collapsed) > uniformity(spread)

    def test_temperature_changes_value(self) -> None:
        result_t2 = uniformity(FIXTURE, t=2.0)
        result_t4 = uniformity(FIXTURE, t=4.0)
        assert result_t2 != pytest.approx(result_t4)

    def test_two_vectors(self) -> None:
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = uniformity(emb)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    def test_empty_pairs_returns_none(self) -> None:
        assert alignment(FIXTURE, []) is None

    def test_identical_embeddings_zero_alignment(self) -> None:
        emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = alignment(emb, [(0, 1)])
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_pair_alignment_equals_two(self) -> None:
        # L2-normalised orthogonal vectors: ||u - v||² = 2(1 - 0) = 2
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = alignment(emb, [(0, 1)])
        assert result == pytest.approx(2.0, abs=1e-4)

    def test_returns_mean_over_pairs(self) -> None:
        # pair (0,1): dist=0; pair (0,2): dist=2 → mean=1
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = alignment(emb, [(0, 1), (0, 2)])
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_non_negative(self) -> None:
        result = alignment(FIXTURE, [(0, 1), (1, 2)])
        assert result is not None
        assert result >= 0.0

    def test_alpha_parameter(self) -> None:
        emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        # alpha=2: ||u-v||² = 2; alpha=1: ||u-v|| = sqrt(2)
        a2 = alignment(emb, [(0, 1)], alpha=2.0)
        a1 = alignment(emb, [(0, 1)], alpha=1.0)
        assert a2 is not None
        assert a1 is not None
        assert a2 != pytest.approx(a1)
