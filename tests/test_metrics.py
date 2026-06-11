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

"""Unit tests for pai.ag_emb.metrics.ranking using dummy vectors."""

from __future__ import annotations

import numpy as np
import pytest

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


class TestL2Normalize:
    def test_unit_vectors_unchanged(self) -> None:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(l2_normalize(vectors), vectors)

    def test_normalized_norms_are_one(self) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.random((10, 64)).astype(np.float32)
        norms = np.linalg.norm(l2_normalize(vectors), axis=1)
        np.testing.assert_allclose(norms, np.ones(10), atol=1e-6)

    def test_zero_vector_does_not_produce_nan(self) -> None:
        vectors = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        assert np.all(np.isfinite(l2_normalize(vectors)))

    def test_variable_dimension(self) -> None:
        rng = np.random.default_rng(1)
        for dim in [4, 16, 128, 512]:
            vectors = rng.random((5, dim)).astype(np.float32)
            result = l2_normalize(vectors)
            assert result.shape == (5, dim)


class TestNormalizeKs:
    def test_clamps_to_pool_size(self) -> None:
        assert normalize_ks([1, 5, 10], 3) == [1, 3]

    def test_deduplicates_after_clamping(self) -> None:
        assert normalize_ks([5, 5, 10], 20) == [5, 10]

    def test_returns_sorted_output(self) -> None:
        assert normalize_ks([10, 1, 5], 20) == [1, 5, 10]

    def test_raises_on_empty_pool(self) -> None:
        with pytest.raises(ValueError, match="candidate"):
            normalize_ks([1], 0)

    def test_raises_on_non_positive_k(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            normalize_ks([0], 10)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k([0, 1, 2], {0, 1, 2}, 3) == 1.0

    def test_none_when_no_relevant(self) -> None:
        assert precision_at_k([0, 1], set(), 2) is None

    def test_partial_overlap(self) -> None:
        assert precision_at_k([0, 1, 2, 3], {0, 2}, 4) == pytest.approx(0.5)


class TestRecallAtK:
    def test_full_recall(self) -> None:
        assert recall_at_k([0, 1, 2], {0, 1}, 3) == 1.0

    def test_none_when_no_relevant(self) -> None:
        assert recall_at_k([0, 1], set(), 2) is None

    def test_partial_recall(self) -> None:
        assert recall_at_k([0, 1, 2, 3], {0, 2, 4}, 2) == pytest.approx(1 / 3)


class TestAveragePrecisionAtK:
    def test_perfect_ranking(self) -> None:
        assert average_precision_at_k([0, 1], {0, 1}, 2) == pytest.approx(1.0)

    def test_none_when_no_relevant(self) -> None:
        assert average_precision_at_k([0, 1], set(), 2) is None

    def test_zero_when_no_hits(self) -> None:
        assert average_precision_at_k([0, 1], {2, 3}, 2) == 0.0


class TestDcgAtK:
    def test_single_hit_at_rank_one(self) -> None:
        assert dcg_at_k([1], 1) == pytest.approx(1.0)

    def test_empty_relevances(self) -> None:
        assert dcg_at_k([], 5) == 0.0

    def test_graded_relevance(self) -> None:
        assert dcg_at_k([3, 2, 1], 3) > dcg_at_k([1, 2, 3], 3)


class TestNdcgAtK:
    def test_perfect_ranking_is_one(self) -> None:
        assert ndcg_at_k([3, 2, 1], [3, 2, 1], 3) == pytest.approx(1.0)

    def test_none_when_no_ideal(self) -> None:
        assert ndcg_at_k([1, 0], [], 2) is None

    def test_imperfect_ranking_below_one(self) -> None:
        score = ndcg_at_k([1, 2, 3], [3, 2, 1], 3)
        assert score is not None
        assert score < 1.0


class TestSafeMean:
    def test_filters_none_values(self) -> None:
        assert safe_mean([1.0, None, 2.0, None]) == pytest.approx(1.5)

    def test_all_none_returns_none(self) -> None:
        assert safe_mean([None, None]) is None

    def test_empty_returns_none(self) -> None:
        assert safe_mean([]) is None


class TestCosineSimilarityPattern:
    """Tests for the cosine similarity pattern used in the evaluation route."""

    def test_orthogonal_vectors_have_zero_similarity(self) -> None:
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        sim = l2_normalize(vectors) @ l2_normalize(vectors).T
        assert sim[0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_identical_vectors_have_unit_similarity(self) -> None:
        vectors = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        sim = l2_normalize(vectors) @ l2_normalize(vectors).T
        assert sim[0, 1] == pytest.approx(1.0, abs=1e-6)

    def test_similarity_bounded_for_variable_dimensions(self) -> None:
        rng = np.random.default_rng(42)
        for dim in [4, 16, 64, 256]:
            vectors = rng.random((8, dim)).astype(np.float32)
            sim = l2_normalize(vectors) @ l2_normalize(vectors).T
            np.fill_diagonal(sim, np.nan)
            off_diag = sim[~np.isnan(sim)]
            assert np.all(off_diag >= -1.0 - 1e-5)
            assert np.all(off_diag <= 1.0 + 1e-5)
