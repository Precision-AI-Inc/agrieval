# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for internal helpers in ``precisionai.agrieval.emb.metrics._utils``."""

from __future__ import annotations

import numpy as np
import pytest

from precisionai.agrieval.emb.metrics._utils import (
    _auto_batch_size,
    _get_pair_similarities,
    _neighbor_stats,
    _percentile_stats,
    _prepare_embeddings,
    _rankdata,
    _validate_embeddings,
)


class TestValidateEmbeddings:
    def test_accepts_2d_finite_input(self) -> None:
        emb = _validate_embeddings([[1.0, 2.0], [3.0, 4.0]], name="x")
        assert emb.shape == (2, 2)
        assert emb.dtype == np.float32

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError, match="2D array"):
            _validate_embeddings([1.0, 2.0, 3.0], name="vectors")

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_embeddings(np.empty((0, 4), dtype=np.float32), name="vectors")

    def test_rejects_non_finite_values(self) -> None:
        with pytest.raises(ValueError, match="NaN or inf"):
            _validate_embeddings([[1.0, np.nan]], name="vectors")


class TestAutoBatchSize:
    def test_returns_at_least_one(self) -> None:
        assert _auto_batch_size(10_000, target_bytes=1) == 1

    def test_zero_items_returns_full_budget_based_batch(self) -> None:
        assert _auto_batch_size(0, target_bytes=1234) == 1234

    def test_matches_expected_budget_formula(self) -> None:
        assert _auto_batch_size(100, target_bytes=4_000) == 10


class TestPercentileStats:
    def test_empty_array_returns_none_stats(self) -> None:
        stats = _percentile_stats(np.array([], dtype=np.float32))
        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["p99"] is None

    def test_non_empty_array_returns_descriptive_stats(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        stats = _percentile_stats(values)
        assert stats["count"] == 4
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(4.0)
        assert stats["p50"] == pytest.approx(2.5)


class TestNeighborStats:
    def test_empty_array_preserves_per_item_and_none_scalars(self) -> None:
        per_item = np.array([], dtype=np.float32)
        stats = _neighbor_stats(per_item)
        assert stats["per_item"] is per_item
        assert stats["mean"] is None
        assert stats["p95"] is None

    def test_non_empty_array_returns_summary(self) -> None:
        per_item = np.array([0.25, 0.5, 0.75], dtype=np.float32)
        stats = _neighbor_stats(per_item)
        assert stats["per_item"] is per_item
        assert stats["mean"] == pytest.approx(0.5)
        assert stats["p50"] == pytest.approx(0.5)


class TestPrepareEmbeddings:
    def test_returns_validated_embeddings_without_normalization(self) -> None:
        raw = np.array([[3.0, 4.0]], dtype=np.float32)
        prepared = _prepare_embeddings(raw, normalize=False)
        np.testing.assert_allclose(prepared, raw)

    def test_normalizes_rows_when_requested(self) -> None:
        prepared = _prepare_embeddings([[3.0, 4.0], [5.0, 12.0]], normalize=True)
        norms = np.linalg.norm(prepared, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)


class TestGetPairSimilarities:
    def test_full_pair_mode_returns_upper_triangle_similarities(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
            ],
            dtype=np.float32,
        )
        sims, total_unique = _get_pair_similarities(embeddings, sample_pairs=None, random_seed=0)
        assert total_unique == 3
        assert sims.dtype == np.float32
        np.testing.assert_allclose(np.sort(sims), np.array([0.0, 0.70710677, 0.70710677], dtype=np.float32))

    def test_sampled_mode_filters_self_pairs(self) -> None:
        embeddings = np.eye(3, dtype=np.float32)
        sims, total_unique = _get_pair_similarities(embeddings, sample_pairs=50, random_seed=42)
        assert total_unique == 3
        assert 0 < len(sims) <= 50
        assert np.all(np.isfinite(sims))
        assert np.all((sims >= -1.0) & (sims <= 1.0))


class TestRankdata:
    def test_assigns_average_ranks_for_ties(self) -> None:
        ranks = _rankdata(np.array([10.0, 20.0, 20.0, 30.0], dtype=np.float32))
        np.testing.assert_allclose(ranks, np.array([1.0, 2.5, 2.5, 4.0]))

    def test_all_equal_values_share_mean_rank(self) -> None:
        ranks = _rankdata(np.array([5.0, 5.0, 5.0], dtype=np.float32))
        np.testing.assert_allclose(ranks, np.array([2.0, 2.0, 2.0]))
