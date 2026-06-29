# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for precisionai.agrieval.emb.metrics.neighbors (P1 diagnostics)."""

from __future__ import annotations

import numpy as np
import pytest

from precisionai.agrieval.emb.metrics.neighbors import (
    gini_coefficient,
    hubness_at_k,
    knn_radius_at_k,
    mean_top_k_similarity,
    outlier_score_at_k,
)
from precisionai.agrieval.emb.metrics.similarity import top_k_neighbors

FIXTURE = np.array(
    [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
    dtype=np.float32,
)


@pytest.fixture
def neighbors() -> dict:
    return top_k_neighbors(FIXTURE, ks=[1, 3])


class TestGiniCoefficient:
    def test_perfect_equality_is_zero(self) -> None:
        assert gini_coefficient([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-5)

    def test_perfect_inequality_near_one(self) -> None:
        g = gini_coefficient([0, 0, 0, 100])
        assert g is not None
        assert g > 0.7

    def test_empty_returns_none(self) -> None:
        assert gini_coefficient([]) is None

    def test_all_zeros_returns_none(self) -> None:
        assert gini_coefficient([0, 0, 0]) is None


class TestMeanTopKSimilarity:
    def test_returns_per_item_array(self, neighbors: dict) -> None:
        result = mean_top_k_similarity(neighbors)
        for k in [1, 3]:
            assert "per_item" in result[k]
            assert len(result[k]["per_item"]) == len(FIXTURE)

    def test_scores_between_neg1_and_1(self, neighbors: dict) -> None:
        result = mean_top_k_similarity(neighbors)
        for k in result:
            per_item = result[k]["per_item"]
            assert np.all(per_item >= -1.0 - 1e-4)
            assert np.all(per_item <= 1.0 + 1e-4)

    def test_k1_mean_equals_radius(self, neighbors: dict) -> None:
        mean_result = mean_top_k_similarity(neighbors)
        radius_result = knn_radius_at_k(neighbors)
        np.testing.assert_allclose(mean_result[1]["per_item"], radius_result[1]["per_item"], atol=1e-5)


class TestKnnRadiusAtK:
    def test_fixture_clusters_visible_in_radius(self, neighbors: dict) -> None:
        result = knn_radius_at_k(neighbors)
        # items 0 & 1 are close; items 2 & 3 are close
        # k=1 radius for item 0 should be high (close to item 1)
        assert result[1]["per_item"][0] > 0.9

    def test_radius_at_k1_vs_k3(self, neighbors: dict) -> None:
        result = knn_radius_at_k(neighbors)
        # k=3 radius (3rd neighbor) should be <= k=1 radius (nearest neighbor)
        for i in range(len(FIXTURE)):
            assert result[3]["per_item"][i] <= result[1]["per_item"][i] + 1e-4


class TestOutlierScoreAtK:
    def test_low_score_for_clustered_items(self, neighbors: dict) -> None:
        result = outlier_score_at_k(neighbors)
        # Items 0 and 1 form a tight cluster — outlier score should be low
        assert result[1]["per_item"][0] < 0.2

    def test_top_outliers_sorted_by_score(self, neighbors: dict) -> None:
        result = outlier_score_at_k(neighbors, top_n=3)
        for k in result:
            scores = [x["score"] for x in result[k]["top_outliers"]]
            assert scores == sorted(scores, reverse=True)

    def test_kth_similarity_method(self, neighbors: dict) -> None:
        result = outlier_score_at_k(neighbors, method="one_minus_kth_similarity")
        for k in result:
            assert "per_item" in result[k]

    def test_invalid_method_raises(self, neighbors: dict) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            outlier_score_at_k(neighbors, method="invalid")


class TestHubnessAtK:
    def test_hub_counts_sum_to_n_times_k(self, neighbors: dict) -> None:
        n = len(FIXTURE)
        result = hubness_at_k(neighbors, n_items=n)
        for k in result:
            assert int(np.sum(result[k]["hub_counts"])) == n * k

    def test_hand_built_example(self) -> None:
        # v0 = [1,0,0]; v1,v2,v3 each have cos-sim 0.8 with v0
        # but only 0.28-0.64 with each other, so all point to v0
        vectors = np.array(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.8, 0.0, 0.6], [0.8, -0.6, 0.0]],
            dtype=np.float32,
        )
        nbrs = top_k_neighbors(vectors, ks=[1])
        result = hubness_at_k(nbrs, n_items=4)
        # Items 1, 2, 3 should all list item 0 as their nearest neighbor
        assert result[1]["hub_counts"][0] == 3

    def test_gini_is_between_0_and_1(self, neighbors: dict) -> None:
        n = len(FIXTURE)
        result = hubness_at_k(neighbors, n_items=n)
        for k in result:
            g = result[k]["gini"]
            if g is not None:
                assert 0.0 <= g <= 1.0 + 1e-5

    def test_anti_hub_count_non_negative(self, neighbors: dict) -> None:
        n = len(FIXTURE)
        result = hubness_at_k(neighbors, n_items=n)
        for k in result:
            assert result[k]["anti_hub_count"] >= 0
