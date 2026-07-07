# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for precisionai.agrieval.emb.metrics.separation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import calinski_harabasz_score, roc_auc_score, silhouette_score

import precisionai.agrieval.emb.metrics.separation as separation_module
from precisionai.agrieval.emb.metrics.separation import (
    calinski_harabasz,
    kmeans_label_agreement,
    pc1_auroc,
    silhouette_cosine,
)


def _two_blobs(n_per: int = 30, dim: int = 8, gap: float = 6.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two direction-separated Gaussian blobs with 0/1 labels.

    The blobs sit along orthogonal axes rather than one being at the
    origin, so they separate under both Euclidean and cosine geometry.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_per, dim)).astype(np.float32)
    b = rng.normal(size=(n_per, dim)).astype(np.float32)
    a[:, 0] += gap
    b[:, 1] += gap
    features = np.vstack([a, b])
    labels = np.array([0] * n_per + [1] * n_per)
    return features, labels


def _random_labeled(n: int = 60, dim: int = 8, k: int = 4, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Random features with random labels covering k classes."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, dim)).astype(np.float32)
    labels = np.concatenate([np.arange(k), rng.integers(0, k, size=n - k)])
    return features, labels


class TestCalinskiHarabasz:
    def test_matches_sklearn(self) -> None:
        features, labels = _random_labeled()
        expected = calinski_harabasz_score(features, labels)
        assert calinski_harabasz(features, labels) == pytest.approx(expected, rel=1e-6)

    def test_separated_blobs_beat_shuffled_labels(self) -> None:
        features, labels = _two_blobs()
        shuffled = np.random.default_rng(2).permutation(labels)
        assert calinski_harabasz(features, labels) > calinski_harabasz(features, shuffled)

    def test_string_labels_accepted(self) -> None:
        features, labels = _two_blobs()
        names = np.where(labels == 0, "soil", "crop")
        assert calinski_harabasz(features, names) == pytest.approx(calinski_harabasz(features, labels), rel=1e-6)

    def test_single_class_raises(self) -> None:
        features, _ = _two_blobs()
        with pytest.raises(ValueError, match="at least 2 distinct labels"):
            calinski_harabasz(features, np.zeros(features.shape[0], dtype=int))

    def test_every_row_its_own_class_raises(self) -> None:
        features, _ = _two_blobs(n_per=3)
        with pytest.raises(ValueError, match="fewer classes than samples"):
            calinski_harabasz(features, np.arange(features.shape[0]))

    def test_zero_within_dispersion_is_inf(self) -> None:
        features = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        assert calinski_harabasz(features, labels) == float("inf")

    def test_all_identical_rows_is_zero(self) -> None:
        features = np.ones((6, 3), dtype=np.float32)
        labels = np.array([0, 0, 0, 1, 1, 1])
        assert calinski_harabasz(features, labels) == 0.0

    def test_label_length_mismatch_raises(self) -> None:
        features, labels = _two_blobs()
        with pytest.raises(ValueError, match="1D array of length"):
            calinski_harabasz(features, labels[:-1])


class TestSilhouetteCosine:
    def test_matches_sklearn(self) -> None:
        features, labels = _random_labeled()
        expected = silhouette_score(features, labels, metric="cosine")
        assert silhouette_cosine(features, labels) == pytest.approx(expected, abs=1e-6)

    def test_matches_sklearn_with_singleton_class(self) -> None:
        features, labels = _random_labeled()
        labels = labels.copy()
        labels[0] = 99  # a class with exactly one member
        expected = silhouette_score(features, labels, metric="cosine")
        assert silhouette_cosine(features, labels) == pytest.approx(expected, abs=1e-6)

    def test_chunked_path_matches_unchunked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        features, labels = _random_labeled()
        full = silhouette_cosine(features, labels)
        monkeypatch.setattr(separation_module, "_SILHOUETTE_CHUNK_ROWS", 7)
        assert silhouette_cosine(features, labels) == pytest.approx(full, abs=1e-12)

    def test_separated_blobs_score_high(self) -> None:
        features, labels = _two_blobs(gap=20.0)
        assert silhouette_cosine(features, labels) > 0.5

    def test_single_class_raises(self) -> None:
        features, _ = _two_blobs()
        with pytest.raises(ValueError, match="2 to n-1 distinct labels"):
            silhouette_cosine(features, np.zeros(features.shape[0], dtype=int))

    def test_every_row_its_own_class_raises(self) -> None:
        features, _ = _two_blobs(n_per=3)
        with pytest.raises(ValueError, match="2 to n-1 distinct labels"):
            silhouette_cosine(features, np.arange(features.shape[0]))


class TestKmeansLabelAgreement:
    def test_perfect_clusters_recovered(self) -> None:
        features, labels = _two_blobs(gap=20.0)
        result = kmeans_label_agreement(features, labels)
        assert result["ari"] == pytest.approx(1.0)
        assert result["nmi"] == pytest.approx(1.0)
        assert result["n_clusters"] == 2

    def test_random_labels_score_near_zero(self) -> None:
        features, labels = _random_labeled()
        result = kmeans_label_agreement(features, labels)
        assert result["ari"] < 0.3
        assert result["nmi"] < 0.3

    def test_deterministic(self) -> None:
        features, labels = _random_labeled()
        first = kmeans_label_agreement(features, labels)
        second = kmeans_label_agreement(features, labels)
        assert first == second

    def test_single_class_raises(self) -> None:
        features, _ = _two_blobs()
        with pytest.raises(ValueError, match="at least 2 distinct labels"):
            kmeans_label_agreement(features, np.zeros(features.shape[0], dtype=int))

    def test_missing_sklearn_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(separation_module, "_SKLEARN_AVAILABLE", False)
        features, labels = _two_blobs()
        with pytest.raises(ImportError, match="scikit-learn is required"):
            kmeans_label_agreement(features, labels)


class TestPc1Auroc:
    def test_matches_sklearn_sign_free(self) -> None:
        features, labels = _two_blobs(gap=2.0)
        fg = labels == 1
        centered = features.astype(np.float64) - features.mean(axis=0)
        _, eigvecs = np.linalg.eigh(centered.T @ centered)
        score = centered @ eigvecs[:, -1]
        expected = roc_auc_score(fg.astype(int), score)
        expected = max(expected, 1.0 - expected)
        assert pc1_auroc(features, fg) == pytest.approx(expected, abs=1e-9)

    def test_separable_partition_scores_near_one(self) -> None:
        features, labels = _two_blobs(gap=20.0)
        assert pc1_auroc(features, labels == 1) > 0.99

    def test_sign_free_under_negation(self) -> None:
        features, labels = _two_blobs(gap=3.0)
        fg = labels == 1
        assert pc1_auroc(-features, fg) == pytest.approx(pc1_auroc(features, fg), abs=1e-9)

    def test_no_signal_scores_near_half(self) -> None:
        rng = np.random.default_rng(5)
        features = rng.normal(size=(200, 8)).astype(np.float32)
        fg = np.arange(200) % 2 == 0
        assert pc1_auroc(features, fg) < 0.65

    def test_all_foreground_raises(self) -> None:
        features, _ = _two_blobs()
        with pytest.raises(ValueError, match=r"both foreground .* and background"):
            pc1_auroc(features, np.ones(features.shape[0], dtype=bool))

    def test_shape_mismatch_raises(self) -> None:
        features, labels = _two_blobs()
        with pytest.raises(ValueError, match="1D boolean array of length"):
            pc1_auroc(features, (labels == 1)[:-1])
