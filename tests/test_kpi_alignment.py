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

"""KPI alignment tests.

Verifies that label-aware KPIs respond correctly to embedding quality:

* **Tight** embeddings — orthogonal class prototypes with tiny per-sample
  noise (scale 0.02).  Every class is a tight cluster; inter-class cosine
  is near zero.  All retrieval metrics should be near-perfect.

* **Sparse** embeddings — nearby class prototypes (~60° apart) with large
  per-sample noise (scale 0.45).  Classes heavily overlap in embedding
  space.  All retrieval metrics should be substantially degraded.

The tests do not load any files; all embeddings are generated
deterministically from fixed seeds so the suite is self-contained and
reproducible.
"""

from __future__ import annotations

import numpy as np

from pai.ag_emb.metrics import (
    intra_inter_similarity_gap,
    knn_label_ndcg_at_k,
    knn_label_purity_at_k,
    knn_map_at_k,
    top_k_neighbors,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_N_PER_CLASS = 10
_DIM = 16
_K = 5
# MAP@K normalises by total relevant items (N-1), so MAP can only reach 1.0
# when K >= N_PER_CLASS - 1.  Use K_ALL for MAP tests.
_K_ALL = _N_PER_CLASS - 1

# ---------------------------------------------------------------------------
# Embedding generators
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return (a / np.linalg.norm(a)).astype(np.float32)


def _build_embeddings(
    proto_a: np.ndarray,
    proto_b: np.ndarray,
    noise_scale: float,
    n_per_class: int = _N_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings [2*N, D], labels [2*N]) for two-class scenarios.

    Each embedding is the class prototype perturbed by Gaussian noise then
    L2-normalised.  Seeds are fixed per sample so output is reproducible.
    """
    vecs: list[np.ndarray] = []
    for i in range(n_per_class):
        rng = np.random.default_rng(i)
        noise = rng.standard_normal(_DIM).astype(np.float32)
        v = proto_a + noise_scale * noise
        vecs.append(v / np.linalg.norm(v))

    for i in range(n_per_class):
        rng = np.random.default_rng(100 + i)
        noise = rng.standard_normal(_DIM).astype(np.float32)
        v = proto_b + noise_scale * noise
        vecs.append(v / np.linalg.norm(v))

    embeddings = np.stack(vecs, axis=0).astype(np.float32)
    labels = np.array(["corn"] * n_per_class + ["soybean"] * n_per_class)
    return embeddings, labels


# Tight: orthogonal prototypes, tiny noise → intra≈0.99, inter≈0.00
_TIGHT_PROTO_A = _unit([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
_TIGHT_PROTO_B = _unit([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
_TIGHT_EMB, _TIGHT_LABELS = _build_embeddings(_TIGHT_PROTO_A, _TIGHT_PROTO_B, noise_scale=0.02)

# Sparse: nearby prototypes (~60° apart), large noise → classes overlap heavily
_SPARSE_PROTO_A = _unit([1.0, 0.4, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
_SPARSE_PROTO_B = _unit([0.4, 1.0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
_SPARSE_EMB, _SPARSE_LABELS = _build_embeddings(_SPARSE_PROTO_A, _SPARSE_PROTO_B, noise_scale=0.45)

# Pre-compute neighbours once at module level (reused across tests).
# K_ALL (= N-1) is used for MAP so every same-class item is reachable and
# MAP can reach 1.0 for perfectly separated embeddings.
_TIGHT_NEIGHBORS = top_k_neighbors(_TIGHT_EMB, ks=[_K, _K_ALL])
_SPARSE_NEIGHBORS = top_k_neighbors(_SPARSE_EMB, ks=[_K, _K_ALL])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gap(emb: np.ndarray, labels: np.ndarray) -> dict:
    return intra_inter_similarity_gap(emb, labels, sample_pairs=None)


def _purity(neighbors: dict, labels: np.ndarray) -> float:
    return knn_label_purity_at_k(neighbors, labels)[_K]["mean"]


def _ndcg(neighbors: dict, labels: np.ndarray) -> float:
    return knn_label_ndcg_at_k(neighbors, labels)[_K]["mean"]


def _map(neighbors: dict, labels: np.ndarray, k: int = _K_ALL) -> float:
    return knn_map_at_k(neighbors, labels)[k]["mean"]


# ---------------------------------------------------------------------------
# Tight embeddings — all KPIs should be near-perfect
# ---------------------------------------------------------------------------


class TestTightEmbeddings:
    """Tight clusters: KPIs must be at or very close to their maximum values."""

    def test_intra_class_similarity_is_high(self) -> None:
        result = _gap(_TIGHT_EMB, _TIGHT_LABELS)
        assert result["mean_intra_class_similarity"] > 0.95

    def test_inter_class_similarity_is_near_zero(self) -> None:
        result = _gap(_TIGHT_EMB, _TIGHT_LABELS)
        assert result["mean_inter_class_similarity"] is not None
        assert result["mean_inter_class_similarity"] < 0.1

    def test_intra_inter_gap_is_large(self) -> None:
        result = _gap(_TIGHT_EMB, _TIGHT_LABELS)
        assert result["gap"] is not None
        assert result["gap"] > 0.85

    def test_knn_purity_is_near_perfect(self) -> None:
        assert _purity(_TIGHT_NEIGHBORS, _TIGHT_LABELS) > 0.95

    def test_knn_ndcg_is_near_perfect(self) -> None:
        assert _ndcg(_TIGHT_NEIGHBORS, _TIGHT_LABELS) > 0.95

    def test_knn_map_is_near_perfect(self) -> None:
        # MAP@K normalises by total relevant items (N_PER_CLASS - 1).
        # Using K_ALL (= N-1) so every same-class neighbour is reachable
        # and MAP can reach 1.0 for perfectly separated embeddings.
        assert _map(_TIGHT_NEIGHBORS, _TIGHT_LABELS) > 0.95

    def test_pair_counts_are_correct(self) -> None:
        result = _gap(_TIGHT_EMB, _TIGHT_LABELS)
        n = _N_PER_CLASS
        # Unordered intra pairs: 2 classes x C(n, 2) = n*(n-1)
        expected_intra = n * (n - 1)
        # Unordered inter pairs: n_class_a x n_class_b
        expected_inter = n * n
        assert result["num_intra_pairs"] == expected_intra
        assert result["num_inter_pairs"] == expected_inter


# ---------------------------------------------------------------------------
# Sparse embeddings — all KPIs should be substantially degraded
# ---------------------------------------------------------------------------


class TestSparseEmbeddings:
    """Overlapping clusters: KPIs must reflect poor separation."""

    def test_intra_class_similarity_is_low(self) -> None:
        result = _gap(_SPARSE_EMB, _SPARSE_LABELS)
        assert result["mean_intra_class_similarity"] < 0.5

    def test_intra_inter_gap_is_small(self) -> None:
        result = _gap(_SPARSE_EMB, _SPARSE_LABELS)
        assert result["gap"] is not None
        assert result["gap"] < 0.2

    def test_knn_purity_is_degraded(self) -> None:
        assert _purity(_SPARSE_NEIGHBORS, _SPARSE_LABELS) < 0.75

    def test_knn_ndcg_is_degraded(self) -> None:
        assert _ndcg(_SPARSE_NEIGHBORS, _SPARSE_LABELS) < 0.75

    def test_knn_map_is_degraded(self) -> None:
        assert _map(_SPARSE_NEIGHBORS, _SPARSE_LABELS) < 0.75


# ---------------------------------------------------------------------------
# Comparative: tight must outperform sparse on every KPI
# ---------------------------------------------------------------------------


class TestTightOutperformsSparse:
    """Tight embeddings must score strictly higher than sparse on every KPI."""

    def test_gap_tight_greater_than_sparse(self) -> None:
        tight_gap = _gap(_TIGHT_EMB, _TIGHT_LABELS)["gap"]
        sparse_gap = _gap(_SPARSE_EMB, _SPARSE_LABELS)["gap"]
        assert tight_gap is not None
        assert sparse_gap is not None
        assert tight_gap > sparse_gap

    def test_intra_similarity_tight_greater_than_sparse(self) -> None:
        tight_intra = _gap(_TIGHT_EMB, _TIGHT_LABELS)["mean_intra_class_similarity"]
        sparse_intra = _gap(_SPARSE_EMB, _SPARSE_LABELS)["mean_intra_class_similarity"]
        assert tight_intra > sparse_intra

    def test_inter_similarity_tight_less_than_sparse(self) -> None:
        tight_inter = _gap(_TIGHT_EMB, _TIGHT_LABELS)["mean_inter_class_similarity"]
        sparse_inter = _gap(_SPARSE_EMB, _SPARSE_LABELS)["mean_inter_class_similarity"]
        assert tight_inter is not None
        assert sparse_inter is not None
        # Tight classes are orthogonal → lower cross-class leakage
        assert tight_inter < sparse_inter

    def test_purity_tight_greater_than_sparse(self) -> None:
        assert _purity(_TIGHT_NEIGHBORS, _TIGHT_LABELS) > _purity(_SPARSE_NEIGHBORS, _SPARSE_LABELS)

    def test_ndcg_tight_greater_than_sparse(self) -> None:
        assert _ndcg(_TIGHT_NEIGHBORS, _TIGHT_LABELS) > _ndcg(_SPARSE_NEIGHBORS, _SPARSE_LABELS)

    def test_map_tight_greater_than_sparse(self) -> None:
        assert _map(_TIGHT_NEIGHBORS, _TIGHT_LABELS) > _map(_SPARSE_NEIGHBORS, _SPARSE_LABELS)
