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

"""Tests for the /v1/embeddings/evaluate endpoint and its service layer."""

from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from pai.ag_emb.api.app import app
from pai.ag_emb.services.evaluate import extract_labels, run_evaluation

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings(
    class_vecs: dict[str, np.ndarray],
    root: str = "dummy",
    camera: str = "HB-25000SBC",
    n_per_class: int = 5,
) -> dict[str, list[float]]:
    """Build a {image_path: embedding} dict using the real path convention.

    Paths follow: ``{root}/{crop}_[{camera}]/img/{timestamp}-{crop}-{camera}-{hash}.png``
    """
    result: dict[str, list[float]] = {}
    rng = np.random.default_rng(0)
    for cls, proto in class_vecs.items():
        for i in range(n_per_class):
            vec = proto + rng.standard_normal(len(proto)).astype(np.float32) * 0.05
            vec = (vec / np.linalg.norm(vec)).astype(np.float32)
            path = f"{root}/{cls}_[{camera}]/img/220101-{i:06d}-{cls}-{camera}-abcd1234.png"
            result[path] = vec.tolist()
    return result


# Two well-separated 8-D class prototypes
_CORN_PROTO = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_SOY_PROTO = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

GOOD_EMBEDDINGS = _make_embeddings({"corn": _CORN_PROTO, "soy": _SOY_PROTO})


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


class TestExtractLabels:
    def test_crop_camera_bracket_convention(self) -> None:
        """Real dataset layout: {root}/{crop}_[{camera}]/img/{image}."""
        paths = [
            "dummy/corn_[HB-25000SBC]/img/220622-img1.png",
            "dummy/corn_[nikon_d610]/img/190627-img2.jpg",
            "dummy/soybean_[HB-25000SBC]/img/220608-img3.png",
            "dummy/soybean_[anafi]/img/210625-img4.jpg",
        ]
        assert extract_labels(paths, dataset_root="dummy") == ["corn", "corn", "soybean", "soybean"]

    def test_crop_camera_auto_root(self) -> None:
        """Root inferred from common prefix when dataset_root is not given."""
        paths = [
            "dummy/corn_[HB-25000SBC]/img/img1.png",
            "dummy/soybean_[anafi]/img/img2.jpg",
        ]
        assert extract_labels(paths) == ["corn", "soybean"]

    def test_plain_folder_convention(self) -> None:
        """Plain {root}/{crop}/{camera}/{image} layout still works."""
        paths = [
            "dataset/corn/cam_a/img1.png",
            "dataset/corn/cam_b/img2.png",
            "dataset/soy/cam_a/img3.png",
        ]
        assert extract_labels(paths) == ["corn", "corn", "soy"]

    def test_explicit_dataset_root(self) -> None:
        paths = [
            "dummy/corn_[HB-25000SBC]/img/img1.png",
            "dummy/soybean_[anafi]/img/img2.jpg",
        ]
        assert extract_labels(paths, dataset_root="dummy") == ["corn", "soybean"]

    def test_no_common_root(self) -> None:
        paths = [
            "corn_[HB-25000SBC]/img/img1.png",
            "soybean_[anafi]/img/img2.jpg",
        ]
        assert extract_labels(paths) == ["corn", "soybean"]

    def test_absolute_paths(self) -> None:
        paths = [
            "/mnt/data/corn_[HB-25000SBC]/img/img1.png",
            "/mnt/data/soybean_[anafi]/img/img2.jpg",
        ]
        labels = extract_labels(paths)
        assert labels == ["corn", "soybean"]

    def test_empty_returns_empty(self) -> None:
        assert extract_labels([]) == []

    def test_single_class_all_same_prefix(self) -> None:
        """All paths belong to the same crop — label extracted from crop_[camera] folder."""
        paths = [
            "dummy/corn_[HB-25000SBC]/img/img1.png",
            "dummy/corn_[nikon_d610]/img/img2.jpg",
        ]
        labels = extract_labels(paths, dataset_root="dummy")
        assert labels == ["corn", "corn"]


# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    def test_returns_expected_top_level_keys(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        for key in ("n_items", "embedding_dim", "classes", "k_values", "global_metrics", "per_class"):
            assert key in result

    def test_n_items_correct(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        assert result["n_items"] == len(GOOD_EMBEDDINGS)

    def test_classes_detected(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        assert sorted(result["classes"]) == ["corn", "soy"]

    def test_per_class_keys_match_classes(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        assert set(result["per_class"].keys()) == {"corn", "soy"}

    def test_per_class_n_items(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        assert result["per_class"]["corn"]["n_items"] == 5
        assert result["per_class"]["soy"]["n_items"] == 5

    def test_global_metrics_keys(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        gm = result["global_metrics"]
        for key in (
            "pairwise_similarity_stats",
            "intra_inter_similarity_gap",
            "knn_label_purity",
            "knn_label_ndcg",
            "knn_map",
            "effective_rank",
            "centroid_similarity_stats",
        ):
            assert key in gm, f"Missing key: {key}"

    def test_separated_classes_positive_gap(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        gap = result["global_metrics"]["intra_inter_similarity_gap"]["gap"]
        assert gap is not None
        assert gap > 0.0, "Well-separated classes should have a positive similarity gap"

    def test_high_knn_purity_for_separated_classes(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        purity = result["global_metrics"]["knn_label_purity"]["3"]["mean"]
        assert purity is not None
        assert purity > 0.8, "KNN purity should be high for well-separated classes"

    def test_no_numpy_arrays_in_output(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        # Should not raise
        json.dumps(result)

    def test_per_class_metrics_keys(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root="dummy",
            sample_pairs=50,
        )
        for cls in ("corn", "soy"):
            cls_data = result["per_class"][cls]
            for key in (
                "n_items",
                "pairwise_similarity_stats",
                "centroid_similarity_stats",
                "effective_rank",
                "knn_label_purity",
                "knn_label_ndcg",
                "knn_map",
            ):
                assert key in cls_data, f"Missing per-class key {key!r} for {cls!r}"


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    def test_returns_200(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": GOOD_EMBEDDINGS, "dataset_root": "dummy", "k_values": [3]},
        )
        assert response.status_code == 200

    def test_response_schema(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": GOOD_EMBEDDINGS, "dataset_root": "dummy", "k_values": [3]},
        )
        data = response.json()
        assert data["n_items"] == len(GOOD_EMBEDDINGS)
        assert "global_metrics" in data
        assert "per_class" in data
        assert sorted(data["classes"]) == ["corn", "soy"]

    def test_single_embedding_rejected(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": {"dataset/corn/c/img.png": [1.0, 0.0]}},
        )
        assert response.status_code == 422

    def test_inconsistent_dims_rejected(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={
                "embeddings": {
                    "dataset/corn/c/img1.png": [1.0, 0.0],
                    "dataset/soy/c/img2.png": [1.0, 0.0, 0.0],
                }
            },
        )
        assert response.status_code == 422

    def test_negative_k_rejected(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": GOOD_EMBEDDINGS, "k_values": [-1]},
        )
        assert response.status_code == 422

    def test_output_is_valid_json(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": GOOD_EMBEDDINGS, "dataset_root": "dummy", "k_values": [3]},
        )
        # response.json() already parses — if it works, output is valid JSON
        assert isinstance(response.json(), dict)
