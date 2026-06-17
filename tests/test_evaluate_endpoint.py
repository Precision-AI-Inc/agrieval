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
import pytest
from fastapi.testclient import TestClient

from pai.ag_emb.api.app import app
from pai.ag_emb.schemas.evaluate import MetadataGroup
from pai.ag_emb.services.evaluate import (
    _labels_from_metadata,
    build_image_items,
    extract_labels,
    run_evaluation,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings(
    class_vecs: dict[str, np.ndarray],
    n_per_class: int = 5,
) -> dict[str, list[float]]:
    """Build a {image_path: embedding} dict using the canonical path convention.

    Paths follow: ``img/{crop}/{timestamp}-{crop}-{hash}.png``
    """
    result: dict[str, list[float]] = {}
    rng = np.random.default_rng(0)
    for cls, proto in class_vecs.items():
        for i in range(n_per_class):
            vec = proto + rng.standard_normal(len(proto)).astype(np.float32) * 0.05
            vec = (vec / np.linalg.norm(vec)).astype(np.float32)
            path = f"img/{cls}/220101-{i:06d}-{cls}-abcd1234.png"
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
    def test_canonical_layout_with_root(self) -> None:
        """Standard layout: {root}/{class_subgroup}/{image}."""
        paths = [
            "images/corn_HB-25000SBC/220622-img1.png",
            "images/corn_nikon_d610/190627-img2.JPG",
            "images/soybean_HB-25000SBC/220608-img3.png",
            "images/soybean_anafi/210625-img4.JPG",
        ]
        assert extract_labels(paths, dataset_root="images") == ["corn", "corn", "soybean", "soybean"]

    def test_auto_root_inferred_from_common_prefix(self) -> None:
        """Root inferred from common prefix when dataset_root is not given."""
        paths = [
            "images/corn_HB-25000SBC/img1.png",
            "images/soybean_anafi/img2.JPG",
        ]
        assert extract_labels(paths) == ["corn", "soybean"]

    def test_explicit_dataset_root(self) -> None:
        paths = [
            "images/corn_HB-25000SBC/img1.png",
            "images/soybean_anafi/img2.JPG",
        ]
        assert extract_labels(paths, dataset_root="images") == ["corn", "soybean"]

    def test_no_common_root(self) -> None:
        paths = [
            "corn_HB-25000SBC/img1.png",
            "soybean_anafi/img2.JPG",
        ]
        assert extract_labels(paths) == ["corn", "soybean"]

    def test_absolute_paths(self) -> None:
        paths = [
            "/mnt/data/corn_HB-25000SBC/img1.png",
            "/mnt/data/soybean_anafi/img2.JPG",
        ]
        assert extract_labels(paths) == ["corn", "soybean"]

    def test_empty_returns_empty(self) -> None:
        assert extract_labels([]) == []

    def test_multiple_subgroups_same_class(self) -> None:
        """Multiple subgroups of the same class all map to the same label."""
        paths = [
            "images/corn_HB-25000SBC/img1.png",
            "images/corn_nikon_d610/img2.JPG",
        ]
        assert extract_labels(paths, dataset_root="images") == ["corn", "corn"]


# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    def test_returns_expected_top_level_keys(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        for key in (
            "n_items",
            "embedding_dim",
            "classes",
            "k_values",
            "item_paths",
            "item_labels",
            "knn_confusion",
            "global_metrics",
            "per_class",
        ):
            assert key in result
        assert "group_analysis" not in result

    def test_n_items_correct(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        assert result["n_items"] == len(GOOD_EMBEDDINGS)

    def test_classes_detected(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        assert sorted(result["classes"]) == ["corn", "soy"]

    def test_per_class_keys_match_classes(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        assert set(result["per_class"].keys()) == {"corn", "soy"}

    def test_per_class_n_items(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        assert result["per_class"]["corn"]["n_items"] == 5
        assert result["per_class"]["soy"]["n_items"] == 5

    def test_global_metrics_keys(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        gm = result["global_metrics"]
        for key in (
            "pairwise_similarity_stats",
            "intra_inter_similarity_gap",
            "knn_label_purity",
            "knn_label_ndcg",
            "knn_map",
            "knn_label_mrr",
            "knn_label_r_precision",
            "effective_rank",
            "centroid_similarity_stats",
            "uniformity",
            "hubness",
            "knn_radius",
            "mean_top_k_sim",
            "outlier_score",
        ):
            assert key in gm, f"Missing key: {key}"

    def test_separated_classes_positive_gap(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        gap = result["global_metrics"]["intra_inter_similarity_gap"]["gap"]
        assert gap is not None
        assert gap > 0.0, "Well-separated classes should have a positive similarity gap"

    def test_high_knn_purity_for_separated_classes(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        purity = result["global_metrics"]["knn_label_purity"]["3"]["mean"]
        assert purity is not None
        assert purity > 0.8, "KNN purity should be high for well-separated classes"

    def test_no_numpy_arrays_in_output(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        # Should not raise
        json.dumps(result)

    def test_per_class_metrics_keys(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
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
                "knn_label_mrr",
                "knn_label_r_precision",
            ):
                assert key in cls_data, f"Missing per-class key {key!r} for {cls!r}"

    def test_mismatched_dims_raises(self) -> None:
        bad = {
            "dataset/corn/cam/img1.png": [1.0, 0.0],
            "dataset/soy/cam/img2.png": [1.0, 0.0, 0.0],
        }
        with pytest.raises(ValueError, match="same dimension"):
            run_evaluation(bad, k_values=[1], dataset_root="dataset", sample_pairs=None)

    def test_too_few_embeddings_raises(self) -> None:
        single = {"dataset/corn/cam/img.png": [1.0, 0.0]}
        with pytest.raises(ValueError, match="At least 2"):
            run_evaluation(single, k_values=[1], dataset_root="dataset", sample_pairs=None)

    def test_empty_vectors_raises(self) -> None:
        empty_vecs: dict[str, list[float]] = {
            "corn_HB-25000SBC/img/img1.png": [],
            "soybean_anafi/img/img2.png": [],
        }
        with pytest.raises(ValueError, match="must not be empty"):
            run_evaluation(empty_vecs, k_values=[1], dataset_root=None, sample_pairs=None)

    def test_unsupported_extension_raises(self) -> None:
        bad = {
            "corn_HB-25000SBC/img/img1.tiff": [1.0, 0.0],
            "soybean_anafi/img/img2.tiff": [0.0, 1.0],
        }
        with pytest.raises(ValueError, match="Unsupported file extension"):
            run_evaluation(bad, k_values=[1], dataset_root=None, sample_pairs=None)

    def test_wrong_case_extension_raises(self) -> None:
        bad = {
            "corn_HB-25000SBC/img/img1.Png": [1.0, 0.0],
            "soybean_anafi/img/img2.Jpg": [0.0, 1.0],
        }
        with pytest.raises(ValueError, match="Unsupported file extension"):
            run_evaluation(bad, k_values=[1], dataset_root=None, sample_pairs=None)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    def test_returns_200(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": GOOD_EMBEDDINGS, "k_values": [3]},
        )
        assert response.status_code == 200

    def test_response_schema(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": GOOD_EMBEDDINGS, "k_values": [3]},
        )
        data = response.json()
        assert data["n_items"] == len(GOOD_EMBEDDINGS)
        assert data["embedding_dim"] == len(_CORN_PROTO)
        assert data["k_values"] == [3]
        assert sorted(data["classes"]) == ["corn", "soy"]
        assert isinstance(data["item_paths"], list)
        assert isinstance(data["item_labels"], list)
        assert len(data["item_paths"]) == len(GOOD_EMBEDDINGS)
        assert "knn_confusion" in data
        assert "global_metrics" in data
        assert "per_class" in data
        assert data.get("group_analysis") is None

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
            json={"embeddings": GOOD_EMBEDDINGS, "k_values": [3]},
        )
        # response.json() already parses — if it works, output is valid JSON
        assert isinstance(response.json(), dict)


# ---------------------------------------------------------------------------
# Metadata helpers (unit)
# ---------------------------------------------------------------------------

# Paths and metadata used in all metadata-related tests.
# Follows the canonical layout: images/class_subgroup/IMAGES, metadata/class_subgroup/
_META_PATHS = [
    "images/corn_HB-25000SBC/img-corn-a.png",
    "images/corn_HB-25000SBC/img-corn-b.png",
    "images/corn_nikon_d610/img-corn-c.JPG",
    "images/soybean_anafi/img-soy-a.JPG",
]
_META_GROUPS = {
    "corn_HB-25000SBC": MetadataGroup(
        images=["images/corn_HB-25000SBC/img-corn-a.png", "images/corn_HB-25000SBC/img-corn-b.png"],
        class_name="corn",
        attributes={"camera": "HB-25000SBC", "growth_stage": "medium"},
    ),
    "corn_nikon_d610": MetadataGroup(
        images=["images/corn_nikon_d610/img-corn-c.JPG"],
        class_name="corn",
        attributes={"camera": "nikon_d610", "growth_stage": "medium"},
    ),
    "soybean_anafi": MetadataGroup(
        images=["images/soybean_anafi/img-soy-a.JPG"],
        class_name="soybean",
        attributes={"camera": "anafi", "growth_stage": "medium"},
    ),
}


class TestLabelsFromMetadata:
    def test_labels_come_from_metadata(self) -> None:
        fallback = ["corn", "corn", "corn", "soybean"]
        labels = _labels_from_metadata(_META_PATHS, _META_GROUPS, fallback)
        assert labels == ["corn", "corn", "corn", "soybean"]

    def test_fallback_used_for_unmatched_paths(self) -> None:
        paths = ["images/unknown/mystery.png", *_META_PATHS[1:]]
        fallback = ["FALLBACK", "corn", "corn", "soybean"]
        labels = _labels_from_metadata(paths, _META_GROUPS, fallback)
        assert labels[0] == "FALLBACK"
        assert labels[1] == "corn"

    def test_exact_path_match(self) -> None:
        groups = {
            "corn_HB-25000SBC": MetadataGroup(images=["images/corn_HB-25000SBC/img-corn-a.png"], class_name="corn"),
        }
        paths = ["images/corn_HB-25000SBC/img-corn-a.png"]
        labels = _labels_from_metadata(paths, groups, ["fallback"])
        assert labels == ["corn"]

    def test_unmatched_path_uses_fallback(self) -> None:
        paths = ["images/unknown/no-match.png"]
        labels = _labels_from_metadata(paths, _META_GROUPS, ["FALLBACK"])
        assert labels == ["FALLBACK"]


class TestBuildImageItems:
    def test_returns_one_item_per_path(self) -> None:
        items = build_image_items(_META_PATHS, _META_GROUPS)
        assert len(items) == len(_META_PATHS)

    def test_image_id_is_exact_path(self) -> None:
        items = build_image_items(_META_PATHS, _META_GROUPS)
        assert items[0].image_id == "images/corn_HB-25000SBC/img-corn-a.png"
        assert items[3].image_id == "images/soybean_anafi/img-soy-a.JPG"

    def test_explicit_positive_ids_exclude_self(self) -> None:
        items = build_image_items(_META_PATHS, _META_GROUPS)
        # corn_HB group has two images: a and b
        assert "images/corn_HB-25000SBC/img-corn-b.png" in items[0].explicit_positive_ids
        assert "images/corn_HB-25000SBC/img-corn-a.png" not in items[0].explicit_positive_ids

    def test_class_name_and_attributes_populated(self) -> None:
        items = build_image_items(_META_PATHS, _META_GROUPS)
        assert items[0].class_name == "corn"
        assert items[0].attributes == {"camera": "HB-25000SBC", "growth_stage": "medium"}
        assert items[3].class_name == "soybean"
        assert items[3].attributes == {"camera": "anafi", "growth_stage": "medium"}

    def test_unmatched_path_has_empty_positives(self) -> None:
        paths = ["images/unknown_cam/mystery.png"]
        items = build_image_items(paths, _META_GROUPS)
        assert items[0].explicit_positive_ids == frozenset()
        assert items[0].class_name is None
        assert items[0].attributes == {}


# ---------------------------------------------------------------------------
# run_evaluation with metadata
# ---------------------------------------------------------------------------


def _make_metadata() -> dict:
    # Split corn into two subgroups with the same growth_stage so that same-class
    # items from different groups receive grade-2 relevance (same class + all
    # query attributes match), while cross-class items stay grade-0.  Soy gets a
    # different growth_stage so knn_attribute_ndcg has a non-trivial signal.
    corn_paths = sorted(p for p in GOOD_EMBEDDINGS if "/corn/" in p)
    soy_paths = [p for p in GOOD_EMBEDDINGS if "/soy/" in p]
    return {
        "corn_a": MetadataGroup(images=corn_paths[:3], class_name="corn", attributes={"growth_stage": "medium"}),
        "corn_b": MetadataGroup(images=corn_paths[3:], class_name="corn", attributes={"growth_stage": "medium"}),
        "soy": MetadataGroup(images=soy_paths, class_name="soy", attributes={"growth_stage": "early"}),
    }


_METADATA_KPI_KEYS = (
    "knn_metadata_map",
    "knn_metadata_mrr",
    "knn_metadata_ndcg",
    "knn_metadata_precision",
    "knn_metadata_r_precision",
)
_LABEL_KPI_KEYS = ("knn_label_purity", "knn_label_ndcg", "knn_map", "knn_label_mrr", "knn_label_r_precision")

_NEIGHBOR_DIAG_KEYS = ("hubness", "knn_radius", "mean_top_k_sim", "outlier_score")


class TestRunEvaluationWithMetadata:
    def test_metadata_kpis_present_in_global_metrics(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        gm = result["global_metrics"]
        for key in _METADATA_KPI_KEYS:
            assert key in gm, f"Missing metadata KPI key: {key}"

    def test_label_kpis_absent_from_global_metrics_when_metadata_provided(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        gm = result["global_metrics"]
        for key in _LABEL_KPI_KEYS:
            assert key not in gm, f"Label KPI should be absent when metadata provided: {key}"

    def test_metadata_kpis_present_in_per_class(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        for cls in ("corn", "soy"):
            for key in _METADATA_KPI_KEYS:
                assert key in result["per_class"][cls], f"Missing {key} in per_class[{cls}]"

    def test_label_kpis_absent_from_per_class_when_metadata_provided(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        for cls in ("corn", "soy"):
            for key in _LABEL_KPI_KEYS:
                assert key not in result["per_class"][cls], f"Label KPI should be absent: {key} in per_class[{cls}]"

    def test_without_metadata_shows_label_kpis_not_metadata_kpis(self) -> None:
        result = run_evaluation(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        gm = result["global_metrics"]
        for key in _LABEL_KPI_KEYS:
            assert key in gm, f"Missing label KPI key: {key}"
        for key in _METADATA_KPI_KEYS:
            assert key not in gm, f"Metadata KPI should be absent without metadata: {key}"

    def test_labels_from_metadata_class_name(self) -> None:
        # Override the path-extracted label "corn" with "maize" via metadata
        paths = list(GOOD_EMBEDDINGS.keys())[:2]
        two_embeddings = {p: GOOD_EMBEDDINGS[p] for p in paths}
        metadata = {
            "maize": MetadataGroup(images=paths, class_name="maize"),
        }
        result = run_evaluation(
            two_embeddings,
            k_values=[1],
            dataset_root=None,
            sample_pairs=None,
            metadata=metadata,
        )
        assert "maize" in result["classes"]

    def test_metadata_endpoint_returns_200(self) -> None:
        corn_paths = [p for p in GOOD_EMBEDDINGS if "/corn/" in p]
        soy_paths = [p for p in GOOD_EMBEDDINGS if "/soy/" in p]
        metadata_payload = {
            "corn": {
                "images": corn_paths,
                "class_name": "corn",
                "attributes": {"growth_stage": "medium"},
            },
            "soy": {
                "images": soy_paths,
                "class_name": "soy",
                "attributes": {"growth_stage": "medium"},
            },
        }
        response = client.post(
            "/v1/embeddings/evaluate",
            json={
                "embeddings": GOOD_EMBEDDINGS,
                "dataset_root": "dummy",
                "k_values": [3],
                "metadata": metadata_payload,
            },
        )
        assert response.status_code == 200
        assert "knn_metadata_ndcg" in response.json()["global_metrics"]

    def test_uniformity_always_present(self) -> None:
        result = run_evaluation(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        assert "uniformity" in result["global_metrics"]

    def test_uniformity_present_with_metadata(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        assert "uniformity" in result["global_metrics"]

    def test_alignment_present_with_metadata(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        assert "alignment" in result["global_metrics"]

    def test_knn_attribute_ndcg_present_in_global_metrics(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        assert "knn_attribute_ndcg" in result["global_metrics"]

    def test_alignment_absent_without_metadata(self) -> None:
        result = run_evaluation(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        assert "alignment" not in result["global_metrics"]

    def test_neighbor_diagnostics_present(self) -> None:
        result = run_evaluation(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        gm = result["global_metrics"]
        for key in _NEIGHBOR_DIAG_KEYS:
            assert key in gm, f"Missing neighbor diagnostic: {key}"

    def test_group_analysis_present_with_metadata(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        assert "group_analysis" in result
        assert isinstance(result["group_analysis"], dict)

    def test_group_analysis_absent_without_metadata(self) -> None:
        result = run_evaluation(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        assert "group_analysis" not in result

    def test_group_analysis_structure(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        for group_key, entry in result["group_analysis"].items():
            for field in (
                "n_images",
                "mean_intra_cosine",
                "std_intra_cosine",
                "cluster_count",
                "noise_count",
                "silhouette_score",
                "suggested_split",
            ):
                assert field in entry, f"Missing field {field!r} in group_analysis[{group_key!r}]"

    def test_per_class_metadata_mrr_and_r_precision(self) -> None:
        result = run_evaluation(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        for cls in ("corn", "soy"):
            cls_data = result["per_class"][cls]
            assert "knn_metadata_mrr" in cls_data, f"Missing knn_metadata_mrr in per_class[{cls}]"
            assert "knn_metadata_r_precision" in cls_data, f"Missing knn_metadata_r_precision in per_class[{cls}]"
