# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the image2image evaluate endpoint, service layer, and schemas."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from precisionai.agrieval.emb.api.app import app
from precisionai.agrieval.emb.schemas.evaluate import EmbeddingEvaluateRequest, MetadataGroup
from precisionai.agrieval.emb.services.evaluate import (
    _jsonify,
    _labels_from_metadata,
    _parse_crop,
    build_image_items,
    extract_labels,
    run_image2image_eval,
)
from tests.test_plant_wirings import _P2I_EMBEDDINGS, _P2I_MAP

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
_A_PROTO = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_B_PROTO = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

GOOD_EMBEDDINGS = _make_embeddings({"A1": _A_PROTO, "B1": _B_PROTO})

_DIM = 8


def _unit(seed: int, dim: int = _DIM) -> list[float]:
    """Return a deterministic L2-normalised vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_two_class(n_per_class: int = 4, dim: int = _DIM, noise: float = 0.05) -> dict[str, list[float]]:
    """Two well-separated classes with n_per_class images each."""
    proto_a = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
    proto_b = np.array([0.0, 1.0] + [0.0] * (dim - 2), dtype=np.float32)
    rng = np.random.default_rng(42)
    result: dict[str, list[float]] = {}
    for i in range(n_per_class):
        v = proto_a + rng.standard_normal(dim).astype(np.float32) * noise
        result[f"ds/A1/img{i:03d}.png"] = (v / np.linalg.norm(v)).tolist()
    for i in range(n_per_class):
        v = proto_b + rng.standard_normal(dim).astype(np.float32) * noise
        result[f"ds/B1/img{i:03d}.png"] = (v / np.linalg.norm(v)).tolist()
    return result


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


class TestExtractLabels:
    def test_canonical_layout_with_root(self) -> None:
        """Canonical layout: {root}/{L2code}/{image}."""
        paths = [
            "images/A1/220622-img1.png",
            "images/A2/190627-img2.JPG",
            "images/B1/220608-img3.png",
            "images/B2/210625-img4.JPG",
        ]
        assert extract_labels(paths, dataset_root="images") == ["A", "A", "B", "B"]

    def test_auto_root_inferred_from_common_prefix(self) -> None:
        """Root inferred from common prefix when dataset_root is not given."""
        paths = [
            "images/A1/img1.png",
            "images/B1/img2.JPG",
        ]
        assert extract_labels(paths) == ["A", "B"]

    def test_explicit_dataset_root(self) -> None:
        paths = [
            "images/A1/img1.png",
            "images/B1/img2.JPG",
        ]
        assert extract_labels(paths, dataset_root="images") == ["A", "B"]

    def test_no_common_root(self) -> None:
        paths = [
            "A1/img1.png",
            "B1/img2.JPG",
        ]
        assert extract_labels(paths) == ["A", "B"]

    def test_absolute_paths(self) -> None:
        paths = [
            "/mnt/data/A1/img1.png",
            "/mnt/data/B1/img2.JPG",
        ]
        assert extract_labels(paths) == ["A", "B"]

    def test_empty_returns_empty(self) -> None:
        assert extract_labels([]) == []

    def test_multiple_l2_groups_same_l1_class(self) -> None:
        """Multiple L2 groups with the same prefix should map to the same L1 label."""
        paths = [
            "images/A1/img1.png",
            "images/A2/img2.JPG",
        ]
        assert extract_labels(paths, dataset_root="images") == ["A", "A"]


# ---------------------------------------------------------------------------
# Service layer
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    def test_returns_expected_top_level_keys(self) -> None:
        result = run_image2image_eval(
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
        result = run_image2image_eval(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        assert result["n_items"] == len(GOOD_EMBEDDINGS)

    def test_classes_detected(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        assert sorted(result["classes"]) == ["A", "B"]

    def test_per_class_keys_match_classes(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        assert set(result["per_class"].keys()) == {"A", "B"}

    def test_per_class_n_items(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        assert result["per_class"]["A"]["n_items"] == 5
        assert result["per_class"]["B"]["n_items"] == 5

    def test_global_metrics_keys(self) -> None:
        result = run_image2image_eval(
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
        result = run_image2image_eval(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        gap = result["global_metrics"]["intra_inter_similarity_gap"]["gap"]
        assert gap is not None
        assert gap > 0.0, "Well-separated classes should have a positive similarity gap"

    def test_high_knn_purity_for_separated_classes(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        purity = result["global_metrics"]["knn_label_purity"]["3"]["mean"]
        assert purity is not None
        assert purity > 0.8, "KNN purity should be high for well-separated classes"

    def test_no_numpy_arrays_in_output(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        # Should not raise
        json.dumps(result)

    def test_per_class_metrics_keys(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS,
            k_values=[3],
            dataset_root=None,
            sample_pairs=50,
        )
        for cls in ("A", "B"):
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
            "dataset/A1/img1.png": [1.0, 0.0],
            "dataset/B1/img2.png": [1.0, 0.0, 0.0],
        }
        with pytest.raises(ValueError, match="same dimension"):
            run_image2image_eval(bad, k_values=[1], dataset_root="dataset", sample_pairs=None)

    def test_too_few_embeddings_raises(self) -> None:
        single = {"dataset/corn/cam/img.png": [1.0, 0.0]}
        with pytest.raises(ValueError, match="At least 2"):
            run_image2image_eval(single, k_values=[1], dataset_root="dataset", sample_pairs=None)

    def test_empty_vectors_raises(self) -> None:
        empty_vecs: dict[str, list[float]] = {
            "A1/img/img1.png": [],
            "B1/img/img2.png": [],
        }
        with pytest.raises(ValueError, match="must not be empty"):
            run_image2image_eval(empty_vecs, k_values=[1], dataset_root=None, sample_pairs=None)

    def test_unsupported_extension_raises(self) -> None:
        bad = {
            "A1/img/img1.tiff": [1.0, 0.0],
            "B1/img/img2.tiff": [0.0, 1.0],
        }
        with pytest.raises(ValueError, match="Unsupported file extension"):
            run_image2image_eval(bad, k_values=[1], dataset_root=None, sample_pairs=None)

    def test_wrong_case_extension_raises(self) -> None:
        bad = {
            "A1/img/img1.Png": [1.0, 0.0],
            "B1/img/img2.Jpg": [0.0, 1.0],
        }
        with pytest.raises(ValueError, match="Unsupported file extension"):
            run_image2image_eval(bad, k_values=[1], dataset_root=None, sample_pairs=None)


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------


class TestAnalyzeEndpoint:
    def test_returns_200(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": GOOD_EMBEDDINGS, "k_values": [3]},
        )
        assert response.status_code == 200

    def test_response_schema(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": GOOD_EMBEDDINGS, "k_values": [3]},
        )
        data = response.json()
        assert data["n_items"] == len(GOOD_EMBEDDINGS)
        assert data["embedding_dim"] == len(_A_PROTO)
        assert data["k_values"] == [3]
        assert sorted(data["classes"]) == ["A", "B"]
        assert isinstance(data["item_paths"], list)
        assert isinstance(data["item_labels"], list)
        assert len(data["item_paths"]) == len(GOOD_EMBEDDINGS)
        assert "knn_confusion" in data
        assert "global_metrics" in data
        assert "per_class" in data
        assert data.get("group_analysis") is None

    def test_single_embedding_rejected(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": {"dataset/A1/img.png": [1.0, 0.0]}},
        )
        assert response.status_code == 422

    def test_inconsistent_dims_rejected(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={
                "embeddings": {
                    "dataset/A1/img1.png": [1.0, 0.0],
                    "dataset/B1/img2.png": [1.0, 0.0, 0.0],
                }
            },
        )
        assert response.status_code == 422

    def test_negative_k_rejected(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": GOOD_EMBEDDINGS, "k_values": [-1]},
        )
        assert response.status_code == 422

    def test_output_is_valid_json(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": GOOD_EMBEDDINGS, "k_values": [3]},
        )
        # response.json() already parses — if it works, output is valid JSON
        assert isinstance(response.json(), dict)


# ---------------------------------------------------------------------------
# Metadata helpers (unit)
# ---------------------------------------------------------------------------

# Paths and metadata used in all metadata-related tests.
_META_PATHS = [
    "images/A1/img-corn-a.png",
    "images/A1/img-corn-b.png",
    "images/A2/img-corn-c.JPG",
    "images/B1/img-soy-a.JPG",
]
_META_GROUPS = {
    "A1": MetadataGroup(
        images=["images/A1/img-corn-a.png", "images/A1/img-corn-b.png"],
        class_name="A1",
        attributes={"camera": "HB-25000SBC", "growth_stage": "medium"},
    ),
    "A2": MetadataGroup(
        images=["images/A2/img-corn-c.JPG"],
        class_name="A2",
        attributes={"camera": "nikon_d610", "growth_stage": "medium"},
    ),
    "B1": MetadataGroup(
        images=["images/B1/img-soy-a.JPG"],
        class_name="B1",
        attributes={"camera": "anafi", "growth_stage": "medium"},
    ),
}


class TestLabelsFromMetadata:
    def test_labels_come_from_metadata(self) -> None:
        fallback = ["A", "A", "A", "B"]
        labels = _labels_from_metadata(_META_PATHS, _META_GROUPS, fallback)
        assert labels == ["A", "A", "A", "B"]

    def test_fallback_used_for_unmatched_paths(self) -> None:
        paths = ["images/unknown/mystery.png", *_META_PATHS[1:]]
        fallback = ["FALLBACK", "A", "A", "B"]
        labels = _labels_from_metadata(paths, _META_GROUPS, fallback)
        assert labels[0] == "FALLBACK"
        assert labels[1] == "A"

    def test_exact_path_match(self) -> None:
        groups = {
            "A1": MetadataGroup(
                images=["images/A1/img-corn-a.png"],
                class_name="A1",
            ),
        }
        paths = ["images/A1/img-corn-a.png"]
        labels = _labels_from_metadata(paths, groups, ["fallback"])
        assert labels == ["A"]

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
        assert items[0].image_id == "images/A1/img-corn-a.png"
        assert items[3].image_id == "images/B1/img-soy-a.JPG"

    def test_explicit_positive_ids_exclude_self(self) -> None:
        items = build_image_items(_META_PATHS, _META_GROUPS)
        # A1 group has two images: a and b
        assert "images/A1/img-corn-b.png" in items[0].explicit_positive_ids
        assert "images/A1/img-corn-a.png" not in items[0].explicit_positive_ids

    def test_class_name_and_attributes_populated(self) -> None:
        items = build_image_items(_META_PATHS, _META_GROUPS)
        assert items[0].class_name == "A"
        assert items[0].attributes == {"camera": "HB-25000SBC", "growth_stage": "medium"}
        assert items[3].class_name == "B"
        assert items[3].attributes == {"camera": "anafi", "growth_stage": "medium"}

    def test_unmatched_path_has_empty_positives(self) -> None:
        paths = ["images/unknown/mystery.png"]
        items = build_image_items(paths, _META_GROUPS)
        assert items[0].explicit_positive_ids == frozenset()
        assert items[0].class_name is None
        assert items[0].attributes == {}


# ---------------------------------------------------------------------------
# run_image2image_eval with metadata
# ---------------------------------------------------------------------------


def _make_metadata() -> dict:
    # Split corn into two subgroups (A1/A2) with the same growth_stage so that
    # same-class items from different groups receive grade-2 relevance (same L1 class
    # + all query attributes match), while cross-class items stay grade-0.  Soy (B1)
    # gets a different growth_stage so knn_attribute_ndcg has a non-trivial signal.
    corn_paths = sorted(p for p in GOOD_EMBEDDINGS if "/A1/" in p)
    soy_paths = [p for p in GOOD_EMBEDDINGS if "/B1/" in p]
    return {
        "A1": MetadataGroup(images=corn_paths[:3], class_name="A1", attributes={"growth_stage": "medium"}),
        "A2": MetadataGroup(images=corn_paths[3:], class_name="A2", attributes={"growth_stage": "medium"}),
        "B1": MetadataGroup(images=soy_paths, class_name="B1", attributes={"growth_stage": "early"}),
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
        result = run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        gm = result["global_metrics"]
        for key in _METADATA_KPI_KEYS:
            assert key in gm, f"Missing metadata KPI key: {key}"

    def test_label_kpis_absent_from_global_metrics_when_metadata_provided(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        gm = result["global_metrics"]
        for key in _LABEL_KPI_KEYS:
            assert key not in gm, f"Label KPI should be absent when metadata provided: {key}"

    def test_metadata_kpis_present_in_per_class(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        for cls in ("A", "B"):
            for key in _METADATA_KPI_KEYS:
                assert key in result["per_class"][cls], f"Missing {key} in per_class[{cls}]"

    def test_label_kpis_absent_from_per_class_when_metadata_provided(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        for cls in ("A", "B"):
            for key in _LABEL_KPI_KEYS:
                assert key not in result["per_class"][cls], f"Label KPI should be absent: {key} in per_class[{cls}]"

    def test_without_metadata_shows_label_kpis_not_metadata_kpis(self) -> None:
        result = run_image2image_eval(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        gm = result["global_metrics"]
        for key in _LABEL_KPI_KEYS:
            assert key in gm, f"Missing label KPI key: {key}"
        for key in _METADATA_KPI_KEYS:
            assert key not in gm, f"Metadata KPI should be absent without metadata: {key}"

    def test_labels_from_metadata_class_name(self) -> None:
        # Override the path-extracted label "A" with "maize" via metadata
        paths = list(GOOD_EMBEDDINGS.keys())[:2]
        two_embeddings = {p: GOOD_EMBEDDINGS[p] for p in paths}
        metadata = {
            "maize": MetadataGroup(images=paths, class_name="maize"),
        }
        result = run_image2image_eval(
            two_embeddings,
            k_values=[1],
            dataset_root=None,
            sample_pairs=None,
            metadata=metadata,
        )
        assert "maize" in result["classes"]

    def test_metadata_endpoint_returns_200(self) -> None:
        corn_paths = [p for p in GOOD_EMBEDDINGS if "/A1/" in p]
        soy_paths = [p for p in GOOD_EMBEDDINGS if "/B1/" in p]
        metadata_payload = {
            "A1": {
                "images": corn_paths,
                "class_name": "A1",
                "attributes": {"growth_stage": "medium"},
            },
            "B1": {
                "images": soy_paths,
                "class_name": "B1",
                "attributes": {"growth_stage": "medium"},
            },
        }
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
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
        result = run_image2image_eval(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        assert "uniformity" in result["global_metrics"]

    def test_uniformity_present_with_metadata(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        assert "uniformity" in result["global_metrics"]

    def test_alignment_present_with_metadata(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        assert "alignment" in result["global_metrics"]

    def test_knn_attribute_ndcg_present_in_global_metrics(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        assert "knn_attribute_ndcg" in result["global_metrics"]

    def test_alignment_absent_without_metadata(self) -> None:
        result = run_image2image_eval(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        assert "alignment" not in result["global_metrics"]

    def test_neighbor_diagnostics_present(self) -> None:
        result = run_image2image_eval(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        gm = result["global_metrics"]
        for key in _NEIGHBOR_DIAG_KEYS:
            assert key in gm, f"Missing neighbor diagnostic: {key}"

    def test_group_analysis_present_with_metadata(self) -> None:
        result = run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        assert "group_analysis" in result
        assert isinstance(result["group_analysis"], dict)

    def test_group_analysis_absent_without_metadata(self) -> None:
        result = run_image2image_eval(GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50)
        assert "group_analysis" not in result

    def test_group_analysis_structure(self) -> None:
        result = run_image2image_eval(
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
        result = run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )
        for cls in ("A", "B"):
            cls_data = result["per_class"][cls]
            assert "knn_metadata_mrr" in cls_data, f"Missing knn_metadata_mrr in per_class[{cls}]"
            assert "knn_metadata_r_precision" in cls_data, f"Missing knn_metadata_r_precision in per_class[{cls}]"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestNanEmbeddingVulnerability:
    """NaN embeddings bypass the schema L2-norm guard at the Python level.

    JSON cannot encode NaN/inf, so this vulnerability is only reachable when
    the Pydantic model is constructed directly in Python (e.g. programmatic
    API clients, tests, internal service-to-service calls).  The guard
    ``abs(norm - 1.0) > tol`` evaluates to ``False`` for NaN because all NaN
    comparisons return ``False`` in Python, so a NaN vector silently passes
    ``validate_embeddings`` and reaches the service layer un-checked.
    """

    def test_nan_comparison_bypass_in_python(self) -> None:
        """Demonstrate that the L2-norm check silently passes NaN."""
        nan = float("nan")
        norm = math.sqrt(nan * nan)
        assert math.isnan(norm)
        assert not (abs(norm - 1.0) > 1e-3)

    def test_nan_embedding_raises_validation_error(self) -> None:
        nan_vec = [float("nan")] + [0.0] * (_DIM - 1)
        embeddings = {
            "ds/A1/nan_img.png": nan_vec,
            "ds/B1/good_img.png": _unit(0),
        }
        with pytest.raises(ValidationError, match="non-finite"):
            EmbeddingEvaluateRequest(embeddings=embeddings, k_values=[1])

    def test_inf_embedding_raises_at_schema_level(self) -> None:
        inf_vec = [1e38, 1e38] + [0.0] * (_DIM - 2)
        embeddings = {
            "ds/A1/large_img.png": inf_vec,
            "ds/B1/good_img.png": _unit(1),
        }
        with pytest.raises(ValidationError, match="normalised"):
            EmbeddingEvaluateRequest(embeddings=embeddings, k_values=[1])

    def test_clearly_unnormalized_embedding_raises(self) -> None:
        half_norm_vec = [0.5 / (_DIM**0.5)] * _DIM
        embeddings = {
            "ds/A1/img.png": half_norm_vec,
            "ds/B1/img.png": _unit(2),
        }
        with pytest.raises(ValidationError, match="normalised"):
            EmbeddingEvaluateRequest(embeddings=embeddings, k_values=[1])


class TestEmptyKValues:
    """k_values=[] passes ``any(k <= 0 for k in [])`` because any([]) is False.

    The schema currently accepts an empty k_values list. This causes the
    metrics pipeline to return empty KNN results with no useful diagnostics.
    Until fixed, the API must not return 500.
    """

    def test_any_of_empty_list_is_false(self) -> None:
        """Demonstrate the Python behaviour: any(k <= 0 for k in []) is False."""
        assert not any(k <= 0 for k in [])

    def test_schema_rejects_empty_k_values(self) -> None:
        with pytest.raises(ValidationError, match="at least one"):
            EmbeddingEvaluateRequest(embeddings=_make_two_class(n_per_class=2), k_values=[])

    def test_empty_k_values_rejected_by_schema(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": _make_two_class(), "k_values": []},
        )
        assert response.status_code == 422


class TestMetadataPathsValidated:
    """EmbeddingEvaluateRequest now validates that metadata.images ⊆ embeddings keys."""

    def test_ghost_path_in_metadata_rejected_by_schema(self) -> None:
        embeddings = _make_two_class(n_per_class=2)
        ghost = "ds/A1/ghost.png"
        with pytest.raises(ValidationError, match="not found in embeddings"):
            EmbeddingEvaluateRequest(
                embeddings=embeddings,
                metadata={"A1": MetadataGroup(images=[ghost], class_name="A1")},
            )

    def test_valid_metadata_paths_accepted(self) -> None:
        embeddings = _make_two_class(n_per_class=2)
        corn_paths = [p for p in embeddings if "/A1/" in p]
        req = EmbeddingEvaluateRequest(
            embeddings=embeddings,
            metadata={"A1": MetadataGroup(images=corn_paths, class_name="A1")},
        )
        assert req.metadata is not None


class TestDuplicateMetadataImage:
    """Same image in two metadata groups is now rejected at schema validation."""

    def test_duplicate_path_in_two_groups_raises(self) -> None:
        embeddings = _make_two_class(n_per_class=2)
        paths = list(embeddings.keys())
        metadata = {
            "group_a": MetadataGroup(images=[paths[0]], class_name="A1"),
            "group_b": MetadataGroup(images=[paths[0]], class_name="B1"),
        }
        with pytest.raises(ValidationError, match="multiple metadata groups"):
            EmbeddingEvaluateRequest(embeddings=embeddings, metadata=metadata)

    def test_non_overlapping_groups_accepted(self) -> None:
        embeddings = _make_two_class(n_per_class=2)
        corn_paths = [p for p in embeddings if "/A1/" in p]
        soy_paths = [p for p in embeddings if "/B1/" in p]
        req = EmbeddingEvaluateRequest(
            embeddings=embeddings,
            metadata={
                "A1": MetadataGroup(images=corn_paths, class_name="A1"),
                "B1": MetadataGroup(images=soy_paths, class_name="B1"),
            },
        )
        assert req.metadata is not None


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


class TestSingleClassDataset:
    """All embeddings belong to one class — inter-class metrics are undefined."""

    @pytest.fixture
    def single_class_embs(self) -> dict[str, list[float]]:
        proto = np.array([1.0] + [0.0] * (_DIM - 1), dtype=np.float32)
        rng = np.random.default_rng(0)
        result: dict[str, list[float]] = {}
        for i in range(6):
            v = proto + rng.standard_normal(_DIM).astype(np.float32) * 0.05
            result[f"ds/A1/img{i:03d}.png"] = (v / np.linalg.norm(v)).tolist()
        return result

    def test_gap_is_none(self, single_class_embs: dict) -> None:
        result = run_image2image_eval(single_class_embs, k_values=[3], dataset_root="ds", sample_pairs=20)
        gap_info = result["global_metrics"]["intra_inter_similarity_gap"]
        assert gap_info["gap"] is None
        assert gap_info["mean_inter_class_similarity"] is None

    def test_mean_intra_is_not_none(self, single_class_embs: dict) -> None:
        result = run_image2image_eval(single_class_embs, k_values=[3], dataset_root="ds", sample_pairs=20)
        assert result["global_metrics"]["intra_inter_similarity_gap"]["mean_intra_class_similarity"] is not None

    def test_per_class_has_exactly_one_entry(self, single_class_embs: dict) -> None:
        result = run_image2image_eval(single_class_embs, k_values=[3], dataset_root="ds", sample_pairs=20)
        assert list(result["per_class"].keys()) == ["A"]

    def test_knn_purity_is_one_for_single_class(self, single_class_embs: dict) -> None:
        result = run_image2image_eval(single_class_embs, k_values=[3], dataset_root="ds", sample_pairs=20)
        purity = result["global_metrics"]["knn_label_purity"]["3"]["mean"]
        assert purity == pytest.approx(1.0, abs=1e-6)

    def test_output_is_json_serialisable(self, single_class_embs: dict) -> None:
        result = run_image2image_eval(single_class_embs, k_values=[3], dataset_root="ds", sample_pairs=20)
        json.dumps(result)

    def test_api_returns_200(self, single_class_embs: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": single_class_embs, "dataset_root": "ds", "k_values": [3]},
        )
        assert response.status_code == 200


class TestMinimumViableDataset:
    """N=2 is the schema minimum. Every metric must produce a valid result."""

    @pytest.fixture
    def two_embs(self) -> dict[str, list[float]]:
        return {
            "ds/A1/img0.png": _unit(0),
            "ds/B1/img0.png": _unit(1),
        }

    def test_produces_valid_output(self, two_embs: dict) -> None:
        result = run_image2image_eval(two_embs, k_values=[1], dataset_root="ds", sample_pairs=None)
        assert result["n_items"] == 2

    def test_json_serialisable(self, two_embs: dict) -> None:
        result = run_image2image_eval(two_embs, k_values=[1], dataset_root="ds", sample_pairs=None)
        json.dumps(result)

    def test_api_returns_200(self, two_embs: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": two_embs, "dataset_root": "ds", "k_values": [1]},
        )
        assert response.status_code == 200

    def test_k_larger_than_n_minus_1_is_clamped(self, two_embs: dict) -> None:
        result = run_image2image_eval(two_embs, k_values=[100], dataset_root="ds", sample_pairs=None)
        assert result["k_values"] == [1]

    def test_purity_is_zero_when_two_different_classes(self, two_embs: dict) -> None:
        result = run_image2image_eval(two_embs, k_values=[1], dataset_root="ds", sample_pairs=None)
        purity = result["global_metrics"]["knn_label_purity"]["1"]["mean"]
        assert purity == pytest.approx(0.0, abs=1e-6)


class TestKValueClamping:
    """Requested k values exceeding the corpus size must be clamped to N-1."""

    def test_single_oversized_k_clamped(self) -> None:
        embs = _make_two_class(n_per_class=3)  # N=6
        result = run_image2image_eval(embs, k_values=[999], dataset_root="ds", sample_pairs=None)
        assert result["k_values"] == [5]

    def test_mixed_k_values_partial_clamping(self) -> None:
        embs = _make_two_class(n_per_class=3)  # N=6
        result = run_image2image_eval(embs, k_values=[2, 5, 999], dataset_root="ds", sample_pairs=None)
        assert 999 not in result["k_values"]
        for k in result["k_values"]:
            assert k <= 5

    def test_k_equals_n_minus_1_is_valid(self) -> None:
        embs = _make_two_class(n_per_class=3)  # N=6
        result = run_image2image_eval(embs, k_values=[5], dataset_root="ds", sample_pairs=None)
        assert result["k_values"] == [5]

    def test_api_does_not_500_for_oversized_k(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": _make_two_class(n_per_class=3), "k_values": [999999], "dataset_root": "ds"},
        )
        assert response.status_code == 200


class TestAllIdenticalEmbeddings:
    """All vectors are the same unit vector — every pairwise similarity is 1.0."""

    @pytest.fixture
    def identical_embs(self) -> dict[str, list[float]]:
        unit_vec = _unit(seed=0)
        return {
            "ds/A1/img0.png": unit_vec,
            "ds/A1/img1.png": unit_vec,
            "ds/B1/img0.png": unit_vec,
            "ds/B1/img1.png": unit_vec,
        }

    def test_accepted_by_schema(self, identical_embs: dict) -> None:
        req = EmbeddingEvaluateRequest(embeddings=identical_embs, k_values=[2])
        assert req is not None

    def test_pairwise_mean_is_one(self, identical_embs: dict) -> None:
        result = run_image2image_eval(identical_embs, k_values=[2], dataset_root="ds", sample_pairs=None)
        mean = result["global_metrics"]["pairwise_similarity_stats"]["mean"]
        assert mean == pytest.approx(1.0, abs=1e-4)

    def test_output_is_json_serialisable(self, identical_embs: dict) -> None:
        result = run_image2image_eval(identical_embs, k_values=[2], dataset_root="ds", sample_pairs=None)
        json.dumps(result)

    def test_api_does_not_500(self, identical_embs: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": identical_embs, "dataset_root": "ds", "k_values": [2]},
        )
        assert response.status_code != 500


# ---------------------------------------------------------------------------
# Internal function tests
# ---------------------------------------------------------------------------


class TestParseCrop:
    def test_l2_folder_returns_full_code(self) -> None:
        assert _parse_crop("A1") == "A1"
        assert _parse_crop("B2") == "B2"
        assert _parse_crop("AB12") == "AB12"
        assert _parse_crop("G7") == "G7"

    def test_plain_name_returns_whole_name(self) -> None:
        assert _parse_crop("corn") == "corn"
        assert _parse_crop("soybean") == "soybean"

    def test_empty_string_returns_empty(self) -> None:
        assert _parse_crop("") == ""

    def test_hyphen_not_treated_as_separator(self) -> None:
        """Hyphens are not separators; the full name is returned when no underscore present."""
        assert _parse_crop("HB-25000SBC") == "HB-25000SBC"


class TestExtractLabelsBoundaries:
    def test_single_path_with_explicit_root(self) -> None:
        assert extract_labels(["images/A1/img.png"], dataset_root="images") == ["A"]

    def test_single_path_auto_root(self) -> None:
        """With one path the common prefix strips the full file component, leaving
        the filename as the 'label'. Auto-root requires ≥2 paths to work correctly."""
        result = extract_labels(["images/A1/img.png"])
        assert len(result) == 1

    def test_dataset_root_with_trailing_slash_not_doubled(self) -> None:
        paths = ["images/A1/img.png", "images/B1/img.png"]
        result = extract_labels(paths, dataset_root="images/")
        assert result == ["A", "B"]

    def test_empty_list_returns_empty(self) -> None:
        assert extract_labels([]) == []

    def test_mixed_class_paths_with_explicit_root(self) -> None:
        paths = [
            "data/train/A1/img1.png",
            "data/train/B1/img2.png",
            "data/train/A1/img3.png",
        ]
        assert extract_labels(paths, dataset_root="data/train") == ["A", "B", "A"]

    def test_windows_style_backslash_paths_normalised(self) -> None:
        paths = [
            "images\\A1\\img.png",
            "images\\B1\\img.png",
        ]
        result = extract_labels(paths, dataset_root="images")
        assert result == ["A", "B"]

    def test_absolute_paths_with_explicit_root(self) -> None:
        paths = [
            "/mnt/data/A1/img1.png",
            "/mnt/data/B1/img2.png",
        ]
        assert extract_labels(paths, dataset_root="/mnt/data") == ["A", "B"]

    def test_dataset_root_as_empty_string_does_not_crash(self) -> None:
        """Empty string root is an edge case; must not raise."""
        paths = ["A1/img.png", "B1/img.png"]
        result = extract_labels(paths, dataset_root="")
        assert len(result) == 2


class TestJsonify:
    def test_nan_becomes_none(self) -> None:
        assert _jsonify(float("nan")) is None

    def test_positive_inf_becomes_none(self) -> None:
        assert _jsonify(float("inf")) is None

    def test_negative_inf_becomes_none(self) -> None:
        assert _jsonify(float("-inf")) is None

    def test_numpy_nan_becomes_none(self) -> None:
        assert _jsonify(np.float32("nan")) is None

    def test_numpy_inf_becomes_none(self) -> None:
        assert _jsonify(np.float32("inf")) is None

    def test_numpy_int64_becomes_python_int(self) -> None:
        result = _jsonify(np.int64(42))
        assert result == 42
        assert type(result) is int

    def test_numpy_float32_becomes_python_float(self) -> None:
        result = _jsonify(np.float32(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01

    def test_numpy_bool_becomes_python_bool(self) -> None:
        assert _jsonify(np.bool_(True)) is True
        assert type(_jsonify(np.bool_(False))) is bool

    def test_numpy_array_is_dropped_returns_none(self) -> None:
        assert _jsonify(np.array([1.0, 2.0, 3.0])) is None

    def test_dict_value_that_is_numpy_array_is_omitted(self) -> None:
        d = {"mean": np.float32(0.5), "per_item": np.array([0.4, 0.6])}
        result = _jsonify(d)
        assert "per_item" not in result
        assert result["mean"] == pytest.approx(0.5, abs=0.01)

    def test_nested_dict_nan_becomes_none(self) -> None:
        d = {"outer": {"inner_nan": float("nan"), "inner_ok": 1.0}}
        result = _jsonify(d)
        assert result["outer"]["inner_nan"] is None
        assert result["outer"]["inner_ok"] == 1.0

    def test_list_with_nan_and_inf(self) -> None:
        lst = [float("nan"), 1.0, float("inf"), -2.5]
        result = _jsonify(lst)
        assert result == [None, 1.0, None, -2.5]

    def test_python_scalars_pass_through(self) -> None:
        assert _jsonify(42) == 42
        assert _jsonify("hello") == "hello"
        assert _jsonify(None) is None
        assert _jsonify(True) is True

    def test_dict_key_converted_to_string(self) -> None:
        """dict keys that are non-string types are cast to str."""
        d = {5: "five"}
        result = _jsonify(d)
        assert "5" in result

    def test_entire_result_is_json_serialisable(self) -> None:
        obj = {
            "a": np.float32(1.0),
            "b": float("nan"),
            "c": np.int64(5),
            "d": np.array([1, 2]),
            "e": [np.float32(0.5), float("inf")],
            "f": {"nested_nan": float("nan")},
        }
        json.dumps(_jsonify(obj))


# ---------------------------------------------------------------------------
# Structural and metric invariants
# ---------------------------------------------------------------------------


class TestSingletonMetadataGroup:
    """A group with one image has no explicit positives — metrics must not crash."""

    @pytest.fixture
    def singleton_setup(self) -> tuple[dict, dict]:
        paths = [f"ds/A1/img{i:03d}.png" for i in range(4)]
        embeddings = {p: _unit(i) for i, p in enumerate(paths)}
        metadata = {
            "solo": MetadataGroup(images=[paths[0]], class_name="A1"),
            "pair": MetadataGroup(images=[paths[1], paths[2]], class_name="A2"),
            "other": MetadataGroup(images=[paths[3]], class_name="B1"),
        }
        return embeddings, metadata

    def test_singleton_has_empty_explicit_positives(self, singleton_setup: tuple) -> None:
        embeddings, metadata = singleton_setup
        items = build_image_items(list(embeddings.keys()), metadata)
        solo = next(it for it in items if it.image_id == "ds/A1/img000.png")
        assert solo.explicit_positive_ids == frozenset()

    def test_singleton_group_does_not_crash_evaluation(self, singleton_setup: tuple) -> None:
        embeddings, metadata = singleton_setup
        result = run_image2image_eval(embeddings, k_values=[2], dataset_root="ds", sample_pairs=None, metadata=metadata)
        json.dumps(result)

    def test_singleton_group_analysis_has_expected_fields(self, singleton_setup: tuple) -> None:
        embeddings, metadata = singleton_setup
        result = run_image2image_eval(embeddings, k_values=[2], dataset_root="ds", sample_pairs=None, metadata=metadata)
        for group_key, entry in result["group_analysis"].items():
            for field in ("n_images", "mean_intra_cosine", "cluster_count", "suggested_split"):
                assert field in entry, f"Missing {field!r} in group_analysis[{group_key!r}]"


class TestConfusionMatrixInvariants:
    """KNN confusion matrix rows must sum to 1.0 for every query class."""

    def test_rows_sum_to_one(self) -> None:
        result = run_image2image_eval(_make_two_class(), k_values=[3], dataset_root="ds", sample_pairs=20)
        for k_str, matrix in result["knn_confusion"].items():
            for true_class, fracs in matrix.items():
                total = sum(fracs.values())
                assert abs(total - 1.0) < 1e-6, f"Row {true_class!r} at k={k_str} sums to {total:.8f} (expected 1.0)"

    def test_keys_are_stringified_k_values(self) -> None:
        result = run_image2image_eval(_make_two_class(), k_values=[3, 5], dataset_root="ds", sample_pairs=20)
        assert set(result["knn_confusion"].keys()) == {"3", "5"}

    def test_all_classes_appear_as_row_keys(self) -> None:
        result = run_image2image_eval(_make_two_class(), k_values=[3], dataset_root="ds", sample_pairs=20)
        for matrix in result["knn_confusion"].values():
            assert "A" in matrix
            assert "B" in matrix


class TestNeighbourDiagnosticRanges:
    @pytest.fixture
    def diag_result(self) -> dict:
        return run_image2image_eval(_make_two_class(), k_values=[3], dataset_root="ds", sample_pairs=20)

    def test_hubness_gini_between_0_and_1(self, diag_result: dict) -> None:
        gini = diag_result["global_metrics"]["hubness"]["3"]["gini"]
        if gini is not None:
            assert 0.0 <= gini <= 1.0

    def test_knn_radius_scores_are_cosine_values(self, diag_result: dict) -> None:
        mean_radius = diag_result["global_metrics"]["knn_radius"]["3"]["mean"]
        if mean_radius is not None:
            assert -1.0 <= mean_radius <= 1.0

    def test_mean_top_k_sim_is_cosine_value(self, diag_result: dict) -> None:
        mean_sim = diag_result["global_metrics"]["mean_top_k_sim"]["3"]["mean"]
        if mean_sim is not None:
            assert -1.0 <= mean_sim <= 1.0

    def test_hubness_present_for_each_requested_k(self, diag_result: dict) -> None:
        assert "3" in diag_result["global_metrics"]["hubness"]


class TestBuildImageItemsStructure:
    def test_item_count_matches_path_count(self) -> None:
        paths = [f"ds/A1/img{i}.png" for i in range(5)]
        items = build_image_items(paths, {})
        assert len(items) == 5

    def test_item_order_matches_path_order(self) -> None:
        paths = ["ds/B1/img1.png", "ds/A1/img0.png", "ds/A1/img2.png"]
        items = build_image_items(paths, {})
        assert [it.image_id for it in items] == paths

    def test_self_not_in_explicit_positives(self) -> None:
        paths = ["ds/A1/img0.png", "ds/A1/img1.png"]
        metadata = {"g": MetadataGroup(images=paths, class_name="A1")}
        items = build_image_items(paths, metadata)
        for item in items:
            assert item.image_id not in item.explicit_positive_ids

    def test_group_members_are_mutual_positives(self) -> None:
        paths = ["ds/A1/img0.png", "ds/A1/img1.png", "ds/A1/img2.png"]
        metadata = {"g": MetadataGroup(images=paths, class_name="A1")}
        items = build_image_items(paths, metadata)
        for item in items:
            expected = frozenset(paths) - {item.image_id}
            assert item.explicit_positive_ids == expected

    def test_unmatched_path_gets_none_class_and_empty_positives(self) -> None:
        unmatched = "ds/unknown/mystery.png"
        items = build_image_items([unmatched], {})
        assert items[0].class_name is None
        assert items[0].explicit_positive_ids == frozenset()
        assert items[0].attributes == {}


class TestWellSeparatedClassMetrics:
    """For perfectly separated classes, specific metric values should be predictable."""

    @pytest.fixture
    def separated_result(self) -> dict:
        embs = _make_two_class(n_per_class=5, noise=0.001)
        return run_image2image_eval(embs, k_values=[3], dataset_root="ds", sample_pairs=50)

    def test_purity_near_one(self, separated_result: dict) -> None:
        purity = separated_result["global_metrics"]["knn_label_purity"]["3"]["mean"]
        assert purity == pytest.approx(1.0, abs=0.05)

    def test_ndcg_near_one(self, separated_result: dict) -> None:
        ndcg = separated_result["global_metrics"]["knn_label_ndcg"]["3"]["mean"]
        assert ndcg is not None
        assert ndcg > 0.9

    def test_positive_intra_inter_gap(self, separated_result: dict) -> None:
        gap = separated_result["global_metrics"]["intra_inter_similarity_gap"]["gap"]
        assert gap is not None
        assert gap > 0.0

    def test_intra_greater_than_inter(self, separated_result: dict) -> None:
        gap_info = separated_result["global_metrics"]["intra_inter_similarity_gap"]
        assert gap_info["mean_intra_class_similarity"] > gap_info["mean_inter_class_similarity"]

    def test_per_class_purity_near_one(self, separated_result: dict) -> None:
        for cls in ("A", "B"):
            purity = separated_result["per_class"][cls]["knn_label_purity"]["3"]["mean"]
            assert purity == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# API boundary conditions
# ---------------------------------------------------------------------------


class TestAPIBoundaryConditions:
    def test_unnormalized_vector_rejected_422(self) -> None:
        embeddings = {
            "ds/A1/img.png": [0.5, 0.5],
            "ds/B1/img.png": [1.0, 0.0],
        }
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": embeddings, "k_values": [1]},
        )
        assert response.status_code == 422

    def test_unnormalized_rejection_message_mentions_norm(self) -> None:
        embeddings = {
            "ds/A1/img.png": [0.5, 0.5],
            "ds/B1/img.png": [1.0, 0.0],
        }
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": embeddings, "k_values": [1]},
        )
        body = response.text.lower()
        assert "norm" in body or "normalised" in body or "normalized" in body

    def test_zero_k_rejected_422(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": _make_two_class(), "k_values": [0]},
        )
        assert response.status_code == 422

    def test_missing_embeddings_field_rejected_422(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"k_values": [3]},
        )
        assert response.status_code == 422

    def test_embeddings_as_list_not_dict_rejected_422(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": [[1.0, 0.0], [0.0, 1.0]], "k_values": [1]},
        )
        assert response.status_code == 422

    def test_unsupported_extension_causes_service_error(self) -> None:
        """Paths ending in .tiff are rejected by the service layer (not the schema).

        The schema does not validate extensions; the service ``_validate_embeddings``
        raises ValueError which propagates as 500.  The schema should also validate
        extensions so clients receive 422 instead.
        """
        lax_client = TestClient(app, raise_server_exceptions=False)
        embeddings = {
            "ds/A1/img.tiff": [1.0, 0.0],
            "ds/B1/img.tiff": [0.0, 1.0],
        }
        response = lax_client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": embeddings, "k_values": [1]},
        )
        assert response.status_code in {400, 422, 500}

    def test_case_sensitive_extension_wrong_case_causes_service_error(self) -> None:
        lax_client = TestClient(app, raise_server_exceptions=False)
        embeddings = {
            "ds/A1/img.Png": [1.0, 0.0],
            "ds/B1/img.Jpg": [0.0, 1.0],
        }
        response = lax_client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": embeddings, "k_values": [1]},
        )
        assert response.status_code in {400, 422, 500}

    def test_sample_pairs_one_does_not_crash(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": _make_two_class(), "k_values": [3], "sample_pairs": 1},
        )
        assert response.status_code == 200

    def test_sample_pairs_null_exhaustive_does_not_crash(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": _make_two_class(), "k_values": [3], "sample_pairs": None},
        )
        assert response.status_code == 200

    def test_large_k_clamped_returns_200(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": _make_two_class(n_per_class=3), "k_values": [999999], "dataset_root": "ds"},
        )
        assert response.status_code == 200

    def test_plant2image_endpoint_with_dataset_root_override(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={
                "embeddings": _P2I_EMBEDDINGS,
                "instance_to_image": _P2I_MAP,
                "dataset_root": "images",
                "k_values": [2],
            },
        )
        assert response.status_code == 200
