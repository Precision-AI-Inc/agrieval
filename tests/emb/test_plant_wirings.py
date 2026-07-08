# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Plant2Image and Plant2Plant data ingestion wirings."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from precisionai.agrieval.emb.api.app import app
from precisionai.agrieval.emb.schemas.evaluate import Plant2ImageRequest, Plant2PlantRequest
from precisionai.agrieval.emb.services.evaluate import (
    run_plant2image_eval,
    run_plant2plant_eval,
)
from precisionai.agrieval.emb.services.labels import (
    plant2image_to_metadata,
    plant2plant_to_metadata,
)

client = TestClient(app)

# ---------------------------------------------------------------------------
# Shared test prototypes (8-D, well-separated)
# ---------------------------------------------------------------------------

_DIM = 8
_A_PROTO = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_B_PROTO = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_C_PROTO = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _l2(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _perturb(proto: np.ndarray, seed: int, scale: float = 0.05) -> list[float]:
    rng = np.random.default_rng(seed)
    v = proto + rng.standard_normal(len(proto)).astype(np.float32) * scale
    return _l2(v).tolist()


# ---------------------------------------------------------------------------
# Plant2Image fixtures
# ---------------------------------------------------------------------------

# Layout:
#   images/A1/field001.png  ← parent (A1)
#   images/A1/field001-0.png  ← instance 0
#   images/A1/field001-1.png  ← instance 1
#   images/B1/field002.png  ← parent (B1)
#   images/B1/field002-0.png  ← instance 0

_P2I_EMBEDDINGS: dict[str, list[float]] = {
    "images/A1/field001.png": _perturb(_A_PROTO, 0),
    "images/A1/field001-0.png": _perturb(_A_PROTO, 1),
    "images/A1/field001-1.png": _perturb(_A_PROTO, 2),
    "images/B1/field002.png": _perturb(_B_PROTO, 3),
    "images/B1/field002-0.png": _perturb(_B_PROTO, 4),
}

_P2I_MAP: dict[str, list[str]] = {
    "images/A1/field001.png": [
        "images/A1/field001-0.png",
        "images/A1/field001-1.png",
    ],
    "images/B1/field002.png": [
        "images/B1/field002-0.png",
    ],
}

# ---------------------------------------------------------------------------
# Plant2Plant fixtures
# ---------------------------------------------------------------------------

# Layout:
#   images/A1/inst-0.png  A1 instance
#   images/A1/inst-1.png  A1 instance
#   images/A1/inst-2.png  A1 instance
#   images/B1/inst-3.png  B1 instance
#   images/B1/inst-4.png  B1 instance
#   images/C1/inst-5.png  C1 instance
#   images/C1/inst-6.png  C1 instance

_P2P_EMBEDDINGS: dict[str, list[float]] = {
    "images/A1/inst-0.png": _perturb(_A_PROTO, 10),
    "images/A1/inst-1.png": _perturb(_A_PROTO, 11),
    "images/A1/inst-2.png": _perturb(_A_PROTO, 12),
    "images/B1/inst-3.png": _perturb(_B_PROTO, 13),
    "images/B1/inst-4.png": _perturb(_B_PROTO, 14),
    "images/C1/inst-5.png": _perturb(_C_PROTO, 15),
    "images/C1/inst-6.png": _perturb(_C_PROTO, 16),
}

_P2P_LABELS: dict[str, str] = {
    "images/A1/inst-0.png": "A1",
    "images/A1/inst-1.png": "A1",
    "images/A1/inst-2.png": "A1",
    "images/B1/inst-3.png": "B1",
    "images/B1/inst-4.png": "B1",
    "images/C1/inst-5.png": "C1",
    "images/C1/inst-6.png": "C1",
}

# ---------------------------------------------------------------------------
# plant2image_to_metadata
# ---------------------------------------------------------------------------


class TestPlant2ImageToMetadata:
    def test_returns_one_group_per_parent(self) -> None:
        meta = plant2image_to_metadata(_P2I_MAP)
        assert set(meta.keys()) == set(_P2I_MAP.keys())

    def test_group_images_contain_parent_and_instances(self) -> None:
        meta = plant2image_to_metadata(_P2I_MAP)
        a1_group = meta["images/A1/field001.png"]
        assert "images/A1/field001.png" in a1_group.images
        assert "images/A1/field001-0.png" in a1_group.images
        assert "images/A1/field001-1.png" in a1_group.images
        assert len(a1_group.images) == 3

    def test_class_name_from_parent_path(self) -> None:
        meta = plant2image_to_metadata(_P2I_MAP)
        assert meta["images/A1/field001.png"].class_name == "A1"
        assert meta["images/A1/field001.png"].l1_cluster == "A"
        assert meta["images/B1/field002.png"].class_name == "B1"
        assert meta["images/B1/field002.png"].l1_cluster == "B"

    def test_dataset_root_overrides_extraction(self) -> None:
        meta = plant2image_to_metadata(_P2I_MAP, dataset_root="images")
        assert meta["images/A1/field001.png"].class_name == "A1"
        assert meta["images/A1/field001.png"].l1_cluster == "A"
        assert meta["images/B1/field002.png"].class_name == "B1"
        assert meta["images/B1/field002.png"].l1_cluster == "B"

    def test_empty_instance_list_is_valid(self) -> None:
        single_map = {"images/A1/solo.png": []}
        meta = plant2image_to_metadata(single_map)
        assert "images/A1/solo.png" in meta
        assert meta["images/A1/solo.png"].images == ["images/A1/solo.png"]

    def test_empty_map_returns_empty(self) -> None:
        assert plant2image_to_metadata({}) == {}


# ---------------------------------------------------------------------------
# plant2plant_to_metadata
# ---------------------------------------------------------------------------


class TestPlant2PlantToMetadata:
    def test_creates_one_group_per_class(self) -> None:
        meta = plant2plant_to_metadata(_P2P_LABELS)
        assert set(meta.keys()) == {"A1", "B1", "C1"}

    def test_all_class_members_in_group(self) -> None:
        meta = plant2plant_to_metadata(_P2P_LABELS)
        a1_members = set(meta["A1"].images)
        assert a1_members == {
            "images/A1/inst-0.png",
            "images/A1/inst-1.png",
            "images/A1/inst-2.png",
        }

    def test_class_name_correct(self) -> None:
        meta = plant2plant_to_metadata(_P2P_LABELS)
        assert meta["A1"].class_name == "A1"
        assert meta["A1"].l1_cluster == "A"
        assert meta["B1"].class_name == "B1"
        assert meta["B1"].l1_cluster == "B"
        assert meta["C1"].class_name == "C1"
        assert meta["C1"].l1_cluster == "C"


# ---------------------------------------------------------------------------
# run_plant2image_eval
# ---------------------------------------------------------------------------


class TestRunPlant2ImageEvaluation:
    def test_returns_expected_keys(self) -> None:
        result = run_plant2image_eval(
            embeddings=_P2I_EMBEDDINGS,
            instance_to_image=_P2I_MAP,
            k_values=[2],
            dataset_root=None,
            sample_pairs=20,
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
            assert key in result, f"Missing key: {key}"

    def test_n_items_correct(self) -> None:
        result = run_plant2image_eval(
            embeddings=_P2I_EMBEDDINGS,
            instance_to_image=_P2I_MAP,
            k_values=[2],
            dataset_root=None,
            sample_pairs=20,
        )
        assert result["n_items"] == len(_P2I_EMBEDDINGS)

    def test_classes_from_parent_paths(self) -> None:
        result = run_plant2image_eval(
            embeddings=_P2I_EMBEDDINGS,
            instance_to_image=_P2I_MAP,
            k_values=[2],
            dataset_root=None,
            sample_pairs=20,
        )
        assert sorted(result["classes"]) == ["A", "B"]

    def test_metadata_kpis_present(self) -> None:
        result = run_plant2image_eval(
            embeddings=_P2I_EMBEDDINGS,
            instance_to_image=_P2I_MAP,
            k_values=[2],
            dataset_root=None,
            sample_pairs=20,
        )
        gm = result["global_metrics"]
        for key in ("knn_metadata_precision", "knn_metadata_ndcg", "knn_metadata_map", "knn_metadata_mrr", "alignment"):
            assert key in gm, f"Missing metadata KPI: {key}"

    def test_label_kpis_absent(self) -> None:
        result = run_plant2image_eval(
            embeddings=_P2I_EMBEDDINGS,
            instance_to_image=_P2I_MAP,
            k_values=[2],
            dataset_root=None,
            sample_pairs=20,
        )
        gm = result["global_metrics"]
        for key in ("knn_label_purity", "knn_label_ndcg", "knn_map"):
            assert key not in gm, f"Label KPI should be absent: {key}"

    def test_group_analysis_present(self) -> None:
        result = run_plant2image_eval(
            embeddings=_P2I_EMBEDDINGS,
            instance_to_image=_P2I_MAP,
            k_values=[2],
            dataset_root=None,
            sample_pairs=20,
        )
        assert "group_analysis" in result

    def test_output_is_json_serializable(self) -> None:
        result = run_plant2image_eval(
            embeddings=_P2I_EMBEDDINGS,
            instance_to_image=_P2I_MAP,
            k_values=[2],
            dataset_root=None,
            sample_pairs=20,
        )
        json.dumps(result)

    def test_per_class_keys_match_classes(self) -> None:
        result = run_plant2image_eval(
            embeddings=_P2I_EMBEDDINGS,
            instance_to_image=_P2I_MAP,
            k_values=[2],
            dataset_root=None,
            sample_pairs=20,
        )
        assert set(result["per_class"].keys()) == {"A", "B"}


# ---------------------------------------------------------------------------
# run_plant2plant_eval
# ---------------------------------------------------------------------------


class TestRunPlant2PlantEvaluation:
    def test_returns_expected_keys(self) -> None:
        result = run_plant2plant_eval(
            embeddings=_P2P_EMBEDDINGS,
            instance_labels=_P2P_LABELS,
            k_values=[3],
            sample_pairs=30,
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

    def test_classes_from_instance_labels(self) -> None:
        result = run_plant2plant_eval(
            embeddings=_P2P_EMBEDDINGS,
            instance_labels=_P2P_LABELS,
            k_values=[3],
            sample_pairs=30,
        )
        assert sorted(result["classes"]) == ["A", "B", "C"]

    def test_metadata_kpis_present(self) -> None:
        result = run_plant2plant_eval(
            embeddings=_P2P_EMBEDDINGS,
            instance_labels=_P2P_LABELS,
            k_values=[3],
            sample_pairs=30,
        )
        gm = result["global_metrics"]
        for key in ("knn_metadata_precision", "knn_metadata_ndcg", "knn_metadata_map", "knn_metadata_mrr", "alignment"):
            assert key in gm

    def test_attribute_ndcg_absent(self) -> None:
        result = run_plant2plant_eval(
            embeddings=_P2P_EMBEDDINGS,
            instance_labels=_P2P_LABELS,
            k_values=[3],
            sample_pairs=30,
        )
        assert "knn_attribute_ndcg" not in result["global_metrics"]

    def test_label_kpis_absent(self) -> None:
        result = run_plant2plant_eval(
            embeddings=_P2P_EMBEDDINGS,
            instance_labels=_P2P_LABELS,
            k_values=[3],
            sample_pairs=30,
        )
        gm = result["global_metrics"]
        for key in ("knn_label_purity", "knn_label_ndcg", "knn_map"):
            assert key not in gm

    def test_output_is_json_serializable(self) -> None:
        result = run_plant2plant_eval(
            embeddings=_P2P_EMBEDDINGS,
            instance_labels=_P2P_LABELS,
            k_values=[3],
            sample_pairs=30,
        )
        json.dumps(result)

    def test_per_class_keys_match_classes(self) -> None:
        result = run_plant2plant_eval(
            embeddings=_P2P_EMBEDDINGS,
            instance_labels=_P2P_LABELS,
            k_values=[3],
            sample_pairs=30,
        )
        assert set(result["per_class"].keys()) == {"A", "B", "C"}

    def test_well_separated_classes_positive_gap(self) -> None:
        result = run_plant2plant_eval(
            embeddings=_P2P_EMBEDDINGS,
            instance_labels=_P2P_LABELS,
            k_values=[3],
            sample_pairs=30,
        )
        gap = result["global_metrics"]["intra_inter_similarity_gap"]["gap"]
        assert gap is not None
        assert gap > 0.0


# ---------------------------------------------------------------------------
# Schema validation — Plant2ImageRequest
# ---------------------------------------------------------------------------


def _norm_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(_DIM).astype(np.float32)
    return _l2(v).tolist()


class TestPlant2ImageRequestValidation:
    def test_valid_request_accepted(self) -> None:
        req = Plant2ImageRequest(
            embeddings=_P2I_EMBEDDINGS,
            instance_to_image=_P2I_MAP,
            k_values=[2],
        )
        assert len(req.embeddings) == len(_P2I_EMBEDDINGS)

    def test_missing_parent_in_embeddings_raises(self) -> None:
        bad_embeddings = {k: v for k, v in _P2I_EMBEDDINGS.items() if "field001.png" not in k or "-" in k}
        with pytest.raises(Exception, match="not found in embeddings"):
            Plant2ImageRequest(embeddings=bad_embeddings, instance_to_image=_P2I_MAP)

    def test_missing_instance_in_embeddings_raises(self) -> None:
        bad_embeddings = {k: v for k, v in _P2I_EMBEDDINGS.items() if "field001-0" not in k}
        with pytest.raises(Exception, match="not found in embeddings"):
            Plant2ImageRequest(embeddings=bad_embeddings, instance_to_image=_P2I_MAP)

    def test_duplicate_instance_across_parents_raises(self) -> None:
        dup_map = {
            "images/A1/field001.png": ["images/A1/field001-0.png"],
            "images/B1/field002.png": ["images/A1/field001-0.png"],  # same instance
        }
        dup_embeddings = {
            "images/A1/field001.png": _norm_vec(0),
            "images/A1/field001-0.png": _norm_vec(1),
            "images/B1/field002.png": _norm_vec(2),
        }
        with pytest.raises(Exception, match="multiple parent groups"):
            Plant2ImageRequest(embeddings=dup_embeddings, instance_to_image=dup_map)

    def test_too_few_embeddings_raises(self) -> None:
        with pytest.raises(Exception, match="At least 2"):
            Plant2ImageRequest(
                embeddings={"images/A1/field001.png": _norm_vec(0)},
                instance_to_image={"images/A1/field001.png": []},
            )

    def test_negative_k_raises(self) -> None:
        with pytest.raises(Exception, match="positive"):
            Plant2ImageRequest(embeddings=_P2I_EMBEDDINGS, instance_to_image=_P2I_MAP, k_values=[-1])

    def test_unsupported_extension_raises(self) -> None:
        bad_embeddings = {
            "images/A1/field001.tiff": _norm_vec(0),
            "images/B1/field002.png": _norm_vec(1),
        }
        with pytest.raises(Exception, match="Unsupported file extension"):
            Plant2ImageRequest(
                embeddings=bad_embeddings,
                instance_to_image={"images/A1/field001.tiff": []},
                k_values=[1],
            )

    def test_dimension_mismatch_raises(self) -> None:
        bad_embeddings = {
            "images/A1/field001.png": _norm_vec(0),
            "images/B1/field002.png": [0.0, 1.0],
        }
        with pytest.raises(Exception, match="same dimension"):
            Plant2ImageRequest(embeddings=bad_embeddings, instance_to_image={"images/A1/field001.png": []})

    def test_empty_vector_raises(self) -> None:
        bad_embeddings = {"images/A1/field001.png": [], "images/B1/field002.png": []}
        with pytest.raises(Exception, match="must not be empty"):
            Plant2ImageRequest(embeddings=bad_embeddings, instance_to_image={"images/A1/field001.png": []})

    def test_non_finite_value_raises(self) -> None:
        bad_embeddings = {
            "images/A1/field001.png": [math.nan] * _DIM,
            "images/B1/field002.png": _norm_vec(1),
        }
        with pytest.raises(Exception, match="non-finite"):
            Plant2ImageRequest(embeddings=bad_embeddings, instance_to_image={"images/A1/field001.png": []})

    def test_not_normalized_raises(self) -> None:
        bad_embeddings = {
            "images/A1/field001.png": [1.0] * _DIM,
            "images/B1/field002.png": _norm_vec(1),
        }
        with pytest.raises(Exception, match="not L2-normalized"):
            Plant2ImageRequest(embeddings=bad_embeddings, instance_to_image={"images/A1/field001.png": []})

    def test_empty_k_values_raises(self) -> None:
        with pytest.raises(Exception, match="at least one value"):
            Plant2ImageRequest(embeddings=_P2I_EMBEDDINGS, instance_to_image=_P2I_MAP, k_values=[])


# ---------------------------------------------------------------------------
# Schema validation — Plant2PlantRequest
# ---------------------------------------------------------------------------


class TestPlant2PlantRequestValidation:
    def test_valid_request(self) -> None:
        req = Plant2PlantRequest(embeddings=_P2P_EMBEDDINGS, instance_labels=_P2P_LABELS)
        assert req.instance_labels == _P2P_LABELS

    def test_unlabeled_embedding_raises(self) -> None:
        bad_labels = {k: v for k, v in _P2P_LABELS.items() if "inst-0" not in k}
        with pytest.raises(Exception, match="without a label"):
            Plant2PlantRequest(embeddings=_P2P_EMBEDDINGS, instance_labels=bad_labels)

    def test_orphan_label_raises(self) -> None:
        extra_labels = {**_P2P_LABELS, "images/A1/ghost.png": "A1"}
        with pytest.raises(Exception, match="not found in embeddings"):
            Plant2PlantRequest(embeddings=_P2P_EMBEDDINGS, instance_labels=extra_labels)

    def test_negative_k_raises(self) -> None:
        with pytest.raises(Exception, match="positive"):
            Plant2PlantRequest(
                embeddings=_P2P_EMBEDDINGS,
                instance_labels=_P2P_LABELS,
                k_values=[0],
            )

    def test_unsupported_extension_raises(self) -> None:
        bad_embeddings = {
            "images/A1/inst-0.tiff": _norm_vec(0),
            "images/B1/inst-1.png": _norm_vec(1),
        }
        with pytest.raises(Exception, match="Unsupported file extension"):
            Plant2PlantRequest(
                embeddings=bad_embeddings,
                instance_labels={"images/A1/inst-0.tiff": "A1", "images/B1/inst-1.png": "B1"},
                k_values=[1],
            )

    def test_dimension_mismatch_raises(self) -> None:
        bad_embeddings = {"images/A1/inst-0.png": _norm_vec(0), "images/B1/inst-1.png": [0.0, 1.0]}
        with pytest.raises(Exception, match="same dimension"):
            Plant2PlantRequest(
                embeddings=bad_embeddings,
                instance_labels={"images/A1/inst-0.png": "A1", "images/B1/inst-1.png": "B1"},
            )

    def test_empty_vector_raises(self) -> None:
        bad_embeddings = {"images/A1/inst-0.png": [], "images/B1/inst-1.png": []}
        with pytest.raises(Exception, match="must not be empty"):
            Plant2PlantRequest(
                embeddings=bad_embeddings,
                instance_labels={"images/A1/inst-0.png": "A1", "images/B1/inst-1.png": "B1"},
            )

    def test_non_finite_value_raises(self) -> None:
        bad_embeddings = {"images/A1/inst-0.png": [math.nan] * _DIM, "images/B1/inst-1.png": _norm_vec(1)}
        with pytest.raises(Exception, match="non-finite"):
            Plant2PlantRequest(
                embeddings=bad_embeddings,
                instance_labels={"images/A1/inst-0.png": "A1", "images/B1/inst-1.png": "B1"},
            )

    def test_not_normalized_raises(self) -> None:
        bad_embeddings = {"images/A1/inst-0.png": [1.0] * _DIM, "images/B1/inst-1.png": _norm_vec(1)}
        with pytest.raises(Exception, match="not L2-normalized"):
            Plant2PlantRequest(
                embeddings=bad_embeddings,
                instance_labels={"images/A1/inst-0.png": "A1", "images/B1/inst-1.png": "B1"},
            )

    def test_empty_k_values_raises(self) -> None:
        with pytest.raises(Exception, match="at least one value"):
            Plant2PlantRequest(embeddings=_P2P_EMBEDDINGS, instance_labels=_P2P_LABELS, k_values=[])


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestPlant2ImageEndpoint:
    def test_returns_200(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={"embeddings": _P2I_EMBEDDINGS, "instance_to_image": _P2I_MAP, "k_values": [2]},
        )
        assert response.status_code == 200

    def test_response_schema(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={"embeddings": _P2I_EMBEDDINGS, "instance_to_image": _P2I_MAP, "k_values": [2]},
        )
        data = response.json()
        assert data["n_items"] == len(_P2I_EMBEDDINGS)
        assert data["embedding_dim"] == _DIM
        assert "knn_metadata_ndcg" in data["global_metrics"]
        assert data.get("group_analysis") is not None

    def test_missing_parent_rejected(self) -> None:
        bad_embeddings = {k: v for k, v in _P2I_EMBEDDINGS.items() if "field001.png" not in k or "-" in k}
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={"embeddings": bad_embeddings, "instance_to_image": _P2I_MAP},
        )
        assert response.status_code == 422

    def test_output_is_valid_json(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={"embeddings": _P2I_EMBEDDINGS, "instance_to_image": _P2I_MAP, "k_values": [2]},
        )
        assert isinstance(response.json(), dict)


class TestPlant2PlantEndpoint:
    def test_returns_200(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={"embeddings": _P2P_EMBEDDINGS, "instance_labels": _P2P_LABELS, "k_values": [3]},
        )
        assert response.status_code == 200

    def test_response_classes_from_labels(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={"embeddings": _P2P_EMBEDDINGS, "instance_labels": _P2P_LABELS, "k_values": [3]},
        )
        data = response.json()
        assert sorted(data["classes"]) == ["A", "B", "C"]

    def test_unlabeled_embedding_rejected(self) -> None:
        bad_labels = {k: v for k, v in _P2P_LABELS.items() if "inst-0" not in k}
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={"embeddings": _P2P_EMBEDDINGS, "instance_labels": bad_labels},
        )
        assert response.status_code == 422

    def test_output_is_valid_json(self) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={"embeddings": _P2P_EMBEDDINGS, "instance_labels": _P2P_LABELS, "k_values": [3]},
        )
        assert isinstance(response.json(), dict)


# ---------------------------------------------------------------------------
# Norm helper (sanity guard so test embeddings are actually L2-normalized)
# ---------------------------------------------------------------------------


def test_fixture_embeddings_are_normalized() -> None:
    for path, vec in {**_P2I_EMBEDDINGS, **_P2P_EMBEDDINGS}.items():
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-3, f"Embedding for {path!r} is not normalized (‖v‖₂={norm:.6f})"


# ---------------------------------------------------------------------------
# plant2image_to_metadata edge cases
# ---------------------------------------------------------------------------


class TestPlant2ImageToMetadataEdgeCases:
    def test_parent_with_no_instances_produces_singleton_group(self) -> None:
        """Parent with empty instance list → group contains only the parent.

        Class extraction requires an explicit dataset_root for a single-parent map;
        auto-root with one path strips too aggressively (strips the filename, leaving
        the directory as root, so the first folder is the bare filename).
        """
        meta = plant2image_to_metadata({"ds/A1/field.png": []}, dataset_root="ds")
        group = meta["ds/A1/field.png"]
        assert group.images == ["ds/A1/field.png"]
        assert group.class_name == "A1"
        assert group.l1_cluster == "A"

    def test_parent_with_no_instances_auto_root_limitation(self) -> None:
        """Without dataset_root a single-parent map cannot reliably extract the class."""
        meta = plant2image_to_metadata({"ds/A1/field.png": []})
        group = meta["ds/A1/field.png"]
        assert "ds/A1/field.png" in group.images

    def test_class_extracted_with_explicit_root(self) -> None:
        p2i = {"images/A1/field.png": ["images/A1/field-0.png"]}
        meta = plant2image_to_metadata(p2i, dataset_root="images")
        assert meta["images/A1/field.png"].class_name == "A1"
        assert meta["images/A1/field.png"].l1_cluster == "A"

    def test_all_instances_appear_in_group_images(self) -> None:
        instances = [f"ds/A1/field-{i}.png" for i in range(3)]
        p2i = {"ds/A1/field.png": instances}
        meta = plant2image_to_metadata(p2i)
        group_images = set(meta["ds/A1/field.png"].images)
        assert "ds/A1/field.png" in group_images
        assert all(inst in group_images for inst in instances)
        assert len(group_images) == 4

    def test_empty_map_returns_empty(self) -> None:
        assert plant2image_to_metadata({}) == {}
