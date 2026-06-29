# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke tests using real downsampled images from tests/data/.

Embeddings are synthetic (class-clustered 16-D vectors from conftest).
The goal is to verify the full request→service→response pipeline works
with realistic path structure and class distribution across all three
evaluation scenarios: image2image, plant2image, and plant2plant.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from precisionai.agrieval.emb.api.app import app

client = TestClient(app)

_IMAGE_DATASET_ROOT = "tests/emb/data/images"


class TestSmokeImage2Image:
    def test_returns_200(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={
                "embeddings": image_embeddings,
                "dataset_root": _IMAGE_DATASET_ROOT,
                "k_values": [3, 5],
            },
        )
        assert response.status_code == 200, response.text

    def test_classes_detected(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": image_embeddings, "dataset_root": _IMAGE_DATASET_ROOT},
        )
        assert sorted(response.json()["classes"]) == ["A", "D"]

    def test_n_items_matches_file_count(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": image_embeddings, "dataset_root": _IMAGE_DATASET_ROOT},
        )
        assert response.json()["n_items"] == len(image_embeddings)

    def test_embedding_dim_is_16(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": image_embeddings, "dataset_root": _IMAGE_DATASET_ROOT},
        )
        assert response.json()["embedding_dim"] == 16

    def test_per_class_keys_present(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": image_embeddings, "dataset_root": _IMAGE_DATASET_ROOT},
        )
        assert set(response.json()["per_class"].keys()) == {"A", "D"}

    def test_output_is_json_serialisable(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": image_embeddings, "dataset_root": _IMAGE_DATASET_ROOT},
        )
        json.dumps(response.json())

    def test_positive_intra_inter_gap(self, image_embeddings: dict) -> None:
        """Clustered synthetic embeddings should have intra > inter similarity."""
        response = client.post(
            "/v1/embeddings/evaluate/image2image",
            json={"embeddings": image_embeddings, "dataset_root": _IMAGE_DATASET_ROOT},
        )
        gap = response.json()["global_metrics"]["intra_inter_similarity_gap"]["gap"]
        assert gap is not None
        assert gap > 0.0, "Synthetic class-clustered embeddings should have a positive gap"


class TestSmokePlant2Image:
    def test_returns_200(self, plant2image_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={
                "embeddings": plant2image_payload["embeddings"],
                "instance_to_image": plant2image_payload["instance_to_image"],
                "dataset_root": _IMAGE_DATASET_ROOT,
                "k_values": [3, 5],
            },
        )
        assert response.status_code == 200, response.text

    def test_n_items_matches_payload(self, plant2image_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={
                "embeddings": plant2image_payload["embeddings"],
                "instance_to_image": plant2image_payload["instance_to_image"],
                "dataset_root": _IMAGE_DATASET_ROOT,
            },
        )
        assert response.json()["n_items"] == len(plant2image_payload["embeddings"])

    def test_classes_detected(self, plant2image_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={
                "embeddings": plant2image_payload["embeddings"],
                "instance_to_image": plant2image_payload["instance_to_image"],
                "dataset_root": _IMAGE_DATASET_ROOT,
            },
        )
        assert sorted(response.json()["classes"]) == ["A", "D"]

    def test_embedding_dim_is_16(self, plant2image_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={
                "embeddings": plant2image_payload["embeddings"],
                "instance_to_image": plant2image_payload["instance_to_image"],
                "dataset_root": _IMAGE_DATASET_ROOT,
            },
        )
        assert response.json()["embedding_dim"] == 16

    def test_metadata_kpis_present(self, plant2image_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={
                "embeddings": plant2image_payload["embeddings"],
                "instance_to_image": plant2image_payload["instance_to_image"],
                "dataset_root": _IMAGE_DATASET_ROOT,
            },
        )
        gm = response.json()["global_metrics"]
        for key in ("knn_metadata_precision", "knn_metadata_ndcg", "knn_metadata_map", "knn_metadata_mrr", "alignment"):
            assert key in gm, f"Missing metadata KPI: {key}"

    def test_group_analysis_present(self, plant2image_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={
                "embeddings": plant2image_payload["embeddings"],
                "instance_to_image": plant2image_payload["instance_to_image"],
                "dataset_root": _IMAGE_DATASET_ROOT,
            },
        )
        assert response.json().get("group_analysis") is not None

    def test_output_is_json_serialisable(self, plant2image_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2image",
            json={
                "embeddings": plant2image_payload["embeddings"],
                "instance_to_image": plant2image_payload["instance_to_image"],
                "dataset_root": _IMAGE_DATASET_ROOT,
            },
        )
        json.dumps(response.json())


class TestSmokePlant2Plant:
    def test_returns_200(self, plant2plant_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={
                "embeddings": plant2plant_payload["embeddings"],
                "instance_labels": plant2plant_payload["instance_labels"],
                "k_values": [3, 5],
            },
        )
        assert response.status_code == 200, response.text

    def test_n_items_matches_payload(self, plant2plant_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={
                "embeddings": plant2plant_payload["embeddings"],
                "instance_labels": plant2plant_payload["instance_labels"],
            },
        )
        assert response.json()["n_items"] == len(plant2plant_payload["embeddings"])

    def test_classes_from_instance_labels(self, plant2plant_payload: dict) -> None:
        unique_labels = sorted(set(plant2plant_payload["instance_labels"].values()))
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={
                "embeddings": plant2plant_payload["embeddings"],
                "instance_labels": plant2plant_payload["instance_labels"],
            },
        )
        assert sorted(response.json()["classes"]) == unique_labels

    def test_embedding_dim_is_16(self, plant2plant_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={
                "embeddings": plant2plant_payload["embeddings"],
                "instance_labels": plant2plant_payload["instance_labels"],
            },
        )
        assert response.json()["embedding_dim"] == 16

    def test_metadata_kpis_present(self, plant2plant_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={
                "embeddings": plant2plant_payload["embeddings"],
                "instance_labels": plant2plant_payload["instance_labels"],
            },
        )
        gm = response.json()["global_metrics"]
        for key in ("knn_metadata_precision", "knn_metadata_ndcg", "knn_metadata_map", "knn_metadata_mrr", "alignment"):
            assert key in gm, f"Missing metadata KPI: {key}"

    def test_positive_intra_inter_gap(self, plant2plant_payload: dict) -> None:
        """Clustered synthetic embeddings should have intra > inter similarity."""
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={
                "embeddings": plant2plant_payload["embeddings"],
                "instance_labels": plant2plant_payload["instance_labels"],
            },
        )
        gap = response.json()["global_metrics"]["intra_inter_similarity_gap"]["gap"]
        assert gap is not None
        assert gap > 0.0, "Synthetic label-clustered embeddings should have a positive gap"

    def test_output_is_json_serialisable(self, plant2plant_payload: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate/plant2plant",
            json={
                "embeddings": plant2plant_payload["embeddings"],
                "instance_labels": plant2plant_payload["instance_labels"],
            },
        )
        json.dumps(response.json())
