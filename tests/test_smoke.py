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

"""End-to-end smoke test using real downsampled images from tests/data/.

Embeddings are synthetic (class-clustered 16-D vectors from conftest).
The goal is to verify the full request→service→response pipeline works
with realistic path structure and class distribution.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from pai.ag_emb.api.app import app

client = TestClient(app)

DATASET_ROOT = "tests/data"


class TestSmokeAnalyze:
    def test_returns_200(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={
                "embeddings": image_embeddings,
                "dataset_root": DATASET_ROOT,
                "k_values": [3, 5],
            },
        )
        assert response.status_code == 200, response.text

    def test_classes_detected(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": image_embeddings, "dataset_root": DATASET_ROOT},
        )
        data = response.json()
        assert sorted(data["classes"]) == ["corn", "soybean"]

    def test_n_items_matches_file_count(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": image_embeddings, "dataset_root": DATASET_ROOT},
        )
        assert response.json()["n_items"] == len(image_embeddings)

    def test_embedding_dim_is_16(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": image_embeddings, "dataset_root": DATASET_ROOT},
        )
        assert response.json()["embedding_dim"] == 16

    def test_per_class_keys_present(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": image_embeddings, "dataset_root": DATASET_ROOT},
        )
        data = response.json()
        assert set(data["per_class"].keys()) == {"corn", "soybean"}

    def test_output_is_json_serialisable(self, image_embeddings: dict) -> None:
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": image_embeddings, "dataset_root": DATASET_ROOT},
        )
        # Will raise if any value is not JSON-serialisable
        json.dumps(response.json())

    def test_positive_intra_inter_gap(self, image_embeddings: dict) -> None:
        """Clustered synthetic embeddings should have intra > inter similarity."""
        response = client.post(
            "/v1/embeddings/evaluate",
            json={"embeddings": image_embeddings, "dataset_root": DATASET_ROOT},
        )
        gap = response.json()["global_metrics"]["intra_inter_similarity_gap"]["gap"]
        assert gap is not None
        assert gap > 0.0, "Synthetic class-clustered embeddings should have a positive gap"
