import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from precisionai.agrieval.api.app import app
from precisionai.agrieval.dpt.schemas.evaluate import DptEvalRequest

client = TestClient(app)

# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------


def test_openapi_contains_dpt_route():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert "/v1/dense-patch-tokens/evaluate" in resp.json()["paths"]


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


def test_unsupervised_evaluate_returns_200(synthetic_tiles):
    resp = client.post("/v1/dense-patch-tokens/evaluate", json={"tiles": synthetic_tiles})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_tiles"] == len(synthetic_tiles)
    assert data["classes"] is None


def test_labeled_evaluate_returns_200(labeled_dataset, classes_path):
    tiles, masks_dir, crop_name = labeled_dataset
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={"tiles": tiles, "masks_dir": str(masks_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["classes"] is not None
    assert crop_name in data["classes"]


def test_dataset_root_resolves_relative_paths(labeled_dataset, classes_path):
    tiles, masks_dir, _ = labeled_dataset
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={
            "tiles": tiles,
            "masks_dir": masks_dir.name,
            "classes_path": str(classes_path),
            "dataset_root": str(masks_dir.parent),
        },
    )
    assert resp.status_code == 200


def test_dataset_root_env_var(labeled_dataset, classes_path, monkeypatch):
    tiles, masks_dir, _ = labeled_dataset
    monkeypatch.setenv("PAI_DATASET_ROOT", str(masks_dir.parent))
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={"tiles": tiles, "masks_dir": masks_dir.name, "classes_path": str(classes_path)},
    )
    assert resp.status_code == 200


def test_single_tile_evaluate_returns_200():
    resp = client.post("/v1/dense-patch-tokens/evaluate", json={"tiles": {"only": [[[1.0, 2.0], [3.0, 4.0]]]}})
    assert resp.status_code == 200
    assert resp.json()["n_tiles"] == 1


def test_tiles_path_npz_returns_200(tmp_path):
    rng = np.random.default_rng(5)
    archive = tmp_path / "tiles.npz"
    np.savez(archive, t0=rng.normal(size=(4, 3, 3)), t1=rng.normal(size=(4, 3, 3)))
    resp = client.post("/v1/dense-patch-tokens/evaluate", json={"tiles_path": str(archive)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_tiles"] == 2
    assert sorted(data["tile_ids"]) == ["t0", "t1"]


def test_tiles_path_npy_dir_with_dataset_root_returns_200(tmp_path):
    rng = np.random.default_rng(6)
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    np.save(tiles_dir / "t0.npy", rng.normal(size=(4, 3, 3)).astype(np.float32))
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={"tiles_path": "tiles", "dataset_root": str(tmp_path)},
    )
    assert resp.status_code == 200
    assert resp.json()["n_tiles"] == 1


# ---------------------------------------------------------------------------
# 422 — pydantic schema validation errors
# ---------------------------------------------------------------------------


def test_empty_tiles_returns_422():
    resp = client.post("/v1/dense-patch-tokens/evaluate", json={"tiles": {}})
    assert resp.status_code == 422


def test_no_tile_source_returns_422():
    resp = client.post("/v1/dense-patch-tokens/evaluate", json={})
    assert resp.status_code == 422


def test_both_tile_sources_returns_422(synthetic_tiles):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={"tiles": synthetic_tiles, "tiles_path": "somewhere"},
    )
    assert resp.status_code == 422


def test_missing_tiles_path_returns_400(tmp_path):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={"tiles_path": str(tmp_path / "does_not_exist")},
    )
    assert resp.status_code == 400
    assert "tiles_path" in resp.json()["detail"]


def test_mismatched_tile_shapes_returns_422():
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={
            "tiles": {
                "a": [[[1.0, 2.0]]],
                "b": [[[1.0, 2.0, 3.0]]],
            }
        },
    )
    assert resp.status_code == 422


def test_non_finite_tile_rejected_by_schema():
    # A NaN payload can't round-trip through TestClient/FastAPI's strict JSON
    # encoders (neither httpx's request encoder nor Starlette's error-response
    # renderer accept it), so this validator is exercised directly instead.
    with pytest.raises(ValidationError, match="non-finite"):
        DptEvalRequest(tiles={"a": [[[1.0, float("nan")]]]})


def test_only_masks_dir_set_returns_422(synthetic_tiles):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={"tiles": synthetic_tiles, "masks_dir": "somewhere"},
    )
    assert resp.status_code == 422


def test_empty_axis_tile_returns_422():
    resp = client.post("/v1/dense-patch-tokens/evaluate", json={"tiles": {"a": [[[]]]}})
    assert resp.status_code == 422


def test_empty_k_values_returns_422(synthetic_tiles):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={"tiles": synthetic_tiles, "k_values": []},
    )
    assert resp.status_code == 422


def test_non_positive_k_value_returns_422(synthetic_tiles):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={"tiles": synthetic_tiles, "k_values": [0]},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 400 — service-level errors
# ---------------------------------------------------------------------------


def test_missing_masks_dir_returns_400(synthetic_tiles, classes_path, tmp_path):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={
            "tiles": synthetic_tiles,
            "masks_dir": str(tmp_path / "does_not_exist"),
            "classes_path": str(classes_path),
        },
    )
    assert resp.status_code == 400


def test_missing_classes_path_returns_400(labeled_dataset, tmp_path):
    tiles, masks_dir, _ = labeled_dataset
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={
            "tiles": tiles,
            "masks_dir": str(masks_dir),
            "classes_path": str(tmp_path / "does_not_exist.json"),
        },
    )
    assert resp.status_code == 400


def test_tile_without_matching_mask_returns_400(labeled_dataset, classes_path):
    tiles, masks_dir, _ = labeled_dataset
    extra_tiles = dict(tiles)
    extra_tiles["tile_without_mask"] = tiles[next(iter(tiles))]
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={"tiles": extra_tiles, "masks_dir": str(masks_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 400


def test_unknown_mask_color_returns_400(tmp_path, classes_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    bad_rgb = np.full((2, 2, 3), 123, dtype=np.uint8)
    Image.fromarray(bad_rgb).save(masks_dir / "tile_0.png")
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={
            "tiles": {"tile_0": [[[1.0, 2.0], [3.0, 4.0]]]},
            "masks_dir": str(masks_dir),
            "classes_path": str(classes_path),
        },
    )
    assert resp.status_code == 400
    assert "unknown color" in resp.json()["detail"].lower()
