import numpy as np
from fastapi.testclient import TestClient

from precisionai.agrieval.api.app import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


def test_unsupervised_evaluate_returns_200(synthetic_tiles):
    resp = client.post("/v1/dense-patch-tokens/evaluate/image", json={"images": synthetic_tiles})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_images"] == len(synthetic_tiles)
    assert data["classes"] is None


def test_labeled_evaluate_returns_200(labeled_dataset, classes_path):
    images, masks_dir, crop_name = labeled_dataset
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate/image",
        json={"images": images, "masks_dir": str(masks_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["classes"] is not None
    assert crop_name in data["classes"]


def test_dataset_root_resolves_relative_paths(labeled_dataset, classes_path):
    images, masks_dir, _ = labeled_dataset
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate/image",
        json={
            "images": images,
            "masks_dir": masks_dir.name,
            "classes_path": str(classes_path),
            "dataset_root": str(masks_dir.parent),
        },
    )
    assert resp.status_code == 200


def test_single_image_evaluate_returns_200():
    resp = client.post("/v1/dense-patch-tokens/evaluate/image", json={"images": {"only": [[[1.0, 2.0], [3.0, 4.0]]]}})
    assert resp.status_code == 200
    assert resp.json()["n_images"] == 1


def _write_images_batch_npz(path, filenames, *, shape=(4, 3, 3), seed=0):
    rng = np.random.default_rng(seed)
    features = np.stack([rng.normal(size=shape).astype(np.float32) for _ in filenames])
    np.savez(path, features=features, filenames=np.array(filenames))
    return path


def test_images_path_npz_returns_200(tmp_path):
    archive = _write_images_batch_npz(tmp_path / "images.npz", ["field_001.png", "field_002.png"], seed=5)
    resp = client.post("/v1/dense-patch-tokens/evaluate/image", json={"images_path": str(archive)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_images"] == 2
    assert sorted(data["image_ids"]) == ["field_001", "field_002"]


def test_images_path_with_dataset_root_returns_200(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    _write_images_batch_npz(images_dir / "batch.npz", ["field_001.png"], seed=6)
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate/image",
        json={"images_path": "images/batch.npz", "dataset_root": str(tmp_path)},
    )
    assert resp.status_code == 200
    assert resp.json()["n_images"] == 1


# ---------------------------------------------------------------------------
# 422 — pydantic schema validation errors
# ---------------------------------------------------------------------------


def test_empty_images_returns_422():
    resp = client.post("/v1/dense-patch-tokens/evaluate/image", json={"images": {}})
    assert resp.status_code == 422


def test_no_image_source_returns_422():
    resp = client.post("/v1/dense-patch-tokens/evaluate/image", json={})
    assert resp.status_code == 422


def test_both_image_sources_returns_422(synthetic_tiles):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate/image",
        json={"images": synthetic_tiles, "images_path": "somewhere"},
    )
    assert resp.status_code == 422


def test_missing_images_path_returns_400(tmp_path):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate/image",
        json={"images_path": str(tmp_path / "does_not_exist")},
    )
    assert resp.status_code == 400
    assert "images_path" in resp.json()["detail"]


def test_images_path_escape_returns_400(tmp_path):
    archive = _write_images_batch_npz(tmp_path / "images.npz", ["field_001.png"], seed=9)
    # A sibling of tmp_path, uniquely named off it — tmp_path.parent is the
    # shared pytest base dir, so a fixed sibling name could collide across tests.
    escaped = tmp_path.parent / f"{tmp_path.name}-escaped.npz"
    escaped.write_bytes(archive.read_bytes())
    resp = client.post("/v1/dense-patch-tokens/evaluate/image", json={"images_path": str(escaped)})
    assert resp.status_code == 400
    assert "images_path" in resp.json()["detail"]


def test_mismatched_image_shapes_returns_422():
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate/image",
        json={"images": {"a": [[[1.0, 2.0]]], "b": [[[1.0, 2.0, 3.0]]]}},
    )
    assert resp.status_code == 422


def test_only_masks_dir_set_returns_422(synthetic_tiles):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate/image",
        json={"images": synthetic_tiles, "masks_dir": "somewhere"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 400 — service-level errors
# ---------------------------------------------------------------------------


def test_missing_masks_dir_returns_400(synthetic_tiles, classes_path, tmp_path):
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate/image",
        json={
            "images": synthetic_tiles,
            "masks_dir": str(tmp_path / "does_not_exist"),
            "classes_path": str(classes_path),
        },
    )
    assert resp.status_code == 400


def test_image_without_matching_mask_returns_400(labeled_dataset, classes_path):
    images, masks_dir, _ = labeled_dataset
    extra_images = dict(images)
    extra_images["image_without_mask"] = images[next(iter(images))]
    resp = client.post(
        "/v1/dense-patch-tokens/evaluate/image",
        json={"images": extra_images, "masks_dir": str(masks_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 400
