import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from precisionai.agrieval.api.app import app

DATA_DIR = Path(__file__).parent.parent / "data"

client = TestClient(app)

_SOY = [49, 140, 101]  # Crop | Soybean, id 9 in class_map.json


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_mask(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _black(shape: tuple[int, int] = (4, 4)) -> np.ndarray:
    return np.zeros((*shape, 3), dtype=np.uint8)


def _post(tmp_path: Path, classes_path: Path, **overrides: object) -> dict:
    """POST /v1/segmentation/evaluate with sensible defaults and black masks."""
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    _make_mask(pred_dir / "t.png", _black())
    _make_mask(gt_dir / "t.png", _black())
    payload: dict = {
        "pred_dir": str(pred_dir),
        "masks_dir": str(gt_dir),
        "classes_path": str(classes_path),
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# OpenAPI — unified app exposes both modalities
# ---------------------------------------------------------------------------


def test_openapi_contains_seg_route():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/v1/segmentation/evaluate" in paths


def test_openapi_contains_emb_routes():
    resp = client.get("/openapi.json")
    paths = resp.json()["paths"]
    assert "/v1/embeddings/evaluate/image2image" in paths
    assert "/v1/embeddings/evaluate/plant2image" in paths
    assert "/v1/embeddings/evaluate/plant2plant" in paths


# ---------------------------------------------------------------------------
# POST /v1/segmentation/evaluate — happy paths
# ---------------------------------------------------------------------------


def test_perfect_prediction_returns_200(tmp_path, classes_path):
    resp = client.post("/v1/segmentation/evaluate", json=_post(tmp_path, classes_path))
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_images"] == 1
    assert data["summary"]["mIoU"] == pytest.approx(1.0)
    assert data["summary"]["mAcc"] == pytest.approx(1.0)
    assert data["summary"]["FWIoU"] == pytest.approx(1.0)


def test_response_contains_image_summary(tmp_path, classes_path):
    resp = client.post("/v1/segmentation/evaluate", json=_post(tmp_path, classes_path))
    data = resp.json()
    assert "image_summary" in data
    assert "t.png" in data["image_summary"]
    img = data["image_summary"]["t.png"]
    assert "classes" in img
    assert "summary" in img


def test_image_summary_key_is_posix(tmp_path, classes_path):
    pred_dir = tmp_path / "pred" / "A1"
    gt_dir = tmp_path / "gt" / "A1"
    pred_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    _make_mask(pred_dir / "t.png", _black())
    _make_mask(gt_dir / "t.png", _black())
    payload = {
        "pred_dir": str(pred_dir.parent),
        "masks_dir": str(gt_dir.parent),
        "classes_path": str(classes_path),
    }
    resp = client.post("/v1/segmentation/evaluate", json=payload)
    assert resp.status_code == 200
    keys = list(resp.json()["image_summary"].keys())
    assert all("/" in k and "\\" not in k for k in keys)
    assert "A1/t.png" in keys


def test_null_metrics_for_absent_class(tmp_path, classes_path):
    # All-black masks → only background present; every other class must be null
    resp = client.post("/v1/segmentation/evaluate", json=_post(tmp_path, classes_path))
    data = resp.json()
    soy_metrics = data["classes"]["Crop | Soybean"]
    assert soy_metrics["iou"] is None
    assert soy_metrics["dice"] is None
    assert soy_metrics["accuracy"] is None


def test_imperfect_prediction_metrics(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    # GT: all bg; pred: one pixel predicted as soy
    pred = _black()
    pred[0, 0] = _SOY
    _make_mask(pred_dir / "t.png", pred)
    _make_mask(gt_dir / "t.png", _black())

    resp = client.post(
        "/v1/segmentation/evaluate",
        json={"pred_dir": str(pred_dir), "masks_dir": str(gt_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 200
    data = resp.json()
    # bg: TP=15, FP=0, FN=1 → IoU=15/16
    assert data["classes"]["background"]["iou"] == pytest.approx(15 / 16)
    # soy: TP=0, FP=1, FN=0 → IoU=0.0
    assert data["classes"]["Crop | Soybean"]["iou"] == pytest.approx(0.0)
    assert data["summary"]["mIoU"] is not None
    assert 0.0 < data["summary"]["mIoU"] < 1.0


def test_with_output_dir_writes_json(tmp_path, classes_path):
    out_dir = tmp_path / "out"
    payload = _post(tmp_path, classes_path)
    payload["output_dir"] = str(out_dir)
    resp = client.post("/v1/segmentation/evaluate", json=payload)
    assert resp.status_code == 200
    assert (out_dir / "output_summary.json").exists()
    assert (out_dir / "image_summary.json").exists()
    with (out_dir / "output_summary.json").open() as f:
        on_disk = json.load(f)
    assert on_disk["n_images"] == resp.json()["n_images"]


def test_custom_output_filenames(tmp_path, classes_path):
    out_dir = tmp_path / "out"
    payload = _post(tmp_path, classes_path)
    payload["output_dir"] = str(out_dir)
    payload["output_summary_name"] = "dataset.json"
    payload["image_summary_name"] = "per_image.json"
    resp = client.post("/v1/segmentation/evaluate", json=payload)
    assert resp.status_code == 200
    assert (out_dir / "dataset.json").exists()
    assert (out_dir / "per_image.json").exists()


def test_dataset_root_resolves_relative_paths(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", _black())
    _make_mask(gt_dir / "t.png", _black())
    payload = {
        "pred_dir": "pred",
        "masks_dir": "gt",
        "classes_path": str(classes_path),
        "dataset_root": str(tmp_path),
    }
    resp = client.post("/v1/segmentation/evaluate", json=payload)
    assert resp.status_code == 200


def test_dataset_root_env_var(tmp_path, classes_path, monkeypatch):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", _black())
    _make_mask(gt_dir / "t.png", _black())
    monkeypatch.setenv("PAI_DATASET_ROOT", str(tmp_path))
    payload = {
        "pred_dir": "pred",
        "masks_dir": "gt",
        "classes_path": str(classes_path),
    }
    resp = client.post("/v1/segmentation/evaluate", json=payload)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /v1/segmentation/evaluate — 400 error paths
# ---------------------------------------------------------------------------


def test_unknown_color_returns_400(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    bad = _black()
    bad[0, 0] = [1, 2, 3]
    _make_mask(pred_dir / "t.png", bad)
    _make_mask(gt_dir / "t.png", _black())
    resp = client.post(
        "/v1/segmentation/evaluate",
        json={"pred_dir": str(pred_dir), "masks_dir": str(gt_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 400
    assert "unknown color" in resp.json()["detail"].lower()


def test_size_mismatch_returns_400(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", _black((4, 4)))
    _make_mask(gt_dir / "t.png", _black((8, 8)))
    resp = client.post(
        "/v1/segmentation/evaluate",
        json={"pred_dir": str(pred_dir), "masks_dir": str(gt_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 400
    assert "size mismatch" in resp.json()["detail"].lower()


def test_missing_gt_mask_returns_400(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", _black())
    # no GT file
    resp = client.post(
        "/v1/segmentation/evaluate",
        json={"pred_dir": str(pred_dir), "masks_dir": str(gt_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 400
    assert "ground-truth" in resp.json()["detail"].lower()


def test_non_contiguous_class_ids_returns_400(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", _black())
    _make_mask(gt_dir / "t.png", _black())
    gap_map = tmp_path / "gap.json"
    gap_map.write_text(
        '{"classes": ['
        '{"id": 0, "name": "bg",   "color": [0,0,0], "hex": "#000000"},'
        '{"id": 2, "name": "crop", "color": [1,0,0], "hex": "#010000"}'
        "]}"
    )
    resp = client.post(
        "/v1/segmentation/evaluate",
        json={"pred_dir": str(pred_dir), "masks_dir": str(gt_dir), "classes_path": str(gap_map)},
    )
    assert resp.status_code == 400
    assert "contiguous" in resp.json()["detail"].lower()


def test_empty_pred_dir_returns_400(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    resp = client.post(
        "/v1/segmentation/evaluate",
        json={"pred_dir": str(pred_dir), "masks_dir": str(gt_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 400
    assert "no supported image files" in resp.json()["detail"].lower()


def test_nonexistent_pred_dir_returns_400(tmp_path, classes_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    resp = client.post(
        "/v1/segmentation/evaluate",
        json={
            "pred_dir": str(tmp_path / "does_not_exist"),
            "masks_dir": str(gt_dir),
            "classes_path": str(classes_path),
        },
    )
    assert resp.status_code == 400
    assert "pred_dir" in resp.json()["detail"]


def test_nonexistent_masks_dir_returns_400(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    resp = client.post(
        "/v1/segmentation/evaluate",
        json={
            "pred_dir": str(pred_dir),
            "masks_dir": str(tmp_path / "does_not_exist"),
            "classes_path": str(classes_path),
        },
    )
    assert resp.status_code == 400
    assert "masks_dir" in resp.json()["detail"]


def test_nonexistent_classes_path_returns_400(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    resp = client.post(
        "/v1/segmentation/evaluate",
        json={
            "pred_dir": str(pred_dir),
            "masks_dir": str(gt_dir),
            "classes_path": str(tmp_path / "does_not_exist.json"),
        },
    )
    assert resp.status_code == 400
    assert "classes_path" in resp.json()["detail"]


def test_os_error_returns_400(tmp_path, classes_path, monkeypatch):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", _black())
    _make_mask(gt_dir / "t.png", _black())

    def _raise_os_error(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("precisionai.agrieval.seg.api.routes.evaluate.run_seg_eval", _raise_os_error)
    resp = client.post(
        "/v1/segmentation/evaluate",
        json={"pred_dir": str(pred_dir), "masks_dir": str(gt_dir), "classes_path": str(classes_path)},
    )
    assert resp.status_code == 400
    assert "i/o error" in resp.json()["detail"].lower()
