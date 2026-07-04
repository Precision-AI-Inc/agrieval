"""End-to-end smoke test for the full dense patch token evaluation pipeline.

Exercises the unified FastAPI app with class-correlated synthetic tiles and
sanity-checks that the returned metrics behave as expected for
well-separated classes (high kNN purity, in-range geometry metrics).
"""

from fastapi.testclient import TestClient

from precisionai.agrieval.api.app import app

client = TestClient(app)


def test_labeled_pipeline_end_to_end(labeled_dataset, classes_path):
    tiles, masks_dir, crop_name = labeled_dataset

    resp = client.post(
        "/v1/dense-patch-tokens/evaluate",
        json={
            "tiles": tiles,
            "masks_dir": str(masks_dir),
            "classes_path": str(classes_path),
            "k_values": [3],
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["n_tiles"] == len(tiles)
    assert data["n_patches"] == data["n_tiles"] * data["grid_height"] * data["grid_width"]
    assert set(data["classes"]) == {"background", crop_name}

    gm = data["global_metrics"]
    assert -1.0 <= gm["mean_patch_smoothness"] <= 1.0
    assert 0.0 <= gm["mean_outlier_fraction"] <= 1.0
    assert gm["effective_rank"]["effective_rank"] <= data["embed_dim"] + 1e-6

    # Well-separated prototypes → each patch's neighbors should mostly share its class.
    purity = data["knn_confusion"]["purity"]["3"]
    assert purity["mean"] > 0.9
