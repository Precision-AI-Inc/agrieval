"""End-to-end smoke tests for the full segmentation evaluation pipeline.

Two test strategies are used:

1. **Real-data tests**: load the ground-truth masks from ``tests/data/masks/``
   and evaluate against a perturbed copy (predictions).  Uses ``class_map.json``
   whose color palette matches the actual mask files.

2. **Synthetic tests**: generate small in-memory masks from ``class_map.json``
   colors.  These are self-contained and act as a fast correctness check.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from precisionai.agrieval.seg.services.evaluate import load_classes, run_seg_eval

DATA_DIR = Path(__file__).parent.parent / "data"
MASKS_DIR = DATA_DIR / "masks"
CLASSES_PATH = DATA_DIR / "class_map.json"

_SYNTH_SIZE = (16, 16)
_SYNTH_N = 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _perturb_mask(
    mask_rgb: np.ndarray,
    classes: list,
    rng: np.random.Generator,
    flip_fraction: float = 0.15,
) -> np.ndarray:
    """Randomly reassign a fraction of pixels to different class colors."""
    h, w = mask_rgb.shape[:2]
    flat = mask_rgb.reshape(-1, 3).copy()
    n_flip = max(1, int(flat.shape[0] * flip_fraction))
    flip_idx = rng.choice(flat.shape[0], size=n_flip, replace=False)
    color_pool = np.array([[c[0], c[1], c[2]] for _, c, _ in classes], dtype=np.uint8)
    flat[flip_idx] = color_pool[rng.integers(0, len(color_pool), size=n_flip)]
    return flat.reshape(h, w, 3)


def _build_synth_mask(classes: list, rng: np.random.Generator) -> np.ndarray:
    """Build a random color-coded mask using known class colors."""
    h, w = _SYNTH_SIZE
    color_pool = np.array([[c[0], c[1], c[2]] for _, c, _ in classes], dtype=np.uint8)
    return color_pool[rng.integers(0, len(color_pool), size=(h, w))]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_mask_files() -> list[Path]:
    files = sorted(MASKS_DIR.rglob("*.png")) + sorted(MASKS_DIR.rglob("*.jpg"))
    if not files:
        pytest.skip("No mask files found in tests/data/masks/")
    return files


@pytest.fixture
def synthetic_dataset(tmp_path):
    classes = load_classes(CLASSES_PATH)
    rng = np.random.default_rng(42)
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    for i in range(_SYNTH_N):
        sub = f"batch_{i // 2}"
        (gt_dir / sub).mkdir(parents=True, exist_ok=True)
        (pred_dir / sub).mkdir(parents=True, exist_ok=True)
        gt_arr = _build_synth_mask(classes, rng)
        pred_arr = _perturb_mask(gt_arr, classes, rng)
        fname = f"mask_{i:04d}.png"
        Image.fromarray(gt_arr).save(gt_dir / sub / fname)
        Image.fromarray(pred_arr).save(pred_dir / sub / fname)
    return pred_dir, gt_dir, CLASSES_PATH


@pytest.fixture
def perfect_dataset(tmp_path):
    classes = load_classes(CLASSES_PATH)
    rng = np.random.default_rng(7)
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    for i in range(_SYNTH_N):
        arr = _build_synth_mask(classes, rng)
        fname = f"mask_{i:04d}.png"
        Image.fromarray(arr).save(gt_dir / fname)
        Image.fromarray(arr).save(pred_dir / fname)
    return pred_dir, gt_dir, CLASSES_PATH


# ---------------------------------------------------------------------------
# real-data smoke tests
# ---------------------------------------------------------------------------


def test_real_masks_imperfect_pipeline(tmp_path, real_mask_files):
    """Perturb real GT masks to produce predictions; evaluate and check KPIs."""
    classes = load_classes(CLASSES_PATH)
    rng = np.random.default_rng(42)

    pred_dir = tmp_path / "pred"
    for mask_path in real_mask_files:
        rel = mask_path.relative_to(MASKS_DIR)
        dest = pred_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        gt_arr = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
        pred_arr = _perturb_mask(gt_arr, classes, rng)
        Image.fromarray(pred_arr).save(dest)

    output_dir = tmp_path / "output"
    dataset_summary, image_summary = run_seg_eval(
        pred_dir=pred_dir,
        masks_dir=MASKS_DIR,
        classes_path=CLASSES_PATH,
        output_dir=output_dir,
        verbose=True,
    )

    assert (output_dir / "output_summary.json").exists()
    assert (output_dir / "image_summary.json").exists()

    with (output_dir / "output_summary.json").open() as f:
        assert json.load(f) == dataset_summary

    assert dataset_summary["n_images"] == len(real_mask_files)
    s = dataset_summary["summary"]
    assert s["mIoU"] is not None
    assert 0.0 <= s["mIoU"] < 1.0, "mIoU must be imperfect due to perturbations"
    assert 0.0 <= s["mAcc"] <= 1.0
    assert 0.0 <= s["FWIoU"] < 1.0

    assert len(image_summary) == len(real_mask_files)
    for img_data in image_summary.values():
        assert "classes" in img_data
        assert "summary" in img_data


def test_real_masks_perfect_prediction(tmp_path, real_mask_files):
    """Copy GT masks verbatim as predictions — all metrics must be 1.0."""
    pred_dir = tmp_path / "pred"
    for mask_path in real_mask_files:
        rel = mask_path.relative_to(MASKS_DIR)
        dest = pred_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.open(mask_path).convert("RGB").save(dest)

    dataset_summary, _ = run_seg_eval(
        pred_dir=pred_dir,
        masks_dir=MASKS_DIR,
        classes_path=CLASSES_PATH,
    )

    s = dataset_summary["summary"]
    assert s["mIoU"] == pytest.approx(1.0)
    assert s["mAcc"] == pytest.approx(1.0)
    assert s["FWIoU"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# synthetic smoke tests
# ---------------------------------------------------------------------------


def test_synthetic_imperfect_pipeline(tmp_path, synthetic_dataset):
    pred_dir, gt_dir, classes_path = synthetic_dataset
    output_dir = tmp_path / "output"

    dataset_summary, image_summary = run_seg_eval(
        pred_dir=pred_dir,
        masks_dir=gt_dir,
        classes_path=classes_path,
        output_dir=output_dir,
        verbose=True,
    )

    assert (output_dir / "output_summary.json").exists()
    assert (output_dir / "image_summary.json").exists()
    with (output_dir / "image_summary.json").open() as f:
        assert json.load(f) == image_summary

    assert dataset_summary["n_images"] == _SYNTH_N
    s = dataset_summary["summary"]
    assert s["mIoU"] is not None
    assert 0.0 <= s["mIoU"] < 1.0
    assert 0.0 <= s["mAcc"] <= 1.0
    assert 0.0 <= s["FWIoU"] < 1.0
    assert len(image_summary) == _SYNTH_N


def test_synthetic_perfect_prediction(perfect_dataset):
    pred_dir, gt_dir, classes_path = perfect_dataset
    dataset_summary, _ = run_seg_eval(pred_dir=pred_dir, masks_dir=gt_dir, classes_path=classes_path)
    s = dataset_summary["summary"]
    assert s["mIoU"] == pytest.approx(1.0)
    assert s["mAcc"] == pytest.approx(1.0)
    assert s["FWIoU"] == pytest.approx(1.0)


def test_all_classes_reported(synthetic_dataset):
    pred_dir, gt_dir, classes_path = synthetic_dataset
    classes = load_classes(classes_path)
    dataset_summary, _ = run_seg_eval(pred_dir=pred_dir, masks_dir=gt_dir, classes_path=classes_path)
    assert set(dataset_summary["classes"].keys()) == {e[0] for e in classes}


def test_per_class_values_in_range(synthetic_dataset):
    pred_dir, gt_dir, classes_path = synthetic_dataset
    dataset_summary, _ = run_seg_eval(pred_dir=pred_dir, masks_dir=gt_dir, classes_path=classes_path)
    for name, metrics in dataset_summary["classes"].items():
        for key in ("iou", "dice", "accuracy"):
            val = metrics[key]
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{name}.{key} out of range: {val}"
