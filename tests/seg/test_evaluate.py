import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from precisionai.agrieval.seg.cli import create_parser, main
from precisionai.agrieval.seg.services.evaluate import (
    _build_color_map,
    _build_lut,
    _discover_pairs,
    _nan_to_none,
    _rgb_to_class_ids,
    _validate_colors,
    load_classes,
    run_seg_eval,
)

# ---------------------------------------------------------------------------
# load_classes
# ---------------------------------------------------------------------------


def test_load_classes_structure(classes_path):
    classes = load_classes(classes_path)
    assert len(classes) == 29  # class_map.json has ids 0-28
    assert classes[0][0] == "background"
    assert classes[0][1] == [0, 0, 0]
    assert classes[0][2] == 0


def test_load_classes_missing_key(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"wrong": []}')
    with pytest.raises(ValueError, match="'classes' key"):
        load_classes(bad)


def test_load_classes_agribench_format(tmp_path):
    agribench = tmp_path / "class_map.json"
    agribench.write_text(
        '{"classes": [{"id": 1, "name": "Crop", "color": [0, 128, 0], "hex": "#008000"},'
        '{"id": 0, "name": "background", "color": [0, 0, 0], "hex": "#000000"}]}'
    )
    classes = load_classes(agribench)
    # Returned sorted by ID
    assert classes[0][0] == "background"
    assert classes[0][2] == 0
    assert classes[1][0] == "Crop"
    assert classes[1][2] == 1


# ---------------------------------------------------------------------------
# _build_color_map
# ---------------------------------------------------------------------------


def test_build_color_map(classes):
    cm = _build_color_map(classes)
    assert (0, 0, 0) in cm
    assert cm[(0, 0, 0)] == 0
    assert (49, 140, 101) in cm  # Crop | Soybean in class_map.json
    assert cm[(49, 140, 101)] == 9
    assert len(cm) == 29


# ---------------------------------------------------------------------------
# _nan_to_none
# ---------------------------------------------------------------------------


def test_nan_to_none_returns_none_for_nan():
    assert _nan_to_none(float("nan")) is None


def test_nan_to_none_preserves_zero():
    # 0.0 is falsy — must not be treated as NaN
    result = _nan_to_none(0.0)
    assert result is not None
    assert result == pytest.approx(0.0)


def test_nan_to_none_preserves_finite():
    assert _nan_to_none(0.75) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# _validate_colors
# ---------------------------------------------------------------------------


def test_validate_colors_valid_mask(color_map):
    mask = np.zeros((4, 4, 3), dtype=np.uint8)  # all background (0,0,0)
    _validate_colors(mask, color_map, Path("test.png"))  # must not raise


def test_validate_colors_unknown_color(color_map):
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[0, 0] = [1, 2, 3]  # not in classes.json
    with pytest.raises(ValueError, match="unknown color"):
        _validate_colors(mask, color_map, Path("test.png"))


def test_validate_colors_multiple_unknown(color_map):
    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[0, 0] = [1, 1, 1]
    mask[1, 1] = [2, 2, 2]
    with pytest.raises(ValueError, match="2 unknown"):
        _validate_colors(mask, color_map, Path("x.png"))


# ---------------------------------------------------------------------------
# _rgb_to_class_ids
# ---------------------------------------------------------------------------


def test_rgb_to_class_ids(color_map):
    lut = _build_lut(color_map)
    mask = np.zeros((2, 2, 3), dtype=np.uint8)
    mask[1, 1] = [49, 140, 101]  # Crop | Soybean in class_map.json → id 9
    ids = _rgb_to_class_ids(mask, lut)
    assert ids.shape == (2, 2)
    assert ids[0, 0] == 0
    assert ids[1, 1] == 9


# ---------------------------------------------------------------------------
# _discover_pairs
# ---------------------------------------------------------------------------


def test_discover_pairs_basic(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    (pred_dir / "img.png").write_bytes(b"x")
    (gt_dir / "img.png").write_bytes(b"x")
    pairs = _discover_pairs(pred_dir, gt_dir)
    assert len(pairs) == 1
    assert pairs[0][0].name == "img.png"
    assert pairs[0][1].name == "img.png"


def test_discover_pairs_cross_extension(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    (pred_dir / "img.png").write_bytes(b"x")
    (gt_dir / "img.jpg").write_bytes(b"x")
    pairs = _discover_pairs(pred_dir, gt_dir)
    assert len(pairs) == 1


def test_discover_pairs_nested(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    (pred_dir / "A1").mkdir(parents=True)
    (gt_dir / "A1").mkdir(parents=True)
    (pred_dir / "A1" / "img.png").write_bytes(b"x")
    (gt_dir / "A1" / "img.png").write_bytes(b"x")
    pairs = _discover_pairs(pred_dir, gt_dir)
    assert len(pairs) == 1


def test_discover_pairs_missing_gt(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    (pred_dir / "img.png").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="no corresponding ground-truth"):
        _discover_pairs(pred_dir, gt_dir)


def test_discover_pairs_empty_pred(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    with pytest.raises(ValueError, match="No supported image files"):
        _discover_pairs(pred_dir, gt_dir)


# ---------------------------------------------------------------------------
# run_seg_eval — error paths
# ---------------------------------------------------------------------------


def _make_mask(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def test_size_mismatch_raises(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _make_mask(gt_dir / "t.png", np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Size mismatch"):
        run_seg_eval(pred_dir, gt_dir, classes_path)


def test_unknown_color_in_pred_raises(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    bad_pred = np.zeros((4, 4, 3), dtype=np.uint8)
    bad_pred[0, 0] = [1, 2, 3]
    _make_mask(pred_dir / "t.png", bad_pred)
    _make_mask(gt_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="unknown color"):
        run_seg_eval(pred_dir, gt_dir, classes_path)


def test_unknown_color_in_gt_raises(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    bad_gt = np.zeros((4, 4, 3), dtype=np.uint8)
    bad_gt[2, 2] = [5, 5, 5]
    _make_mask(pred_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _make_mask(gt_dir / "t.png", bad_gt)
    with pytest.raises(ValueError, match="unknown color"):
        run_seg_eval(pred_dir, gt_dir, classes_path)


# ---------------------------------------------------------------------------
# run_seg_eval — happy paths
# ---------------------------------------------------------------------------


def test_run_seg_eval_perfect(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    black = np.zeros((4, 4, 3), dtype=np.uint8)  # background only
    _make_mask(pred_dir / "t.png", black)
    _make_mask(gt_dir / "t.png", black)

    ds, imgs = run_seg_eval(pred_dir, gt_dir, classes_path)

    assert ds["n_images"] == 1
    assert ds["summary"]["mIoU"] == pytest.approx(1.0)
    assert ds["summary"]["FWIoU"] == pytest.approx(1.0)
    assert "t.png" in imgs
    assert imgs["t.png"]["summary"]["mIoU"] == pytest.approx(1.0)


def test_run_seg_eval_no_output_dir(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    _make_mask(pred_dir / "t.png", black)
    _make_mask(gt_dir / "t.png", black)

    ds, _ = run_seg_eval(pred_dir, gt_dir, classes_path, output_dir=None)
    assert "summary" in ds
    assert not list(tmp_path.glob("*.json"))


def test_run_seg_eval_writes_json(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    out_dir = tmp_path / "out"
    pred_dir.mkdir()
    gt_dir.mkdir()
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    _make_mask(pred_dir / "t.png", black)
    _make_mask(gt_dir / "t.png", black)

    run_seg_eval(pred_dir, gt_dir, classes_path, output_dir=out_dir)

    assert (out_dir / "output_summary.json").exists()
    assert (out_dir / "image_summary.json").exists()

    with (out_dir / "output_summary.json").open() as f:
        data = json.load(f)
    assert "n_images" in data
    assert "summary" in data


def test_run_seg_eval_custom_filenames(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _make_mask(gt_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))

    run_seg_eval(
        pred_dir,
        gt_dir,
        classes_path,
        output_dir=tmp_path,
        output_summary_name="my_summary.json",
        image_summary_name="my_images.json",
    )
    assert (tmp_path / "my_summary.json").exists()
    assert (tmp_path / "my_images.json").exists()


def test_run_seg_eval_verbose(tmp_path, classes_path, capsys):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _make_mask(gt_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))

    run_seg_eval(pred_dir, gt_dir, classes_path, verbose=True)
    captured = capsys.readouterr()
    assert "mIoU" in captured.out
    assert "mAcc" in captured.out
    assert "FWIoU" in captured.out


def test_run_seg_eval_multiple_classes(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    # 2x2 mask: top-left background (0,0,0), rest Soybean (49,140,101) per class_map.json
    soy = [49, 140, 101]
    mask = np.array([[[0, 0, 0], soy], [soy, soy]], dtype=np.uint8)
    _make_mask(pred_dir / "t.png", mask)
    _make_mask(gt_dir / "t.png", mask)

    ds, _ = run_seg_eval(pred_dir, gt_dir, classes_path)
    assert ds["summary"]["mIoU"] == pytest.approx(1.0)
    assert ds["classes"]["background"]["iou"] == pytest.approx(1.0)
    assert ds["classes"]["Crop | Soybean"]["iou"] == pytest.approx(1.0)


def test_run_seg_eval_imperfect(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    gt = np.zeros((2, 2, 3), dtype=np.uint8)  # all background
    pred = np.zeros((2, 2, 3), dtype=np.uint8)
    pred[0, 0] = [49, 140, 101]  # one pixel wrong: Soybean instead of background

    _make_mask(pred_dir / "t.png", pred)
    _make_mask(gt_dir / "t.png", gt)

    ds, _ = run_seg_eval(pred_dir, gt_dir, classes_path)
    # background: TP=3, FP=0, FN=1 → IoU = 3/(3+1+0) = 0.75
    assert ds["classes"]["background"]["iou"] == pytest.approx(0.75)
    # Soybean: tp=0, fp=1, fn=0 → IoU = 0/(0+1+0) = 0.0
    assert ds["classes"]["Crop | Soybean"]["iou"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_parser():
    parser = create_parser()
    assert parser.description is not None


def test_cli_main(tmp_path, classes_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    out_dir = tmp_path / "out"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _make_mask(gt_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))

    sys.argv = [
        "precisionai-agrieval-seg",
        "--pred",
        str(pred_dir),
        "--masks",
        str(gt_dir),
        "--classes",
        str(classes_path),
        "--output-dir",
        str(out_dir),
    ]
    main()
    assert (out_dir / "output_summary.json").exists()


# ---------------------------------------------------------------------------
# run_seg_eval — additional edge cases
# ---------------------------------------------------------------------------


def test_non_contiguous_class_ids_raises(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    _make_mask(pred_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _make_mask(gt_dir / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))
    # IDs [0, 2] — missing 1; n_classes=2, range(2)={0,1} ≠ {0,2}
    gap_map = tmp_path / "gap.json"
    gap_map.write_text(
        '{"classes": ['
        '{"id": 0, "name": "bg",   "color": [0, 0, 0], "hex": "#000000"},'
        '{"id": 2, "name": "crop", "color": [1, 0, 0], "hex": "#010000"}'
        "]}"
    )
    with pytest.raises(ValueError, match="contiguous"):
        run_seg_eval(pred_dir, gt_dir, gap_map)


def test_image_summary_key_uses_posix_path(tmp_path, classes_path):
    # On Windows, relative_to() produces backslash paths; as_posix() must give forward slashes
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    (pred_dir / "A1").mkdir(parents=True)
    (gt_dir / "A1").mkdir(parents=True)
    _make_mask(pred_dir / "A1" / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _make_mask(gt_dir / "A1" / "t.png", np.zeros((4, 4, 3), dtype=np.uint8))

    _, imgs = run_seg_eval(pred_dir, gt_dir, classes_path)
    assert "A1/t.png" in imgs
    assert not any("\\" in k for k in imgs)


def test_fwiou_differs_from_miou_for_imbalanced_classes(tmp_path, classes_path):
    # Dominant background class pulls FWIoU above mIoU.
    # GT: 3 bg + 1 soy; pred: all bg.
    # The missed soy pixel is a FP for bg and FN for soy, so:
    #   bg:  TP=3, FP=1, FN=0 → IoU = 3/4 = 0.75; freq = 3/4
    #   soy: TP=0, FP=0, FN=1 → IoU = 0.0;         freq = 1/4
    #   mIoU  = (0.75 + 0.0) / 2 = 0.375
    #   FWIoU = (3/4)*0.75 + (1/4)*0.0 = 0.5625 > mIoU
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    soy = [49, 140, 101]
    gt = np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], soy]], dtype=np.uint8)
    _make_mask(pred_dir / "t.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _make_mask(gt_dir / "t.png", gt)

    ds, _ = run_seg_eval(pred_dir, gt_dir, classes_path)
    assert ds["classes"]["background"]["iou"] == pytest.approx(0.75)
    assert ds["classes"]["Crop | Soybean"]["iou"] == pytest.approx(0.0)
    assert ds["summary"]["mIoU"] == pytest.approx(0.375)
    assert ds["summary"]["FWIoU"] == pytest.approx(0.5625)
    assert ds["summary"]["FWIoU"] > ds["summary"]["mIoU"]


def test_dataset_metrics_aggregated_not_per_image_average(tmp_path, classes_path):
    # Dataset mIoU must come from the aggregated CM, not mean(per-image mIoU).
    # Image 1: all bg, perfect → per-image mIoU = 1.0
    # Image 2: 2 bg + 2 soy, all swapped → per-image mIoU = 0.0
    # Mean of per-image mIoU = 0.5; aggregated mIoU = 0.25 (bg IoU=0.5, soy IoU=0.0)
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    soy = [49, 140, 101]
    _make_mask(pred_dir / "img1.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _make_mask(gt_dir / "img1.png", np.zeros((2, 2, 3), dtype=np.uint8))
    gt2 = np.array([[[0, 0, 0], soy], [[0, 0, 0], soy]], dtype=np.uint8)
    # Every bg pixel → soy, every soy pixel → bg (all wrong)
    pred2 = np.array([soy, [0, 0, 0], soy, [0, 0, 0]], dtype=np.uint8).reshape(2, 2, 3)
    _make_mask(pred_dir / "img2.png", pred2)
    _make_mask(gt_dir / "img2.png", gt2)

    ds, imgs = run_seg_eval(pred_dir, gt_dir, classes_path)

    assert imgs["img1.png"]["summary"]["mIoU"] == pytest.approx(1.0)
    assert imgs["img2.png"]["summary"]["mIoU"] == pytest.approx(0.0)
    # Aggregated dataset mIoU (0.25) must differ from the naive per-image mean (0.5)
    assert ds["summary"]["mIoU"] == pytest.approx(0.25)
