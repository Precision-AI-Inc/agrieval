import numpy as np
import pytest

from precisionai.agrieval.seg.metrics import (
    confusion_matrix,
    frequency_weighted_iou,
    mean_accuracy,
    mean_iou,
    per_class_accuracy,
    per_class_dice,
    per_class_iou,
)


def test_confusion_matrix_perfect():
    pred = np.array([[0, 1], [2, 0]])
    gt = np.array([[0, 1], [2, 0]])
    cm = confusion_matrix(pred, gt, n_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1
    assert cm.sum() == 4


def test_confusion_matrix_errors():
    # gt=[[0,0],[1,1]], pred=[[0,1],[1,0]]
    pred = np.array([[0, 1], [1, 0]])
    gt = np.array([[0, 0], [1, 1]])
    cm = confusion_matrix(pred, gt, n_classes=2)
    assert cm[0, 0] == 1  # true 0, pred 0
    assert cm[0, 1] == 1  # true 0, pred 1
    assert cm[1, 0] == 1  # true 1, pred 0
    assert cm[1, 1] == 1  # true 1, pred 1


def test_confusion_matrix_ignores_out_of_range():
    pred = np.array([[0, 5], [1, 1]])  # 5 is out of range for n_classes=2
    gt = np.array([[0, 0], [1, 1]])
    cm = confusion_matrix(pred, gt, n_classes=2)
    assert cm.sum() == 3  # pixel (0,1) with pred=5 is skipped


def test_per_class_iou_perfect():
    cm = np.array([[2, 0], [0, 3]], dtype=np.int64)
    iou = per_class_iou(cm)
    assert np.allclose(iou, [1.0, 1.0])


def test_per_class_iou_absent_from_both():
    cm = np.array([[2, 0], [0, 0]], dtype=np.int64)
    iou = per_class_iou(cm)
    assert iou[0] == pytest.approx(1.0)
    assert np.isnan(iou[1])


def test_per_class_iou_predicted_but_not_in_gt():
    # Class 1 predicted (FP) but absent from gt — IoU should be 0, not NaN
    cm = np.array([[2, 1], [0, 0]], dtype=np.int64)
    iou = per_class_iou(cm)
    assert iou[0] == pytest.approx(2 / 3)
    assert iou[1] == pytest.approx(0.0)


def test_per_class_dice_relationship_to_iou():
    # Dice = 2*IoU / (1 + IoU) — must hold element-wise
    cm = np.array([[10, 2], [3, 15]], dtype=np.int64)
    iou = per_class_iou(cm)
    dice = per_class_dice(cm)
    expected = 2 * iou / (1 + iou)
    assert np.allclose(dice, expected, equal_nan=True)


def test_per_class_dice_absent_class():
    cm = np.array([[4, 0], [0, 0]], dtype=np.int64)
    dice = per_class_dice(cm)
    assert dice[0] == pytest.approx(1.0)
    assert np.isnan(dice[1])


def test_per_class_accuracy():
    cm = np.array([[3, 1], [2, 4]], dtype=np.int64)
    acc = per_class_accuracy(cm)
    assert acc[0] == pytest.approx(3 / 4)
    assert acc[1] == pytest.approx(4 / 6)


def test_per_class_accuracy_absent_gt():
    cm = np.array([[5, 0], [0, 0]], dtype=np.int64)
    acc = per_class_accuracy(cm)
    assert acc[0] == pytest.approx(1.0)
    assert np.isnan(acc[1])


def test_mean_iou_excludes_nan():
    iou = np.array([0.5, 0.8, np.nan])
    assert mean_iou(iou) == pytest.approx(0.65)


def test_mean_iou_all_nan():
    assert np.isnan(mean_iou(np.array([np.nan, np.nan])))


def test_mean_accuracy_excludes_nan():
    acc = np.array([0.9, np.nan, 0.7])
    assert mean_accuracy(acc) == pytest.approx(0.8)


def test_mean_accuracy_all_nan():
    assert np.isnan(mean_accuracy(np.array([np.nan])))


def test_fwiou_perfect_balanced():
    cm = np.array([[10, 0], [0, 10]], dtype=np.int64)
    iou = per_class_iou(cm)
    assert frequency_weighted_iou(cm, iou) == pytest.approx(1.0)


def test_fwiou_perfect_imbalanced():
    cm = np.array([[8, 0], [0, 2]], dtype=np.int64)
    iou = per_class_iou(cm)
    assert frequency_weighted_iou(cm, iou) == pytest.approx(1.0)


def test_fwiou_zero_cm():
    cm = np.zeros((2, 2), dtype=np.int64)
    iou = per_class_iou(cm)
    assert np.isnan(frequency_weighted_iou(cm, iou))


def test_fwiou_nan_class_contributes_zero():
    # Class 1 is absent — its NaN IoU should not corrupt the sum
    cm = np.array([[10, 0], [0, 0]], dtype=np.int64)
    iou = per_class_iou(cm)
    assert np.isnan(iou[1])
    fwiou = frequency_weighted_iou(cm, iou)
    assert fwiou == pytest.approx(1.0)


def test_fwiou_concrete_weighted_sum():
    # Class 0: 8 GT pixels, perfect prediction; class 1: 2 GT pixels, 1 TP / 1 FN / 0 FP
    # Class 0: IoU = 8/(8+1+0) = 8/9 (there's 1 FP from class 1's FN)
    # Class 1: IoU = 1/(1+0+1) = 0.5
    # FWIoU = (8/10) * (8/9) + (2/10) * 0.5
    cm = np.array([[8, 0], [1, 1]], dtype=np.int64)
    iou = per_class_iou(cm)
    result = frequency_weighted_iou(cm, iou)
    assert result == pytest.approx(0.8 * (8 / 9) + 0.2 * 0.5)


def test_confusion_matrix_gt_out_of_range():
    # GT pixel >= n_classes must be filtered the same as pred out-of-range
    pred = np.array([[0, 0], [1, 1]])
    gt = np.array([[0, 5], [1, 1]])  # 5 is out of range for n_classes=2
    cm = confusion_matrix(pred, gt, n_classes=2)
    assert cm.sum() == 3  # pixel at (0,1) with gt=5 is skipped


def test_per_class_iou_in_gt_not_pred():
    # Class 1 present in GT only (FN > 0, FP = 0, TP = 0) — must be 0.0, not NaN
    # cm: bg TP=3, bg misclassified as cls1: none; cls1 misclassified as bg: 2, cls1 TP: 0
    cm = np.array([[3, 0], [2, 0]], dtype=np.int64)
    iou = per_class_iou(cm)
    # class 0: TP=3, FP=2 (from row1→col0), FN=0 → 3/5
    assert iou[0] == pytest.approx(3 / 5)
    # class 1: TP=0, FP=0, FN=2 → denom=2 → 0.0 (not NaN)
    assert iou[1] == pytest.approx(0.0)
    assert not np.isnan(iou[1])


def test_per_class_iou_three_classes_with_absent():
    # 3-class matrix; class 2 absent from both pred and GT
    cm = np.array([[5, 1, 0], [2, 3, 0], [0, 0, 0]], dtype=np.int64)
    iou = per_class_iou(cm)
    # class 0: TP=5, FP=cm[1,0]=2, FN=cm[0,1]=1 → 5/8
    assert iou[0] == pytest.approx(5 / 8)
    # class 1: TP=3, FP=cm[0,1]=1, FN=cm[1,0]=2 → 3/6 = 0.5
    assert iou[1] == pytest.approx(0.5)
    # class 2: completely absent → NaN
    assert np.isnan(iou[2])


def test_per_class_accuracy_zero():
    # Class 0 has GT pixels but all misclassified — accuracy must be 0.0, not NaN
    cm = np.array([[0, 3], [0, 2]], dtype=np.int64)
    acc = per_class_accuracy(cm)
    assert acc[0] == pytest.approx(0.0)
    assert acc[1] == pytest.approx(1.0)
