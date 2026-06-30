"""Pure pixel-level segmentation metrics computed from confusion matrices."""

import numpy as np


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute a pixel-level confusion matrix.

    Parameters
    ----------
    pred : np.ndarray
        Predicted class-ID mask, shape (H, W), integer dtype.
    gt : np.ndarray
        Ground-truth class-ID mask, shape (H, W), integer dtype.
    n_classes : int
        Total number of classes (IDs must be in [0, n_classes)).

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes), dtype int64.
        ``cm[i, j]`` is the count of pixels truly belonging to class *i*
        that were predicted as class *j*.
    """
    valid = (gt >= 0) & (gt < n_classes) & (pred >= 0) & (pred < n_classes)
    encoded = n_classes * gt[valid].astype(np.int64) + pred[valid].astype(np.int64)
    cm = np.bincount(encoded, minlength=n_classes * n_classes)
    return cm.reshape(n_classes, n_classes).astype(np.int64)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """Compute per-class Intersection over Union (Jaccard index).

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes, n_classes) where ``cm[i, j]`` is the
        count of pixels truly class *i* predicted as class *j*.

    Returns
    -------
    np.ndarray
        IoU per class, shape (n_classes,).
        ``NaN`` for classes whose denominator is zero (absent from both
        prediction and ground truth).
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return iou.astype(np.float64)


def per_class_dice(cm: np.ndarray) -> np.ndarray:
    """Compute per-class Dice coefficient (F1 score).

    Dice is monotonically related to IoU via ``Dice = 2·IoU / (1 + IoU)``,
    so it produces identical model rankings while yielding more lenient values.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes, n_classes).

    Returns
    -------
    np.ndarray
        Dice per class, shape (n_classes,).
        ``NaN`` for classes with a zero denominator.
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = np.where(denom > 0, 2 * tp / denom, np.nan)
    return dice.astype(np.float64)


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Compute per-class recall (pixel accuracy within each class).

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes, n_classes).

    Returns
    -------
    np.ndarray
        Per-class recall, shape (n_classes,).
        ``NaN`` for classes absent from the ground truth (row sum is zero).
    """
    tp = np.diag(cm)
    row_sums = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.where(row_sums > 0, tp / row_sums, np.nan)
    return acc.astype(np.float64)


def mean_iou(iou: np.ndarray) -> float:
    """Compute mean IoU (mIoU), ignoring NaN classes.

    Parameters
    ----------
    iou : np.ndarray
        Per-class IoU values, shape (n_classes,).

    Returns
    -------
    float
        Macro-average IoU over valid classes.
        ``NaN`` if every class is NaN (degenerate input).
    """
    valid = iou[~np.isnan(iou)]
    return float(valid.mean()) if valid.size > 0 else float("nan")


def mean_accuracy(acc: np.ndarray) -> float:
    """Compute mean per-class accuracy (mAcc), ignoring NaN classes.

    Parameters
    ----------
    acc : np.ndarray
        Per-class accuracy values, shape (n_classes,).

    Returns
    -------
    float
        Macro-average accuracy over valid classes.
        ``NaN`` if every class is NaN (degenerate input).
    """
    valid = acc[~np.isnan(acc)]
    return float(valid.mean()) if valid.size > 0 else float("nan")


def frequency_weighted_iou(cm: np.ndarray, iou: np.ndarray) -> float:
    """Compute frequency-weighted IoU (FWIoU).

    FWIoU = Σ_c freq_c · IoU_c, where freq_c is the fraction of ground-truth
    pixels belonging to class *c*.  Classes absent from the ground truth
    (NaN IoU) contribute zero to the sum.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes, n_classes).
    iou : np.ndarray
        Per-class IoU values, shape (n_classes,), typically from
        :func:`per_class_iou`.

    Returns
    -------
    float
        Frequency-weighted IoU.
        ``NaN`` if the confusion matrix is all zeros (no pixels evaluated).
    """
    total = float(cm.sum())
    if total == 0.0:
        return float("nan")
    freq = cm.sum(axis=1) / total
    valid_iou = np.where(np.isnan(iou), 0.0, iou)
    return float((freq * valid_iou).sum())
