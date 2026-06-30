"""Semantic segmentation evaluation service.

Loads colour-coded prediction and ground-truth masks, validates them against a
class definition file, computes per-image and dataset-level KPIs, and writes
the results to JSON.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from precisionai.agrieval.seg.metrics.segmentation import (
    confusion_matrix,
    frequency_weighted_iou,
    mean_accuracy,
    mean_iou,
    per_class_accuracy,
    per_class_dice,
    per_class_iou,
)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"})


def load_classes(classes_path: Path | str) -> list[Any]:
    """Load an AgriBench class-definition JSON and return entries as ``[name, [R,G,B], id]`` lists.

    Expected format (``"classes"`` key)::

        {"classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "hex": "..."}, ...]}

    Parameters
    ----------
    classes_path : Path | str
        Path to the class-definition JSON file.

    Returns
    -------
    list[Any]
        Entries as ``[name, [R, G, B], id]``, sorted by class ID.

    Raises
    ------
    ValueError
        If the file does not contain a ``"classes"`` key.
    """
    path = Path(classes_path)
    with path.open("r") as f:
        data = json.load(f)

    if "classes" not in data:
        raise ValueError(
            f"Unrecognised class-definition format in '{path}'. Expected a 'classes' key; got: {list(data.keys())}"
        )

    entries: list[Any] = [[e["name"], e["color"], e["id"]] for e in data["classes"]]
    return sorted(entries, key=lambda e: int(e[2]))


def _build_color_map(classes: list[Any]) -> dict[tuple[int, int, int], int]:
    """Build a colour → class-ID lookup from the classes list."""
    return {(int(rgb[0]), int(rgb[1]), int(rgb[2])): int(cid) for _, rgb, cid in classes}


def _load_mask(path: Path) -> np.ndarray:
    """Load a mask image file as an (H, W, 3) uint8 RGB array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _validate_colors(
    mask_rgb: np.ndarray,
    color_map: dict[tuple[int, int, int], int],
    path: Path,
) -> None:
    """Raise ValueError if any pixel colour in *mask_rgb* is absent from *color_map*.

    Parameters
    ----------
    mask_rgb : np.ndarray
        RGB mask array, shape (H, W, 3).
    color_map : dict[tuple[int, int, int], int]
        Mapping of valid colours to class IDs.
    path : Path
        Source path used in the error message.

    Raises
    ------
    ValueError
        When one or more pixel colours are not defined in the class map.
    """
    flat = mask_rgb.reshape(-1, 3)
    unique = set(map(tuple, np.unique(flat, axis=0).tolist()))
    unknown = unique - set(color_map.keys())
    if unknown:
        samples = ", ".join(str(c) for c in sorted(unknown)[:5])
        raise ValueError(f"Mask '{path}' contains {len(unknown)} unknown color(s) not in classes.json: {samples}")


def _build_lut(color_map: dict[tuple[int, int, int], int]) -> np.ndarray:
    """Build a 24-bit packed-RGB lookup table mapping colour → class ID.

    Packs each ``(R, G, B)`` triple as ``R<<16 | G<<8 | B`` to index a flat
    array of length 2²⁴ (16 M entries, 64 MB).  Colours not in *color_map*
    default to 0 (background); ``_validate_colors`` guarantees they never
    appear in a mask passed to :func:`_rgb_to_class_ids`.

    Parameters
    ----------
    color_map : dict[tuple[int, int, int], int]
        Colour → class-ID mapping from :func:`_build_color_map`.

    Returns
    -------
    np.ndarray
        Lookup table of shape (2²⁴,), dtype int32.
    """
    lut = np.zeros(1 << 24, dtype=np.int32)
    for (r, g, b), cid in color_map.items():
        lut[(int(r) << 16) | (int(g) << 8) | int(b)] = cid
    return lut


def _rgb_to_class_ids(mask_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Convert an RGB mask to a class-ID mask of shape (H, W).

    Uses a pre-built packed-RGB lookup table for O(H·W) decoding — one
    vectorised index operation with no per-class Python loops.

    Parameters
    ----------
    mask_rgb : np.ndarray
        RGB mask array, shape (H, W, 3), validated against the colour map
        used to build *lut*.
    lut : np.ndarray
        Lookup table from :func:`_build_lut`.

    Returns
    -------
    np.ndarray
        Integer class-ID array, shape (H, W), dtype int32.
    """
    flat = mask_rgb.reshape(-1, 3).astype(np.uint32)
    packed = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    return lut[packed].reshape(mask_rgb.shape[:2])


def _discover_pairs(pred_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """Match prediction files to ground-truth files by relative path and stem.

    Parameters
    ----------
    pred_dir : Path
        Root directory of predicted masks.
    masks_dir : Path
        Root directory of ground-truth masks.

    Returns
    -------
    list[tuple[Path, Path]]
        Sorted list of ``(pred_path, gt_path)`` pairs.

    Raises
    ------
    ValueError
        If no supported image files are found in *pred_dir*.
    FileNotFoundError
        If a prediction file has no corresponding ground-truth mask.
    """
    gt_index: dict[tuple[str, str], Path] = {}
    for gt_file in masks_dir.rglob("*"):
        if gt_file.suffix in _SUPPORTED_EXTENSIONS:
            rel_parent = gt_file.parent.relative_to(masks_dir).as_posix()
            gt_index[(rel_parent, gt_file.stem)] = gt_file

    pairs: list[tuple[Path, Path]] = []
    for pred_file in sorted(pred_dir.rglob("*")):
        if pred_file.suffix not in _SUPPORTED_EXTENSIONS:
            continue
        rel_parent = pred_file.parent.relative_to(pred_dir).as_posix()
        key = (rel_parent, pred_file.stem)
        if key not in gt_index:
            raise FileNotFoundError(f"Prediction '{pred_file}' has no corresponding ground-truth mask in '{masks_dir}'")
        pairs.append((pred_file, gt_index[key]))

    if not pairs:
        raise ValueError(f"No supported image files found in '{pred_dir}'")
    return pairs


def _nan_to_none(v: float) -> float | None:
    """Return None for NaN, preserving finite floats for JSON serialisation."""
    return None if np.isnan(v) else v


def _metrics_from_cm(cm: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Build the per-image or per-dataset metrics dict from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (n_classes, n_classes).
    class_names : list[str]
        Ordered class name labels.

    Returns
    -------
    dict[str, Any]
        Dict with ``"classes"`` (per-class IoU/Dice/accuracy) and
        ``"summary"`` (mIoU, mAcc, FWIoU) keys.
    """
    iou = per_class_iou(cm)
    dice = per_class_dice(cm)
    acc = per_class_accuracy(cm)

    classes_out: dict[str, dict[str, float | None]] = {
        name: {
            "iou": _nan_to_none(float(iou[i])),
            "dice": _nan_to_none(float(dice[i])),
            "accuracy": _nan_to_none(float(acc[i])),
        }
        for i, name in enumerate(class_names)
    }

    return {
        "classes": classes_out,
        "summary": {
            "mIoU": _nan_to_none(float(mean_iou(iou))),
            "mAcc": _nan_to_none(float(mean_accuracy(acc))),
            "FWIoU": _nan_to_none(float(frequency_weighted_iou(cm, iou))),
        },
    }


def run_seg_eval(
    pred_dir: Path | str,
    masks_dir: Path | str,
    classes_path: Path | str,
    output_dir: Path | str | None = None,
    *,
    output_summary_name: str = "output_summary.json",
    image_summary_name: str = "image_summary.json",
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run semantic segmentation evaluation over a directory of masks.

    Parameters
    ----------
    pred_dir : Path | str
        Directory containing predicted colour-coded masks.
    masks_dir : Path | str
        Directory containing ground-truth colour-coded masks.
    classes_path : Path | str
        Path to ``classes.json`` defining class names and RGB colours.
    output_dir : Path | str | None
        Directory to write JSON output files.  ``None`` skips file output.
    output_summary_name : str
        Filename for the dataset-level summary.  Default: ``"output_summary.json"``.
    image_summary_name : str
        Filename for the per-image summary.  Default: ``"image_summary.json"``.
    verbose : bool
        Print summary metrics to stdout after evaluation.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        ``(dataset_summary, image_summary)`` — both are JSON-serialisable dicts.

    Raises
    ------
    ValueError
        If spatial dimensions of a pred/gt pair do not match, if unknown
        colours appear in any mask, or if class IDs in classes.json are not
        contiguous from 0.
    FileNotFoundError
        If a predicted mask has no corresponding ground-truth mask.
    """
    pred_dir = Path(pred_dir)
    masks_dir = Path(masks_dir)
    classes_path = Path(classes_path)

    classes = load_classes(classes_path)
    color_map = _build_color_map(classes)
    lut = _build_lut(color_map)
    class_names: list[str] = [entry[0] for entry in classes]
    class_ids: list[int] = [int(entry[2]) for entry in classes]
    n_classes = len(classes)

    if set(class_ids) != set(range(n_classes)):
        raise ValueError(f"Class IDs in classes.json must be contiguous 0..{n_classes - 1}; got: {sorted(class_ids)}")

    pairs = _discover_pairs(pred_dir, masks_dir)

    agg_cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    image_summary: dict[str, Any] = {}

    for pred_path, gt_path in pairs:
        pred_rgb = _load_mask(pred_path)
        gt_rgb = _load_mask(gt_path)

        if pred_rgb.shape != gt_rgb.shape:
            raise ValueError(
                f"Size mismatch for '{pred_path.name}': "
                f"prediction {pred_rgb.shape[:2]} vs ground truth {gt_rgb.shape[:2]}"
            )

        _validate_colors(pred_rgb, color_map, pred_path)
        _validate_colors(gt_rgb, color_map, gt_path)

        pred_ids = _rgb_to_class_ids(pred_rgb, lut)
        gt_ids = _rgb_to_class_ids(gt_rgb, lut)

        cm_i = confusion_matrix(pred_ids, gt_ids, n_classes)
        agg_cm += cm_i

        rel_key = pred_path.relative_to(pred_dir).as_posix()
        image_summary[rel_key] = _metrics_from_cm(cm_i, class_names)

    dataset_summary: dict[str, Any] = {
        "n_images": len(pairs),
        **_metrics_from_cm(agg_cm, class_names),
    }

    if verbose:
        s = dataset_summary["summary"]

        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "N/A"

        print(f"Evaluated {len(pairs)} image(s)")
        print(f"  mIoU:  {_fmt(s['mIoU'])}")
        print(f"  mAcc:  {_fmt(s['mAcc'])}")
        print(f"  FWIoU: {_fmt(s['FWIoU'])}")

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / output_summary_name).open("w") as f:
            json.dump(dataset_summary, f, indent=2)
        with (out / image_summary_name).open("w") as f:
            json.dump(image_summary, f, indent=2)

    return dataset_summary, image_summary
