"""Semantic segmentation evaluation service.

Loads colour-coded prediction and ground-truth masks, validates them against a
class definition file, computes per-image and dataset-level KPIs, and writes
the results to JSON.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

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


def _evaluate_pair_cm(
    pred_path: Path,
    gt_path: Path,
    pred_dir: Path,
    color_map: dict[tuple[int, int, int], int],
    lut: np.ndarray,
    n_classes: int,
) -> tuple[str, np.ndarray]:
    """Load, validate, decode, and score one prediction/ground-truth pair."""
    pred_rgb = _load_mask(pred_path)
    gt_rgb = _load_mask(gt_path)

    if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
        raise ValueError(
            f"Size mismatch for '{pred_path.name}': prediction {pred_rgb.shape[:2]} vs ground truth {gt_rgb.shape[:2]}"
        )

    _validate_colors(pred_rgb, color_map, pred_path)
    _validate_colors(gt_rgb, color_map, gt_path)

    pred_ids = _rgb_to_class_ids(pred_rgb, lut)
    gt_ids = _rgb_to_class_ids(gt_rgb, lut)

    rel_key = pred_path.relative_to(pred_dir).as_posix()
    return rel_key, confusion_matrix(pred_ids, gt_ids, n_classes)


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
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"pred_dir does not exist or is not a directory: '{pred_dir}'")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks_dir does not exist or is not a directory: '{masks_dir}'")

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
            raise FileNotFoundError(
                f"no corresponding ground-truth mask found for '{pred_file.name}' "
                f"(stem='{pred_file.stem}', subdir='{rel_parent}') in '{masks_dir}'. "
                "Prediction and ground-truth files must share the same relative path and stem."
            )
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


def _fmt_metric(v: float | None) -> str:
    """Format a metric value for human-readable logging."""
    return f"{v:.4f}" if v is not None else "N/A"


def _log(message: str, show_progress: bool) -> None:
    """Write a message without corrupting an active tqdm progress bar."""
    if show_progress:
        tqdm.write(message)
    else:
        print(message)


def _resolve_worker_count(num_workers: int | None, pair_count: int) -> int:
    """Return a validated worker count bounded by the number of image pairs."""
    if num_workers is None:
        worker_count = min(4, os.cpu_count() or 1)
    elif num_workers < 1:
        raise ValueError(f"num_workers must be at least 1; got {num_workers}")
    else:
        worker_count = int(num_workers)
    return min(worker_count, pair_count)


def _log_eval_start(
    *,
    pred_dir: Path,
    masks_dir: Path,
    classes_path: Path,
    output_dir: Path | str | None,
    pair_count: int,
    n_classes: int,
    worker_count: int,
    show_progress: bool,
) -> None:
    """Log input context before evaluation starts."""
    _log("Segmentation evaluation", show_progress)
    _log(f"  predictions : {pred_dir}", show_progress)
    _log(f"  ground truth: {masks_dir}", show_progress)
    _log(f"  classes     : {classes_path}", show_progress)
    _log(f"  image pairs : {pair_count}", show_progress)
    _log(f"  class count : {n_classes}", show_progress)
    _log(f"  workers     : {worker_count}", show_progress)
    if output_dir is not None:
        _log(f"  output dir  : {output_dir}", show_progress)


def _evaluate_pair_from_tuple(
    pair: tuple[Path, Path],
    *,
    pred_dir: Path,
    color_map: dict[tuple[int, int, int], int],
    lut: np.ndarray,
    n_classes: int,
) -> tuple[str, np.ndarray]:
    """Evaluate one pair from a tuple so it can be submitted to an executor."""
    pred_path, gt_path = pair
    return _evaluate_pair_cm(
        pred_path=pred_path,
        gt_path=gt_path,
        pred_dir=pred_dir,
        color_map=color_map,
        lut=lut,
        n_classes=n_classes,
    )


def _evaluate_pairs(
    *,
    pairs: list[tuple[Path, Path]],
    pred_dir: Path,
    color_map: dict[tuple[int, int, int], int],
    lut: np.ndarray,
    n_classes: int,
    worker_count: int,
    show_progress: bool,
) -> list[tuple[str, np.ndarray]]:
    """Evaluate all image pairs, optionally using worker threads."""
    score_pair = partial(
        _evaluate_pair_from_tuple,
        pred_dir=pred_dir,
        color_map=color_map,
        lut=lut,
        n_classes=n_classes,
    )
    if worker_count == 1:
        results_iter = map(score_pair, pairs)
        return list(_progress(results_iter, total=len(pairs), show_progress=show_progress))
    return _evaluate_pairs_threaded(score_pair, pairs, worker_count, show_progress)


def _progress(iterable: Any, *, total: int, show_progress: bool) -> Any:
    """Wrap an iterable in the standard mask-evaluation progress bar."""
    return tqdm(
        iterable,
        total=total,
        desc="Evaluating masks",
        unit="image",
        dynamic_ncols=True,
        disable=not show_progress,
    )


def _evaluate_pairs_threaded(
    score_pair: Any,
    pairs: list[tuple[Path, Path]],
    worker_count: int,
    show_progress: bool,
) -> list[tuple[str, np.ndarray]]:
    """Evaluate image pairs with a thread pool while preserving pair order."""
    completed: list[tuple[str, np.ndarray] | None] = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        future_to_index = {pool.submit(score_pair, pair): idx for idx, pair in enumerate(pairs)}
        for future in _progress(as_completed(future_to_index), total=len(pairs), show_progress=show_progress):
            completed[future_to_index[future]] = future.result()
    return [result for result in completed if result is not None]


def _summarise_pair_results(
    *,
    results: list[tuple[str, np.ndarray]],
    class_names: list[str],
    verbose: bool,
    show_progress: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build dataset-level and per-image summaries from pair confusion matrices."""
    n_classes = len(class_names)
    agg_cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    image_summary: dict[str, Any] = {}
    for rel_key, cm_i in results:
        agg_cm += cm_i
        image_summary[rel_key] = _metrics_from_cm(cm_i, class_names)
        if verbose:
            _log_image_summary(rel_key, image_summary[rel_key]["summary"], show_progress)

    dataset_summary: dict[str, Any] = {
        "n_images": len(results),
        **_metrics_from_cm(agg_cm, class_names),
    }
    return dataset_summary, image_summary


def _log_image_summary(rel_key: str, summary: dict[str, float | None], show_progress: bool) -> None:
    """Log one per-image metric summary."""
    _log(
        f"  {rel_key}: "
        f"mIoU={_fmt_metric(summary['mIoU'])} "
        f"mAcc={_fmt_metric(summary['mAcc'])} "
        f"FWIoU={_fmt_metric(summary['FWIoU'])}",
        show_progress,
    )


def _log_dataset_summary(dataset_summary: dict[str, Any], show_progress: bool) -> None:
    """Log dataset-level and per-class metrics."""
    summary = dataset_summary["summary"]
    _log(f"Evaluated {dataset_summary['n_images']} image(s)", show_progress)
    _log(f"  mIoU:  {_fmt_metric(summary['mIoU'])}", show_progress)
    _log(f"  mAcc:  {_fmt_metric(summary['mAcc'])}", show_progress)
    _log(f"  FWIoU: {_fmt_metric(summary['FWIoU'])}", show_progress)
    _log("Per-class metrics:", show_progress)
    for cls_name, metrics in dataset_summary["classes"].items():
        _log(
            f"  {cls_name}: "
            f"IoU={_fmt_metric(metrics['iou'])} "
            f"Dice={_fmt_metric(metrics['dice'])} "
            f"Acc={_fmt_metric(metrics['accuracy'])}",
            show_progress,
        )


def _write_json_outputs(
    *,
    output_dir: Path | str,
    output_summary_name: str,
    image_summary_name: str,
    dataset_summary: dict[str, Any],
    image_summary: dict[str, Any],
) -> None:
    """Write dataset and per-image summaries to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / output_summary_name).open("w") as f:
        json.dump(dataset_summary, f, indent=2)
    with (out / image_summary_name).open("w") as f:
        json.dump(image_summary, f, indent=2)


def run_seg_eval(
    pred_dir: Path | str,
    masks_dir: Path | str,
    classes_path: Path | str,
    output_dir: Path | str | None = None,
    *,
    output_summary_name: str = "output_summary.json",
    image_summary_name: str = "image_summary.json",
    verbose: bool = False,
    show_progress: bool = True,
    num_workers: int | None = None,
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
        Print detailed evaluation progress and metrics to stdout.
    show_progress : bool
        Display a tqdm progress bar while image pairs are evaluated. Default: ``True``.
    num_workers : int | None
        Number of worker threads for independent mask-pair processing.
        ``None`` uses up to 4 threads, matching the embedding evaluator.
        Use ``1`` for sequential processing.

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
    worker_count = _resolve_worker_count(num_workers, len(pairs))

    if verbose:
        _log_eval_start(
            pred_dir=pred_dir,
            masks_dir=masks_dir,
            classes_path=classes_path,
            output_dir=output_dir,
            pair_count=len(pairs),
            n_classes=n_classes,
            worker_count=worker_count,
            show_progress=show_progress,
        )

    pair_results = _evaluate_pairs(
        pairs=pairs,
        pred_dir=pred_dir,
        color_map=color_map,
        lut=lut,
        n_classes=n_classes,
        worker_count=worker_count,
        show_progress=show_progress,
    )
    dataset_summary, image_summary = _summarise_pair_results(
        results=pair_results,
        class_names=class_names,
        verbose=verbose,
        show_progress=show_progress,
    )

    if verbose:
        _log_dataset_summary(dataset_summary, show_progress)

    if output_dir is not None:
        _write_json_outputs(
            output_dir=output_dir,
            output_summary_name=output_summary_name,
            image_summary_name=image_summary_name,
            dataset_summary=dataset_summary,
            image_summary=image_summary,
        )

    return dataset_summary, image_summary
