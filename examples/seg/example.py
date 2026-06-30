# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Run the segmentation evaluation service directly — no HTTP server required.

The example defaults to the bundled test data shipped with this repository
(``tests/data/``), so it works out of the box after ``pip install -e .``.
Point ``--pred``, ``--masks``, and ``--classes`` at your own directories to
evaluate real model outputs.

Usage (from repo root)::

    python examples/seg/example.py
    python examples/seg/example.py --pred path/to/predictions --masks path/to/gt --classes path/to/class_map.json
    python examples/seg/example.py --output-dir output --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from precisionai.agrieval.seg.services.evaluate import run_seg_eval

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_PRED = _REPO_ROOT / "tests" / "data" / "masks"
_DEFAULT_MASKS = _REPO_ROOT / "tests" / "data" / "masks"
_DEFAULT_CLASSES = _REPO_ROOT / "tests" / "data" / "class_map.json"


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


def _print_summary_line(summary: dict) -> None:  # type: ignore[type-arg]
    print(f"  mIoU : {_fmt(summary['mIoU'])}")
    print(f"  mAcc : {_fmt(summary['mAcc'])}")
    print(f"  FWIoU: {_fmt(summary['FWIoU'])}")


def _print_image_entry(img_key: str, img_result: dict) -> None:  # type: ignore[type-arg]
    img_s = img_result["summary"]
    print(f"  {img_key}")
    print(f"    mIoU={_fmt(img_s['mIoU'])}  mAcc={_fmt(img_s['mAcc'])}  FWIoU={_fmt(img_s['FWIoU'])}")


def main() -> None:
    """Run segmentation evaluation and print a summary of the results.

    Recognised arguments
    --------------------
    --pred : str
        Directory containing predicted colour-coded masks.  Defaults to
        ``tests/data/masks`` (ground-truth masks used as a stand-in so the
        example produces perfect scores out of the box).
    --masks : str
        Directory containing ground-truth colour-coded masks.
    --classes : str
        Path to the AgriBench class-definition JSON (``class_map.json``).
    --output-dir : str
        Directory to write ``output_summary.json`` and ``image_summary.json``.
        Omit to skip file output.
    --verbose
        Print summary metrics to stdout after evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate semantic segmentation predictions against ground-truth masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=_DEFAULT_PRED,
        metavar="DIR",
        help="Directory of predicted colour-coded masks.",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=_DEFAULT_MASKS,
        metavar="DIR",
        help="Directory of ground-truth colour-coded masks.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=_DEFAULT_CLASSES,
        metavar="FILE",
        help="Path to the AgriBench class-definition JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to write JSON output files.  Omit to skip file output.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print summary metrics to stdout.")
    args = parser.parse_args()

    print(f"Predictions : {args.pred}")
    print(f"Ground truth: {args.masks}")
    print(f"Classes     : {args.classes}")
    print()

    dataset_summary, image_summary = run_seg_eval(
        pred_dir=args.pred,
        masks_dir=args.masks,
        classes_path=args.classes,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    n = dataset_summary["n_images"]
    print(f"Evaluated {n} image pair(s)")
    _print_summary_line(dataset_summary["summary"])
    print()

    print("Per-class results:")
    for cls_name, metrics in dataset_summary["classes"].items():
        iou = _fmt(metrics["iou"])
        dice = _fmt(metrics["dice"])
        acc = _fmt(metrics["accuracy"])
        print(f"  {cls_name:<20} IoU={iou}  Dice={dice}  Acc={acc}")

    if args.output_dir:
        print()
        print(f"Results written to: {args.output_dir}")

    print()
    if len(image_summary) <= 5:
        print("Per-image summary:")
        for img_key, img_result in image_summary.items():
            _print_image_entry(img_key, img_result)
    else:
        print(f"Per-image summary: {len(image_summary)} images evaluated.")
        print("Run with --output-dir to write full per-image results to disk.")
        print()
        print("First 5 images:")
        for img_key, img_result in list(image_summary.items())[:5]:
            _print_image_entry(img_key, img_result)

    if args.output_dir:
        out = Path(args.output_dir)
        summary_path = out / "output_summary.json"
        if summary_path.exists():
            print()
            print("Dataset summary (JSON):")
            with summary_path.open() as f:
                print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    main()
