"""Command-line interface for semantic segmentation evaluation."""

import argparse
from pathlib import Path

from precisionai.agrieval.seg.services.evaluate import run_seg_eval


def create_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the seg evaluation CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate semantic segmentation predictions against ground-truth masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred", required=True, type=Path, metavar="DIR", help="Directory of predicted masks.")
    parser.add_argument("--masks", required=True, type=Path, metavar="DIR", help="Directory of ground-truth masks.")
    parser.add_argument("--classes", required=True, type=Path, metavar="FILE", help="Path to classes.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Output directory for JSON results.",
    )
    parser.add_argument(
        "--output-summary",
        default="output_summary.json",
        metavar="NAME",
        help="Filename for the dataset-level summary.",
    )
    parser.add_argument(
        "--image-summary",
        default="image_summary.json",
        metavar="NAME",
        help="Filename for the per-image summary.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print summary metrics to stdout.")
    return parser


def main() -> None:
    """Entry point for the ``precisionai-agrieval-seg`` CLI."""
    parser = create_parser()
    args = parser.parse_args()
    run_seg_eval(
        pred_dir=args.pred,
        masks_dir=args.masks,
        classes_path=args.classes,
        output_dir=args.output_dir,
        output_summary_name=args.output_summary,
        image_summary_name=args.image_summary,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
