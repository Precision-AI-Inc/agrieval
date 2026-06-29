# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0
"""Run the evaluation service directly — no HTTP server required.

Four example modes are bundled, one per retrieval wiring:

* **image2image** — Image→Image, embeddings only.
* **image2image_meta** — Image→Image with metadata groups, enabling graded nDCG
  and per-attribute KPIs.
* **plant2image** — Plant→Image; mixed corpus of full-field images and
  per-plant instance crops with an ``instance_to_image`` mapping.
* **plant2plant** — Plant→Plant; instance crops with class labels.

Usage (from repo root)::

    python examples/emb/example.py
    python examples/emb/example.py --mode image2image_meta
    python examples/emb/example.py --mode plant2image
    python examples/emb/example.py --mode plant2plant
    python examples/emb/example.py --k-values 1 5 10
    python examples/emb/example.py --output-dir output --tsne-dimensions 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent))

from precisionai.agrieval.emb.schemas.evaluate import MetadataGroup
from precisionai.agrieval.emb.services.evaluate import (
    run_image2image_eval,
    run_plant2image_eval,
    run_plant2plant_eval,
)
from precisionai.agrieval.emb.services.reporting import (
    plot_cosine_similarity,
    plot_knn_confusion,
    plot_lle,
    plot_tsne,
    print_result,
)

_SCENARIO_FILES: dict[str, str] = {
    "image2image": "emb_image2image.json",
    "image2image_meta": "emb_image2image_meta.json",
    "plant2image": "emb_plant2image.json",
    "plant2plant": "emb_plant2plant.json",
}


def main() -> None:
    """Load an example input file, run the evaluation service, and print results.

    Optionally writes visualization HTML files to an output directory.

    Recognised arguments
    --------------------
    --mode : {image2image, image2image_meta, plant2image, plant2plant}
        Which bundled mode to run.

        ``image2image`` uses embeddings only (Image→Image).
        ``image2image_meta`` adds metadata groups for graded nDCG.
        ``plant2image`` demonstrates the Plant→Image wiring with
        ``instance_to_image`` mapping.
        ``plant2plant`` demonstrates the Plant→Plant wiring with
        ``instance_labels``.
    --dataset-root : str
        Dataset root for Image→Image label extraction.  Defaults to
        ``"images"`` to match the ``images/{L2}/`` key prefix.
    --k-values : list[int]
        One or more K cutoffs for nearest-neighbour metrics.  Defaults to
        ``[5, 10]``.
    --sample-pairs : int | None
        Maximum number of pairs for pairwise similarity stats.  Omit for an
        exact (exhaustive) computation.
    --output-dir : str
        Directory to write visualization files.  Defaults to ``"output"``.
    --tsne-dimensions : {2, 3}
        Dimensionality for the t-SNE scatter plot.  Defaults to ``3``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(_SCENARIO_FILES),
        default="image2image",
        help="Mode to run (default: image2image)",
    )
    parser.add_argument(
        "--dataset-root",
        default="images",
        help="Root prefix for Image→Image label extraction (default: images)",
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10])
    parser.add_argument(
        "--sample-pairs",
        type=int,
        default=None,
        help="Max pairs for pairwise similarity stats; omit for exact (default: None)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Directory to write visualizations",
    )
    parser.add_argument(
        "--tsne-dimensions", type=int, default=3, choices=[2, 3], help="t-SNE dimensionality: 2 or 3 (default: 3)"
    )
    args = parser.parse_args()

    payload_path = Path(__file__).parent / _SCENARIO_FILES[args.mode]
    with open(payload_path) as f:
        payload = json.load(f)

    embeddings: dict[str, list[float]] = payload["embeddings"]
    print(f"Mode      : {args.mode}")
    print(f"Embeddings: {len(embeddings)} items, dim={len(next(iter(embeddings.values())))}")

    if args.mode == "plant2image":
        instance_to_image: dict[str, list[str]] = payload["instance_to_image"]
        print(f"Parents   : {len(instance_to_image)} full-field images")
        print(f"Instances : {sum(len(v) for v in instance_to_image.values())} crop instances")
        print()
        result = run_plant2image_eval(
            embeddings=embeddings,
            instance_to_image=instance_to_image,
            k_values=args.k_values,
            dataset_root=args.dataset_root,
            sample_pairs=args.sample_pairs,
        )

    elif args.mode == "plant2plant":
        instance_labels: dict[str, str] = payload["instance_labels"]
        print(f"Labels    : {len(instance_labels)} labelled instances")
        print()
        result = run_plant2plant_eval(
            embeddings=embeddings,
            instance_labels=instance_labels,
            k_values=args.k_values,
            sample_pairs=args.sample_pairs,
        )

    else:
        metadata: dict[str, MetadataGroup] | None = None
        if "metadata" in payload:
            metadata = {key: MetadataGroup(**group) for key, group in payload["metadata"].items()}
        if metadata is not None:
            print(f"Metadata  : {len(metadata)} groups")
        print()
        result = run_image2image_eval(
            image_embeddings=embeddings,
            k_values=args.k_values,
            dataset_root=args.dataset_root,
            sample_pairs=args.sample_pairs,
            metadata=metadata,
        )

    print_result(result)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_knn_confusion(result, output_path=str(out / "class_confusion_matrix.html"))
        plot_cosine_similarity(embeddings, result, output_path=str(out / "cosine_similarity.html"))
        plot_tsne(embeddings, result, output_path=str(out / "tsne.html"), dimensions=args.tsne_dimensions)
        plot_lle(embeddings, result, output_path=str(out / "lle.html"))


if __name__ == "__main__":
    main()
