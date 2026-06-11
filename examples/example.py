# ======================================================================
#  CONFIDENTIAL — © Precision AI 2025. All Rights Reserved.
#
#  This source code and any accompanying documentation contain
#  confidential and proprietary information of Precision AI.
#
#  Unauthorized reproduction, disclosure, modification, or distribution
#  of this material is strictly prohibited and will be prosecuted to the
#  fullest extent of the law.
# ======================================================================
"""Run the evaluation service directly — no HTTP server required.

Loads dummy_input.json, calls run_evaluation(), and prints selected KPIs.

Usage (from repo root):
    python examples/example.py
    python examples/example.py --dataset-root /path/to/dataset
    python examples/example.py --k-values 1 5 10
    python examples/example.py --output-dir output --tsne-dimensions 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--sample-pairs", type=int, default=None,
                        help="Max pairs for pairwise similarity stats; omit for exact (default: None)")
    parser.add_argument("--output-dir", default="output", metavar="DIR",
                        help="Directory to write visualizations (confusion matrix + t-SNE)")
    parser.add_argument("--tsne-dimensions", type=int, default=3, choices=[2, 3],
                        help="t-SNE dimensionality: 2 or 3 (default: 3)")
    args = parser.parse_args()

    payload_path = Path(__file__).parent.parent / "dummy_input.json"
    with open(payload_path) as f:
        payload = json.load(f)

    embeddings: dict[str, list[float]] = payload["embeddings"]
    print(f"Loaded {len(embeddings)} embeddings, "
          f"dim={len(next(iter(embeddings.values())))}")
    print()

    from pai.ag_emb.services.reporting import print_result
    from pai.ag_emb.services.evaluate import run_evaluation

    result = run_evaluation(
        image_embeddings=embeddings,
        k_values=args.k_values,
        dataset_root=args.dataset_root,
        sample_pairs=args.sample_pairs,
    )

    print_result(result)

    if args.output_dir:
        from pai.ag_emb.services.reporting import (
            plot_cosine_similarity,
            plot_knn_confusion,
            plot_lle,
            plot_tsne,
        )
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_knn_confusion(result, output_path=str(out / "class_confusion_matrix.html"))
        plot_cosine_similarity(embeddings, result, output_path=str(out / "cosine_similarity.html"))
        plot_tsne(embeddings, result, output_path=str(out / "tsne.html"),
                  dimensions=args.tsne_dimensions)
        plot_lle(embeddings, result, output_path=str(out / "lle.html"))


if __name__ == "__main__":
    main()
