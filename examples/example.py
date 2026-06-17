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

Four example scenarios are bundled:

* **tight** (default) — embeddings only, well-separated classes
  (intra-class cosine ≈ 0.99, inter-class ≈ 0.00).
* **sparse** — embeddings only, heavily overlapping classes
  (gap ≈ 0.07, purity@5 ≈ 0.56).
* **tight_meta** — same tight embeddings with metadata groups, enabling
  graded nDCG and per-attribute KPIs.
* **sparse_meta** — same sparse embeddings with metadata groups.

Usage (from repo root):
    python examples/example.py
    python examples/example.py --scenario sparse
    python examples/example.py --scenario tight_meta
    python examples/example.py --scenario sparse_meta
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

from pai.ag_emb.schemas.evaluate import MetadataGroup
from pai.ag_emb.services.evaluate import run_evaluation
from pai.ag_emb.services.reporting import (
    plot_cosine_similarity,
    plot_knn_confusion,
    plot_lle,
    plot_tsne,
    print_result,
)

_SCENARIO_FILES: dict[str, str] = {
    "tight": "emb_tight.json",
    "sparse": "emb_sparse.json",
    "tight_meta": "emb_tight_meta.json",
    "sparse_meta": "emb_sparse_meta.json",
}


def main() -> None:
    """Load an example input file, run the evaluation service, and print results.

    Optionally writes visualization HTML files to an output directory.

    Recognised arguments
    --------------------
    --scenario : {tight, sparse, tight_meta, sparse_meta}
        Which bundled scenario to run.  ``tight``/``sparse`` use embeddings
        only; ``tight_meta``/``sparse_meta`` additionally supply metadata
        groups for graded nDCG and per-attribute KPIs.
    --dataset-root : str
        Dataset root for Wiring 1 label extraction.  Defaults to ``"images"``
        to match the ``images/class_subgroup/`` embedding key prefix.
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
        "--scenario",
        choices=list(_SCENARIO_FILES),
        default="tight",
        help="Scenario to run (default: tight)",
    )
    parser.add_argument(
        "--dataset-root",
        default="images",
        help="Root prefix for Wiring 1 label extraction (default: images)",
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

    payload_path = Path(__file__).parent / _SCENARIO_FILES[args.scenario]
    with open(payload_path) as f:
        payload = json.load(f)

    embeddings: dict[str, list[float]] = payload["embeddings"]
    metadata: dict[str, MetadataGroup] | None = None
    if "metadata" in payload:
        metadata = {key: MetadataGroup(**group) for key, group in payload["metadata"].items()}

    print(f"Scenario  : {args.scenario}")
    print(f"Embeddings: {len(embeddings)} items, dim={len(next(iter(embeddings.values())))}")
    if metadata is not None:
        print(f"Metadata  : {len(metadata)} groups")
    print()

    result = run_evaluation(
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
