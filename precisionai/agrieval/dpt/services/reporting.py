# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Human-readable display of dense patch token evaluation results."""

from __future__ import annotations

from typing import Any


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


def print_result(result: dict[str, Any]) -> None:
    """Pretty-print the output of ``run_dpt_eval()`` or ``run_dpt_image_eval()``.

    Handles both wirings — tile-keyed (``n_tiles``/``tile_ids``/``per_tile``)
    and whole-image-keyed (``n_images``/``image_ids``/``per_image``) result
    dicts — printing every global metric and, when present, per-class patch
    counts, kNN purity, and warnings.

    Parameters
    ----------
    result : dict[str, Any]
        Return value of ``run_dpt_eval()`` or ``run_dpt_image_eval()``.
    """
    n_entries = result["n_tiles"] if "n_tiles" in result else result["n_images"]

    print(f"n_entries: {n_entries}")
    print(f"embed_dim: {result['embed_dim']}")
    print(f"grid     : {result['grid_height']}x{result['grid_width']}")
    print(f"n_patches: {result['n_patches']}")
    print(f"k_values : {result['k_values']}")
    print(f"classes  : {result['classes']}")
    print()

    _print_global(result["global_metrics"])

    if result["per_class"] is not None:
        print()
        _print_per_class(result["per_class"])

    if result["knn_confusion"] is not None:
        print()
        _print_knn_confusion(result["knn_confusion"], result["k_values"])

    if result["warnings"]:
        print()
        print("Warnings:")
        for w in result["warnings"]:
            print(f"  - {w}")


def _print_global(gm: dict[str, Any]) -> None:
    """Print the global_metrics section of a dpt evaluation result."""
    print("── global_metrics ──────────────────────────────────────────────────────")
    er = gm["effective_rank"]
    pca = gm["pca_explained_variance"]
    pw = gm["pairwise_similarity_stats"]
    ctr = gm["centroid_similarity_stats"]
    print(
        f"  effective_rank        : {_fmt(er['effective_rank'])}  "
        f"(ratio={_fmt(er['effective_rank_ratio'])}  dim={er['embedding_dim']})"
    )
    print(
        f"  pca_explained_variance: pc1={_fmt(pca['pc1'])}  top_10={_fmt(pca['top_10'])}  "
        f"top_50={_fmt(pca['top_50'])}  top_100={_fmt(pca['top_100'])}"
    )
    print(
        f"  pairwise cosine       : mean={_fmt(pw['mean'])}  std={_fmt(pw['std'])}  "
        f"(p05={_fmt(pw['p05'])}  p50={_fmt(pw['p50'])}  p95={_fmt(pw['p95'])})"
    )
    print(
        f"  centroid cosine       : mean={_fmt(ctr['mean_cosine_to_centroid'])}  "
        f"std={_fmt(ctr['std_cosine_to_centroid'])}  norm={_fmt(ctr['centroid_norm'])}"
    )
    print(f"  uniformity            : {_fmt(gm['uniformity'])}")
    print(f"  mean_patch_smoothness : {_fmt(gm['mean_patch_smoothness'])}")
    print(f"  mean_outlier_fraction : {_fmt(gm['mean_outlier_fraction'])}")


def _print_per_class(per_class: dict[str, Any]) -> None:
    """Print the per_class section of a dpt evaluation result."""
    print("── per_class ───────────────────────────────────────────────────────────")
    for cls_name, metrics in per_class.items():
        er = metrics.get("effective_rank")
        pca = metrics.get("pca_explained_variance")
        suffix = f"  effective_rank={_fmt(er['effective_rank'])}  pc1={_fmt(pca['pc1'])}" if er is not None else ""
        print(f"  {cls_name:<20} n_patches={metrics['n_patches']}{suffix}")


def _print_knn_confusion(knn_confusion: dict[str, Any], k_values: list[int]) -> None:
    """Print the knn_confusion purity section of a dpt evaluation result.

    A requested ``k`` may be absent from ``purity`` when subsampling shrank
    the corpus below that ``k`` — skipped rather than treated as an error.
    """
    print("── knn_confusion ───────────────────────────────────────────────────────")
    purity = knn_confusion["purity"]
    for k in k_values:
        p = purity.get(str(k))
        if p is not None:
            print(f"  purity@{k:<3}: mean={_fmt(p['mean'])}  std={_fmt(p['std'])}")
