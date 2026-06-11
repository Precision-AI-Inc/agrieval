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

"""Service layer: orchestrates embedding metrics into the fixed analysis JSON."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from pai.ag_emb.metrics import (
    centroid_similarity_stats,
    effective_rank,
    intra_inter_similarity_gap,
    knn_confusion_matrix,
    knn_label_ndcg_at_k,
    knn_label_purity_at_k,
    knn_map_at_k,
    pairwise_similarity_stats,
    top_k_neighbors,
)

# Matches folder names of the form  crop_[camera]  e.g.  corn_[HB-25000SBC]
_CROP_CAMERA_RE = re.compile(r"^(.+?)_\[.+\]$")

# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------


def _parse_crop(folder_name: str) -> str:
    """Extract the crop name from a folder that may follow the ``crop_[camera]`` convention.

    Examples
    --------
    ``corn_[HB-25000SBC]``  →  ``corn``
    ``soybean_[anafi]``     →  ``soybean``
    ``corn``                →  ``corn``   (plain folder, returned as-is)
    """
    match = _CROP_CAMERA_RE.match(folder_name)
    return match.group(1) if match else folder_name


def extract_labels(paths: list[str], dataset_root: str | None = None) -> list[str]:
    """Infer crop class from image paths.

    The class is extracted from the first path component after the dataset
    root.  Two folder naming conventions are supported:

    * ``{root}/{crop}_{[camera]}/img/{image}``  — real dataset layout, e.g.
      ``dummy/corn_[HB-25000SBC]/img/220622-img.png``
    * ``{root}/{crop}/{camera}/{image}``  — plain layout without camera tags.

    If ``dataset_root`` is not given the longest common directory prefix of
    all paths is used as the root automatically.

    Parameters
    ----------
    paths : list[str]
        Image paths as they appear in the embeddings dictionary.
    dataset_root : str | None
        Explicit root to strip before label extraction.

    Returns
    -------
    list[str]
        Crop class label per path (same order as ``paths``).
    """
    if not paths:
        return []

    if dataset_root is not None:
        root_prefix = dataset_root.rstrip("/") + "/"
    else:
        raw = os.path.commonprefix(paths)
        # Trim to the last directory separator so we don't clip mid-word
        root_prefix = raw[: raw.rfind("/") + 1] if "/" in raw else ""

    labels: list[str] = []
    for path in paths:
        remainder = path[len(root_prefix) :] if path.startswith(root_prefix) else path
        parts = Path(remainder).parts
        folder = parts[0] if parts else "unknown"
        labels.append(_parse_crop(folder))

    return labels


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _jsonify(obj: object) -> Any:  # noqa: PLR0911
    """Recursively convert a metrics result to JSON-serialisable types.

    numpy arrays are dropped because they are per-item visualisation artefacts
    that do not belong in a summary JSON.  numpy scalars are converted to
    Python native types.  ``NaN`` and ``inf`` become ``None``.

    Parameters
    ----------
    obj : object
        Arbitrary metrics result object — dict, list, numpy array/scalar, or
        Python scalar.

    Returns
    -------
    object
        JSON-serialisable equivalent of ``obj``.  ``None`` is returned for
        numpy arrays (sentinel value; callers omit these keys).
    """
    if isinstance(obj, np.ndarray):
        return None  # sentinel — callers filter this key out
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                continue  # drop per_item / hub_counts etc.
            serialised = _jsonify(v)
            out[str(k)] = serialised
        return out
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return obj


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def run_evaluation(
    image_embeddings: dict[str, list[float]],
    k_values: list[int],
    dataset_root: str | None,
    sample_pairs: int | None,
) -> dict:
    """Compute the full fixed-schema embedding evaluation.

    Parameters
    ----------
    image_embeddings : dict[str, list[float]]
        Mapping of image path to embedding vector.
    k_values : list[int]
        K cutoffs for nearest-neighbour metrics.
    dataset_root : str | None
        Optional explicit dataset root for label extraction.
    sample_pairs : int | None
        Pair-sampling budget for global similarity stats.

    Returns
    -------
    dict
        Fixed-schema result ready for JSON serialisation.
    """
    paths = list(image_embeddings.keys())
    vectors = list(image_embeddings.values())

    labels_list = extract_labels(paths, dataset_root)
    labels_arr = np.array(labels_list)

    embeddings = np.array(vectors, dtype=np.float32)
    n, d = embeddings.shape
    unique_classes = sorted(set(labels_list))

    # --- Global nearest-neighbour computation (shared across all metrics) ---
    neighbors = top_k_neighbors(embeddings, ks=k_values)
    effective_ks = sorted(neighbors.keys())

    purity_raw = knn_label_purity_at_k(neighbors, labels_arr)
    ndcg_raw = knn_label_ndcg_at_k(neighbors, labels_arr)
    map_raw = knn_map_at_k(neighbors, labels_arr)

    # Preserve per_item arrays before _jsonify strips them (needed for per-class slices)
    purity_per_item: dict[int, np.ndarray] = {k: stats["per_item"] for k, stats in purity_raw.items()}
    ndcg_per_item: dict[int, np.ndarray] = {k: stats["per_item"] for k, stats in ndcg_raw.items()}
    map_per_item: dict[int, np.ndarray] = {k: stats["per_item"] for k, stats in map_raw.items()}

    # --- Global metrics ---
    global_metrics: dict = {
        "pairwise_similarity_stats": pairwise_similarity_stats(embeddings, sample_pairs=sample_pairs),
        "intra_inter_similarity_gap": intra_inter_similarity_gap(embeddings, labels_arr, sample_pairs=sample_pairs),
        "knn_label_purity": purity_raw,
        "knn_label_ndcg": ndcg_raw,
        "knn_map": map_raw,
        "effective_rank": effective_rank(embeddings),
        "centroid_similarity_stats": centroid_similarity_stats(embeddings),
    }

    # --- Per-class metrics ---
    per_class: dict = {}
    for cls in unique_classes:
        cls_mask = labels_arr == cls
        cls_indices = np.where(cls_mask)[0]
        cls_embeddings = embeddings[cls_mask]
        cls_n = int(np.sum(cls_mask))

        cls_metrics: dict = {"n_items": cls_n}

        if cls_n >= 2:
            cls_sample = min(sample_pairs, 100_000) if sample_pairs is not None else None
            cls_metrics["pairwise_similarity_stats"] = pairwise_similarity_stats(
                cls_embeddings, sample_pairs=cls_sample
            )
            cls_metrics["centroid_similarity_stats"] = centroid_similarity_stats(cls_embeddings)
            cls_metrics["effective_rank"] = effective_rank(cls_embeddings)

        def _slice_per_item(per_item_by_k: dict[int, np.ndarray], _idx: np.ndarray = cls_indices) -> dict:
            out: dict = {}
            for k, per_item in per_item_by_k.items():
                vals = per_item[_idx].astype(np.float64)
                p05, p50, p95 = np.percentile(vals, [5, 50, 95]).tolist()
                out[str(k)] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "p05": p05,
                    "p50": p50,
                    "p95": p95,
                }
            return out

        cls_metrics["knn_label_purity"] = _slice_per_item(purity_per_item)
        cls_metrics["knn_label_ndcg"] = _slice_per_item(ndcg_per_item)
        cls_metrics["knn_map"] = _slice_per_item(map_per_item)

        per_class[cls] = cls_metrics

    confusion_raw = knn_confusion_matrix(neighbors, labels_arr)

    return {
        "n_items": n,
        "embedding_dim": d,
        "classes": unique_classes,
        "k_values": effective_ks,
        "item_paths": paths,
        "item_labels": labels_list,
        "knn_confusion": _jsonify(confusion_raw),
        "global_metrics": _jsonify(global_metrics),
        "per_class": {cls: _jsonify(m) for cls, m in per_class.items()},
    }
