# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Service layer: orchestrates dense patch token metrics into the fixed analysis JSON."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from precisionai.agrieval.dpt.metrics import outlier_fraction, patch_norm_stats, patch_smoothness
from precisionai.agrieval.emb.metrics import (
    centroid_similarity_stats,
    effective_rank,
    knn_confusion_matrix,
    knn_label_purity_at_k,
    pairwise_similarity_stats,
    pca_explained_variance,
    top_k_neighbors,
    uniformity,
)
from precisionai.agrieval.emb.services.serialization import _jsonify
from precisionai.agrieval.seg.services.evaluate import (
    _build_color_map,
    _build_lut,
    _load_mask,
    _rgb_to_class_ids,
    _validate_colors,
    load_classes,
)

_SUBSAMPLE_SEED = 42


def _validate_tile_arrays(tiles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate loaded tile arrays and return them converted to float32.

    Applies the same rules as the request schema's inline-tile validator:
    non-empty dict, safe tile IDs, 3-D ``[C, H, W]`` shape with no empty
    axis, finite values at float32 precision, and a uniform shape across
    all tiles.

    Parameters
    ----------
    tiles : dict[str, np.ndarray]
        Map of tile ID to a loaded array.

    Returns
    -------
    dict[str, np.ndarray]
        The same mapping with every array converted to float32.

    Raises
    ------
    ValueError
        If any rule is violated.
    """
    if not tiles:
        raise ValueError("At least 1 tile is required.")
    validated: dict[str, np.ndarray] = {}
    shapes: set[tuple[int, ...]] = set()
    for tile_id, arr in tiles.items():
        if not tile_id or "/" in tile_id or "\\" in tile_id:
            raise ValueError(f"Invalid tile ID {tile_id!r}: must be non-empty and must not contain path separators.")
        try:
            with np.errstate(over="ignore"):
                converted = np.asarray(arr, dtype=np.float32)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Tile '{tile_id}' could not be converted to a float32 array.") from err
        if converted.ndim != 3:
            raise ValueError(f"Tile '{tile_id}' must be 3-dimensional [C, H, W]; got {converted.ndim} dimension(s).")
        if 0 in converted.shape:
            raise ValueError(f"Tile '{tile_id}' must not be empty along any axis. Got shape {converted.shape}.")
        if not np.all(np.isfinite(converted)):
            raise ValueError(f"Tile '{tile_id}' contains non-finite values (NaN or inf).")
        shapes.add(converted.shape)
        validated[tile_id] = converted
    if len(shapes) > 1:
        raise ValueError(f"All tiles must share the same (C, H, W) shape. Found: {sorted(shapes)}")
    return validated


def load_tiles(tiles_path: Path | str) -> dict[str, np.ndarray]:
    """Load dense patch token tiles from disk.

    Two layouts are supported:

    - a **directory** containing one ``.npy`` file per tile (searched
      recursively); each file's stem becomes the tile ID;
    - a single **``.npz`` archive** whose member names are the tile IDs.

    Arrays are loaded with ``allow_pickle=False``, so archives containing
    pickled objects are rejected rather than executed. Tiles produced as
    torch tensors should be saved via ``tensor.numpy()`` → ``np.save`` —
    loading ``.pt`` files would require a torch dependency and is not
    supported.

    Parameters
    ----------
    tiles_path : Path | str
        Directory of ``.npy`` files or path to a ``.npz`` archive.

    Returns
    -------
    dict[str, np.ndarray]
        Map of tile ID to a validated float32 ``[C, H, W]`` array, ready for
        :func:`run_dpt_eval`.

    Raises
    ------
    FileNotFoundError
        If ``tiles_path`` is neither an existing directory nor a ``.npz`` file.
    ValueError
        If no tiles are found, tile IDs collide, or any array fails the
        shape/finiteness rules.
    """
    path = Path(tiles_path)
    raw: dict[str, np.ndarray] = {}
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.npy") if p.is_file())
        if not files:
            raise ValueError(f"No .npy tile files found under '{path}'.")
        for f in files:
            if f.stem in raw:
                raise ValueError(f"Duplicate tile ID '{f.stem}': multiple .npy files share this stem under '{path}'.")
            raw[f.stem] = np.load(f, allow_pickle=False)
    elif path.is_file() and path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            raw = {name: archive[name] for name in archive.files}
        if not raw:
            raise ValueError(f"The .npz archive '{path}' contains no arrays.")
    else:
        raise FileNotFoundError(f"tiles_path must be an existing directory of .npy files or a .npz archive: '{path}'")
    return _validate_tile_arrays(raw)


def _stack_tiles(tiles: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Convert a ``tile_id -> array-like`` dict into ordered IDs and a stacked array.

    Parameters
    ----------
    tiles : dict[str, Any]
        Map of tile ID to a ``[C, H, W]`` array-like (nested lists, numpy
        arrays, or anything ``np.asarray`` accepts).

    Returns
    -------
    tuple[list[str], np.ndarray]
        Sorted tile IDs and a ``[N, C, H, W]`` float32 array in that order.
    """
    tile_ids = sorted(tiles)
    arr = np.asarray([np.asarray(tiles[tile_id], dtype=np.float32) for tile_id in tile_ids], dtype=np.float32)
    return tile_ids, arr


def _flatten_patches(tiles_arr: np.ndarray) -> np.ndarray:
    """Flatten ``[N, C, H, W]`` tiles into a ``[N*H*W, C]`` patch-token matrix."""
    n, c, h, w = tiles_arr.shape
    return tiles_arr.transpose(0, 2, 3, 1).reshape(n * h * w, c)


def _downsample_mask_to_grid(class_ids: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Majority-vote downsample a full-resolution class-ID mask to a patch grid.

    Each patch cell is assigned the class covering the most pixels in its
    corresponding mask region, matching the label-assignment convention of
    the upstream dense-feature benchmark (``pai-vision-feature-map-eval``).
    Cells that receive no pixels — possible only when the mask is smaller
    than the grid along an axis — fall back to nearest-neighbor sampling.

    Parameters
    ----------
    class_ids : np.ndarray
        Full-resolution class-ID mask, shape ``[H, W]``, non-negative IDs.
    grid_h : int
        Target patch-grid height.
    grid_w : int
        Target patch-grid width.

    Returns
    -------
    np.ndarray
        Class-ID mask resampled to shape ``[grid_h, grid_w]``.
    """
    mask_h, mask_w = class_ids.shape
    n_classes = int(class_ids.max()) + 1
    n_cells = grid_h * grid_w

    cell_r = (np.arange(mask_h) * grid_h) // mask_h
    cell_c = (np.arange(mask_w) * grid_w) // mask_w
    cell_idx = (cell_r[:, None] * grid_w + cell_c[None, :]).ravel()
    flat_labels = class_ids.ravel().astype(np.int64)

    combined = cell_idx * n_classes + flat_labels
    counts = np.bincount(combined, minlength=n_cells * n_classes).reshape(n_cells, n_classes)
    majority = counts.argmax(axis=1).reshape(grid_h, grid_w).astype(np.int64)

    empty = (counts.sum(axis=1) == 0).reshape(grid_h, grid_w)
    if np.any(empty):
        img = Image.fromarray(class_ids.astype(np.int32), mode="I")
        nearest = np.array(img.resize((grid_w, grid_h), Image.Resampling.NEAREST), dtype=np.int64)
        majority[empty] = nearest[empty]
    return majority


def _find_mask_file(masks_dir: Path, tile_id: str) -> Path:
    """Find the ground-truth mask file whose stem matches ``tile_id``.

    The tile ID is glob-escaped so IDs containing ``*``, ``?``, or ``[`` are
    matched literally rather than interpreted as wildcard patterns, and IDs
    containing path separators are rejected outright — a tile ID names a
    file stem, never a path.

    Parameters
    ----------
    masks_dir : Path
        Directory containing ground-truth colour-coded masks.
    tile_id : str
        Tile ID to match against a file stem.

    Returns
    -------
    Path
        The matched mask file (lexicographically first when multiple
        extensions exist for the same stem).

    Raises
    ------
    ValueError
        If ``tile_id`` is empty or contains a path separator.
    FileNotFoundError
        If no file with a matching stem is found under ``masks_dir``.
    """
    if not tile_id or "/" in tile_id or "\\" in tile_id:
        raise ValueError(f"Invalid tile ID {tile_id!r}: must be non-empty and must not contain path separators.")
    matches = sorted(p for p in masks_dir.rglob(f"{glob.escape(tile_id)}.*") if p.is_file())
    if not matches:
        raise FileNotFoundError(f"No ground-truth mask found for tile '{tile_id}' under '{masks_dir}'.")
    return matches[0]


def _compute_label_metrics(
    *,
    tile_ids: list[str],
    tiles_arr: np.ndarray,
    patch_tokens: np.ndarray,
    masks_dir: str | Path,
    classes_path: str | Path,
    k_values: list[int],
    max_patches: int,
    warnings_out: list[str],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Decode ground-truth masks, align them to the patch grid, and score label-aware metrics.

    ``masks_dir`` and ``classes_path`` must already be resolved to real
    locations — path resolution against ``dataset_root`` is an API-layer
    concern, matching ``run_seg_eval``.

    Returns
    -------
    tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]
        ``(classes, per_class, knn_confusion)``.
    """
    _, _, h, w = tiles_arr.shape

    classes_entries = load_classes(classes_path)
    color_map = _build_color_map(classes_entries)
    lut = _build_lut(color_map)
    class_names = [entry[0] for entry in classes_entries]
    id_to_name = {int(entry[2]): entry[0] for entry in classes_entries}

    masks_root = Path(masks_dir)
    if not masks_root.is_dir():
        raise FileNotFoundError(f"masks_dir does not exist or is not a directory: '{masks_root}'")

    labels_per_tile: list[np.ndarray] = []
    for tile_id in tile_ids:
        mask_path = _find_mask_file(masks_root, tile_id)
        mask_rgb = _load_mask(mask_path)
        _validate_colors(mask_rgb, color_map, mask_path)
        class_ids_full = _rgb_to_class_ids(mask_rgb, lut)
        class_ids_grid = _downsample_mask_to_grid(class_ids_full, h, w)
        labels_per_tile.append(class_ids_grid.reshape(-1))
    patch_labels = np.concatenate(labels_per_tile)

    n_patches_total = patch_tokens.shape[0]
    if n_patches_total > max_patches:
        rng = np.random.default_rng(_SUBSAMPLE_SEED)
        idx = rng.choice(n_patches_total, size=max_patches, replace=False)
        sub_tokens, sub_labels = patch_tokens[idx], patch_labels[idx]
        warnings_out.append(f"Subsampled {max_patches} of {n_patches_total} patches for label-aware kNN metrics.")
    else:
        sub_tokens, sub_labels = patch_tokens, patch_labels

    sub_label_names = np.array([id_to_name.get(int(cid), str(cid)) for cid in sub_labels])
    neighbors_by_k = top_k_neighbors(sub_tokens, k_values)
    knn_confusion = {
        "confusion": knn_confusion_matrix(neighbors_by_k, sub_label_names),
        "purity": knn_label_purity_at_k(neighbors_by_k, sub_label_names),
    }

    present_class_ids = sorted({cid for cid in patch_labels.tolist() if 0 <= cid < len(class_names)})
    classes = [class_names[cid] for cid in present_class_ids]

    per_class: dict[str, dict[str, Any]] = {}
    for cid in present_class_ids:
        class_tokens = patch_tokens[patch_labels == cid]
        class_metrics: dict[str, Any] = {"n_patches": int(class_tokens.shape[0])}
        if class_tokens.shape[0] >= 2:
            class_metrics["effective_rank"] = effective_rank(class_tokens, normalize=False)
            class_metrics["pca_explained_variance"] = pca_explained_variance(class_tokens, normalize=False)
        per_class[class_names[cid]] = class_metrics

    return classes, per_class, knn_confusion


def run_dpt_eval(
    tiles: dict[str, Any],
    *,
    k_values: list[int] | None = None,
    sample_pairs: int | None = 1_000_000,
    max_patches: int = 20_000,
    masks_dir: str | Path | None = None,
    classes_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run dense patch token evaluation over a batch of tiles.

    Parameters
    ----------
    tiles : dict[str, Any]
        Map of tile ID to a ``[C, H, W]`` channels-first feature map. Values
        may be nested lists, numpy arrays (e.g. from :func:`load_tiles`), or
        any array-like ``np.asarray`` accepts — CPU torch tensors work
        without a torch dependency here. All tiles must share the same
        ``C``, ``H``, and ``W``.
    k_values : list[int] | None
        K cutoffs for nearest-neighbour label metrics. Defaults to ``[5, 10, 20]``.
    sample_pairs : int | None
        Max random pairs for global pairwise similarity stats. ``None`` computes exactly.
    max_patches : int
        Max patches used for O(N²) label-aware kNN computation. Larger corpora
        are randomly subsampled (seeded) and the drop is reported in ``warnings``.
    masks_dir : str | Path | None
        Directory of ground-truth colour-coded masks, one per tile (filename
        stem must match the tile ID). Must already be resolved to a real
        location — dataset_root resolution is an API-layer concern, matching
        ``run_seg_eval``. Requires ``classes_path``.
    classes_path : str | Path | None
        Path to an AgriBench ``class_map.json``, same format as ``seg``.
        Must already be resolved. Requires ``masks_dir``.

    Returns
    -------
    dict[str, Any]
        Fixed analysis JSON — see :class:`precisionai.agrieval.dpt.schemas.evaluate.DptEvalResponse`.

    Raises
    ------
    ValueError
        If tiles do not share a uniform ``(C, H, W)`` shape.
    FileNotFoundError
        If ``masks_dir`` does not exist, or a tile has no matching ground-truth mask.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    tile_ids, tiles_arr = _stack_tiles(tiles)
    n, c, h, w = tiles_arr.shape
    patch_tokens = _flatten_patches(tiles_arr)

    warnings_list: list[str] = []

    # Spectrum metrics run on raw centered features (normalize=False) to match
    # the upstream dense-feature benchmark; cosine-based metrics are inherently
    # computed on L2-normalized tokens.
    global_metrics: dict[str, Any] = {
        "effective_rank": effective_rank(patch_tokens, normalize=False),
        "pca_explained_variance": pca_explained_variance(patch_tokens, normalize=False),
        "pairwise_similarity_stats": pairwise_similarity_stats(patch_tokens, sample_pairs=sample_pairs),
        "centroid_similarity_stats": centroid_similarity_stats(patch_tokens),
        "uniformity": uniformity(patch_tokens),
    }

    all_norms = np.linalg.norm(patch_tokens, axis=1)
    # Corpus-wide mean + 3 sigma, matching the benchmark's artifact-patch definition.
    outlier_threshold = float(all_norms.mean() + 3.0 * all_norms.std())

    hw = h * w
    per_tile: dict[str, dict[str, Any]] = {}
    for i, tile_id in enumerate(tile_ids):
        tile_tokens = patch_tokens[i * hw : (i + 1) * hw]
        tile_norms = all_norms[i * hw : (i + 1) * hw]
        per_tile[tile_id] = {
            "patch_norm_stats": patch_norm_stats(tile_tokens),
            "patch_smoothness": patch_smoothness(tiles_arr[i]),
            "outlier_fraction": outlier_fraction(tile_norms, threshold=outlier_threshold),
        }

    global_metrics["mean_patch_smoothness"] = float(np.mean([m["patch_smoothness"] for m in per_tile.values()]))
    global_metrics["mean_outlier_fraction"] = float(np.mean([m["outlier_fraction"] for m in per_tile.values()]))

    classes: list[str] | None = None
    per_class: dict[str, dict[str, Any]] | None = None
    knn_confusion: dict[str, Any] | None = None

    if masks_dir is not None and classes_path is not None:
        classes, per_class, knn_confusion = _compute_label_metrics(
            tile_ids=tile_ids,
            tiles_arr=tiles_arr,
            patch_tokens=patch_tokens,
            masks_dir=masks_dir,
            classes_path=classes_path,
            k_values=k_values,
            max_patches=max_patches,
            warnings_out=warnings_list,
        )

    return {
        "n_tiles": n,
        "embed_dim": c,
        "grid_height": h,
        "grid_width": w,
        "n_patches": n * h * w,
        "tile_ids": tile_ids,
        "k_values": k_values,
        "global_metrics": _jsonify(global_metrics),
        "per_tile": _jsonify(per_tile),
        "classes": classes,
        "per_class": _jsonify(per_class) if per_class is not None else None,
        "knn_confusion": _jsonify(knn_confusion) if knn_confusion is not None else None,
        "warnings": warnings_list or None,
    }
