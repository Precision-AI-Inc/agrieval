# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Service layer: orchestrates dense patch token metrics into the fixed analysis JSON."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, NamedTuple

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


_TILE_BATCH_KEYS = ("feature_maps", "tile_image_id", "tile_index", "tile_y0", "tile_x0", "filenames")
_IMAGE_BATCH_KEYS = ("features", "filenames")


class TilePlacement(NamedTuple):
    """A tile's source image and exact pixel rectangle within it."""

    image_stem: str
    y0: int
    x0: int
    height: int
    width: int


def _open_npz(path: Path | str, *, kind: str) -> Path:
    """Resolve ``path`` to an existing ``.npz`` archive, raising a clear error otherwise."""
    path = Path(path)
    if not (path.is_file() and path.suffix.lower() == ".npz"):
        raise FileNotFoundError(f"{kind}s_path must be an existing .npz archive: '{path}'")
    return path


def _require_keys(archive_files: set[str], required: tuple[str, ...], *, path: Path) -> None:
    """Raise ``ValueError`` naming exactly which required archive keys are missing."""
    missing = [key for key in required if key not in archive_files]
    if missing:
        raise ValueError(f"Archive '{path}' is missing required key(s): {missing}.")


def _synthesize_tile_id(image_stem: str, tile_index: int) -> str:
    """Build a deterministic tile ID matching the producer's own ``tile_00.npz``-style naming."""
    return f"{image_stem}_tile_{tile_index:02d}"


def _parse_tile_pixel_size(meta_json: str, *, path: Path) -> tuple[int, int]:
    """Parse the ``"WxH"``-formatted ``tile`` field out of an archive's ``meta`` JSON string.

    ``meta`` is otherwise decorative (model name, embed dim, grid size, ...)
    but this one field is load-bearing: it's the only place a tile's pixel
    footprint is recorded, needed to crop a whole-image mask to a tile's
    exact ``(tile_y0, tile_x0)`` rectangle.

    Returns
    -------
    tuple[int, int]
        ``(height, width)`` in pixels.

    Raises
    ------
    ValueError
        If ``meta`` isn't valid JSON, has no ``tile`` field, or ``tile``
        isn't formatted ``"<width>x<height>"``.
    """
    try:
        meta = json.loads(meta_json)
    except json.JSONDecodeError as err:
        raise ValueError(f"Archive '{path}' has a malformed 'meta' JSON string: {err}") from err
    tile = meta.get("tile")
    if not isinstance(tile, str) or "x" not in tile:
        raise ValueError(f"Archive '{path}' meta.tile must be a \"<width>x<height>\" string; got {tile!r}.")
    width_str, _, height_str = tile.partition("x")
    try:
        return int(height_str), int(width_str)
    except ValueError as err:
        raise ValueError(f"Archive '{path}' meta.tile must be a \"<width>x<height>\" string; got {tile!r}.") from err


def _load_tile_batch_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Open, validate, and return the raw component arrays of a tile-batch ``.npz`` archive.

    Shared by :func:`load_tiles` and :func:`load_tile_placement` so both
    read the exact same validated data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(feature_maps, tile_image_id, tile_index, tile_y0, tile_x0, filenames)``.

    Raises
    ------
    ValueError
        If a required key is missing, ``feature_maps`` isn't 4-D with no
        empty axis, a companion array's length doesn't match
        ``feature_maps``, a ``tile_image_id`` value is out of range, a
        ``tile_index``/``tile_y0``/``tile_x0`` value is negative, two tiles
        share the same ``(tile_image_id, tile_index)``, or two filenames
        share the same stem (which would collide in the synthesized tile IDs).
    """
    with np.load(path, allow_pickle=False) as archive:
        _require_keys(set(archive.files), _TILE_BATCH_KEYS, path=path)
        feature_maps = np.asarray(archive["feature_maps"])
        tile_image_id = np.asarray(archive["tile_image_id"])
        tile_index = np.asarray(archive["tile_index"])
        tile_y0 = np.asarray(archive["tile_y0"])
        tile_x0 = np.asarray(archive["tile_x0"])
        filenames = np.asarray(archive["filenames"])

    if feature_maps.ndim != 4:
        raise ValueError(f"'feature_maps' must be 4-dimensional [T, P, H, W]; got {feature_maps.ndim} dimension(s).")
    if 0 in feature_maps.shape:
        raise ValueError(f"'feature_maps' must not be empty along any axis. Got shape {feature_maps.shape}.")

    t = feature_maps.shape[0]
    companions = (
        ("tile_image_id", tile_image_id),
        ("tile_index", tile_index),
        ("tile_y0", tile_y0),
        ("tile_x0", tile_x0),
    )
    for name, arr in companions:
        if arr.shape != (t,):
            raise ValueError(f"'{name}' must have shape ({t},) to match 'feature_maps'; got {arr.shape}.")
    for name, arr in companions[1:]:
        if np.any(arr < 0):
            raise ValueError(f"'{name}' must not contain negative values.")

    n_images = filenames.shape[0]
    out_of_range = sorted({int(v) for v in tile_image_id if not (0 <= v < n_images)})
    if out_of_range:
        raise ValueError(f"'tile_image_id' contains value(s) out of range for {n_images} filenames: {out_of_range}.")

    _validate_unique_stems(filenames)
    _validate_unique_tile_positions(tile_image_id, tile_index)

    return feature_maps, tile_image_id, tile_index, tile_y0, tile_x0, filenames


def _validate_unique_stems(filenames: np.ndarray) -> None:
    """Reject filenames whose stems collide — stems key the synthesized tile IDs."""
    stems: dict[str, str] = {}
    for raw_filename in filenames:
        filename = str(raw_filename)
        stem = Path(filename).stem
        if stem in stems:
            raise ValueError(
                f"Duplicate image stem '{stem}' ('{stems[stem]}' vs '{filename}'): "
                f"filename stems must be unique, as they key the synthesized tile IDs."
            )
        stems[stem] = filename


def _validate_unique_tile_positions(tile_image_id: np.ndarray, tile_index: np.ndarray) -> None:
    """Reject two tiles sharing the same ``(tile_image_id, tile_index)`` pair."""
    seen: set[tuple[int, int]] = set()
    for img_id, idx in zip(tile_image_id, tile_index, strict=True):
        key = (int(img_id), int(idx))
        if key in seen:
            raise ValueError(f"Duplicate tile position: tile_image_id={key[0]}, tile_index={key[1]}.")
        seen.add(key)


def load_tiles(tiles_path: Path | str) -> dict[str, np.ndarray]:
    """Load dense patch token tiles from a batched ``.npz`` archive.

    The archive must contain six arrays: ``feature_maps`` (shape
    ``[T, P, H, W]``, one ``[P, H, W]`` slice per tile), ``tile_image_id``
    (shape ``[T]``, indexing into ``filenames``), ``tile_index`` (shape
    ``[T]``, 0-based sequence number of each tile within its source
    image), ``tile_y0``/``tile_x0`` (shape ``[T]``, pixel offset of each
    tile's top-left corner within its source image), and ``filenames``
    (shape ``[N]``, one source-image filename per ``tile_image_id``
    index). Arrays are loaded with ``allow_pickle=False``, so archives
    containing pickled objects are rejected rather than executed.

    Each tile is keyed by a synthesized ID — ``f"{image_stem}_tile_{tile_index:02d}"``
    — where ``image_stem`` is the matching ``filenames`` entry with its
    extension stripped, matching the naming convention of a per-tile
    ``tile_00.npz`` export. Use :func:`load_tile_placement` to recover each
    tile's exact pixel rectangle for label-aware evaluation.

    Parameters
    ----------
    tiles_path : Path | str
        Path to a ``.npz`` archive matching the schema above.

    Returns
    -------
    dict[str, np.ndarray]
        Map of synthesized tile ID to a validated float32 ``[P, H, W]``
        array, ready for :func:`run_dpt_eval`.

    Raises
    ------
    FileNotFoundError
        If ``tiles_path`` is not an existing ``.npz`` file.
    ValueError
        If a required key is missing or any array fails validation — see
        :func:`_load_tile_batch_arrays`.
    """
    path = _open_npz(tiles_path, kind="tile")
    feature_maps, tile_image_id, tile_index, _, _, filenames = _load_tile_batch_arrays(path)

    with np.errstate(over="ignore"):
        feature_maps = feature_maps.astype(np.float32)
    if not np.all(np.isfinite(feature_maps)):
        raise ValueError("'feature_maps' contains non-finite values (NaN or inf).")

    tiles: dict[str, np.ndarray] = {}
    for i in range(feature_maps.shape[0]):
        image_stem = Path(str(filenames[tile_image_id[i]])).stem
        tile_id = _synthesize_tile_id(image_stem, int(tile_index[i]))
        tiles[tile_id] = feature_maps[i]
    return tiles


def load_tile_placement(tiles_path: Path | str) -> dict[str, TilePlacement]:
    """Recover each tile's source image and exact pixel rectangle from a batched ``.npz`` archive.

    Reads the same archive as :func:`load_tiles` and produces a map with
    identical keys, suitable for passing as ``run_dpt_eval``'s
    ``tile_placement`` argument to enable crop-aware label alignment
    against whole-image ground-truth masks. The tile's pixel height/width
    is read from the archive's ``meta.tile`` field (``"<width>x<height>"``)
    — the only place this producer records it — so ``meta`` must be present
    and well-formed for this function, even though :func:`load_tiles`
    itself doesn't need it.

    Parameters
    ----------
    tiles_path : Path | str
        Path to the same ``.npz`` archive passed to :func:`load_tiles`.

    Returns
    -------
    dict[str, TilePlacement]
        Map of synthesized tile ID to ``TilePlacement(image_stem, y0, x0, height, width)``.

    Raises
    ------
    FileNotFoundError
        If ``tiles_path`` is not an existing ``.npz`` file.
    ValueError
        If a required key is missing, any array fails validation (see
        :func:`_load_tile_batch_arrays`), or ``meta`` is missing/malformed
        (see :func:`_parse_tile_pixel_size`).
    """
    path = _open_npz(tiles_path, kind="tile")
    _, tile_image_id, tile_index, tile_y0, tile_x0, filenames = _load_tile_batch_arrays(path)

    with np.load(path, allow_pickle=False) as archive:
        if "meta" not in archive.files:
            raise ValueError(f"Archive '{path}' is missing required key(s): ['meta'].")
        height, width = _parse_tile_pixel_size(str(archive["meta"]), path=path)

    placement: dict[str, TilePlacement] = {}
    for i in range(tile_image_id.shape[0]):
        image_stem = Path(str(filenames[tile_image_id[i]])).stem
        tile_id = _synthesize_tile_id(image_stem, int(tile_index[i]))
        placement[tile_id] = TilePlacement(image_stem, int(tile_y0[i]), int(tile_x0[i]), height, width)
    return placement


def load_images(images_path: Path | str) -> dict[str, np.ndarray]:
    """Load whole-image dense feature maps from a batched ``.npz`` archive.

    The archive must contain two arrays: ``features`` (shape
    ``[N, P, H, W]``, one ``[P, H, W]`` feature map per image) and
    ``filenames`` (shape ``[N]``, one source-image filename per entry).
    Arrays are loaded with ``allow_pickle=False``, so archives containing
    pickled objects are rejected rather than executed. Unlike tiles, images
    have no position-within-something-bigger — each entry is keyed directly
    by its ``filenames`` entry's stem, matching the convention
    :func:`run_dpt_image_eval` already uses for mask alignment.

    Parameters
    ----------
    images_path : Path | str
        Path to a ``.npz`` archive matching the schema above.

    Returns
    -------
    dict[str, np.ndarray]
        Map of image filename stem to a validated float32 ``[P, H, W]``
        array, ready for :func:`run_dpt_image_eval`.

    Raises
    ------
    FileNotFoundError
        If ``images_path`` is not an existing ``.npz`` file.
    ValueError
        If a required key is missing, ``features`` isn't 4-D with no empty
        axis, ``filenames``'s length doesn't match ``features``, two
        entries share the same filename stem, or any value is non-finite.
    """
    path = _open_npz(images_path, kind="image")
    with np.load(path, allow_pickle=False) as archive:
        _require_keys(set(archive.files), _IMAGE_BATCH_KEYS, path=path)
        features = np.asarray(archive["features"])
        filenames = np.asarray(archive["filenames"])

    if features.ndim != 4:
        raise ValueError(f"'features' must be 4-dimensional [N, P, H, W]; got {features.ndim} dimension(s).")
    if 0 in features.shape:
        raise ValueError(f"'features' must not be empty along any axis. Got shape {features.shape}.")

    n = features.shape[0]
    if filenames.shape != (n,):
        raise ValueError(f"'filenames' must have shape ({n},) to match 'features'; got {filenames.shape}.")

    with np.errstate(over="ignore"):
        features = features.astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("'features' contains non-finite values (NaN or inf).")

    images: dict[str, np.ndarray] = {}
    for i in range(n):
        stem = Path(str(filenames[i])).stem
        if stem in images:
            raise ValueError(f"Duplicate image ID '{stem}': multiple entries share this filename stem.")
        images[stem] = features[i]
    return images


def _stack_tiles(tiles: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Convert a ``tile_id -> array-like`` dict into ordered IDs and a stacked array.

    Parameters
    ----------
    tiles : dict[str, Any]
        Map of tile ID to a ``[P, H, W]`` array-like (nested lists, numpy
        arrays, or anything ``np.asarray`` accepts).

    Returns
    -------
    tuple[list[str], np.ndarray]
        Sorted tile IDs and a ``[N, P, H, W]`` float32 array in that order.
    """
    tile_ids = sorted(tiles)
    arr = np.asarray([np.asarray(tiles[tile_id], dtype=np.float32) for tile_id in tile_ids], dtype=np.float32)
    return tile_ids, arr


def _flatten_patches(tiles_arr: np.ndarray) -> np.ndarray:
    """Flatten ``[N, P, H, W]`` tiles into a ``[N*H*W, P]`` patch-token matrix."""
    n, c, h, w = tiles_arr.shape
    return tiles_arr.transpose(0, 2, 3, 1).reshape(n * h * w, c)


def _downsample_mask_to_grid(class_ids: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Majority-vote downsample a full-resolution class-ID mask to a patch grid.

    Each patch cell is assigned the class covering the most pixels in its
    corresponding mask region. Cells that receive no pixels — possible only
    when the mask is smaller than the grid along an axis — fall back to
    nearest-neighbor sampling.

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

    return _assemble_label_metrics(
        patch_tokens=patch_tokens,
        patch_labels=patch_labels,
        class_names=class_names,
        id_to_name=id_to_name,
        k_values=k_values,
        max_patches=max_patches,
        warnings_out=warnings_out,
    )


def _compute_label_metrics_placed(
    *,
    tile_ids: list[str],
    tiles_arr: np.ndarray,
    patch_tokens: np.ndarray,
    tile_placement: dict[str, TilePlacement],
    masks_dir: str | Path,
    classes_path: str | Path,
    k_values: list[int],
    max_patches: int,
    warnings_out: list[str],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Decode one whole-image mask per source image, crop it per tile, and score label-aware metrics.

    Same rules and outputs as :func:`_compute_label_metrics`, but ground
    truth is one mask per *source image* rather than one mask per tile.
    Each tile's exact pixel rectangle (``tile_placement[tile_id].y0/x0/height/width``)
    is cropped out of its image's mask before the existing
    :func:`_downsample_mask_to_grid` runs — this works correctly even when
    tiles overlap, since each tile's rectangle is used independently and
    isn't assumed to partition the image into a non-overlapping grid.
    Masks are assumed to be at the same pixel resolution as the original
    (untiled) source image the pixel offsets were computed against; a crop
    rectangle extending past the mask's bounds is clipped to what's
    available rather than raising.

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

    missing = [tile_id for tile_id in tile_ids if tile_id not in tile_placement]
    if missing:
        raise ValueError(f"tile_placement is missing entries for tile ID(s): {missing}.")

    tiles_by_image: dict[str, list[str]] = {}
    for tile_id in tile_ids:
        tiles_by_image.setdefault(tile_placement[tile_id].image_stem, []).append(tile_id)

    labels_by_tile: dict[str, np.ndarray] = {}
    for image_stem, image_tile_ids in tiles_by_image.items():
        mask_path = _find_mask_file(masks_root, image_stem)
        mask_rgb = _load_mask(mask_path)
        _validate_colors(mask_rgb, color_map, mask_path)
        class_ids_full = _rgb_to_class_ids(mask_rgb, lut)
        mask_h, mask_w = class_ids_full.shape

        for tile_id in image_tile_ids:
            placement = tile_placement[tile_id]
            y0, x0 = max(placement.y0, 0), max(placement.x0, 0)
            y1 = min(placement.y0 + placement.height, mask_h)
            x1 = min(placement.x0 + placement.width, mask_w)
            if y0 >= y1 or x0 >= x1:
                raise ValueError(
                    f"Tile '{tile_id}' pixel rectangle (y0={placement.y0}, x0={placement.x0}, "
                    f"height={placement.height}, width={placement.width}) lies entirely outside "
                    f"its {mask_h}x{mask_w} ground-truth mask '{mask_path.name}'."
                )
            crop = class_ids_full[y0:y1, x0:x1]
            class_ids_grid = _downsample_mask_to_grid(crop, h, w)
            labels_by_tile[tile_id] = class_ids_grid.reshape(-1)

    patch_labels = np.concatenate([labels_by_tile[tile_id] for tile_id in tile_ids])

    return _assemble_label_metrics(
        patch_tokens=patch_tokens,
        patch_labels=patch_labels,
        class_names=class_names,
        id_to_name=id_to_name,
        k_values=k_values,
        max_patches=max_patches,
        warnings_out=warnings_out,
    )


def _assemble_label_metrics(
    *,
    patch_tokens: np.ndarray,
    patch_labels: np.ndarray,
    class_names: list[str],
    id_to_name: dict[int, str],
    k_values: list[int],
    max_patches: int,
    warnings_out: list[str],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Subsample if needed, then score kNN confusion/purity and per-class spectrum metrics.

    Shared tail for :func:`_compute_label_metrics` and
    :func:`_compute_label_metrics_placed` — both hand off an already
    patch-aligned label array; only how that array gets built differs.

    Returns
    -------
    tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]
        ``(classes, per_class, knn_confusion)``.
    """
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
    tile_placement: dict[str, TilePlacement] | None = None,
) -> dict[str, Any]:
    """Run dense patch token evaluation over a batch of tiles.

    Parameters
    ----------
    tiles : dict[str, Any]
        Map of tile ID to a ``[P, H, W]`` patch-first feature map. Values
        may be nested lists, numpy arrays (e.g. from :func:`load_tiles`), or
        any array-like ``np.asarray`` accepts — CPU torch tensors work
        without a torch dependency here. All tiles must share the same
        ``P``, ``H``, and ``W``.
    k_values : list[int] | None
        K cutoffs for nearest-neighbour label metrics. Defaults to ``[5, 10, 20]``.
    sample_pairs : int | None
        Max random pairs for global pairwise similarity stats. ``None`` computes exactly.
    max_patches : int
        Max patches used for O(N²) label-aware kNN computation. Larger corpora
        are randomly subsampled (seeded) and the drop is reported in ``warnings``.
    masks_dir : str | Path | None
        Directory of ground-truth colour-coded masks. When ``tile_placement``
        is ``None``, one mask per tile (filename stem must match the tile
        ID); when given, one mask per source image instead (filename stem
        must match each tile's ``image_stem``), cropped to each tile's exact
        pixel rectangle before downsampling. Must already be resolved to a
        real location — dataset_root resolution is an API-layer concern,
        matching ``run_seg_eval``. Requires ``classes_path``.
    classes_path : str | Path | None
        Path to an AgriBench ``class_map.json``, same format as ``seg``.
        Must already be resolved. Requires ``masks_dir``.
    tile_placement : dict[str, TilePlacement] | None
        Map of tile ID to :class:`TilePlacement`, as returned by
        :func:`load_tile_placement`. When given (and ``masks_dir``/
        ``classes_path`` are set), ground-truth masks are resolved per
        source image and cropped to each tile's exact pixel rectangle
        instead of matched one-to-one by tile ID. ``None`` for inline
        ``tiles`` dicts, which have no placement metadata and keep the
        one-mask-per-tile convention.

    Returns
    -------
    dict[str, Any]
        Fixed analysis JSON — see :class:`precisionai.agrieval.dpt.schemas.evaluate.DptEvalResponse`.

    Raises
    ------
    ValueError
        If tiles do not share a uniform ``(P, H, W)`` shape.
    FileNotFoundError
        If ``masks_dir`` does not exist, or a tile (or its source image,
        when ``tile_placement`` is given) has no matching ground-truth mask.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    tile_ids, tiles_arr = _stack_tiles(tiles)
    n, c, h, w = tiles_arr.shape
    patch_tokens = _flatten_patches(tiles_arr)

    warnings_list: list[str] = []

    # Spectrum metrics run on raw centered features (normalize=False);
    # cosine-based metrics are inherently computed on L2-normalized tokens.
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
        if tile_placement is not None:
            classes, per_class, knn_confusion = _compute_label_metrics_placed(
                tile_ids=tile_ids,
                tiles_arr=tiles_arr,
                patch_tokens=patch_tokens,
                tile_placement=tile_placement,
                masks_dir=masks_dir,
                classes_path=classes_path,
                k_values=k_values,
                max_patches=max_patches,
                warnings_out=warnings_list,
            )
        else:
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


def run_dpt_image_eval(
    images: dict[str, Any],
    *,
    k_values: list[int] | None = None,
    sample_pairs: int | None = 1_000_000,
    max_patches: int = 20_000,
    masks_dir: str | Path | None = None,
    classes_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run dense patch token evaluation over a batch of whole-image feature maps.

    Identical computation to :func:`run_dpt_eval` — every entry must share
    the same ``[P, H, W]`` shape, since every image, whole or tiled, is
    processed at the backbone's fixed native input resolution. The only
    difference is naming: this wiring is for callers whose entries each
    represent one whole (untiled) image rather than an arbitrary tile crop.

    Parameters
    ----------
    images : dict[str, Any]
        Map of image ID to a ``[P, H, W]`` patch-first feature map. Values
        may be nested lists, numpy arrays (e.g. from :func:`load_images`), or
        any array-like ``np.asarray`` accepts. All images must share the same
        ``P``, ``H``, and ``W``.
    k_values : list[int] | None
        K cutoffs for nearest-neighbour label metrics. Defaults to ``[5, 10, 20]``.
    sample_pairs : int | None
        Max random pairs for global pairwise similarity stats. ``None`` computes exactly.
    max_patches : int
        Max patches used for O(N²) label-aware kNN computation. Larger corpora
        are randomly subsampled (seeded) and the drop is reported in ``warnings``.
    masks_dir : str | Path | None
        Directory of ground-truth colour-coded masks, one per image (filename
        stem must match the image ID). Must already be resolved to a real
        location — dataset_root resolution is an API-layer concern. Requires
        ``classes_path``.
    classes_path : str | Path | None
        Path to an AgriBench ``class_map.json``, same format as ``seg``.
        Must already be resolved. Requires ``masks_dir``.

    Returns
    -------
    dict[str, Any]
        Fixed analysis JSON — see :class:`precisionai.agrieval.dpt.schemas.evaluate.DptImageEvalResponse`.

    Raises
    ------
    ValueError
        If images do not share a uniform ``(P, H, W)`` shape.
    FileNotFoundError
        If ``masks_dir`` does not exist, or an image has no matching ground-truth mask.
    """
    result = run_dpt_eval(
        tiles=images,
        k_values=k_values,
        sample_pairs=sample_pairs,
        max_patches=max_patches,
        masks_dir=masks_dir,
        classes_path=classes_path,
    )
    return {
        "n_images": result["n_tiles"],
        "embed_dim": result["embed_dim"],
        "grid_height": result["grid_height"],
        "grid_width": result["grid_width"],
        "n_patches": result["n_patches"],
        "image_ids": result["tile_ids"],
        "k_values": result["k_values"],
        "global_metrics": result["global_metrics"],
        "per_image": result["per_tile"],
        "classes": result["classes"],
        "per_class": result["per_class"],
        "knn_confusion": result["knn_confusion"],
        "warnings": result["warnings"],
    }
