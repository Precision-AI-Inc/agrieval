# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Label extraction, metadata wiring, and dataset loading for embedding evaluation."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from precisionai.agrieval.emb.metrics import ImageItem
from precisionai.agrieval.emb.schemas.evaluate import MetadataGroup

# Matches a class name whose leading letters define the coarse L1 label.
_L1_LABEL_RE = re.compile(r"^([A-Za-z]+)\d+$")

# ---------------------------------------------------------------------------
# Path parsing
# ---------------------------------------------------------------------------


def _parse_crop(folder_name: str) -> str:
    """Extract the class name from a folder name.

    The canonical dataset layout uses bare L2 folder names such as ``A1`` or
    ``BC14``. In that case the folder name itself is returned unchanged.

    Plain names with no trailing digits are returned as-is.

    Examples
    --------
    ``A1``   →  ``A1``
    ``G7``   →  ``G7``
    ``BC14`` →  ``BC14``
    ``corn`` →  ``corn``
    """
    if "_" in folder_name:
        return folder_name.split("_", maxsplit=1)[0]
    return folder_name


def _derive_l1_label(class_name: str) -> str:
    """Return the coarse L1 label derived from a class name."""
    m = _L1_LABEL_RE.match(class_name)
    return m.group(1) if m else class_name


def _extract_class_names(paths: list[str], dataset_root: str | None = None) -> list[str]:
    """Extract per-path class names from the first folder after ``dataset_root``."""
    if not paths:
        return []

    posix_paths = [Path(p.replace("\\", "/")).as_posix() for p in paths]

    if dataset_root is not None:
        root_prefix = Path(dataset_root.replace("\\", "/")).as_posix().rstrip("/") + "/"
        if not any(posix_path.startswith(root_prefix) for posix_path in posix_paths):
            raw = os.path.commonprefix(posix_paths)
            root_prefix = raw[: raw.rfind("/") + 1] if "/" in raw else ""
    else:
        raw = os.path.commonprefix(posix_paths)
        # Trim to the last directory separator so we don't clip mid-word
        root_prefix = raw[: raw.rfind("/") + 1] if "/" in raw else ""

    class_names: list[str] = []
    for posix_path in posix_paths:
        remainder = posix_path[len(root_prefix) :] if posix_path.startswith(root_prefix) else posix_path
        parts = Path(remainder).parts
        folder = parts[0] if parts else "unknown"
        class_names.append(_parse_crop(folder))

    return class_names


def extract_labels(paths: list[str], dataset_root: str | None = None) -> list[str]:
    """Infer coarse L1 class labels from image paths.

    The class name is extracted from the first path component after the
    dataset root using ``_parse_crop``. Bare L2 folders such as ``A1`` are
    treated as the canonical layout. The returned labels are the derived L1
    classes, so ``A1`` becomes ``A`` and ``BC14`` becomes ``BC``. Plain names
    without trailing digits are returned unchanged.

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
        Coarse L1 class label per path (same order as ``paths``).
    """
    return [_derive_l1_label(class_name) for class_name in _extract_class_names(paths, dataset_root)]


# ---------------------------------------------------------------------------
# Metadata wiring
# ---------------------------------------------------------------------------


def _labels_from_metadata(
    paths: list[str],
    metadata: dict[str, MetadataGroup],
    fallback_labels: list[str],
) -> list[str]:
    """Return per-path class labels sourced from metadata, falling back to path-extracted labels.

    Matching is performed by exact path key comparison.  The ``images`` entries
    in each group must exactly match the keys used in the embeddings dictionary.

    Parameters
    ----------
    paths : list[str]
        Image paths as they appear in the embeddings dictionary.
    metadata : dict[str, MetadataGroup]
        Metadata groups keyed by arbitrary group ID.
    fallback_labels : list[str]
        Labels to use when a path has no matching metadata group.

    Returns
    -------
    list[str]
        One label per path, in the same order as ``paths``.
    """
    path_to_class: dict[str, str] = {}
    for group in metadata.values():
        for img in group.images:
            path_to_class[img] = group.l1_cluster

    return [path_to_class.get(p, fallback) for p, fallback in zip(paths, fallback_labels, strict=True)]


def _normalise_attributes(raw: dict[str, Any]) -> dict[str, str]:
    """Convert a MetadataGroup attributes dict to a flat ``dict[str, str]``.

    List values are sorted and joined with ``","`` for stable canonical form.
    Null values are excluded so they do not participate in attribute matching.

    Parameters
    ----------
    raw : dict[str, Any]
        Attributes as stored in a :class:`MetadataGroup` (may contain lists or ``None``).

    Returns
    -------
    dict[str, str]
        Normalised attributes suitable for :class:`~precisionai.agrieval.emb.metrics.ImageItem`.
    """
    result: dict[str, str] = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, list):
            result[k] = ",".join(sorted(str(x) for x in v))
        else:
            result[k] = str(v)
    return result


def build_image_items(
    paths: list[str],
    metadata: dict[str, MetadataGroup],
) -> list[ImageItem]:
    """Build :class:`~precisionai.agrieval.emb.metrics.ImageItem` objects from embedding paths and metadata.

    Each item's ``image_id`` is the exact embedding path key.
    The ``explicit_positive_ids`` are the exact paths of all other
    images in the same metadata group.  Items with no matching group receive
    empty ``explicit_positive_ids``, ``None`` for ``class_name``, and an
    empty ``attributes`` dict.

    Parameters
    ----------
    paths : list[str]
        Image paths as they appear in the embeddings dictionary.
    metadata : dict[str, MetadataGroup]
        Metadata groups keyed by arbitrary group ID.

    Returns
    -------
    list[ImageItem]
        One :class:`~precisionai.agrieval.emb.metrics.ImageItem` per path, in the same order.
    """
    # exact path → (l1_cluster, normalised str attributes, frozenset of all member paths)
    path_to_group: dict[str, tuple[str, dict[str, str], frozenset[str]]] = {}
    for group in metadata.values():
        member_ids = frozenset(group.images)
        norm_attrs = _normalise_attributes(group.attributes)
        for img in group.images:
            path_to_group[img] = (group.l1_cluster, norm_attrs, member_ids)

    items: list[ImageItem] = []
    for path in paths:
        if path in path_to_group:
            l1_cluster, norm_attrs, member_ids = path_to_group[path]
            items.append(
                ImageItem(
                    image_id=path,
                    explicit_positive_ids=member_ids - {path},
                    class_name=l1_cluster,
                    attributes=norm_attrs,
                )
            )
        else:
            items.append(ImageItem(image_id=path))
    return items


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------


def load_image2image_metadata(json_path: str) -> dict[str, MetadataGroup]:
    """Load cluster metadata from an ``image2image.json`` file.

    Reads the top-level ``"metadata"`` dict from the JSON file and returns one
    :class:`MetadataGroup` per cluster, keyed by ``class_name``
    (e.g. ``"A1"``).

    The expected file structure is::

        {
          "metadata": {
            "A1": {
              "class_name": "A1",
              "images": ["images/A1/pai-abc.png", ...],
              "attributes": {"plants": ["Crop | Soybean"], ...}
            },
            ...
          }
        }

    Image paths are stored exactly as written in the JSON file.  Pass the same
    relative paths as embedding keys when calling the evaluation functions.

    Parameters
    ----------
    json_path : str
        Path to the ``image2image.json`` file (e.g. ``"dataset/image2image.json"``).

    Returns
    -------
    dict[str, MetadataGroup]
        Keyed by ``class_name`` (e.g. ``"A1"``).  Each value is a
        :class:`MetadataGroup` with ``class_name``, ``images``, and
        ``attributes`` populated from the JSON.

    Raises
    ------
    FileNotFoundError
        If ``json_path`` does not exist.
    ValueError
        If the file does not contain a top-level ``"metadata"`` key.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"image2image metadata file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if "metadata" not in data:
        raise ValueError(f"Expected a top-level 'metadata' key in {path}")
    return {cls: MetadataGroup(**group) for cls, group in data["metadata"].items()}


# ---------------------------------------------------------------------------
# Plant wiring adapters
# ---------------------------------------------------------------------------


def plant2image_to_metadata(
    instance_to_image: dict[str, list[str]],
    dataset_root: str | None = None,
) -> dict[str, MetadataGroup]:
    """Convert a Plant2Image wiring map to :class:`MetadataGroup` form.

    Each parent full-image and all its instance crops form one group whose
    members are mutual explicit positives.  The crop class is extracted from
    the parent image path.  The crop class is extracted from the parent image
    path and used as ``class_name``.

    Parameters
    ----------
    instance_to_image : dict[str, list[str]]
        Mapping from parent full-image path to a list of its instance crop
        paths.
    dataset_root : str | None
        Optional dataset root prefix for extracting the crop class label from
        parent image paths.  Defaults to the longest common directory prefix.

    Returns
    -------
    dict[str, MetadataGroup]
        One :class:`MetadataGroup` per parent image, keyed by the parent path.
        Each group's ``images`` list contains the parent path followed by all
        its instance paths.
    """
    parent_paths = list(instance_to_image.keys())
    if not parent_paths:
        return {}
    parent_labels = _extract_class_names(parent_paths, dataset_root)
    return {
        parent_path: MetadataGroup(
            images=[parent_path, *instance_paths],
            class_name=class_name,
        )
        for (parent_path, instance_paths), class_name in zip(instance_to_image.items(), parent_labels, strict=True)
    }


def plant2plant_to_metadata(
    instance_labels: dict[str, str],
) -> dict[str, MetadataGroup]:
    """Convert a Plant2Plant wiring map to :class:`MetadataGroup` form.

    All instances sharing the same class label are grouped into one
    :class:`MetadataGroup`, making them mutual explicit positives.

    Parameters
    ----------
    instance_labels : dict[str, str]
        Mapping from instance path to its crop/weed class label.

    Returns
    -------
    dict[str, MetadataGroup]
        :class:`MetadataGroup` objects ready for
        :func:`~precisionai.agrieval.emb.services.evaluate.run_image2image_eval`.
    """
    class_to_instances: dict[str, list[str]] = {}
    for inst, cls in instance_labels.items():
        class_to_instances.setdefault(cls, []).append(inst)
    return {cls: MetadataGroup(images=instances, class_name=cls) for cls, instances in class_to_instances.items()}
