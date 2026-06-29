# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures.

``image_embeddings`` — session-scoped fixture that walks ``tests/emb/data/images/``,
infers crop class from each path, and generates synthetic 16-D embeddings
where images from the same crop cluster together.

``plant2image_payload`` — session-scoped fixture that loads
``tests/emb/data/plant2image.json`` and generates synthetic embeddings for both
parent full-field images and instance crops.

``plant2plant_payload`` — session-scoped fixture that loads
``tests/emb/data/plant2plant.json`` and generates synthetic embeddings for instance
crops, clustered by their species label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from precisionai.agrieval.emb.services.evaluate import _parse_crop

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TESTS_DATA = Path(__file__).parent / "data"
_PROJECT_ROOT = Path(__file__).parent.parent
EMB_DIM = 16
_IMAGE_EXTS = {".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_paths() -> list[Path]:
    """Return image paths under tests/emb/data/images/ sorted for reproducibility."""
    return sorted(p for p in (TESTS_DATA / "images").rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def _crop_from_path(path: Path) -> str:
    """Extract crop label from a tests/emb/data/ path using the service convention."""
    # Layout: tests/emb/data/images/{L2}/{filename}
    # parts[0] = "images", parts[1] = L2 folder (e.g. "A1"), etc.
    rel = path.relative_to(TESTS_DATA)
    idx = 1 if len(rel.parts) > 1 and rel.parts[0] == "images" else 0
    return _parse_crop(rel.parts[idx])


def _class_prototypes(classes: list[str]) -> dict[str, np.ndarray]:
    """Build one unit-norm 16-D prototype per crop class.

    Uses a fixed seed so class vectors are stable across test runs.
    Seeds each class independently by hashing its name so adding a new class
    doesn't shift existing prototypes.
    """
    protos: dict[str, np.ndarray] = {}
    for cls in classes:
        seed = int.from_bytes(cls.encode(), "little") & 0xFFFF_FFFF
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(EMB_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        protos[cls] = v
    return protos


def _make_embedding(path: Path, prototype: np.ndarray) -> list[float]:
    """Perturb the class prototype with tiny per-image noise (seed = filename hash)."""
    seed = hash(path.name) & 0xFFFF_FFFF
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(EMB_DIM).astype(np.float32) * 0.05
    v = prototype + noise
    v /= np.linalg.norm(v)
    return v.tolist()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def image_embeddings() -> dict[str, list[float]]:
    """Map of ``tests/emb/data``-relative image path → 16-D embedding.

    Same-crop images cluster tightly; different crops are well-separated.
    Suitable for end-to-end smoke tests of the analysis endpoint.
    """
    paths = _collect_paths()
    if not paths:
        pytest.skip("tests/emb/data/images/ contains no images")

    crops = sorted({_crop_from_path(p) for p in paths})
    protos = _class_prototypes(crops)

    return {str(p.relative_to(_PROJECT_ROOT)): _make_embedding(p, protos[_crop_from_path(p)]) for p in paths}


@pytest.fixture(scope="session")
def plant2image_payload() -> dict[str, object]:
    """Mixed embeddings payload for the plant2image smoke tests.

    Loads the instance-to-image mapping from ``tests/data/plant2image.json``
    and generates synthetic 16-D embeddings for every parent full-field image
    and every instance crop, clustered by L2 crop class.

    Returns a dict with keys ``embeddings`` and ``instance_to_image``, both
    using project-root-relative path strings as keys.
    """
    p2i_path = TESTS_DATA / "plant2image.json"
    if not p2i_path.exists():
        pytest.skip("tests/emb/data/plant2image.json not found")

    with p2i_path.open() as f:
        raw: dict[str, list[str]] = json.load(f)["instance_to_image"]

    # Translate JSON-relative paths (e.g. "images/A1/…") to project-root paths.
    instance_to_image: dict[str, list[str]] = {}
    path_to_cls: dict[str, str] = {}

    for parent_json, instances_json in raw.items():
        parent_full = TESTS_DATA / parent_json
        parent_key = str(parent_full.relative_to(_PROJECT_ROOT))
        inst_keys = [str((TESTS_DATA / inst).relative_to(_PROJECT_ROOT)) for inst in instances_json]
        instance_to_image[parent_key] = inst_keys

        cls = _crop_from_path(parent_full)  # parent is always under images/ — safe
        path_to_cls[parent_key] = cls
        for k in inst_keys:
            path_to_cls[k] = cls  # instances inherit parent class

    protos = _class_prototypes(sorted(set(path_to_cls.values())))
    embeddings = {k: _make_embedding(Path(k), protos[path_to_cls[k]]) for k in path_to_cls}

    return {"embeddings": embeddings, "instance_to_image": instance_to_image}


@pytest.fixture(scope="session")
def plant2plant_payload() -> dict[str, object]:
    """Instance embeddings payload for the plant2plant smoke tests.

    Loads instance labels from ``tests/data/plant2plant.json`` and generates
    synthetic 16-D embeddings for every instance crop, clustered by species label
    (e.g. ``"Crop | Soybean"``, ``"Weed | Weed"``).

    Returns a dict with keys ``embeddings`` and ``instance_labels``, both
    using project-root-relative path strings as keys.
    """
    p2p_path = TESTS_DATA / "plant2plant.json"
    if not p2p_path.exists():
        pytest.skip("tests/emb/data/plant2plant.json not found")

    with p2p_path.open() as f:
        raw: dict[str, str] = json.load(f)["instance_labels"]

    # Translate JSON-relative paths to project-root paths.
    instance_labels: dict[str, str] = {
        str((TESTS_DATA / inst_json).relative_to(_PROJECT_ROOT)): label for inst_json, label in raw.items()
    }

    unique_labels = sorted(set(instance_labels.values()))
    protos = _class_prototypes(unique_labels)
    embeddings = {k: _make_embedding(Path(k), protos[label]) for k, label in instance_labels.items()}

    return {"embeddings": embeddings, "instance_labels": instance_labels}
