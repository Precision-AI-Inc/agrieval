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

"""Shared pytest fixtures.

``image_embeddings`` — session-scoped fixture that walks ``tests/data/``,
infers crop class from each path, and generates synthetic 16-D embeddings
where images from the same crop cluster together.  This lets the full
analysis endpoint run as an integration smoke test without a real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pai.ag_emb.services.evaluate import _parse_crop

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
    """Return all image paths under tests/data/ sorted for reproducibility."""
    return sorted(p for p in TESTS_DATA.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def _crop_from_path(path: Path) -> str:
    """Extract crop label from a tests/data/ path using the service convention."""
    # Layout: tests/data/images/class_subgroup/filename
    # parts[0] = "images", parts[1] = "corn_HB-25000SBC", etc.
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
    """Map of ``tests/data``-relative image path → 16-D embedding.

    Same-crop images cluster tightly; different crops are well-separated.
    Suitable for end-to-end smoke tests of the analysis endpoint.
    """
    paths = _collect_paths()
    if not paths:
        pytest.skip("tests/data/ contains no images — run scripts/build_test_data.py first")

    crops = sorted({_crop_from_path(p) for p in paths})
    protos = _class_prototypes(crops)

    return {str(p.relative_to(_PROJECT_ROOT)): _make_embedding(p, protos[_crop_from_path(p)]) for p in paths}
