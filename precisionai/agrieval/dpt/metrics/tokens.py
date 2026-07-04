# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Dense patch token diagnostics not covered by the generic embedding-space metrics."""

from __future__ import annotations

import numpy as np


def patch_norm_stats(patch_tokens: object) -> dict:
    """Summarise the L2 norm distribution of a set of patch tokens.

    Raw dense patch features are not pre-normalised the way whole-image
    embeddings are, so their norm distribution is itself a useful health
    check (collapse, dead patches, exploding activations).

    Parameters
    ----------
    patch_tokens : array-like
        Shape ``[N, C]``.

    Returns
    -------
    dict
        Keys: ``mean``, ``std``, ``p05``, ``p50``, ``p95``.
    """
    arr = np.asarray(patch_tokens, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    p05, p50, p95 = np.percentile(norms, [5, 50, 95]).tolist()
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "p05": p05,
        "p50": p50,
        "p95": p95,
    }


def patch_smoothness(tile: object) -> float:
    """Adjacent-patch cosine smoothness of a tile's feature map.

    Computes the mean cosine similarity of horizontally adjacent (right)
    patch pairs and of vertically adjacent (down) patch pairs, then averages
    the two directional means — matching the smoothness definition of the
    upstream dense-feature benchmark (``pai-vision-feature-map-eval``), where
    the two directions are weighted equally regardless of grid aspect ratio.

    High spatial smoothness is expected for natural imagery — a tile whose
    neighboring patches are nearly uncorrelated suggests noisy or
    misregistered features.

    Parameters
    ----------
    tile : array-like
        Single tile, shape ``[C, H, W]`` (channels-first).

    Returns
    -------
    float
        Average of the horizontal-pair mean and vertical-pair mean cosine
        similarity. When only one direction has pairs (``H == 1`` or
        ``W == 1``) that direction's mean is returned; ``0.0`` when the grid
        has no neighbor pairs at all (``1 x 1``).
    """
    arr = np.asarray(tile, dtype=np.float64)
    _, h, w = arr.shape
    norms = np.linalg.norm(arr, axis=0)
    safe_norms = np.where(norms > 0, norms, 1.0)
    unit = arr / safe_norms

    direction_means: list[float] = []
    if w > 1:
        right = np.einsum("chw,chw->hw", unit[:, :, :-1], unit[:, :, 1:])
        direction_means.append(float(right.mean()))
    if h > 1:
        down = np.einsum("chw,chw->hw", unit[:, :-1, :], unit[:, 1:, :])
        direction_means.append(float(down.mean()))

    if not direction_means:
        return 0.0
    return float(np.mean(direction_means))


def outlier_fraction(norms: object, *, threshold: float) -> float:
    """Fraction of values strictly above a given threshold.

    Used to flag tiles with a disproportionate share of high-norm "artifact"
    patches relative to the full patch-token corpus. The threshold is
    computed once across *all* tiles — mean + 3 standard deviations of every
    patch norm in the request, matching the artifact-patch definition of the
    upstream dense-feature benchmark — and passed in here per tile, so
    fractions are comparable across tiles instead of being tautological.

    Parameters
    ----------
    norms : array-like
        Shape ``[N]``. Per-patch L2 norms for a single tile.
    threshold : float
        Norm value strictly above which a patch counts as an outlier.

    Returns
    -------
    float
        Fraction of ``norms`` strictly above ``threshold``. ``0.0`` for an
        empty input.
    """
    arr = np.asarray(norms, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr > threshold))
