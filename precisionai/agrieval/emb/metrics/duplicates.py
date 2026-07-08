# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Duplicate and near-duplicate detection at configurable similarity thresholds."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from precisionai.agrieval.emb.metrics._utils import _auto_batch_size, _prepare_embeddings


class _UnionFind:
    """Union-Find with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        """Initialize a disjoint-set forest for ``n`` elements.

        Parameters
        ----------
        n : int
            Number of elements.  Each element starts as its own root.
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Return the root of the component containing ``x``.

        Uses path-halving compression to keep the tree shallow.

        Parameters
        ----------
        x : int
            Element index.

        Returns
        -------
        int
            Root index of the component.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge the components containing ``x`` and ``y``.

        Uses union-by-rank to keep trees balanced.

        Parameters
        ----------
        x : int
            First element index.
        y : int
            Second element index.
        """
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def duplicate_pairs_at_threshold(
    embeddings: object,
    thresholds: Sequence[float] = (0.95, 0.98, 0.99),
    *,
    normalize: bool = True,
    max_pairs_returned: int = 10_000,
) -> dict:
    """Return pairs of vectors with cosine similarity at or above each threshold.

    Parameters
    ----------
    embeddings : array-like
        Shape ``[N, D]``.
    thresholds : sequence of float
        Similarity cutoffs. Only unique unordered pairs ``(i < j)`` are counted.
    normalize : bool
        L2-normalize rows before computing similarity.
    max_pairs_returned : int
        Cap on returned pairs per threshold. Total ``pair_count`` is always exact.

    Returns
    -------
    dict
        Keyed by threshold. Each value: ``pair_count``, ``pairs`` (list of dicts
        with ``i``, ``j``, ``similarity``).
    """
    emb = _prepare_embeddings(embeddings, normalize)
    n = emb.shape[0]
    sorted_thresholds = sorted(thresholds)
    min_threshold = sorted_thresholds[0]

    total_by_t: dict[float, int] = {t: 0 for t in sorted_thresholds}
    pairs_by_t: dict[float, list] = {t: [] for t in sorted_thresholds}

    batch_size = _auto_batch_size(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = (emb[start:end] @ emb.T).astype(np.float32)  # [B, N]

        for local_i in range(end - start):
            global_i = start + local_i
            row = sims[local_i, global_i + 1 :]  # upper triangle only
            if len(row) == 0:
                continue

            above_min = row >= min_threshold
            if not np.any(above_min):
                continue

            j_base = global_i + 1
            js = np.where(above_min)[0] + j_base
            ss = row[above_min]

            for t in sorted_thresholds:
                mask = ss >= t
                count = int(np.sum(mask))
                total_by_t[t] += count
                remaining = max_pairs_returned - len(pairs_by_t[t])
                if remaining > 0 and count > 0:
                    for j, s in zip(js[mask][:remaining].tolist(), ss[mask][:remaining].tolist(), strict=False):
                        pairs_by_t[t].append({"i": global_i, "j": int(j), "similarity": float(s)})

    return {
        float(t): {
            "pair_count": total_by_t[t],
            "pairs": pairs_by_t[t],
        }
        for t in sorted_thresholds
    }


def duplicate_groups_at_threshold(
    duplicate_pairs_by_threshold: dict,
    *,
    n_items: int,
) -> dict:
    """Group near-duplicate pairs into connected components.

    Parameters
    ----------
    duplicate_pairs_by_threshold : dict
        Output of :func:`duplicate_pairs_at_threshold`.
    n_items : int
        Total number of items.

    Returns
    -------
    dict
        Keyed by threshold. Each value: ``num_groups``, ``largest_group_size``,
        ``mean_group_size``, ``groups`` (list of lists of indices).
    """
    result = {}
    for threshold, data in duplicate_pairs_by_threshold.items():
        uf = _UnionFind(n_items)
        for pair in data["pairs"]:
            uf.union(pair["i"], pair["j"])

        components: dict = defaultdict(list)
        for idx in range(n_items):
            components[uf.find(idx)].append(idx)

        groups = [g for g in components.values() if len(g) >= 2]
        if groups:
            sizes = [len(g) for g in groups]
            result[threshold] = {
                "num_groups": len(groups),
                "largest_group_size": max(sizes),
                "mean_group_size": float(sum(sizes) / len(sizes)),
                "groups": groups,
            }
        else:
            result[threshold] = {
                "num_groups": 0,
                "largest_group_size": 0,
                "mean_group_size": 0.0,
                "groups": [],
            }
    return result
