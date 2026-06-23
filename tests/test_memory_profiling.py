# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Runtime memory profiling tests using tracemalloc.

These tests verify that the evaluation pipeline stays within reasonable
memory bounds and does not exhibit super-linear memory growth with dataset
size.  They are intentionally fast (small synthetic datasets) and serve as
regression guards rather than production benchmarks.
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np

from precisionai.agrieval.emb.schemas.evaluate import MetadataGroup
from precisionai.agrieval.emb.services.evaluate import run_image2image_eval

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings(n: int, d: int, n_classes: int = 3, seed: int = 42) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    class_names = ["corn", "soy", "wheat"]
    # Use orthogonal basis vectors as prototypes for clean class separation.
    prototypes = np.eye(d, dtype=np.float32)[:n_classes]
    result: dict[str, list[float]] = {}
    for i in range(n):
        cls_idx = i % n_classes
        vec = prototypes[cls_idx] + rng.standard_normal(d).astype(np.float32) * 0.1
        vec /= float(np.linalg.norm(vec))
        path = f"img/{class_names[cls_idx]}/{i:06d}.png"
        result[path] = vec.tolist()
    return result


def _peak_mb(embeddings: dict, k_values: list, sample_pairs: int) -> float:
    gc.collect()
    tracemalloc.start()
    run_image2image_eval(embeddings, k_values=k_values, dataset_root=None, sample_pairs=sample_pairs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPeakMemorySmallDataset:
    def test_100_items_64d_under_100mb(self) -> None:
        embeddings = _make_embeddings(n=100, d=64, n_classes=2)
        peak_mb = _peak_mb(embeddings, k_values=[5], sample_pairs=500)
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f} MB exceeds 100 MB for N=100, D=64"

    def test_result_is_correct(self) -> None:
        embeddings = _make_embeddings(n=60, d=32, n_classes=3)
        result = run_image2image_eval(embeddings, k_values=[3], dataset_root=None, sample_pairs=200)
        assert result["n_items"] == 60
        assert result["embedding_dim"] == 32
        assert len(result["classes"]) == 3

    def test_multiple_k_values_small_dataset(self) -> None:
        embeddings = _make_embeddings(n=50, d=16, n_classes=2)
        peak_mb = _peak_mb(embeddings, k_values=[3, 5, 10], sample_pairs=100)
        assert peak_mb < 100, f"Peak memory {peak_mb:.1f} MB for N=50, D=16"


class TestMemoryScaling:
    def test_10x_items_less_than_1000x_memory(self) -> None:
        small = _make_embeddings(n=50, d=32, n_classes=2, seed=1)
        large = _make_embeddings(n=500, d=32, n_classes=2, seed=1)

        peak_small = _peak_mb(small, k_values=[3], sample_pairs=200)
        peak_large = _peak_mb(large, k_values=[3], sample_pairs=2000)

        # 10x more items must not cause more than 1000x memory growth.
        ratio = peak_large / max(peak_small, 0.1)
        assert ratio < 1000, f"Memory scaled by {ratio:.0f}x for 10x items ({peak_small:.1f} MB -> {peak_large:.1f} MB)"

    def test_peak_per_item_under_1mb(self) -> None:
        n, d = 200, 128
        embeddings = _make_embeddings(n=n, d=d, n_classes=3)
        peak_mb = _peak_mb(embeddings, k_values=[5, 10], sample_pairs=1000)
        per_item_mb = peak_mb / n
        assert per_item_mb < 1.0, (
            f"Peak memory per item {per_item_mb:.3f} MB exceeds 1 MB (total {peak_mb:.1f} MB for N={n}, D={d})"
        )


class TestMemoryWithMetadata:
    def test_metadata_path_memory_reasonable(self) -> None:
        embeddings = _make_embeddings(n=60, d=32, n_classes=2)
        corn_paths = [p for p in embeddings if "/corn/" in p]
        soy_paths = [p for p in embeddings if "/soy/" in p]
        metadata = {
            "corn": MetadataGroup(
                images=corn_paths, l1_cluster="corn", l2_cluster="corn", attributes={"growth_stage": "medium"}
            ),
            "soy": MetadataGroup(
                images=soy_paths, l1_cluster="soy", l2_cluster="soy", attributes={"growth_stage": "early"}
            ),
        }

        gc.collect()
        tracemalloc.start()
        result = run_image2image_eval(
            embeddings,
            k_values=[3],
            dataset_root=None,
            sample_pairs=200,
            metadata=metadata,
        )
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        assert result["n_items"] == 60
        assert "group_analysis" in result
        assert peak_mb < 200, f"Peak memory {peak_mb:.1f} MB with metadata path"
