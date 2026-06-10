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

"""Tests for label-aware metrics: intra_inter_similarity_gap,
knn_label_purity_at_k, knn_label_ndcg_at_k, knn_map_at_k."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from pai.ag_emb.metrics import (
    intra_inter_similarity_gap,
    knn_label_ndcg_at_k,
    knn_label_purity_at_k,
    knn_map_at_k,
    top_k_neighbors,
)
from pai.ag_emb.services.evaluate import _jsonify, _parse_crop, extract_labels


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).astype(np.float32)


# Two well-separated classes: 4 corn + 4 soy in 4-D space
_CORN = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
_SOY  = np.array([_unit([0, 1, 0, 0])] * 4, dtype=np.float32)
_MIXED = np.vstack([_CORN, _SOY])                     # [8, 4]
_MIXED_LABELS = np.array(["corn"] * 4 + ["soy"] * 4)

# Single-class dataset
_SINGLE = _CORN
_SINGLE_LABELS = np.array(["corn"] * 4)


# ---------------------------------------------------------------------------
# intra_inter_similarity_gap
# ---------------------------------------------------------------------------

class TestIntraInterSimilarityGap:
    def test_positive_gap_for_separated_classes(self) -> None:
        result = intra_inter_similarity_gap(_MIXED, _MIXED_LABELS)
        assert result["gap"] is not None
        assert result["gap"] > 0.0

    def test_single_class_has_no_inter_pairs(self) -> None:
        result = intra_inter_similarity_gap(_SINGLE, _SINGLE_LABELS)
        assert result["mean_inter_class_similarity"] is None
        assert result["gap"] is None
        assert result["num_inter_pairs"] == 0

    def test_expected_keys(self) -> None:
        result = intra_inter_similarity_gap(_MIXED, _MIXED_LABELS)
        for key in ("mean_intra_class_similarity", "mean_inter_class_similarity",
                    "gap", "num_intra_pairs", "num_inter_pairs"):
            assert key in result

    def test_exact_computation(self) -> None:
        result = intra_inter_similarity_gap(_MIXED, _MIXED_LABELS, sample_pairs=None)
        assert result["gap"] is not None
        assert result["gap"] > 0.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            intra_inter_similarity_gap(_MIXED, _MIXED_LABELS[:3])


# ---------------------------------------------------------------------------
# knn_label_purity_at_k
# ---------------------------------------------------------------------------

class TestKnnLabelPurityAtK:
    def test_perfect_purity_same_class(self) -> None:
        neighbors = top_k_neighbors(_SINGLE, ks=[3])
        result = knn_label_purity_at_k(neighbors, _SINGLE_LABELS)
        assert result[3]["mean"] == pytest.approx(1.0)

    def test_all_different_classes(self) -> None:
        # 1 item per class — each item's only neighbor is the other class
        emb = np.array([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])], dtype=np.float32)
        labels = np.array(["a", "b"])
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_label_purity_at_k(neighbors, labels)
        assert result[1]["mean"] == pytest.approx(0.0)

    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3])
        result = knn_label_purity_at_k(neighbors, _MIXED_LABELS)
        entry = result[3]
        assert "mean" in entry and "std" in entry and "per_item" in entry
        assert entry["per_item"].shape == (len(_MIXED),)


# ---------------------------------------------------------------------------
# knn_label_ndcg_at_k
# ---------------------------------------------------------------------------

class TestKnnLabelNdcgAtK:
    def test_perfect_ndcg_same_class(self) -> None:
        neighbors = top_k_neighbors(_SINGLE, ks=[3])
        result = knn_label_ndcg_at_k(neighbors, _SINGLE_LABELS)
        assert result[3]["mean"] == pytest.approx(1.0)

    def test_zero_ndcg_all_different_classes(self) -> None:
        # Each item's only neighbor is a different class
        emb = np.array([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])], dtype=np.float32)
        labels = np.array(["a", "b"])
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_label_ndcg_at_k(neighbors, labels)
        assert result[1]["mean"] == pytest.approx(0.0)

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3, 5])
        result = knn_label_ndcg_at_k(neighbors, _MIXED_LABELS)
        for k, stats in result.items():
            assert 0.0 <= stats["mean"] <= 1.0
            assert stats["per_item"].min() >= 0.0
            assert stats["per_item"].max() <= 1.0 + 1e-6

    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3])
        result = knn_label_ndcg_at_k(neighbors, _MIXED_LABELS)
        entry = result[3]
        assert "mean" in entry and "std" in entry and "per_item" in entry
        assert entry["per_item"].shape == (len(_MIXED),)

    def test_multiple_k_values(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3, 5])
        result = knn_label_ndcg_at_k(neighbors, _MIXED_LABELS)
        assert set(result.keys()) == {3, 5}

    def test_higher_than_purity_when_relevant_ranked_first(self) -> None:
        # With perfectly separated classes, nDCG should be perfect (1.0)
        neighbors = top_k_neighbors(_MIXED, ks=[3])
        ndcg = knn_label_ndcg_at_k(neighbors, _MIXED_LABELS)
        purity = knn_label_purity_at_k(neighbors, _MIXED_LABELS)
        # Both should be 1.0 for perfectly separated classes
        assert ndcg[3]["mean"] == pytest.approx(purity[3]["mean"], abs=1e-5)


# ---------------------------------------------------------------------------
# knn_map_at_k
# ---------------------------------------------------------------------------

class TestKnnMapAtK:
    def test_perfect_map_same_class(self) -> None:
        neighbors = top_k_neighbors(_SINGLE, ks=[3])
        result = knn_map_at_k(neighbors, _SINGLE_LABELS)
        assert result[3]["mean"] == pytest.approx(1.0)

    def test_zero_map_all_different_classes(self) -> None:
        emb = np.array([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])], dtype=np.float32)
        labels = np.array(["a", "b"])
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_map_at_k(neighbors, labels)
        assert result[1]["mean"] == pytest.approx(0.0)

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3, 5])
        result = knn_map_at_k(neighbors, _MIXED_LABELS)
        for k, stats in result.items():
            assert 0.0 <= stats["mean"] <= 1.0 + 1e-6
            assert stats["per_item"].min() >= 0.0
            assert stats["per_item"].max() <= 1.0 + 1e-6

    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3])
        result = knn_map_at_k(neighbors, _MIXED_LABELS)
        entry = result[3]
        assert "mean" in entry and "std" in entry and "per_item" in entry
        assert entry["per_item"].shape == (len(_MIXED),)

    def test_multiple_k_values(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3, 5])
        result = knn_map_at_k(neighbors, _MIXED_LABELS)
        assert set(result.keys()) == {3, 5}


# ---------------------------------------------------------------------------
# _jsonify
# ---------------------------------------------------------------------------

class TestJsonify:
    def test_numpy_array_dropped_from_dict(self) -> None:
        result = _jsonify({"a": np.array([1.0, 2.0]), "b": 3.0})
        assert "a" not in result
        assert result["b"] == 3.0

    def test_numpy_float32_converted(self) -> None:
        result = _jsonify(np.float32(1.5))
        assert result == pytest.approx(1.5)
        assert isinstance(result, float)

    def test_numpy_int64_converted(self) -> None:
        result = _jsonify(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_bool_converted(self) -> None:
        assert _jsonify(np.bool_(True)) is True
        assert _jsonify(np.bool_(False)) is False
        assert isinstance(_jsonify(np.bool_(True)), bool)

    def test_nan_becomes_none(self) -> None:
        assert _jsonify(float("nan")) is None
        assert _jsonify(np.float32("nan")) is None

    def test_inf_becomes_none(self) -> None:
        assert _jsonify(float("inf")) is None
        assert _jsonify(float("-inf")) is None

    def test_nested_structure(self) -> None:
        data = {"outer": {"inner": np.float32(2.0), "arr": np.array([1, 2])}}
        result = _jsonify(data)
        assert result["outer"]["inner"] == pytest.approx(2.0)
        assert "arr" not in result["outer"]

    def test_output_is_json_serialisable(self) -> None:
        data = {
            "a": np.float32(1.0),
            "b": np.int64(2),
            "c": np.bool_(True),
            "d": float("nan"),
            "e": np.array([1.0, 2.0]),
            "f": [np.float32(3.0), np.int32(4)],
        }
        json.dumps(_jsonify(data))  # must not raise


# ---------------------------------------------------------------------------
# _parse_crop
# ---------------------------------------------------------------------------

class TestParseCrop:
    def test_crop_camera_bracket_format(self) -> None:
        assert _parse_crop("corn_[HB-25000SBC]") == "corn"
        assert _parse_crop("soybean_[anafi]") == "soybean"
        assert _parse_crop("wheat_[nikon_d610]") == "wheat"

    def test_plain_folder_returned_as_is(self) -> None:
        assert _parse_crop("corn") == "corn"
        assert _parse_crop("soybean") == "soybean"

    def test_no_closing_bracket_returned_as_is(self) -> None:
        assert _parse_crop("corn_[unclosed") == "corn_[unclosed"

    def test_empty_string(self) -> None:
        assert _parse_crop("") == ""


# ---------------------------------------------------------------------------
# extract_labels
# ---------------------------------------------------------------------------

class TestExtractLabelsEdgeCases:
    def test_single_path(self) -> None:
        paths = ["dataset/corn_[HB-25000SBC]/img/img1.png"]
        assert extract_labels(paths, dataset_root="dataset") == ["corn"]

    def test_no_separator_falls_back_gracefully(self) -> None:
        # Paths that are just filenames — parts[0] is the filename itself
        paths = ["img1.png", "img2.png"]
        labels = extract_labels(paths)
        assert len(labels) == 2  # doesn't crash

    def test_trailing_slash_on_root(self) -> None:
        paths = ["dataset/corn_[HB-25000SBC]/img/img1.png"]
        # dataset_root with or without trailing slash should give same result
        assert (extract_labels(paths, dataset_root="dataset") ==
                extract_labels(paths, dataset_root="dataset/"))


# ---------------------------------------------------------------------------
# get_dataset_root
# ---------------------------------------------------------------------------

class TestGetDatasetRoot:
    def test_default_is_dataset(self) -> None:
        env_backup = os.environ.pop("PAI_DATASET_ROOT", None)
        try:
            from pai.ag_emb.api.config import get_dataset_root
            assert get_dataset_root() == "dataset"
        finally:
            if env_backup is not None:
                os.environ["PAI_DATASET_ROOT"] = env_backup

    def test_env_var_override(self) -> None:
        os.environ["PAI_DATASET_ROOT"] = "/mnt/storage/crops"
        try:
            from pai.ag_emb.api.config import get_dataset_root
            assert get_dataset_root() == "/mnt/storage/crops"
        finally:
            del os.environ["PAI_DATASET_ROOT"]
