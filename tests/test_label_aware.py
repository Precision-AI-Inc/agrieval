# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for label-aware metrics: intra_inter_similarity_gap,
knn_label_purity_at_k, knn_label_ndcg_at_k, knn_map_at_k,
knn_label_mrr_at_k, knn_label_r_precision,
ImageItem, relevance_grade, knn_metadata_ndcg_at_k, knn_per_attribute_ndcg_at_k,
knn_metadata_precision_at_k, knn_metadata_map_at_k,
knn_metadata_mrr_at_k, knn_metadata_r_precision."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from precisionai.agrieval.emb.api.config import get_dataset_root
from precisionai.agrieval.emb.metrics import (
    ImageItem,
    intra_inter_similarity_gap,
    knn_label_mrr_at_k,
    knn_label_ndcg_at_k,
    knn_label_purity_at_k,
    knn_label_r_precision,
    knn_map_at_k,
    knn_metadata_map_at_k,
    knn_metadata_mrr_at_k,
    knn_metadata_ndcg_at_k,
    knn_metadata_precision_at_k,
    knn_metadata_r_precision,
    knn_per_attribute_ndcg_at_k,
    relevance_grade,
    top_k_neighbors,
)
from precisionai.agrieval.emb.metrics.label_aware import (
    _build_class_instance_sets,
    _build_explicit_positive_mask,
    _build_grade_matrix,
    _encode_class_names,
)
from precisionai.agrieval.emb.services.evaluate import _jsonify, _parse_crop, extract_labels

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).astype(np.float32)


# Two well-separated classes: 4 corn + 4 soy in 4-D space
_CORN = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
_SOY = np.array([_unit([0, 1, 0, 0])] * 4, dtype=np.float32)
_MIXED = np.vstack([_CORN, _SOY])  # [8, 4]
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
        for key in (
            "mean_intra_class_similarity",
            "mean_inter_class_similarity",
            "gap",
            "num_intra_pairs",
            "num_inter_pairs",
        ):
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
        assert "mean" in entry
        assert "std" in entry
        assert "per_item" in entry
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
        for stats in result.values():
            assert 0.0 <= stats["mean"] <= 1.0
            assert stats["per_item"].min() >= 0.0
            assert stats["per_item"].max() <= 1.0 + 1e-6

    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3])
        result = knn_label_ndcg_at_k(neighbors, _MIXED_LABELS)
        entry = result[3]
        assert "mean" in entry
        assert "std" in entry
        assert "per_item" in entry
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
        for stats in result.values():
            assert 0.0 <= stats["mean"] <= 1.0 + 1e-6
            assert stats["per_item"].min() >= 0.0
            assert stats["per_item"].max() <= 1.0 + 1e-6

    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3])
        result = knn_map_at_k(neighbors, _MIXED_LABELS)
        entry = result[3]
        assert "mean" in entry
        assert "std" in entry
        assert "per_item" in entry
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
    def test_l2_folder_format(self) -> None:
        assert _parse_crop("A1") == "A1"
        assert _parse_crop("B1") == "B1"
        assert _parse_crop("BC14") == "BC14"
        assert _parse_crop("AB12") == "AB12"

    def test_plain_folder_returned_as_is(self) -> None:
        assert _parse_crop("corn") == "corn"
        assert _parse_crop("soybean") == "soybean"

    def test_empty_string(self) -> None:
        assert _parse_crop("") == ""


# ---------------------------------------------------------------------------
# extract_labels
# ---------------------------------------------------------------------------


class TestExtractLabelsEdgeCases:
    def test_single_path(self) -> None:
        paths = ["images/A1/img1.png"]
        assert extract_labels(paths, dataset_root="images") == ["A"]

    def test_no_separator_falls_back_gracefully(self) -> None:
        # Paths that are just filenames — parts[0] is the filename itself
        paths = ["img1.png", "img2.png"]
        labels = extract_labels(paths)
        assert len(labels) == 2  # doesn't crash

    def test_trailing_slash_on_root(self) -> None:
        paths = ["images/A1/img1.png"]
        # dataset_root with or without trailing slash should give same result
        assert extract_labels(paths, dataset_root="images") == extract_labels(paths, dataset_root="images/")


# ---------------------------------------------------------------------------
# get_dataset_root
# ---------------------------------------------------------------------------


class TestGetDatasetRoot:
    def test_default_is_dataset(self) -> None:
        env_backup = os.environ.pop("PAI_DATASET_ROOT", None)
        try:
            assert get_dataset_root() == "dataset"
        finally:
            if env_backup is not None:
                os.environ["PAI_DATASET_ROOT"] = env_backup

    def test_env_var_override(self) -> None:
        os.environ["PAI_DATASET_ROOT"] = "/mnt/storage/crops"
        try:
            assert get_dataset_root() == "/mnt/storage/crops"
        finally:
            del os.environ["PAI_DATASET_ROOT"]


# ---------------------------------------------------------------------------
# ImageItem
# ---------------------------------------------------------------------------


class TestImageItem:
    def test_default_fields(self) -> None:
        item = ImageItem(image_id="img.png")
        assert item.image_id == "img.png"
        assert item.explicit_positive_ids == frozenset()
        assert item.class_name is None
        assert item.attributes == {}

    def test_explicit_positives(self) -> None:
        item = ImageItem(
            image_id="a.png",
            explicit_positive_ids=frozenset({"b.png", "c.png"}),
            class_name="corn",
            attributes={"growth_stage": "medium"},
        )
        assert "b.png" in item.explicit_positive_ids
        assert "c.png" in item.explicit_positive_ids
        assert "a.png" not in item.explicit_positive_ids

    def test_mutable_attributes(self) -> None:
        item = ImageItem(image_id="img.png", attributes={"growth_stage": "medium"})
        assert item.attributes["growth_stage"] == "medium"


# ---------------------------------------------------------------------------
# relevance_grade
# ---------------------------------------------------------------------------


class TestRelevanceGrade:
    def _item(
        self,
        image_id: str,
        positives: frozenset[str] | None = None,
        class_name: str | None = None,
        attrs: dict[str, str] | None = None,
    ) -> ImageItem:
        return ImageItem(
            image_id=image_id,
            explicit_positive_ids=positives or frozenset(),
            class_name=class_name,
            attributes=attrs or {},
        )

    def test_self_is_zero(self) -> None:
        item = self._item("a.png", class_name="corn", attrs={"growth_stage": "medium"})
        assert relevance_grade(item, item) == 0

    def test_explicit_positive_is_three(self) -> None:
        query = self._item("a.png", positives=frozenset({"b.png"}), class_name="corn", attrs={"growth_stage": "medium"})
        candidate = self._item("b.png", class_name="corn", attrs={"growth_stage": "medium"})
        assert relevance_grade(query, candidate) == 3

    def test_same_class_is_two(self) -> None:
        query = self._item("a.png", class_name="corn")
        candidate = self._item("b.png", class_name="corn")
        assert relevance_grade(query, candidate) == 2

    def test_same_class_with_attributes_is_still_two(self) -> None:
        query = self._item("a.png", class_name="corn", attrs={"growth_stage": "medium", "camera": "anafi"})
        candidate = self._item("b.png", class_name="corn", attrs={"growth_stage": "early", "camera": "nikon"})
        assert relevance_grade(query, candidate) == 2

    def test_different_class_shared_class_instances_is_one(self) -> None:
        ci = "Crop | Corn,Weed | Waterhemp"
        query = self._item("a.png", class_name="A", attrs={"class_instances": ci})
        candidate = self._item(
            "b.png", class_name="B", attrs={"class_instances": "Weed | Waterhemp,Weed | Lambsquarters"}
        )
        assert relevance_grade(query, candidate) == 1

    def test_different_class_no_shared_class_instances_is_zero(self) -> None:
        query = self._item("a.png", class_name="A", attrs={"class_instances": "Crop | Corn"})
        candidate = self._item("b.png", class_name="B", attrs={"class_instances": "Weed | Waterhemp"})
        assert relevance_grade(query, candidate) == 0

    def test_different_class_missing_class_instances_is_zero(self) -> None:
        query = self._item("a.png", class_name="corn", attrs={"growth_stage": "medium"})
        candidate = self._item("b.png", class_name="soybean", attrs={"growth_stage": "medium"})
        assert relevance_grade(query, candidate) == 0

    def test_no_class_info_is_zero(self) -> None:
        query = self._item("a.png")
        candidate = self._item("b.png")
        assert relevance_grade(query, candidate) == 0

    def test_explicit_positive_takes_priority(self) -> None:
        query = self._item("a.png", positives=frozenset({"b.png"}), class_name="corn", attrs={"growth_stage": "medium"})
        candidate = self._item("b.png", class_name="corn", attrs={"growth_stage": "medium"})
        assert relevance_grade(query, candidate) == 3


# ---------------------------------------------------------------------------
# _build_grade_matrix
# ---------------------------------------------------------------------------


class TestBuildGradeMatrix:
    def test_encodes_all_grade_levels_and_precedence(self) -> None:
        items = [
            ImageItem(
                "a0.png",
                explicit_positive_ids=frozenset({"a1.png"}),
                class_name="A",
                attributes={"class_instances": "Crop | Corn"},
            ),
            ImageItem(
                "a1.png",
                explicit_positive_ids=frozenset({"a0.png"}),
                class_name="A",
                attributes={"class_instances": "Crop | Corn"},
            ),
            ImageItem(
                "a2.png",
                class_name="A",
                attributes={"class_instances": "Crop | Corn"},
            ),
            ImageItem(
                "b0.png",
                class_name="B",
                attributes={"class_instances": "Crop | Corn,Weed | Waterhemp"},
            ),
            ImageItem(
                "c0.png",
                class_name="C",
                attributes={"class_instances": "Weed | Lambsquarters"},
            ),
        ]

        grades = _build_grade_matrix(items)

        assert grades.shape == (5, 5)
        assert np.all(np.diag(grades) == 0)
        assert grades[0, 1] == 3
        assert grades[0, 2] == 2
        assert grades[0, 3] == 1
        assert grades[0, 4] == 0

    def test_same_l1_overrides_class_instance_overlap(self) -> None:
        items = [
            ImageItem("a0.png", class_name="A", attributes={"class_instances": "Crop | Corn"}),
            ImageItem("a1.png", class_name="A", attributes={"class_instances": "Crop | Corn,Weed | Waterhemp"}),
        ]

        grades = _build_grade_matrix(items)

        assert grades[0, 1] == 2
        assert grades[1, 0] == 2


class TestLabelAwareHelpers:
    def test_explicit_positive_mask_ignores_unknown_ids(self) -> None:
        items = [
            ImageItem("a.png", explicit_positive_ids=frozenset({"b.png", "missing.png"})),
            ImageItem("b.png"),
        ]

        mask = _build_explicit_positive_mask(items)

        expected = np.array([[False, True], [False, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_encode_class_names_and_class_instance_sets_handle_missing_values(self) -> None:
        items = [
            ImageItem("a.png", class_name="A", attributes={"class_instances": "Crop | Corn,Weed | Waterhemp"}),
            ImageItem("b.png", class_name="A", attributes={}),
            ImageItem("c.png", class_name=None, attributes={"class_instances": ""}),
        ]

        class_ints, has_class = _encode_class_names(items)
        ci_sets = _build_class_instance_sets(items)

        assert class_ints[0] == class_ints[1]
        assert has_class.tolist() == [True, True, False]
        assert ci_sets[0] == {"Crop | Corn", "Weed | Waterhemp"}
        assert ci_sets[1] == set()
        assert ci_sets[2] == set()


# ---------------------------------------------------------------------------
# knn_metadata_ndcg_at_k
# ---------------------------------------------------------------------------


# Four items: two corn (tight cluster), two soybean (tight cluster)
_META_CORN = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
_META_SOY = np.array([_unit([0, 1, 0, 0])] * 4, dtype=np.float32)
_META_EMB = np.vstack([_META_CORN, _META_SOY])  # [8, 4]

# Metadata items: corn images are explicit positives for each other
_CORN_IDS = frozenset({"c0.png", "c1.png", "c2.png", "c3.png"})
_SOY_IDS = frozenset({"s0.png", "s1.png", "s2.png", "s3.png"})
_META_ITEMS = [
    ImageItem("c0.png", _CORN_IDS - {"c0.png"}, class_name="corn", attributes={"growth_stage": "medium"}),
    ImageItem("c1.png", _CORN_IDS - {"c1.png"}, class_name="corn", attributes={"growth_stage": "medium"}),
    ImageItem("c2.png", _CORN_IDS - {"c2.png"}, class_name="corn", attributes={"growth_stage": "medium"}),
    ImageItem("c3.png", _CORN_IDS - {"c3.png"}, class_name="corn", attributes={"growth_stage": "medium"}),
    ImageItem("s0.png", _SOY_IDS - {"s0.png"}, class_name="soybean", attributes={"growth_stage": "medium"}),
    ImageItem("s1.png", _SOY_IDS - {"s1.png"}, class_name="soybean", attributes={"growth_stage": "medium"}),
    ImageItem("s2.png", _SOY_IDS - {"s2.png"}, class_name="soybean", attributes={"growth_stage": "medium"}),
    ImageItem("s3.png", _SOY_IDS - {"s3.png"}, class_name="soybean", attributes={"growth_stage": "medium"}),
]


class TestKnnMetadataNdcgAtK:
    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_ndcg_at_k(neighbors, _META_ITEMS)
        entry = result[3]
        assert "mean" in entry
        assert "std" in entry
        assert "per_item" in entry
        assert entry["per_item"].shape == (len(_META_ITEMS),)

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_ndcg_at_k(neighbors, _META_ITEMS)
        for stats in result.values():
            assert 0.0 <= stats["mean"] <= 1.0 + 1e-6
            assert stats["per_item"].min() >= 0.0
            assert stats["per_item"].max() <= 1.0 + 1e-6

    def test_multiple_k_values(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3, 5])
        result = knn_metadata_ndcg_at_k(neighbors, _META_ITEMS)
        assert set(result.keys()) == {3, 5}

    def test_perfect_score_for_separated_explicit_positives(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_ndcg_at_k(neighbors, _META_ITEMS)
        assert result[3]["mean"] == pytest.approx(1.0, abs=1e-5)

    def test_items_without_positives_score_zero(self) -> None:
        # Items with no explicit positives and no crop info → all relevance=0
        bare_items = [ImageItem(f"img{i}.png") for i in range(4)]
        emb = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_metadata_ndcg_at_k(neighbors, bare_items)
        assert result[1]["mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# knn_per_attribute_ndcg_at_k
# ---------------------------------------------------------------------------


class TestKnnPerAttributeNdcgAtK:
    def test_returns_dict_keyed_by_attribute(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_per_attribute_ndcg_at_k(neighbors, _META_ITEMS)
        assert "growth_stage" in result

    def test_empty_when_no_attributes(self) -> None:
        bare_items = [ImageItem(f"img{i}.png") for i in range(4)]
        emb = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
        neighbors = top_k_neighbors(emb, ks=[3])
        result = knn_per_attribute_ndcg_at_k(neighbors, bare_items)
        assert result == {}

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_per_attribute_ndcg_at_k(neighbors, _META_ITEMS)
        for k_stats in result.values():
            for stats in k_stats.values():
                assert 0.0 <= stats["mean"] <= 1.0 + 1e-6
                assert stats["per_item"].min() >= 0.0
                assert stats["per_item"].max() <= 1.0 + 1e-6

    def test_multiple_attributes_each_has_own_entry(self) -> None:
        items = [
            ImageItem(f"img{i}.png", class_name="corn", attributes={"growth_stage": "medium", "camera": "anafi"})
            for i in range(4)
        ]
        emb = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
        neighbors = top_k_neighbors(emb, ks=[3])
        result = knn_per_attribute_ndcg_at_k(neighbors, items)
        assert "growth_stage" in result
        assert "camera" in result

    def test_perfect_score_when_all_same_attribute(self) -> None:
        # All items same class and attribute — each item's neighbors are all same label
        items = [ImageItem(f"img{i}.png", class_name="corn", attributes={"growth_stage": "medium"}) for i in range(4)]
        emb = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
        neighbors = top_k_neighbors(emb, ks=[3])
        result = knn_per_attribute_ndcg_at_k(neighbors, items)
        assert result["growth_stage"][3]["mean"] == pytest.approx(1.0)

    def test_zero_score_when_all_different_attribute(self) -> None:
        # Each item has unique attribute value — no neighbor matches
        items = [ImageItem(f"img{i}.png", class_name="corn", attributes={"growth_stage": str(i)}) for i in range(4)]
        emb = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_per_attribute_ndcg_at_k(neighbors, items)
        assert result["growth_stage"][1]["mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# knn_metadata_precision_at_k
# ---------------------------------------------------------------------------


class TestKnnMetadataPrecisionAtK:
    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_precision_at_k(neighbors, _META_ITEMS)
        entry = result[3]
        assert "mean" in entry
        assert "std" in entry
        assert entry["per_item"].shape == (len(_META_ITEMS),)

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_precision_at_k(neighbors, _META_ITEMS)
        for stats in result.values():
            assert 0.0 <= stats["mean"] <= 1.0 + 1e-6
            assert stats["per_item"].min() >= 0.0
            assert stats["per_item"].max() <= 1.0 + 1e-6

    def test_multiple_k_values(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[1, 3])
        result = knn_metadata_precision_at_k(neighbors, _META_ITEMS)
        assert set(result.keys()) == {1, 3}

    def test_perfect_precision_for_separated_explicit_positives(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_precision_at_k(neighbors, _META_ITEMS)
        assert result[3]["mean"] == pytest.approx(1.0, abs=1e-5)

    def test_zero_for_items_without_positives(self) -> None:
        bare_items = [ImageItem(f"img{i}.png") for i in range(4)]
        emb = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_metadata_precision_at_k(neighbors, bare_items)
        assert result[1]["mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# knn_metadata_map_at_k
# ---------------------------------------------------------------------------


class TestKnnMetadataMapAtK:
    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_map_at_k(neighbors, _META_ITEMS)
        entry = result[3]
        assert "mean" in entry
        assert "std" in entry
        assert entry["per_item"].shape == (len(_META_ITEMS),)

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_map_at_k(neighbors, _META_ITEMS)
        for stats in result.values():
            assert 0.0 <= stats["mean"] <= 1.0 + 1e-6
            assert stats["per_item"].min() >= 0.0
            assert stats["per_item"].max() <= 1.0 + 1e-6

    def test_multiple_k_values(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[1, 3])
        result = knn_metadata_map_at_k(neighbors, _META_ITEMS)
        assert set(result.keys()) == {1, 3}

    def test_perfect_map_for_separated_explicit_positives(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_map_at_k(neighbors, _META_ITEMS)
        assert result[3]["mean"] == pytest.approx(1.0, abs=1e-5)

    def test_zero_for_items_without_positives(self) -> None:
        bare_items = [ImageItem(f"img{i}.png") for i in range(4)]
        emb = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_metadata_map_at_k(neighbors, bare_items)
        assert result[1]["mean"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# knn_label_mrr_at_k
# ---------------------------------------------------------------------------


class TestKnnLabelMrrAtK:
    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3])
        result = knn_label_mrr_at_k(neighbors, _MIXED_LABELS)
        entry = result[3]
        assert "mean" in entry
        assert "std" in entry
        assert "per_item" in entry
        assert entry["per_item"].shape == (len(_MIXED),)

    def test_perfect_mrr_single_class(self) -> None:
        neighbors = top_k_neighbors(_SINGLE, ks=[3])
        result = knn_label_mrr_at_k(neighbors, _SINGLE_LABELS)
        assert result[3]["mean"] == pytest.approx(1.0)

    def test_zero_mrr_all_different_classes(self) -> None:
        emb = np.array([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])], dtype=np.float32)
        labels = np.array(["a", "b"])
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_label_mrr_at_k(neighbors, labels)
        assert result[1]["mean"] == pytest.approx(0.0)

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3])
        result = knn_label_mrr_at_k(neighbors, _MIXED_LABELS)
        assert 0.0 <= result[3]["mean"] <= 1.0 + 1e-6
        assert result[3]["per_item"].min() >= 0.0
        assert result[3]["per_item"].max() <= 1.0 + 1e-6

    def test_multiple_k_values(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3, 5])
        result = knn_label_mrr_at_k(neighbors, _MIXED_LABELS)
        assert set(result.keys()) == {3, 5}

    def test_separated_classes_high_mrr(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[3])
        result = knn_label_mrr_at_k(neighbors, _MIXED_LABELS)
        assert result[3]["mean"] > 0.8

    def test_first_relevant_at_rank_two_gives_half(self) -> None:
        # 3 items: [1,0] [0,1] [1,0]  — labels a, a, b
        # item 2 (label b): nearest neighbor is item 0 (label a, rank-1), then item 2
        # is itself excluded; item 1 (label a) is rank-2 away. No same-label item is
        # available at rank 1, so first relevant for item 0 is at rank 2 → RR = 0.5.
        emb = np.array(
            [_unit([1, 0, 0, 0]), _unit([0.98, 0.2, 0, 0]), _unit([0, 1, 0, 0])],
            dtype=np.float32,
        )
        labels = np.array(["a", "b", "a"])
        neighbors = top_k_neighbors(emb, ks=[2])
        result = knn_label_mrr_at_k(neighbors, labels)
        # item 1 (label b): no other b → RR = 0
        # item 0 (label a): nearest is item 1 (b, wrong), second is item 2 (a, correct) → RR = 0.5
        # item 2 (label a): nearest is item 1 (b, wrong), second is item 0 (a, correct) → RR = 0.5 (approx)
        # mean ≈ (0.5 + 0 + 0.5) / 3 ≈ 0.333
        assert result[2]["mean"] == pytest.approx(1.0 / 3.0, abs=0.01)


# ---------------------------------------------------------------------------
# knn_label_r_precision
# ---------------------------------------------------------------------------


class TestKnnLabelRPrecision:
    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[7])
        result = knn_label_r_precision(neighbors, _MIXED_LABELS)
        assert "mean" in result
        assert "std" in result
        assert "per_item" in result
        assert result["per_item"].shape == (len(_MIXED),)

    def test_perfect_r_precision_single_class(self) -> None:
        neighbors = top_k_neighbors(_SINGLE, ks=[3])
        result = knn_label_r_precision(neighbors, _SINGLE_LABELS)
        assert result["mean"] == pytest.approx(1.0)

    def test_separated_classes_high_r_precision(self) -> None:
        # With well-separated 4-corn / 4-soy, R=3 for each item
        neighbors = top_k_neighbors(_MIXED, ks=[7])
        result = knn_label_r_precision(neighbors, _MIXED_LABELS)
        assert result["mean"] > 0.8

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[7])
        result = knn_label_r_precision(neighbors, _MIXED_LABELS)
        assert 0.0 <= result["mean"] <= 1.0 + 1e-6

    def test_flat_not_nested_by_k(self) -> None:
        neighbors = top_k_neighbors(_MIXED, ks=[7])
        result = knn_label_r_precision(neighbors, _MIXED_LABELS)
        assert "mean" in result
        assert not isinstance(result.get("mean"), dict)


# ---------------------------------------------------------------------------
# knn_metadata_mrr_at_k
# ---------------------------------------------------------------------------


class TestKnnMetadataMrrAtK:
    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_mrr_at_k(neighbors, _META_ITEMS)
        entry = result[3]
        assert "mean" in entry
        assert "std" in entry
        assert "per_item" in entry
        assert entry["per_item"].shape == (len(_META_ITEMS),)

    def test_perfect_mrr_for_separated_explicit_positives(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_mrr_at_k(neighbors, _META_ITEMS)
        assert result[3]["mean"] == pytest.approx(1.0, abs=1e-5)

    def test_zero_for_items_without_positives(self) -> None:
        bare_items = [ImageItem(f"img{i}.png") for i in range(4)]
        emb = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_metadata_mrr_at_k(neighbors, bare_items)
        assert result[1]["mean"] == pytest.approx(0.0)

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_mrr_at_k(neighbors, _META_ITEMS)
        assert 0.0 <= result[3]["mean"] <= 1.0 + 1e-6
        assert result[3]["per_item"].min() >= 0.0
        assert result[3]["per_item"].max() <= 1.0 + 1e-6

    def test_multiple_k_values(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[1, 3])
        result = knn_metadata_mrr_at_k(neighbors, _META_ITEMS)
        assert set(result.keys()) == {1, 3}

    def test_mrr_degrades_when_first_relevant_not_rank_one(self) -> None:
        # Two items: query item 0 with positive = item 3 only, but item 1 and 2 are nearer
        # Use a layout where the positive pair is far apart
        emb = np.array(
            [
                _unit([1, 0, 0, 0]),  # item 0 (query)
                _unit([1, 0, 0, 0]),  # item 1 (not positive of 0)
                _unit([1, 0, 0, 0]),  # item 2 (not positive of 0)
                _unit([0, 1, 0, 0]),  # item 3 (positive of 0, but far away)
            ],
            dtype=np.float32,
        )
        items = [
            ImageItem("a", frozenset({"d"})),
            ImageItem("b"),
            ImageItem("c"),
            ImageItem("d", frozenset({"a"})),
        ]
        neighbors = top_k_neighbors(emb, ks=[3])
        result = knn_metadata_mrr_at_k(neighbors, items)
        # item 0: positive (d) is at rank 3 among 3 neighbors → MRR = 1/3
        assert result[3]["per_item"][0] == pytest.approx(1.0 / 3.0, abs=1e-5)


# ---------------------------------------------------------------------------
# knn_metadata_r_precision
# ---------------------------------------------------------------------------


class TestKnnMetadataRPrecision:
    def test_returns_mean_std_per_item(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_r_precision(neighbors, _META_ITEMS)
        assert "mean" in result
        assert "std" in result
        assert "per_item" in result
        assert result["per_item"].shape == (len(_META_ITEMS),)

    def test_perfect_r_precision_for_separated_explicit_positives(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_r_precision(neighbors, _META_ITEMS)
        assert result["mean"] == pytest.approx(1.0, abs=1e-5)

    def test_zero_for_items_without_positives(self) -> None:
        bare_items = [ImageItem(f"img{i}.png") for i in range(4)]
        emb = np.array([_unit([1, 0, 0, 0])] * 4, dtype=np.float32)
        neighbors = top_k_neighbors(emb, ks=[1])
        result = knn_metadata_r_precision(neighbors, bare_items)
        assert result["mean"] == pytest.approx(0.0)

    def test_values_in_unit_interval(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_r_precision(neighbors, _META_ITEMS)
        assert 0.0 <= result["mean"] <= 1.0 + 1e-6

    def test_flat_not_nested_by_k(self) -> None:
        neighbors = top_k_neighbors(_META_EMB, ks=[3])
        result = knn_metadata_r_precision(neighbors, _META_ITEMS)
        assert "mean" in result
        assert not isinstance(result.get("mean"), dict)
