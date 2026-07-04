# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for precisionai.agrieval.emb.services.reporting — print and visualization functions."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from precisionai.agrieval.emb.services.evaluate import run_image2image_eval
from precisionai.agrieval.emb.services.reporting import (
    _build_scatter3d,
    _print_global,
    _print_per_class,
    _write_plotly,
    plot_cosine_similarity,
    plot_knn_confusion,
    plot_lle,
    plot_tsne,
    print_result,
)
from tests.emb.test_evaluate_endpoint import GOOD_EMBEDDINGS, _make_metadata

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result(image_embeddings):
    return run_image2image_eval(
        image_embeddings=image_embeddings,
        k_values=[5, 10],
        dataset_root=None,
        sample_pairs=None,
    )


# ---------------------------------------------------------------------------
# print_result / _print_global / _print_per_class
# ---------------------------------------------------------------------------


class TestPrintResult:
    def test_outputs_header(self, capsys, result):
        print_result(result)
        out = capsys.readouterr().out
        assert "n_items" in out
        assert "embedding_dim" in out
        assert "classes" in out
        assert "k_values" in out

    def test_outputs_global_metrics(self, capsys, result):
        print_result(result)
        out = capsys.readouterr().out
        assert "global_metrics" in out
        assert "pairwise cosine" in out
        assert "intra/inter gap" in out
        assert "KNN purity" in out
        assert "nDCG" in out
        assert "MAP" in out
        assert "effective_rank" in out

    def test_outputs_per_class(self, capsys, result):
        print_result(result)
        out = capsys.readouterr().out
        assert "per_class" in out
        for cls in result["classes"]:
            assert cls in out

    def test_print_global_directly(self, capsys, result):
        _print_global(result["global_metrics"])
        out = capsys.readouterr().out
        assert "pairwise cosine" in out

    def test_print_per_class_directly(self, capsys, result):
        _print_per_class(result["per_class"])
        out = capsys.readouterr().out
        assert "per_class" in out

    def test_single_class_no_inter_gap_printed(self, capsys):
        single_class_result = run_image2image_eval(
            image_embeddings={
                "dataset/A1/img/a.png": [1.0] + [0.0] * 15,
                "dataset/A1/img/b.png": [1.0] + [0.0] * 15,
            },
            k_values=[1],
            dataset_root="dataset",
            sample_pairs=None,
        )
        print_result(single_class_result)
        out = capsys.readouterr().out
        assert "A" in out
        assert "intra/inter gap" not in out  # gap=None for single class


# ---------------------------------------------------------------------------
# _write_plotly
# ---------------------------------------------------------------------------


class TestWritePlotly:
    def test_writes_html(self, tmp_path):
        fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
        out = tmp_path / "test.html"
        _write_plotly(fig, str(out))
        assert out.exists()
        content = out.read_text()
        assert "plotly" in content.lower()

    def test_falls_back_to_html_when_kaleido_missing(self, tmp_path, capsys):
        fig = go.Figure(data=go.Scatter(x=[1], y=[1]))
        out = tmp_path / "test.png"
        html_out = tmp_path / "test.html"
        _write_plotly(fig, str(out))
        # kaleido not installed → should fall back to .html
        assert html_out.exists()
        captured = capsys.readouterr().out
        assert "Saved:" in captured


# ---------------------------------------------------------------------------
# plot_knn_confusion
# ---------------------------------------------------------------------------


class TestPlotKnnConfusion:
    def test_writes_html(self, tmp_path, result):
        out = tmp_path / "confusion.html"
        plot_knn_confusion(result, output_path=str(out))
        assert out.exists()
        assert "plotly" in out.read_text().lower()

    def test_default_k_is_largest(self, tmp_path, result):
        out = tmp_path / "confusion.html"
        plot_knn_confusion(result, output_path=str(out))
        assert out.exists()

    def test_explicit_k(self, tmp_path, result):
        out = tmp_path / "confusion_k5.html"
        plot_knn_confusion(result, k=5, output_path=str(out))
        assert out.exists()

    def test_missing_confusion_raises(self, tmp_path):
        bad_result = {"k_values": [5], "classes": ["a"], "knn_confusion": {}}
        with pytest.raises(ValueError, match="knn_confusion not found"):
            plot_knn_confusion(bad_result, k=5, output_path=str(tmp_path / "x.html"))

    def test_prints_saved_path(self, tmp_path, result, capsys):
        out = tmp_path / "confusion.html"
        plot_knn_confusion(result, output_path=str(out))
        assert "Saved:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# plot_cosine_similarity
# ---------------------------------------------------------------------------


class TestPlotCosineSimilarity:
    def test_writes_html(self, tmp_path, image_embeddings, result):
        out = tmp_path / "cosine.html"
        plot_cosine_similarity(image_embeddings, result, output_path=str(out))
        assert out.exists()
        assert "plotly" in out.read_text().lower()

    def test_sorted_by_class(self, tmp_path, image_embeddings, result):
        out = tmp_path / "cosine.html"
        plot_cosine_similarity(image_embeddings, result, output_path=str(out))
        # tick labels are filenames, not class names — just verify the file was written
        assert out.exists()

    def test_prints_saved_path(self, tmp_path, image_embeddings, result, capsys):
        out = tmp_path / "cosine.html"
        plot_cosine_similarity(image_embeddings, result, output_path=str(out))
        assert "Saved:" in capsys.readouterr().out

    def test_missing_item_labels_falls_back_to_extract_labels(self, tmp_path, image_embeddings, result):
        """When the result dict has no item_labels key, labels are derived from paths."""
        result_without_labels = {k: v for k, v in result.items() if k != "item_labels"}
        out = tmp_path / "cosine.html"
        plot_cosine_similarity(image_embeddings, result_without_labels, output_path=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# _build_scatter3d
# ---------------------------------------------------------------------------


class TestBuildScatter3d:
    def test_adds_traces_per_class(self, image_embeddings, result):
        paths = list(image_embeddings.keys())
        n = len(paths)
        coords = np.zeros((n, 3), dtype=np.float64)
        classes = result["classes"]
        label_arr = np.array(result["item_labels"])

        fig = go.Figure()
        _build_scatter3d(fig, coords, classes, label_arr, paths, axis_prefix="test")
        assert len(fig.data) == len(classes)  # type: ignore[arg-type]

    def test_trace_names_match_classes(self, image_embeddings, result):
        paths = list(image_embeddings.keys())
        n = len(paths)
        coords = np.zeros((n, 3), dtype=np.float64)
        classes = result["classes"]
        label_arr = np.array(result["item_labels"])

        fig = go.Figure()
        _build_scatter3d(fig, coords, classes, label_arr, paths, axis_prefix="test")
        trace_names = [t.name for t in fig.data]  # type: ignore[union-attr]
        assert set(trace_names) == set(classes)


# ---------------------------------------------------------------------------
# plot_tsne
# ---------------------------------------------------------------------------


class TestPlotTsne:
    def test_writes_html_3d(self, tmp_path, image_embeddings, result):
        out = tmp_path / "tsne3d.html"
        plot_tsne(image_embeddings, result, output_path=str(out), dimensions=3)
        assert out.exists()
        assert "plotly" in out.read_text().lower()

    def test_writes_html_2d(self, tmp_path, image_embeddings, result):
        out = tmp_path / "tsne2d.html"
        plot_tsne(image_embeddings, result, output_path=str(out), dimensions=2)
        assert out.exists()

    def test_invalid_dimensions_raises(self, tmp_path, image_embeddings, result):
        with pytest.raises(ValueError, match="dimensions must be 2 or 3"):
            plot_tsne(image_embeddings, result, output_path=str(tmp_path / "x.html"), dimensions=4)

    def test_prints_saved_path(self, tmp_path, image_embeddings, result, capsys):
        out = tmp_path / "tsne.html"
        plot_tsne(image_embeddings, result, output_path=str(out))
        assert "Saved:" in capsys.readouterr().out

    def test_item_labels_fallback(self, tmp_path, image_embeddings, result):
        result_no_labels = {k: v for k, v in result.items() if k != "item_labels"}
        out = tmp_path / "tsne_fallback.html"
        plot_tsne(image_embeddings, result_no_labels, output_path=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_lle
# ---------------------------------------------------------------------------


class TestPlotLle:
    def test_writes_html(self, tmp_path, image_embeddings, result):
        out = tmp_path / "lle.html"
        plot_lle(image_embeddings, result, output_path=str(out))
        assert out.exists()
        assert "plotly" in out.read_text().lower()

    def test_custom_n_neighbors(self, tmp_path, image_embeddings, result):
        out = tmp_path / "lle_custom.html"
        plot_lle(image_embeddings, result, output_path=str(out), n_neighbors=5)
        assert out.exists()

    def test_prints_saved_path(self, tmp_path, image_embeddings, result, capsys):
        out = tmp_path / "lle.html"
        plot_lle(image_embeddings, result, output_path=str(out))
        assert "Saved:" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# print_result — metadata mode (alignment, attribute nDCG, group_analysis)
# ---------------------------------------------------------------------------


class TestPrintResultMetadataMode:
    @pytest.fixture(scope="class")
    def metadata_result(self):
        return run_image2image_eval(
            GOOD_EMBEDDINGS, k_values=[3], dataset_root=None, sample_pairs=50, metadata=_make_metadata()
        )

    def test_prints_alignment(self, capsys, metadata_result):
        print_result(metadata_result)
        assert "alignment" in capsys.readouterr().out

    def test_prints_attribute_ndcg(self, capsys, metadata_result):
        print_result(metadata_result)
        assert "attr_nDCG" in capsys.readouterr().out

    def test_prints_group_analysis(self, capsys, metadata_result):
        print_result(metadata_result)
        assert "group_analysis" in capsys.readouterr().out
