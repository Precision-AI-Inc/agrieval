import numpy as np
import pytest

from precisionai.agrieval.dpt.metrics import outlier_fraction, patch_norm_stats, patch_smoothness

# ---------------------------------------------------------------------------
# patch_norm_stats
# ---------------------------------------------------------------------------


def test_patch_norm_stats_known_values():
    patch_tokens = np.array([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]])
    stats = patch_norm_stats(patch_tokens)
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["p50"] == pytest.approx(5.0)


def test_patch_norm_stats_keys():
    stats = patch_norm_stats(np.ones((4, 3)))
    assert set(stats) == {"mean", "std", "p05", "p50", "p95"}


def test_patch_norm_stats_single_patch():
    stats = patch_norm_stats(np.array([[3.0, 4.0]]))
    assert stats["mean"] == pytest.approx(5.0)
    assert stats["std"] == pytest.approx(0.0)
    assert stats["p05"] == stats["p50"] == stats["p95"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# patch_smoothness
# ---------------------------------------------------------------------------


def test_patch_smoothness_identical_patches_is_one():
    tile = np.ones((3, 4, 5))
    assert patch_smoothness(tile) == pytest.approx(1.0)


def test_patch_smoothness_orthogonal_patches_is_zero():
    tile = np.zeros((2, 1, 2))
    tile[:, 0, 0] = [1.0, 0.0]
    tile[:, 0, 1] = [0.0, 1.0]
    assert patch_smoothness(tile) == pytest.approx(0.0)


def test_patch_smoothness_single_patch_returns_zero():
    tile = np.ones((3, 1, 1))
    assert patch_smoothness(tile) == 0.0


def test_patch_smoothness_zero_vector_patch_no_division_error():
    # A dead (all-zero) patch must not raise (division-by-zero guard).
    tile = np.zeros((2, 1, 2))
    tile[:, 0, 0] = [1.0, 0.0]
    tile[:, 0, 1] = [0.0, 0.0]
    assert patch_smoothness(tile) == pytest.approx(0.0)


def test_patch_smoothness_width_only_grid():
    # H=1: only right-neighbor pairs exist. sim(p0,p1)=1, sim(p1,p2)=0 → mean 0.5.
    tile = np.zeros((2, 1, 3))
    tile[:, 0, 0] = [1.0, 0.0]
    tile[:, 0, 1] = [1.0, 0.0]
    tile[:, 0, 2] = [0.0, 1.0]
    assert patch_smoothness(tile) == pytest.approx(0.5)


def test_patch_smoothness_height_only_grid():
    # W=1: only down-neighbor pairs exist. sim(p0,p1)=1, sim(p1,p2)=0 → mean 0.5.
    tile = np.zeros((2, 3, 1))
    tile[:, 0, 0] = [1.0, 0.0]
    tile[:, 1, 0] = [1.0, 0.0]
    tile[:, 2, 0] = [0.0, 1.0]
    assert patch_smoothness(tile) == pytest.approx(0.5)


def test_patch_smoothness_averages_directional_means_not_pooled_pairs():
    # 2x3 grid: rows are constant (all 4 right-pairs have sim 1) and the two
    # rows are orthogonal (all 3 down-pairs have sim 0). The benchmark-aligned
    # definition averages the two directional means → (1 + 0) / 2 = 0.5.
    # A pooled mean over all 7 pairs would give 4/7 instead.
    tile = np.zeros((2, 2, 3))
    tile[:, 0, :] = np.array([1.0, 0.0])[:, None]
    tile[:, 1, :] = np.array([0.0, 1.0])[:, None]
    assert patch_smoothness(tile) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# outlier_fraction
# ---------------------------------------------------------------------------


def test_outlier_fraction_known_values():
    # Strictly-above comparison: only 5.0 exceeds 4.0.
    norms = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert outlier_fraction(norms, threshold=4.0) == pytest.approx(1 / 5)


def test_outlier_fraction_empty_returns_zero():
    assert outlier_fraction(np.array([]), threshold=1.0) == 0.0


def test_outlier_fraction_no_values_above_threshold():
    norms = np.array([1.0, 1.0, 1.0])
    assert outlier_fraction(norms, threshold=10.0) == 0.0


def test_outlier_fraction_values_equal_to_threshold_are_not_outliers():
    norms = np.array([5.0, 5.0, 5.0])
    assert outlier_fraction(norms, threshold=5.0) == pytest.approx(0.0)


def test_outlier_fraction_negative_threshold_includes_all():
    norms = np.array([1.0, 2.0, 3.0])
    assert outlier_fraction(norms, threshold=-1.0) == pytest.approx(1.0)


def test_outlier_fraction_single_value():
    assert outlier_fraction(np.array([5.0]), threshold=4.9) == pytest.approx(1.0)
    assert outlier_fraction(np.array([5.0]), threshold=5.0) == pytest.approx(0.0)
