import pytest
from pydantic import ValidationError

from precisionai.agrieval.dpt.schemas.evaluate import DptEvalRequest, DptEvalResponse

_VALID_TILE = [[[1.0, 2.0], [3.0, 4.0]]]  # (C=1, H=2, W=2)

# ---------------------------------------------------------------------------
# tiles validator
# ---------------------------------------------------------------------------


def test_valid_request_minimal():
    req = DptEvalRequest(tiles={"a": _VALID_TILE})
    assert req.k_values == [5, 10, 20]
    assert req.sample_pairs == 1_000_000
    assert req.max_patches == 20_000
    assert req.masks_dir is None
    assert req.classes_path is None


def test_empty_tiles_rejected():
    with pytest.raises(ValidationError, match="At least 1 tile"):
        DptEvalRequest(tiles={})


def test_tiles_path_only_accepted():
    req = DptEvalRequest(tiles_path="tiles/batch_01.npz")
    assert req.tiles is None
    assert req.tiles_path == "tiles/batch_01.npz"


def test_tiles_explicit_none_with_tiles_path_accepted():
    """Pydantic skips validators on defaulted fields, so pass tiles=None explicitly
    to exercise the tiles validator's own None short-circuit."""
    req = DptEvalRequest(tiles=None, tiles_path="tiles/batch_01.npz")
    assert req.tiles is None


def test_both_tile_sources_rejected():
    with pytest.raises(ValidationError, match="exactly one"):
        DptEvalRequest(tiles={"a": _VALID_TILE}, tiles_path="tiles.npz")


def test_no_tile_source_rejected():
    with pytest.raises(ValidationError, match="exactly one"):
        DptEvalRequest()


def test_empty_tile_id_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        DptEvalRequest(tiles={"": _VALID_TILE})


@pytest.mark.parametrize("bad_id", ["a/b", "a\\b", "..\\up", "nested/deep/id"])
def test_tile_id_with_path_separator_rejected(bad_id):
    with pytest.raises(ValidationError, match="path separators"):
        DptEvalRequest(tiles={bad_id: _VALID_TILE})


def test_ragged_tile_rejected():
    ragged = [[[1.0, 2.0], [3.0]]]  # second row shorter
    with pytest.raises(ValidationError, match="rectangular"):
        DptEvalRequest(tiles={"a": ragged})


def test_two_dimensional_tile_rejected():
    # Pydantic's type layer enforces exactly three nesting levels, so a 2-D
    # tile fails coercion before the custom validator runs. The nesting depth
    # is intentionally wrong here, so it mismatches the declared type.
    with pytest.raises(ValidationError, match="list_type"):
        DptEvalRequest(tiles={"a": [[1.0, 2.0]]})  # type: ignore[arg-type]


def test_four_dimensional_tile_rejected():
    with pytest.raises(ValidationError, match=r"float_type|list_type"):
        DptEvalRequest(tiles={"a": [[[[1.0, 2.0]]]]})  # type: ignore[arg-type]


def test_empty_axis_tile_rejected():
    with pytest.raises(ValidationError, match="empty along any axis"):
        DptEvalRequest(tiles={"a": [[[]]]})


def test_mismatched_shapes_rejected():
    with pytest.raises(ValidationError, match="same \\(C, H, W\\) shape"):
        DptEvalRequest(tiles={"a": _VALID_TILE, "b": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]})


def test_nan_tile_rejected():
    with pytest.raises(ValidationError, match="non-finite"):
        DptEvalRequest(tiles={"a": [[[1.0, float("nan")], [3.0, 4.0]]]})


def test_inf_tile_rejected():
    with pytest.raises(ValidationError, match="non-finite"):
        DptEvalRequest(tiles={"a": [[[1.0, float("inf")], [3.0, 4.0]]]})


def test_float32_overflow_rejected():
    # Finite in float64 but overflows to inf at the float32 evaluation dtype.
    with pytest.raises(ValidationError, match="non-finite"):
        DptEvalRequest(tiles={"a": [[[1.0, 1e39], [3.0, 4.0]]]})


# ---------------------------------------------------------------------------
# k_values validator
# ---------------------------------------------------------------------------


def test_empty_k_values_rejected():
    with pytest.raises(ValidationError, match="at least one value"):
        DptEvalRequest(tiles={"a": _VALID_TILE}, k_values=[])


def test_non_positive_k_rejected():
    with pytest.raises(ValidationError, match="positive"):
        DptEvalRequest(tiles={"a": _VALID_TILE}, k_values=[5, 0])


# ---------------------------------------------------------------------------
# label wiring validator
# ---------------------------------------------------------------------------


def test_masks_dir_without_classes_path_rejected():
    with pytest.raises(ValidationError, match="both"):
        DptEvalRequest(tiles={"a": _VALID_TILE}, masks_dir="masks/")


def test_classes_path_without_masks_dir_rejected():
    with pytest.raises(ValidationError, match="both"):
        DptEvalRequest(tiles={"a": _VALID_TILE}, classes_path="class_map.json")


def test_both_label_fields_accepted():
    req = DptEvalRequest(tiles={"a": _VALID_TILE}, masks_dir="masks/", classes_path="class_map.json")
    assert req.masks_dir == "masks/"


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


def test_negative_sample_pairs_rejected():
    with pytest.raises(ValidationError, match="greater than or equal"):
        DptEvalRequest(tiles={"a": _VALID_TILE}, sample_pairs=-1)


def test_max_patches_below_two_rejected():
    with pytest.raises(ValidationError, match="greater than or equal"):
        DptEvalRequest(tiles={"a": _VALID_TILE}, max_patches=1)


# ---------------------------------------------------------------------------
# response model
# ---------------------------------------------------------------------------


def test_response_optional_fields_default_none():
    resp = DptEvalResponse(
        n_tiles=1,
        embed_dim=1,
        grid_height=2,
        grid_width=2,
        n_patches=4,
        tile_ids=["a"],
        k_values=[5],
        global_metrics={},
        per_tile={"a": {}},
    )
    assert resp.classes is None
    assert resp.per_class is None
    assert resp.knn_confusion is None
    assert resp.warnings is None
