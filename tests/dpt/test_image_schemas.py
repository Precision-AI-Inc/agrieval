import pytest
from pydantic import ValidationError

from precisionai.agrieval.dpt.schemas.evaluate import DptImageEvalRequest, DptImageEvalResponse

_VALID_IMAGE = [[[1.0, 2.0], [3.0, 4.0]]]  # (P=1, H=2, W=2)

# ---------------------------------------------------------------------------
# images validator — shares _validate_dense_entries with DptEvalRequest.tiles;
# these tests focus on the "image"-worded branch and the class's own wiring,
# not re-proving every rule already covered by test_schemas.py under kind="tile".
# ---------------------------------------------------------------------------


def test_valid_request_minimal():
    req = DptImageEvalRequest(images={"a": _VALID_IMAGE})
    assert req.k_values == [5, 10, 20]
    assert req.sample_pairs == 1_000_000
    assert req.max_patches == 20_000
    assert req.masks_dir is None
    assert req.classes_path is None


def test_empty_images_rejected():
    with pytest.raises(ValidationError, match="At least 1 image"):
        DptImageEvalRequest(images={})


def test_images_path_only_accepted():
    req = DptImageEvalRequest(images_path="images/batch_01.npz")
    assert req.images is None
    assert req.images_path == "images/batch_01.npz"


def test_both_image_sources_rejected():
    with pytest.raises(ValidationError, match="exactly one"):
        DptImageEvalRequest(images={"a": _VALID_IMAGE}, images_path="images.npz")


def test_no_image_source_rejected():
    with pytest.raises(ValidationError, match="exactly one"):
        DptImageEvalRequest()


def test_empty_image_id_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        DptImageEvalRequest(images={"": _VALID_IMAGE})


@pytest.mark.parametrize("bad_id", ["a/b", "a\\b", "nested/deep/id"])
def test_image_id_with_path_separator_rejected(bad_id):
    with pytest.raises(ValidationError, match="path separators"):
        DptImageEvalRequest(images={bad_id: _VALID_IMAGE})


def test_ragged_image_rejected():
    ragged = [[[1.0, 2.0], [3.0]]]  # second row shorter
    with pytest.raises(ValidationError, match="rectangular"):
        DptImageEvalRequest(images={"a": ragged})


def test_empty_axis_image_rejected():
    with pytest.raises(ValidationError, match="empty along any axis"):
        DptImageEvalRequest(images={"a": [[[]]]})


def test_mismatched_shapes_rejected():
    with pytest.raises(ValidationError, match=r"All images must share the same \(P, H, W\) shape"):
        DptImageEvalRequest(images={"a": _VALID_IMAGE, "b": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]})


def test_nan_image_rejected():
    with pytest.raises(ValidationError, match="non-finite"):
        DptImageEvalRequest(images={"a": [[[1.0, float("nan")], [3.0, 4.0]]]})


# ---------------------------------------------------------------------------
# k_values validator (duplicated per wiring, matching the tiles convention)
# ---------------------------------------------------------------------------


def test_empty_k_values_rejected():
    with pytest.raises(ValidationError, match="at least one value"):
        DptImageEvalRequest(images={"a": _VALID_IMAGE}, k_values=[])


def test_non_positive_k_rejected():
    with pytest.raises(ValidationError, match="positive"):
        DptImageEvalRequest(images={"a": _VALID_IMAGE}, k_values=[5, 0])


# ---------------------------------------------------------------------------
# label wiring validator
# ---------------------------------------------------------------------------


def test_masks_dir_without_classes_path_rejected():
    with pytest.raises(ValidationError, match="both"):
        DptImageEvalRequest(images={"a": _VALID_IMAGE}, masks_dir="masks/")


def test_classes_path_without_masks_dir_rejected():
    with pytest.raises(ValidationError, match="both"):
        DptImageEvalRequest(images={"a": _VALID_IMAGE}, classes_path="class_map.json")


def test_both_label_fields_accepted():
    req = DptImageEvalRequest(images={"a": _VALID_IMAGE}, masks_dir="masks/", classes_path="class_map.json")
    assert req.masks_dir == "masks/"


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


def test_negative_sample_pairs_rejected():
    with pytest.raises(ValidationError, match="greater than or equal"):
        DptImageEvalRequest(images={"a": _VALID_IMAGE}, sample_pairs=-1)


def test_max_patches_below_two_rejected():
    with pytest.raises(ValidationError, match="greater than or equal"):
        DptImageEvalRequest(images={"a": _VALID_IMAGE}, max_patches=1)


# ---------------------------------------------------------------------------
# response model
# ---------------------------------------------------------------------------


def test_response_optional_fields_default_none():
    resp = DptImageEvalResponse(
        n_images=1,
        embed_dim=1,
        grid_height=2,
        grid_width=2,
        n_patches=4,
        image_ids=["a"],
        k_values=[5],
        global_metrics={},
        per_image={"a": {}},
    )
    assert resp.classes is None
    assert resp.per_class is None
    assert resp.knn_confusion is None
    assert resp.warnings is None
