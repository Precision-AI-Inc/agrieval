from pathlib import Path

import numpy as np
import pytest

from precisionai.agrieval.seg.services.evaluate import _build_color_map, load_classes

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(autouse=True)
def _default_dataset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the API's default dataset_root at tmp_path for every seg test.

    The route layer now rejects any path that resolves outside dataset_root
    (see ``precisionai.agrieval.api.paths.resolve_under_root``), so tests
    that post tmp_path-based directories need tmp_path itself to be — or be
    under — the effective root. Tests that set ``dataset_root`` explicitly
    simply override this default.
    """
    monkeypatch.setenv("PAI_DATASET_ROOT", str(tmp_path))


@pytest.fixture
def classes_path(tmp_path: Path) -> Path:
    """A copy of the shared class-definition fixture under tmp_path.

    Copied (rather than referenced in place under ``tests/data/``) so it
    resolves under the per-test ``dataset_root`` set by
    ``_default_dataset_root`` above.
    """
    dest = tmp_path / "class_map.json"
    dest.write_bytes((DATA_DIR / "class_map.json").read_bytes())
    return dest


@pytest.fixture
def classes(classes_path: Path) -> list:
    return load_classes(classes_path)


@pytest.fixture
def color_map(classes: list) -> dict:
    return _build_color_map(classes)


@pytest.fixture
def masks_dir() -> Path:
    return DATA_DIR / "masks"


@pytest.fixture
def n_classes(classes: list) -> int:
    return len(classes)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
