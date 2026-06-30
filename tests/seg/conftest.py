from pathlib import Path

import numpy as np
import pytest

from precisionai.agrieval.seg.services.evaluate import _build_color_map, load_classes

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def classes_path() -> Path:
    return DATA_DIR / "class_map.json"


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
