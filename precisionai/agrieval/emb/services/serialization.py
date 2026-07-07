# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""JSON serialisation helpers shared by the evaluation services."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _jsonify(obj: Any) -> Any:
    """Recursively convert a metrics result to JSON-serialisable types.

    numpy arrays are dropped because they are per-item visualisation artefacts
    that do not belong in a summary JSON.  numpy scalars are converted to
    Python native types.  ``NaN`` and ``inf`` become ``None``.

    Parameters
    ----------
    obj : Any
        Arbitrary metrics result object — dict, list, numpy array/scalar, or
        Python scalar.

    Returns
    -------
    object
        JSON-serialisable equivalent of ``obj``.  ``None`` is returned for
        numpy arrays (sentinel value; callers omit these keys).
    """
    if isinstance(obj, np.ndarray):
        return None  # sentinel — callers filter this key out
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, np.floating | float):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.generic):
        return obj.item()  # np.bool_, np.integer, and any other numpy scalar
    return obj
