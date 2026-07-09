# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Safe resolution of request-supplied paths against a dataset root."""

from __future__ import annotations

from pathlib import Path


def resolve_under_root(root: Path | str, value: str, *, field_name: str) -> Path:
    """Resolve ``value`` against ``root``, rejecting a result that escapes it.

    Plain ``Path`` joining silently discards ``root`` when ``value`` is
    absolute, and does not catch ``..`` segments that climb back out of
    ``root`` — both let a caller reach any path the server process can
    access. This function closes both gaps: regardless of whether ``value``
    is relative or absolute, the resolved result must be ``root`` itself or
    one of its descendants.

    Parameters
    ----------
    root : Path | str
        Dataset root the resolved path must stay within.
    value : str
        Request-supplied path, relative or absolute.
    field_name : str
        Name of the request field ``value`` came from, used in the error
        message.

    Returns
    -------
    Path
        The resolved, absolute path.

    Raises
    ------
    ValueError
        If the resolved path is not ``root`` itself or a descendant of it.
    """
    root_resolved = Path(root).resolve()
    candidate = (root_resolved / value).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"{field_name} '{value}' must resolve within dataset_root ('{root_resolved}').")
    return candidate
