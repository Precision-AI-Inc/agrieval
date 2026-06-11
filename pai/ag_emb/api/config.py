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

"""Runtime configuration resolved from environment variables.

``PAI_DATASET_ROOT`` — default dataset root used when the request does not
supply ``dataset_root``.  Set by the ``--dataset-root`` CLI argument before
uvicorn spawns worker processes so the value is inherited.

Defaults to ``"dataset"`` so a repository checked out next to a ``dataset/``
directory works out of the box.
"""

from __future__ import annotations

import os


def get_dataset_root() -> str:
    """Return the server-level default dataset root.

    The value is read from the ``PAI_DATASET_ROOT`` environment variable at
    request time so that it reflects any changes made after server start.

    Returns
    -------
    str
        Dataset root path, defaulting to ``"dataset"`` when the environment
        variable is not set.
    """
    return os.environ.get("PAI_DATASET_ROOT", "dataset")
