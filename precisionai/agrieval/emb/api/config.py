# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Re-export of the shared dataset-root configuration for the emb sub-package."""

from precisionai.agrieval.api.config import get_dataset_root

__all__ = ["get_dataset_root"]
