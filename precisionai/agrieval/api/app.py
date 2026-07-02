# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unified FastAPI application factory and CLI entry point for all AgriEval wirings."""

from __future__ import annotations

import argparse
import os

import uvicorn
from fastapi import FastAPI

from precisionai.agrieval.emb.api.routes.evaluate import router as emb_router
from precisionai.agrieval.seg.api.routes.evaluate import router as seg_router

_DESCRIPTION = """
Precision AI AgriEval — evaluation framework for agricultural computer vision models.

## Embedding evaluation

Benchmark how well image embeddings cluster by L1/L2 crop-class hierarchy.

```
POST /v1/embeddings/evaluate/image2image
POST /v1/embeddings/evaluate/plant2image
POST /v1/embeddings/evaluate/plant2plant
```

## Segmentation evaluation

Evaluate predicted colour-coded masks against ground-truth masks.  Returns
per-class IoU, Dice/F1, accuracy, mIoU, mAcc, and FWIoU.

```
POST /v1/segmentation/evaluate
```

## Path resolution

All path arguments (``pred_dir``, ``masks_dir``, ``classes_path``, etc.) are
resolved against the ``dataset_root`` field in the request body, which falls
back to the ``PAI_DATASET_ROOT`` environment variable and then ``"dataset"``.
Absolute paths bypass ``dataset_root`` entirely.
"""

app = FastAPI(
    title="precisionai-agrieval",
    description=_DESCRIPTION,
    version="0.1.0",
)

app.include_router(emb_router, prefix="/v1")
app.include_router(seg_router, prefix="/v1")


def main() -> None:
    """Parse CLI arguments and start the uvicorn ASGI server.

    Recognised arguments
    --------------------
    --dataset-root : str
        Default dataset root used when a request does not supply one.  Also
        settable via the ``PAI_DATASET_ROOT`` environment variable.  Defaults
        to ``"dataset"``.
    --host : str
        Bind address for the server.  Defaults to ``"0.0.0.0"``.
    --port : int
        TCP port to listen on.  Defaults to ``8000``.
    --no-reload : flag
        Disable uvicorn auto-reload (recommended in production).
    """
    parser = argparse.ArgumentParser(description="precisionai-agrieval unified API server")
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        metavar="PATH",
        help="Default dataset root used when the request omits dataset_root. "
        "Can also be set via PAI_DATASET_ROOT env var. (default: dataset)",
    )
    parser.add_argument("--host", default="0.0.0.0", metavar="HOST")
    parser.add_argument("--port", type=int, default=8000, metavar="PORT")
    parser.add_argument(
        "--no-reload",
        dest="reload",
        action="store_false",
        default=True,
        help="Disable auto-reload (recommended in production)",
    )
    args = parser.parse_args()

    os.environ.setdefault("PAI_DATASET_ROOT", args.dataset_root)

    uvicorn.run(
        "precisionai.agrieval.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
