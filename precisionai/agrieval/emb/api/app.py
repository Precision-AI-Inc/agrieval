# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory and entry point."""

from __future__ import annotations

import argparse
import os

import uvicorn
from fastapi import FastAPI

from precisionai.agrieval.emb.api.routes.evaluate import router as evaluate_router

_DESCRIPTION = """
Evaluate the quality of image model embeddings — how well they cluster by L1/L2 cluster.

## Usage

Send a single JSON object with your embeddings dict to **POST /v1/embeddings/evaluate/image2image**.
Everything else is optional.

```json
{
  "embeddings": {
    "images/A1/220622-corn-54e94639.png":  [0.12, -0.31, 0.27, ...],
    "images/A2/190627-corn-bded43cb.JPG":  [0.11, -0.30, 0.26, ...],
    "images/B1/210625-soybeans-fce5a850.png": [-0.45, 0.18, -0.09, ...]
  }
}
```

**Embedding requirements**
- Flat 1-D array (length = `embedding_dim`)
- L2-normalised float32 (`‖v‖₂ = 1.0 ± 0.001`)
- All vectors must share the same length
- Minimum 2 embeddings per request

**Path convention**

The L1 cluster label is extracted automatically from the L2 folder name:
```
images / A1 / 220622-img.png
         ^^
         L2 folder  →  L1 = "A"  (leading letters)
```

Supply a `metadata` dict (loaded from the dataset's `metadata/` directory) to
enable graded nDCG and per-attribute KPIs based on the explicit L1/L2 cluster structure.

**Response**

Returns a fixed JSON report with a `global_metrics` summary and a
`per_class` breakdown for every detected L1 cluster.
"""

app = FastAPI(
    title="precisionai-agrieval-emb",
    description=_DESCRIPTION,
    version="0.1.0",
)

app.include_router(evaluate_router, prefix="/v1")


def main() -> None:
    """Parse CLI arguments and start the uvicorn ASGI server.

    Recognised arguments
    --------------------
    --dataset-root : str
        Default dataset root passed to the evaluation service when a request
        does not supply one.  Also settable via the ``PAI_DATASET_ROOT``
        environment variable.  Defaults to ``"dataset"``.
    --host : str
        Bind address for the server.  Defaults to ``"0.0.0.0"``.
    --port : int
        TCP port to listen on.  Defaults to ``8000``.
    --no-reload : flag
        Disable uvicorn auto-reload (recommended in production).
    """
    parser = argparse.ArgumentParser(description="precisionai-agrieval-emb API server")
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

    # Propagate to child worker processes via env var before uvicorn forks.
    os.environ.setdefault("PAI_DATASET_ROOT", args.dataset_root)

    uvicorn.run(
        "precisionai.agrieval.emb.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
