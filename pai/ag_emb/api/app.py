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

"""FastAPI application factory and entry point."""

from __future__ import annotations

import argparse
import os

import uvicorn
from fastapi import FastAPI

from pai.ag_emb.api.routes.evaluate import router as evaluate_router

_DESCRIPTION = """
Evaluate the quality of image model embeddings — how well they cluster by crop class.

## Usage

Send a single JSON object with your embeddings dict to **POST /v1/embeddings/evaluate**.
Everything else is optional.

```json
{
  "embeddings": {
    "dataset/corn_[HB-25000SBC]/img/220622-corn-54e94639.png": [0.12, -0.31, 0.27, ...],
    "dataset/corn_[nikon_d610]/img/190627-corn-bded43cb.JPG":  [0.11, -0.30, 0.26, ...],
    "dataset/soybean_[anafi]/img/210625-soybeans-fce5a850.JPG": [-0.45, 0.18, -0.09, ...]
  }
}
```

**Embedding requirements**
- Flat 1-D array (length = `embedding_dim`)
- L2-normalised float32 (`‖v‖₂ = 1.0 ± 0.001`)
- All vectors must share the same length
- Minimum 2 embeddings per request

**Path convention**

Crop class is extracted automatically from the folder name:
```
dataset / corn_[HB-25000SBC] / img / 220622-img.png
          ^^^^^^^^^^^^^^^^^^^^^^^^
          crop_[camera]  →  class = "corn"
```

**Response**

Returns a fixed JSON report with a `global_metrics` summary and a
`per_class` breakdown for every detected crop.
"""

app = FastAPI(
    title="pai-ag-emb",
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
    parser = argparse.ArgumentParser(description="pai-ag-emb API server")
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
        "pai.ag_emb.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
