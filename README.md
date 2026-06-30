<p align="center">
  <img src="docs/assets/logo.png" alt="Precision AI Logo" width="120"/>
</p>

# Precision AI AgriEval

AgriEval is Precision AI's evaluation framework for agricultural computer vision models. It answers the question: **how well do your model outputs support the downstream retrieval and analysis tasks that agronomic workflows depend on?**

You bring model outputs — embeddings, segmentation masks, detections — alongside your dataset's ground-truth cluster annotations. AgriEval produces a structured report of retrieval KPIs, geometry diagnostics, and per-class breakdowns, accessible through a FastAPI server or directly from Python.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

---

## Subpackages

| Subpackage | Domain | Docs |
|---|---|---|
| `precisionai.agrieval.emb` | Embedding evaluation — KNN retrieval benchmarking, geometry diagnostics, interactive visualizations | [EMBEDDING.md](docs/EMBEDDING.md) |

---

## Data conventions

All AgriEval subpackages share a common **L1/L2 cluster convention** for structuring dataset annotations.

Images are organised into **L2 folders** — letter-number identifiers such as `A1`, `A2`, `D1` — where the leading letters define the coarse **L1 class** (`A1` and `A2` both belong to class `A`). This two-level hierarchy encodes both fine-grained subgroup identity and coarse crop-class membership, and is the ground truth against which all retrieval and classification metrics are computed.

```
images/
  A1/          ← L2 subgroup; L1 class "A"
    img001.png
    img002.png
  A2/          ← different subgroup, same L1 class "A"
    img003.png
  D1/          ← L2 subgroup; L1 class "D"
    img004.png
```

Each subpackage extends this foundation with domain-specific annotation files (embedding vectors, segmentation masks, instance crops) and evaluation wirings suited to that modality. See each subpackage's documentation for the full data format.

Repository image fixtures and other committed data files must stay at or below **1800 KB** each. Test image fixtures should use the shared `1332x540` resolution and optimized RGB PNG encoding. The size limit is enforced for newly added files by the `check-added-large-files` pre-commit hook.

---

## Subpackage layout

```
precisionai/agrieval/
  emb/          # Embedding evaluation
    api/        # FastAPI app and route handlers
    metrics/    # Pure computation — ranking, similarity, geometry, …
    schemas/    # Pydantic request/response models
    services/   # Evaluation orchestration and reporting

tests/
  emb/          # Tests for precisionai.agrieval.emb

examples/
  emb/          # Runnable scripts and notebooks for emb
```

---

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Full development (all subpackages, tests, visualization, pre-commit)
pip install -e ".[dev]"

# Runtime only
pip install -r requirements.txt
```

---

## Quick start

### Embedding evaluation

See **[docs/EMBEDDING.md](docs/EMBEDDING.md)** for the full reference — data format, all four retrieval wirings, API endpoints, metrics, and visualizations.

```python
from precisionai.agrieval.emb.services.evaluate import run_image2image_eval
from precisionai.agrieval.emb.services.reporting import print_result

embeddings = {
    "images/A1/img1.png": [...],  # L2-normalised float32 vectors
    "images/D2/img2.JPG": [...],
}

result = run_image2image_eval(
    image_embeddings=embeddings,
    k_values=[5, 10],
    dataset_root="images",
)
print_result(result)
```

```bash
# Start the embedding evaluation API server (default port 8000)
precisionai-agrieval

# Interactive API docs
open http://localhost:8000/docs
```

---

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, branching, and PR guidelines. [CLAUDE.md](CLAUDE.md) documents the code style and conventions enforced in this repo.

```bash
# Run embedding evaluation tests
python -m pytest tests/emb/

# Run pre-commit hooks manually
pre-commit run --all-files
```

Pre-commit rejects newly added files larger than **1800 KB**. Downsample test image fixtures to `1332x540` and save them as optimized RGB PNGs before committing them.

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE.md](LICENSE.md).
