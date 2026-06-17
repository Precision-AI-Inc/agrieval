# Precision AI Agricultural Embedding Evaluation API

Agricultural embedding evaluation API for Precision AI. Measures how well image model embeddings cluster by crop class using KNN-based metrics, geometry diagnostics, and interactive visualizations.

[![Tests](https://github.com/Precision-AI-Inc/pai-ag-emb-eval/actions/workflows/workflow.yml/badge.svg)](https://github.com/Precision-AI-Inc/pai-ag-emb-eval/actions/workflows/workflow.yml) [![Coverage](.github/badges/coverage.svg)](https://github.com/Precision-AI-Inc/pai-ag-emb-eval/actions/workflows/workflow.yml)

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate

# Runtime only (API server)
pip install -r requirements.txt

# Full development (analysis, visualization, tests)
pip install -r requirements-dev.txt

# Or via pyproject extras
pip install -e ".[dev]"
```

---

## Data format

### Canonical directory layout

```
images/
  corn_HB-25000SBC/
    220622-img.png
    220622-img2.png
  corn_nikon_d610/
    190627-img.JPG
  soybean_HB-25000SBC/
    220608-img.png
  soybean_anafi/
    210625-img.JPG
metadata/
  corn_HB-25000SBC/
    metadata.json
  corn_nikon_d610/
    metadata.json
  soybean_HB-25000SBC/
    metadata.json
  soybean_anafi/
    metadata.json
```

Subgroup directory names follow strict `class_subgroup` convention — the part before the first `_` is the class label. Both `images/` and `metadata/` mirror the same subgroup structure.

Each `metadata.json` identifies which images belong to the group and carries shared metadata for all of them:

```json
{
  "class_name": "corn",
  "attributes": { "camera": "HB-25000SBC", "growth_stage": "medium" },
  "images": [
    "images/corn_HB-25000SBC/220622-img1.png",
    "images/corn_HB-25000SBC/220622-img2.png"
  ]
}
```

The `images` list uses the same path strings that will be used as embedding keys in the API request. All images listed in the same `metadata.json` are considered explicit positives of each other.

**Supported image extensions (case-sensitive):** `.jpg`, `.JPG`, `.jpeg`, `.JPEG`, `.png`, `.PNG`. Requests with any other extension are rejected with a validation error listing the offending paths.

---

### Wiring 1 — embeddings only

Pass a dict of `image_path → embedding`. The crop class label is inferred automatically from the subgroup folder name using `dataset_root` as the strip prefix.

**Path convention** — `{dataset_root}/{class_subgroup}/{image}`:

```
images/
  corn_HB-25000SBC/220622-img.png    →  class: corn
  corn_nikon_d610/190627-img.JPG     →  class: corn
  soybean_anafi/210625-img.JPG       →  class: soybean
```

`dataset_root` should point to the `images/` directory (direct parent of subgroup folders). Unbalanced subgroup sizes are expected and accepted.

**Embedding vector requirements:**

- Flat 1-D array of length `embedding_dim`
- **L2-normalised float32** — `‖v‖₂ = 1.0 ± 0.001`
- All vectors must share the same length
- Minimum 2 embeddings per request

---

### Wiring 2 — embeddings + metadata

Add a `metadata` field alongside `embeddings`. Each entry in `metadata` is a **group** of images that are semantically similar to each other. Use the subgroup directory name as the group key.

```json
{
  "embeddings": {
    "images/corn_HB-25000SBC/220622-img.png":  [0.12, -0.31, "..."],
    "images/corn_nikon_d610/190627-img.JPG":   [0.11, -0.29, "..."],
    "images/soybean_HB-25000SBC/220608-img.png": [-0.45, 0.18, "..."],
    "images/soybean_anafi/210625-img.JPG":      [-0.44, 0.17, "..."]
  },
  "metadata": {
    "corn_HB-25000SBC": {
      "images": ["images/corn_HB-25000SBC/220622-img.png"],
      "class_name": "corn",
      "attributes": { "camera": "HB-25000SBC", "growth_stage": "medium" }
    },
    "corn_nikon_d610": {
      "images": ["images/corn_nikon_d610/190627-img.JPG"],
      "class_name": "corn",
      "attributes": { "camera": "nikon_d610", "growth_stage": "medium" }
    },
    "soybean_HB-25000SBC": {
      "images": ["images/soybean_HB-25000SBC/220608-img.png"],
      "class_name": "soybean",
      "attributes": { "camera": "HB-25000SBC", "growth_stage": "medium" }
    },
    "soybean_anafi": {
      "images": ["images/soybean_anafi/210625-img.JPG"],
      "class_name": "soybean",
      "attributes": { "camera": "anafi", "growth_stage": "medium" }
    }
  }
}
```

**`metadata` field contract:**

| Key | Type | Required | Description |
|---|---|---|---|
| `images` | `list[str]` | yes | Exact embedding path keys — must match the keys used in `embeddings`. |
| `class_name` | `str` | **yes** | Class label for all images in this group. Used as the class label for all label-aware metrics. |
| `attributes` | `dict[str, str]` | no | Open key-value pairs shared by all images in this group. Any keys are accepted — common examples: `growth_stage`, `camera`, `sunlight`, `crop-field`. One `knn_attribute_ndcg` metric is produced **per attribute key** present across all groups. |

**Group key** — use the subgroup directory name (e.g. `corn_HB-25000SBC`) as a convention; any unique string is accepted.

**Multiple clusters per subgroup** — a single subgroup directory can be split into multiple metadata groups by using distinct group keys and partitioning the `images` lists. This is useful when images within one subgroup vary across an attribute (e.g. different growth stages captured in the same camera session):

```json
{
  "embeddings": {
    "images/corn_HB-25000SBC/220622-img1.png": [0.99, -0.01, "..."],
    "images/corn_HB-25000SBC/220622-img2.png": [0.98, -0.02, "..."],
    "images/corn_HB-25000SBC/220622-img3.png": [0.97,  0.03, "..."],
    "images/corn_HB-25000SBC/220622-img4.png": [0.96,  0.04, "..."]
  },
  "metadata": {
    "cluster_1": {
      "images": ["images/corn_HB-25000SBC/220622-img1.png", "images/corn_HB-25000SBC/220622-img2.png"],
      "class_name": "corn",
      "attributes": { "camera": "HB-25000SBC", "growth_stage": "low" }
    },
    "cluster_2": {
      "images": ["images/corn_HB-25000SBC/220622-img3.png", "images/corn_HB-25000SBC/220622-img4.png"],
      "class_name": "corn",
      "attributes": { "camera": "HB-25000SBC", "growth_stage": "medium" }
    }
  }
}
```

All four images live in the same `corn_HB-25000SBC/` directory and share `class_name: corn` — the group key (`cluster_1`, `cluster_2`) is just a unique identifier, not a directory name. Images within the same group are explicit positives of each other (grade 3). Querying a `low` image against a `medium` image yields grade 2 (same class, `camera` matches, `growth_stage` differs) — the attribute nDCG for `growth_stage` captures this distinction.

Each group defines its own explicit-positive set and `attributes` dict. The number of groups, their sizes, and the number of attribute keys are all open — KPIs automatically scale to however many attribute keys appear across all groups.

**Exact path matching** — each string in `images` must be the exact key used in the `embeddings` dictionary. The client assembling the JSON has full context over the paths, so no normalisation is performed server-side.

**`dataset_root`** — used only for Wiring 1 label extraction from paths. When `metadata` is provided, class labels come from `class_name` in each group; `dataset_root` has no effect on metadata matching.

**Unmatched images** — embedding paths with no entry in any metadata group fall back to path-extracted class labels and receive relevance grade 0 in the metadata-aware nDCG.

**Graded relevance** — when `metadata` is provided, the `knn_metadata_ndcg` metric uses a 4-level relevance scheme:

| Grade | Condition |
|---|---|
| `3` | Explicit positive — candidate is in the same metadata group as the query |
| `2` | Same `class_name` **and** all query attributes match the candidate |
| `1` | Same `class_name`, no attributes or at least one attribute differs |
| `0` | Self, different class, or no class information |

---

## Running the API server

```bash
# Default: dataset root = ./dataset, port 8000
pai-ag-emb

# Or explicitly
python -m pai.ag_emb.api.app --dataset-root /path/to/dataset --port 8000

# Interactive docs
open http://localhost:8000/docs
```

### POST `/v1/embeddings/evaluate`

Only `embeddings` is required. All other fields use server defaults.

**Wiring 1 — embeddings only:**

```json
{
  "embeddings": {
    "images/corn_HB-25000SBC/220622-img.png": [0.12, -0.31, 0.27, "..."],
    "images/soybean_anafi/210625-img.JPG":    [-0.45, 0.18, -0.09, "..."]
  },
  "dataset_root": "images",
  "k_values": [5, 10],
  "sample_pairs": 1000000
}
```

**Wiring 2 — embeddings + metadata:**

```json
{
  "embeddings": {
    "images/corn_HB-25000SBC/220622-img.png": [0.12, -0.31, "..."],
    "images/soybean_anafi/210625-img.JPG":    [-0.45, 0.18, "..."]
  },
  "metadata": {
    "corn_HB-25000SBC": { "images": ["images/corn_HB-25000SBC/220622-img.png"], "class_name": "corn",    "attributes": { "camera": "HB-25000SBC", "growth_stage": "medium" } },
    "soybean_anafi":    { "images": ["images/soybean_anafi/210625-img.JPG"],    "class_name": "soybean", "attributes": { "camera": "anafi",       "growth_stage": "medium" } }
  },
  "k_values": [5, 10]
}
```

**Optional request fields**

| Field | Default | Description |
|---|---|---|
| `metadata` | `null` | Similarity groups — enables metadata-aware graded nDCG. See [Data format](#data-format). |
| `dataset_root` | server default | Override dataset root for label extraction (ignored when `metadata` is provided) |
| `k_values` | `[5, 10, 20]` | K cutoffs for KNN metrics |
| `sample_pairs` | `1 000 000` | Pair budget for pairwise stats; `null` = exact |

---

## Python / CLI usage

```bash
# Embeddings only — tight scenario (well-separated classes)
python examples/example.py

# Embeddings only — sparse scenario (overlapping classes, degraded KPIs)
python examples/example.py --scenario sparse

# With metadata — enables graded nDCG and per-attribute KPIs
python examples/example.py --scenario tight_meta
python examples/example.py --scenario sparse_meta

# With options
python examples/example.py \
  --scenario tight_meta \
  --k-values 5 10 20 \
  --sample-pairs 500000 \
  --output-dir output \
  --tsne-dimensions 3
```

```python
from pai.ag_emb.services.evaluate import run_evaluation
from pai.ag_emb.services.reporting import print_result
from pai.ag_emb.schemas.evaluate import MetadataGroup

embeddings = {
    "images/corn_HB-25000SBC/img1.png":  [...],  # L2-normalised float32
    "images/soybean_anafi/img2.JPG":     [...],
}

# Wiring 1 — embeddings only (class inferred from subgroup folder)
result = run_evaluation(
    image_embeddings=embeddings,
    k_values=[5, 10],
    dataset_root="images",
    sample_pairs=None,
)

# Wiring 2 — embeddings + metadata (graded nDCG, explicit positives)
metadata = {
    "corn_HB-25000SBC": MetadataGroup(
        images=["images/corn_HB-25000SBC/img1.png"],
        class_name="corn",
        attributes={"camera": "HB-25000SBC", "growth_stage": "medium"},
    ),
    "soybean_anafi": MetadataGroup(
        images=["images/soybean_anafi/img2.JPG"],
        class_name="soybean",
        attributes={"camera": "anafi", "growth_stage": "medium"},
    ),
}
result = run_evaluation(
    image_embeddings=embeddings,
    k_values=[5, 10],
    dataset_root=None,
    sample_pairs=None,
    metadata=metadata,
)

print_result(result)
```

---

## Metrics

When metadata is provided, retrieval KPIs use explicit-positive ground truth from your declared similarity groups. Without metadata, class labels inferred from the path structure are used — useful for a quick sanity check but less precise. Geometry and neighbour diagnostics are always computed regardless of whether metadata is supplied.

### Quick reference — what each metric means

| Metric | Description | Better when |
|---|---|---|
| `pairwise_similarity_stats` | How similar images are to each other across the whole dataset (mean, std, percentiles). A high mean means embeddings are generally close; a low std means they are consistently so. | Context-dependent — high mean with high gap (below) is ideal |
| `centroid_similarity_stats` | How close each embedding sits to the dataset's "centre of mass." Very high uniformity here can signal the space is lop-sided rather than well spread. | Lower (more spread from centre) |
| `intra_inter_similarity_gap` | Same-class similarity minus different-class similarity. Tells you how much more similar images of the same crop are compared to images of different crops. | **Higher** — a large positive gap means the model clearly separates crops |
| `effective_rank` | How many of the model's embedding dimensions are meaningfully used. A model that squashes everything into a few dimensions wastes capacity. | **Higher** — more expressive use of the embedding space |
| `uniformity` | How evenly embeddings are spread across the space (Wang & Isola 2020). A clumped space means many images share the same neighbourhood. | **More negative** — embeddings should cover the space, not all pile up |
| `hubness` | Whether a small number of images are disproportionately the nearest neighbour of everyone else. High hubness is a geometry warning sign. | **Lower** mean, std, and Gini coefficient |
| `knn_radius@k` | How far (in embedding distance) you need to reach to find k neighbours. Useful for tuning retrieval thresholds. | Context-dependent |
| `mean_top_k_sim@k` | Average cosine similarity to your k nearest neighbours. Higher means tighter local clusters. | **Higher** for well-separated classes |
| `outlier_score@k` | How isolated each image is from its neighbours. High scores flag images that don't fit neatly with anything else. | **Lower** for a coherent dataset |
| `knn_label_purity@k` | Of the k nearest neighbours for each image, what fraction share the same crop class? 1.0 = every neighbour is the right crop. | **Higher** — closer to 1.0 is better |
| `knn_label_ndcg@k` | Are same-class images ranked near the top of the neighbour list? Penalises correct matches that appear late in the ranking. | **Higher** — 1.0 is perfect ranking |
| `knn_map@k` | Can the model consistently find *all* images of the same class, not just the first few? | **Higher** |
| `knn_label_mrr@k` | On average, how far down the ranked list is the first correct (same-class) match? MRR of 1.0 means it is always the very first result. | **Higher** |
| `knn_label_r_precision` | Precision when you retrieve exactly as many results as there are images of that class. | **Higher** |
| `alignment` | How close explicitly declared positives are to each other in embedding space. 0 = perfectly aligned; only available when metadata is provided. | **Lower** |
| `knn_metadata_precision@k` | Of the top-k results, what fraction are declared explicit positives (from your metadata groups)? | **Higher** |
| `knn_metadata_ndcg@k` | Graded ranking quality using the 0–3 relevance scale (explicit positive > same class + attributes > same class > different class). | **Higher** — 1.0 is perfect |
| `knn_metadata_map@k` | How consistently does the model surface *all* explicit positives early in the ranked list? | **Higher** |
| `knn_metadata_mrr@k` | How far down before the first explicit positive appears? | **Higher** |
| `knn_metadata_r_precision` | Precision at R, where R equals the number of explicit positives declared for that image. | **Higher** |
| `knn_attribute_ndcg@k` | One score per metadata attribute (e.g. `camera`, `growth_stage`). Measures whether images sharing the same attribute value are ranked above those that differ on it. | **Higher** per attribute |

---

### Always present

| Metric | Description |
|---|---|
| `pairwise_similarity_stats` | Mean / std / percentiles of all pairwise cosine similarities |
| `centroid_similarity_stats` | Cosine similarity of each embedding to the dataset centroid (anisotropy signal) |
| `intra_inter_similarity_gap` | Mean intra-class cosine − mean inter-class cosine |
| `effective_rank` | Participation-ratio effective dimensionality of the embedding space |
| `uniformity` | Wang & Isola (2020) uniformity — `log E[exp(-t‖u−v‖²)]`; more negative = better spread |
| `hubness` | K-occurrence statistics: mean, std, Gini coefficient, top hubs |
| `knn_radius@k` | Mean distance to the k-th nearest neighbour (search radius) |
| `mean_top_k_sim@k` | Mean cosine similarity to the k nearest neighbours |
| `outlier_score@k` | Per-item outlier signal based on neighbour distances |

### Without metadata

| Metric | Description |
|---|---|
| `knn_label_purity@k` | Fraction of each item's k nearest neighbours sharing its path-inferred class label |
| `knn_label_ndcg@k` | nDCG with binary relevance — rewards same-class neighbours ranked at the top |
| `knn_map@k` | MAP — rewards finding *all* same-class neighbours early |
| `knn_label_mrr@k` | **MRR** — reciprocal rank of the first same-class neighbour |
| `knn_label_r_precision` | **R-Precision** — precision at R where R = class size − 1 |

### With metadata

| Metric | Description |
|---|---|
| `alignment` | Wang & Isola (2020) alignment — mean `‖u−v‖²` over explicit positive pairs; lower = better |
| `knn_metadata_precision@k` | **P@K** — fraction of top-k retrieved items that are explicit positives |
| `knn_metadata_ndcg@k` | **nDCG@K** — graded relevance (0–3) by explicit group, class, and attributes |
| `knn_metadata_map@k` | **MAP@K** — average precision at finding all explicit positives early |
| `knn_metadata_mrr@k` | **MRR@K** — reciprocal rank of the first explicit positive |
| `knn_metadata_r_precision` | **R-Precision** — precision at R where R = number of explicit positives |
| `knn_attribute_ndcg@k` | One binary nDCG entry **per attribute key** (e.g. `growth_stage`, `camera`) — only present when at least one metadata group has a non-empty `attributes` dict |

> **Graded relevance for nDCG** — grade 3: same explicit group; grade 2: same class + all attributes match; grade 1: same class; grade 0: different class or no match.

### Group analysis (with metadata only)

`group_analysis` is a top-level key containing one entry per declared metadata group. It is a **diagnostic**, not a KPI — it flags groups whose embeddings may be too coarse for reliable retrieval benchmarking.

| Field | Description |
|---|---|
| `n_images` | Number of images in the group |
| `mean_intra_cosine` | Mean pairwise cosine similarity within the group |
| `std_intra_cosine` | Std of pairwise cosine similarities within the group |
| `cluster_count` | Number of stable sub-clusters detected by HDBSCAN (requires `scikit-learn>=1.3`) |
| `noise_count` | Number of images not assigned to any sub-cluster |
| `silhouette_score` | Mean silhouette score of the detected sub-clusters |
| `suggested_split` | `true` when `cluster_count > 1` and `silhouette_score > 0.25` |

### Per-class

Each class reports geometry metrics (`pairwise_similarity_stats`, `centroid_similarity_stats`, `effective_rank`) plus the same retrieval KPI tier sliced to that class — label-based without metadata (`knn_label_purity`, `knn_label_ndcg`, `knn_map`, `knn_label_mrr`, `knn_label_r_precision`), metadata-based with metadata (`knn_metadata_precision`, `knn_metadata_ndcg`, `knn_metadata_map`, `knn_metadata_mrr`, `knn_metadata_r_precision`).

---

## Visualizations

Generated by `examples/example.py --output-dir output` (or called directly from Python):

| File | Function | Description |
|---|---|---|
| `class_confusion_matrix.html` | `plot_knn_confusion` | KNN confusion matrix — which classes get misidentified as which |
| `cosine_similarity.html` | `plot_cosine_similarity` | Full N×N pairwise cosine similarity heatmap, sorted by class |
| `tsne.html` | `plot_tsne` | Rotatable 3D t-SNE scatter coloured by class |
| `lle.html` | `plot_lle` | Rotatable 3D LLE scatter — preserves local neighbourhood structure |

All outputs are interactive Plotly HTML. Static PNG export requires `pip install kaleido`.

```python
from pai.ag_emb.services.reporting import (
    plot_knn_confusion,
    plot_cosine_similarity,
    plot_tsne,
    plot_lle,
)

plot_knn_confusion(result, k=10, output_path="output/confusion.html")
plot_cosine_similarity(embeddings, result, output_path="output/cosine.html")
plot_tsne(embeddings, result, output_path="output/tsne.html", dimensions=3)
plot_lle(embeddings, result, output_path="output/lle.html")
```

---

## Example notebooks

Two notebooks are provided — start Jupyter from the `examples/` directory for both.

**[`examples/example.ipynb`](examples/example.ipynb)** — embeddings only (`emb_tight.json` / `emb_sparse.json`).
Covers KNN purity, nDCG, MAP, pairwise similarity, and all visualizations.

**[`examples/example_meta.ipynb`](examples/example_meta.ipynb)** — embeddings + metadata (`emb_tight_meta.json` / `emb_sparse_meta.json`).
Adds graded `knn_metadata_ndcg` and per-attribute `knn_attribute_ndcg` (one entry per attribute key).

```bash
# From the project root
jupyter notebook examples/example.ipynb
jupyter notebook examples/example_meta.ipynb
```

---

## Development

```bash
# Run tests
python -m pytest tests/

# Start API with live reload
pai-ag-emb --no-reload  # disable reload for production
```

Environment variable override for dataset root:

```bash
PAI_DATASET_ROOT=/data/crops pai-ag-emb
```
