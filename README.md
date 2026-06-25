# Precision AI Agricultural Embedding Evaluation API

Agricultural embedding evaluation API for Precision AI. Measures how well image model embeddings support three retrieval wirings using KNN-based metrics, geometry diagnostics, and interactive visualizations.

| Wiring | Endpoint | Query type | Retrieval target |
|---|---|---|---|
| **Image→Image** | `/v1/embeddings/evaluate/image2image` | Full field image | Full field image |
| **Plant→Image** | `/v1/embeddings/evaluate/plant2image` | Instance crop | Full field image |
| **Plant→Plant** | `/v1/embeddings/evaluate/plant2plant` | Instance crop | Instance crop |

[![Coverage](.github/badges/coverage.svg)](https://github.com/Precision-AI-Inc/precisionai-agrieval-emb/actions/workflows/workflow.yml) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)

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
  A1/
    220622-img.png
    220622-img2.png
  A2/
    190627-img.JPG
  B1/
    220608-img.png
  B2/
    210625-img.JPG
metadata/
  A1/
    metadata.json
  A2/
    metadata.json
  B1/
    metadata.json
  B2/
    metadata.json
```

Directory names follow the **L1/L2 cluster convention**: L2 folders are letter+number identifiers (e.g. `A1`, `A2`, `B1`). The L1 cluster label is the leading letter(s) of the L2 name — `A1` and `A2` both belong to L1 cluster `A`. Both `images/` and `metadata/` mirror the same L2 folder structure.

Each `metadata.json` identifies which images belong to the group and carries shared metadata for all of them:

```json
{
  "class_name": "A1",
  "attributes": { "class_instances": ["Crop | Corn"], "camera_source": "HB-25000SBC", "time_period": "ss" },
  "images": [
    "images/A1/220622-img1.png",
    "images/A1/220622-img2.png"
  ]
}
```

`class_name` is the cluster identifier (same as the folder name, e.g. `A1`). The leading letters form the coarse L1 class label (`A1` → `A`), shared by all subgroups with the same plant-species composition.

The `images` list uses the same path strings that will be used as embedding keys in the API request. All images listed in the same `metadata.json` are considered explicit positives of each other.

**Supported image extensions (case-sensitive):** `.jpg`, `.JPG`, `.jpeg`, `.JPEG`, `.png`, `.PNG`. Requests with any other extension are rejected with a validation error listing the offending paths.

---

### Wiring 1 — embeddings only

Pass a dict of `image_path → embedding`. The crop class label is inferred automatically from the L2 folder name using `dataset_root` as the strip prefix — the leading letters of the L2 folder become the L1 class label.

**Path convention** — `{dataset_root}/{L2}/{image}`:

```
images/
  A1/220622-img.png    →  class: A  (leading letters of "A1")
  A2/190627-img.JPG   →  class: A  (leading letters of "A2")
  B2/210625-img.JPG   →  class: B  (leading letters of "B2")
```

`dataset_root` should point to the `images/` directory (direct parent of L2 folders). Unbalanced subgroup sizes are expected and accepted.

**Embedding vector requirements:**

- Flat 1-D array of length `embedding_dim`
- **L2-normalised float32** — `‖v‖₂ = 1.0 ± 0.001`
- All vectors must share the same length
- Minimum 2 embeddings per request

---

### Wiring 2 — embeddings + metadata

Add a `metadata` field alongside `embeddings`. Each entry in `metadata` is a cluster group of images sharing a `class_name` and optional attributes. Use the cluster name as the group key.

```json
{
  "embeddings": {
    "images/A1/220622-img.png":  [0.12, -0.31, "..."],
    "images/A2/190627-img.JPG":  [0.11, -0.29, "..."],
    "images/B1/220608-img.png":  [-0.45, 0.18, "..."],
    "images/B2/210625-img.JPG":  [-0.44, 0.17, "..."]
  },
  "metadata": {
    "A1": {
      "class_name": "A1",
      "images": ["images/A1/220622-img.png"],
      "attributes": { "class_instances": ["Crop | Corn"], "camera_source": "HB-25000SBC", "time_period": "ss" }
    },
    "A2": {
      "class_name": "A2",
      "images": ["images/A2/190627-img.JPG"],
      "attributes": { "class_instances": ["Crop | Corn"], "camera_source": "nikon_d610", "time_period": "ss" }
    },
    "B1": {
      "class_name": "B1",
      "images": ["images/B1/220608-img.png"],
      "attributes": { "class_instances": ["Crop | Soybean"], "camera_source": "HB-25000SBC", "time_period": "ss" }
    },
    "B2": {
      "class_name": "B2",
      "images": ["images/B2/210625-img.JPG"],
      "attributes": { "class_instances": ["Crop | Soybean"], "camera_source": "anafi", "time_period": "ss" }
    }
  }
}
```

**`metadata` field contract:**

| Key | Type | Required | Description |
|---|---|---|---|
| `images` | `list[str]` | yes | Exact embedding path keys — must match the keys used in `embeddings`. |
| `class_name` | `str` | **yes** | Cluster identifier (e.g. `"A1"`). The L1 class label is derived from its leading letters (`"A1"` → `"A"`). All images in this group are treated as explicit positives (grade 3). |
| `attributes` | `dict[str, Any]` | no | Open key-value pairs shared by all images in this group. Values may be strings, lists, or `null`. Common examples: `class_instances`, `camera_source`, `time_period`, `collecting_date`. One `knn_attribute_ndcg` metric is produced **per attribute key** present across all groups. |

**Group key** — use the cluster name (e.g. `A1`) as a convention; any unique string is accepted.

**Multiple groups per L1 class** — all groups whose `class_name` shares the same leading letters belong to the same coarse class. Images within the same group are explicit positives of each other (grade 3); images in different groups of the same L1 class score grade 2; images from a different L1 class score grade 1 if their `class_instances` lists overlap, or grade 0 otherwise:

```json
{
  "embeddings": {
    "images/A1/220622-img1.png": [0.99, -0.01, "..."],
    "images/A1/220622-img2.png": [0.98, -0.02, "..."],
    "images/A2/190627-img1.JPG": [0.97,  0.03, "..."],
    "images/A2/190627-img2.JPG": [0.96,  0.04, "..."]
  },
  "metadata": {
    "A1": {
      "class_name": "A1",
      "images": ["images/A1/220622-img1.png", "images/A1/220622-img2.png"],
      "attributes": { "class_instances": ["Crop | Corn"], "camera_source": "HB-25000SBC", "time_period": "ss" }
    },
    "A2": {
      "class_name": "A2",
      "images": ["images/A2/190627-img1.JPG", "images/A2/190627-img2.JPG"],
      "attributes": { "class_instances": ["Crop | Corn"], "camera_source": "nikon_d610", "time_period": "ss" }
    }
  }
}
```

Both groups share L1 class `A` (derived from their `class_name` leading letters) — the group key (`A1`, `A2`) is the cluster identifier. Images within the same group are explicit positives of each other (grade 3). An A1 image queried against an A2 image yields grade 2 (same L1 class). Images from a different L1 class score grade 1 if their `class_instances` lists share at least one entry, or grade 0 otherwise.

Each group defines its own explicit-positive set and `attributes` dict. The number of groups, their sizes, and the number of attribute keys are all open — KPIs automatically scale to however many attribute keys appear across all groups.

**Exact path matching** — each string in `images` must be the exact key used in the `embeddings` dictionary. The client assembling the JSON has full context over the paths, so no normalisation is performed server-side.

**`dataset_root`** — used only for Wiring 1 label extraction from paths. When `metadata` is provided, class labels come from `class_name` in each group (L1 is derived from its leading letters); `dataset_root` has no effect on metadata matching.

**Unmatched images** — embedding paths with no entry in any metadata group fall back to path-extracted class labels and receive relevance grade 0 in the metadata-aware nDCG.

**Graded relevance** — when `metadata` is provided, the `knn_metadata_ndcg` metric uses a 4-level relevance scheme:

| Grade | Condition |
|---|---|
| `3` | Explicit positive — candidate is in the same L2 group as the query |
| `2` | Same L1 class (different L2 group) |
| `1` | Different L1 class, but at least one `class_instances` value overlaps |
| `0` | Different class with no `class_instances` overlap, or no class information |

---

### Wiring 3 — Plant→Image

Pass a **mixed corpus** of full-field image embeddings and instance crop embeddings together with an `instance_to_image` mapping that declares which crops were taken from which parent image.

```json
{
  "embeddings": {
    "images/A1/field001.png":   [...],
    "images/A1/field001-0.png": [...],
    "images/A1/field001-1.png": [...],
    "images/B1/field002.png":   [...],
    "images/B1/field002-0.png": [...]
  },
  "instance_to_image": {
    "images/A1/field001.png": [
      "images/A1/field001-0.png",
      "images/A1/field001-1.png"
    ],
    "images/B1/field002.png": [
      "images/B1/field002-0.png"
    ]
  },
  "k_values": [5, 10]
}
```

**Ground truth:** when an instance is the query, its parent full image is grade-3; when a full image is the query, all its instances are grade-3. Different parent groups under the same class folder are grade-2 matches, and different class folders are grade-0.

Instance paths must follow the naming scheme `{original_image_name}-{ID}{ext}`, where `ID` is a zero-based integer counter or a short UUID suffix.

---

### Wiring 4 — Plant→Plant

Pass **instance crop embeddings only** with an `instance_labels` mapping that assigns each instance its class label. All instances sharing the same label are mutual grade-3 positives.

```json
{
  "embeddings": {
    "images/A1/inst-0.png": [...],
    "images/A1/inst-1.png": [...],
    "images/B1/inst-2.png": [...]
  },
  "instance_labels": {
    "images/A1/inst-0.png": "A1",
    "images/A1/inst-1.png": "A1",
    "images/B1/inst-2.png": "B1"
  },
  "k_values": [5, 10]
}
```

Every embedding key must have a label and every label key must exist in `embeddings` (complete labelling is enforced).

---

## Running the API server

```bash
# Default: dataset root = ./dataset, port 8000
precisionai-agrieval-emb

# Or explicitly
python -m precisionai.agrieval.emb.api.app --dataset-root /path/to/dataset --port 8000

# Interactive docs
open http://localhost:8000/docs
```

### Endpoints

| Method | Path | Wiring |
|---|---|---|
| `POST` | `/v1/embeddings/evaluate/image2image` | Image→Image (Wiring 1 / 2) |
| `POST` | `/v1/embeddings/evaluate/plant2image` | Plant→Image (Wiring 3) |
| `POST` | `/v1/embeddings/evaluate/plant2plant` | Plant→Plant (Wiring 4) |

### POST `/v1/embeddings/evaluate/image2image`

Only `embeddings` is required. All other fields use server defaults.

**Wiring 1 — embeddings only:**

```json
{
  "embeddings": {
    "images/A1/220622-img.png": [0.12, -0.31, 0.27, "..."],
    "images/B2/210625-img.JPG": [-0.45, 0.18, -0.09, "..."]
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
    "images/A1/220622-img.png": [0.12, -0.31, "..."],
    "images/B2/210625-img.JPG": [-0.45, 0.18, "..."]
  },
  "metadata": {
    "A1": { "class_name": "A1", "images": ["images/A1/220622-img.png"], "attributes": { "class_instances": ["Crop | Corn"], "camera_source": "HB-25000SBC", "time_period": "ss" } },
    "B2": { "class_name": "B2", "images": ["images/B2/210625-img.JPG"], "attributes": { "class_instances": ["Crop | Soybean"], "camera_source": "anafi", "time_period": "ss" } }
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
# Image→Image — embeddings only
python examples/example.py

# Image→Image — with metadata (graded nDCG, per-attribute KPIs)
python examples/example.py --mode image2image_meta

# Plant→Image — mixed corpus with instance_to_image mapping
python examples/example.py --mode plant2image

# Plant→Plant — instance crops with class labels
python examples/example.py --mode plant2plant

# With options
python examples/example.py \
  --mode image2image_meta \
  --k-values 5 10 20 \
  --sample-pairs 500000 \
  --output-dir output \
  --tsne-dimensions 3
```

```python
from precisionai.agrieval.emb.services.evaluate import (
    run_image2image_eval,
    run_plant2image_eval,
    run_plant2plant_eval,
)
from precisionai.agrieval.emb.services.reporting import print_result
from precisionai.agrieval.emb.schemas.evaluate import MetadataGroup

embeddings = {
    "images/A1/img1.png":  [...],  # L2-normalised float32
    "images/B2/img2.JPG":  [...],
}

# Wiring 1 — Image→Image, L1 class inferred from L2 folder name
result = run_image2image_eval(
    image_embeddings=embeddings,
    k_values=[5, 10],
    dataset_root="images",
    sample_pairs=None,
)

# Wiring 2 — Image→Image with explicit groups (graded nDCG, explicit positives)
metadata = {
    "A1": MetadataGroup.model_validate({
        "class_name": "A1",
        "images": ["images/A1/img1.png"],
        "attributes": {"class_instances": ["Crop | Corn"], "camera_source": "HB-25000SBC", "time_period": "ss"},
    }),
    "B2": MetadataGroup.model_validate({
        "class_name": "B2",
        "images": ["images/B2/img2.JPG"],
        "attributes": {"class_instances": ["Crop | Soybean"], "camera_source": "anafi", "time_period": "ss"},
    }),
}
result = run_image2image_eval(
    image_embeddings=embeddings,
    k_values=[5, 10],
    dataset_root=None,
    sample_pairs=None,
    metadata=metadata,
)

# Wiring 3 — Plant→Image (mixed corpus of full images and instance crops)
mixed_embeddings = {
    "images/A1/field001.png":   [...],
    "images/A1/field001-0.png": [...],
    "images/A1/field001-1.png": [...],
    "images/B1/field002.png":   [...],
    "images/B1/field002-0.png": [...],
}
result = run_plant2image_eval(
    embeddings=mixed_embeddings,
    instance_to_image={
        "images/A1/field001.png": ["images/A1/field001-0.png", "images/A1/field001-1.png"],
        "images/B1/field002.png": ["images/B1/field002-0.png"],
    },
    k_values=[5, 10],
    dataset_root=None,
    sample_pairs=None,
)

# Wiring 4 — Plant→Plant (instance crops with explicit class labels)
instance_embeddings = {
    "images/A1/inst-0.png": [...],
    "images/A1/inst-1.png": [...],
    "images/B1/inst-2.png": [...],
}
result = run_plant2plant_eval(
    embeddings=instance_embeddings,
    instance_labels={
        "images/A1/inst-0.png": "A1",
        "images/A1/inst-1.png": "A1",
        "images/B1/inst-2.png": "B1",
    },
    k_values=[5, 10],
    sample_pairs=None,
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
| `knn_attribute_ndcg@k` | One score per metadata attribute (e.g. `class_instances`, `camera_source`). Measures whether images sharing the same attribute value are ranked above those that differ on it. | **Higher** per attribute |

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
| `knn_attribute_ndcg@k` | One binary nDCG entry **per attribute key** (e.g. `class_instances`, `camera_source`) — only present when at least one metadata group has a non-empty `attributes` dict |

> **Graded relevance for nDCG** — grade 3: same L2 group (explicit positive); grade 2: same L1 class; grade 1: different L1 but overlapping `class_instances`; grade 0: no relationship.

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
from precisionai.agrieval.emb.services.reporting import (
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

Two notebooks are provided — start Jupyter from the `examples/` directory.

**[`examples/example.ipynb`](examples/example.ipynb)** — Image→Image, two parts.
- **Part 1 — Basic** (`emb_image2image.json`): class labels inferred from the L2 path hierarchy. Covers KNN purity, nDCG, MAP, pairwise similarity, and all visualizations.
- **Part 2 — With Metadata** (`emb_image2image_meta.json`): explicit `MetadataGroup` clusters (A1, A2, B1, B2). Adds graded `knn_metadata_ndcg` and per-attribute `knn_attribute_ndcg`.

**[`examples/example_plant.ipynb`](examples/example_plant.ipynb)** — Plant wirings (`emb_plant2image.json` / `emb_plant2plant.json`).
Covers Plant→Image with `instance_to_image` mapping and Plant→Plant with `instance_labels`.

```bash
# From the project root
jupyter notebook examples/example.ipynb
jupyter notebook examples/example_plant.ipynb
```

---

## Development

```bash
# Run tests
python -m pytest tests/

# Start API (no live reload by default; add --reload for development)
precisionai-agrieval-emb
```

Environment variable override for dataset root:

```bash
PAI_DATASET_ROOT=/data/crops precisionai-agrieval-emb
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE.md](LICENSE.md).
