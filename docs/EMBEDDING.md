# Embedding Evaluation — `precisionai.agrieval.emb`

KNN-based retrieval benchmarking for image model embeddings. Measures how well embeddings support four retrieval wirings using purity, nDCG, MAP, MRR, geometry diagnostics, and interactive visualizations.

| Wiring | Endpoint | Query type | Retrieval target |
|---|---|---|---|
| **Image→Image** | `/v1/embeddings/evaluate/image2image` | Full field image | Full field image |
| **Plant→Image** | `/v1/embeddings/evaluate/plant2image` | Instance crop | Full field image |
| **Plant→Plant** | `/v1/embeddings/evaluate/plant2plant` | Instance crop | Instance crop |

---

## Data format

### Canonical directory layout

```
images/
  A1/
    pai-abc123.png
    pai-def456.png
  A2/
    pai-ghi789.png
  D1/
    pai-jkl012.png
  D2/
    pai-mno345.png
instances/
  A1/
    pai-abc123-1.png
    pai-abc123-2.png
  D1/
    pai-jkl012-1.png
image2image.json
plant2image.json
plant2plant.json
```

Directory names follow the **L1/L2 cluster convention**: L2 folders are letter+number identifiers (e.g. `A1`, `A2`, `D1`). The L1 cluster label is the leading letter(s) of the L2 name — `A1` and `A2` both belong to L1 cluster `A`.

Committed image fixtures should be downsampled/compressed before commit. Test image fixtures use `1332x540` optimized RGB PNGs and must stay at or below **1800 KB per file**, enforced for newly added files by pre-commit.

`image2image.json` identifies which images belong to each cluster and carries shared metadata:

```json
{
  "metadata": {
    "A1": {
      "class_name": "A1",
      "attributes": { "plants": ["Crop | Soybean", "Weed | Weed"], "time_period": "ss_sr", "camera_model": "HB-25000SBC", "camera_source": "drone", "collecting_date": null },
      "images": ["images/A1/pai-abc123.png", "images/A1/pai-def456.png"]
    }
  }
}
```

`plant2image.json` maps each parent full-field image to its instance crops:

```json
{
  "instance_to_image": {
    "images/A1/pai-abc123.png": ["instances/A1/pai-abc123-1.png", "instances/A1/pai-abc123-2.png"]
  }
}
```

`plant2plant.json` assigns a species label to each instance crop:

```json
{
  "instance_labels": {
    "instances/A1/pai-abc123-1.png": "Crop | Soybean",
    "instances/A1/pai-abc123-2.png": "Weed | Weed"
  }
}
```

`class_name` is the cluster identifier (e.g. `A1`). The leading letters form the coarse L1 class label (`A1` → `A`), shared by all subgroups with the same plant-species composition.

The `images` list uses the same path strings that will be used as embedding keys in the API request. All images listed in the same cluster group are considered explicit positives of each other.

**Supported image extensions (case-sensitive):** `.jpg`, `.JPG`, `.jpeg`, `.JPEG`, `.png`, `.PNG`. Requests with any other extension are rejected with a validation error listing the offending paths.

---

### Wiring 1 — embeddings only

Pass a dict of `image_path → embedding`. The crop class label is inferred automatically from the L2 folder name using `dataset_root` as the strip prefix — the leading letters of the L2 folder become the L1 class label.

**Path convention** — `{dataset_root}/{L2}/{image}`:

```
images/
  A1/220622-img.png    →  class: A  (leading letters of "A1")
  A2/190627-img.JPG   →  class: A  (leading letters of "A2")
  D2/210625-img.JPG   →  class: D  (leading letters of "D2")
```

`dataset_root` should point to the `images/` directory (direct parent of L2 folders). Unbalanced subgroup sizes are expected and accepted.

**Embedding vector requirements:**

- Flat 1-D array of length `embedding_dim`
- **L2-normalized float32** — `‖v‖₂ = 1.0 ± 0.001`
- All vectors must share the same length
- Between 2 and 50,000 embeddings per request (all three wirings — sized for
  the ~50k-image datasets this service targets; also bounds the O(n²) work
  done downstream)

---

### Wiring 2 — embeddings + metadata

Add a `metadata` field alongside `embeddings`. Each entry in `metadata` is a cluster group of images sharing a `class_name` and optional attributes. Use the cluster name as the group key.

```json
{
  "embeddings": {
    "images/A1/220622-img.png":  [0.12, -0.31, "..."],
    "images/A2/190627-img.JPG":  [0.11, -0.29, "..."],
    "images/D1/220608-img.png":  [-0.45, 0.18, "..."],
    "images/D2/210625-img.JPG":  [-0.44, 0.17, "..."]
  },
  "metadata": {
    "A1": {
      "class_name": "A1",
      "images": ["images/A1/220622-img.png"],
      "attributes": { "plants": ["Crop | Corn"], "time_period": "ss", "camera_model": "HB-25000SBC", "camera_source": "drone", "collecting_date": null }
    },
    "A2": {
      "class_name": "A2",
      "images": ["images/A2/190627-img.JPG"],
      "attributes": { "plants": ["Crop | Corn"], "time_period": "ss", "camera_model": "nikon d610", "camera_source": "ground", "collecting_date": null }
    },
    "D1": {
      "class_name": "D1",
      "images": ["images/D1/220608-img.png"],
      "attributes": { "plants": ["Crop | Soybean"], "time_period": "ss", "camera_model": "HB-25000SBC", "camera_source": "drone", "collecting_date": null }
    },
    "D2": {
      "class_name": "D2",
      "images": ["images/D2/210625-img.JPG"],
      "attributes": { "plants": ["Crop | Soybean"], "time_period": "ss", "camera_model": "Anafi", "camera_source": "drone", "collecting_date": null }
    }
  }
}
```

**`metadata` field contract:**

| Key | Type | Required | Description |
|---|---|---|---|
| `images` | `list[str]` | yes | Exact embedding path keys — must match the keys used in `embeddings`. |
| `class_name` | `str` | **yes** | Cluster identifier (e.g. `"A1"`). The L1 class label is derived from its leading letters (`"A1"` → `"A"`). All images in this group are treated as explicit positives (grade 3). |
| `attributes` | `dict[str, Any]` | no | Open key-value pairs shared by all images in this group. Values may be strings, lists, or `null`. Common examples: `plants`, `time_period`, `camera_model`, `camera_source`, `collecting_date`. One `knn_attribute_ndcg` metric is produced **per attribute key** present across all groups. |

**Group key** — use the cluster name (e.g. `A1`) as a convention; any unique string is accepted.

**Multiple groups per L1 class** — all groups whose `class_name` shares the same leading letters belong to the same coarse class. Images within the same group are explicit positives of each other (grade 3); images in different groups of the same L1 class score grade 2; images from a different L1 class score grade 1 if their `plants` lists overlap, or grade 0 otherwise.

**Exact path matching** — each string in `images` must be the exact key used in the `embeddings` dictionary. No normalization is performed server-side.

**`dataset_root`** — used only for Wiring 1 label extraction from paths. When `metadata` is provided, class labels come from `class_name` in each group; `dataset_root` has no effect on metadata matching.

**Unmatched images** — embedding paths with no entry in any metadata group fall back to path-extracted class labels and receive relevance grade 0 in the metadata-aware nDCG.

**Graded relevance** — when `metadata` is provided, the `knn_metadata_ndcg` metric uses a 4-level relevance scheme:

| Grade | Condition |
|---|---|
| `3` | Explicit positive — candidate is in the same L2 group as the query |
| `2` | Same L1 class (different L2 group) |
| `1` | Different L1 class, but at least one `plants` value overlaps |
| `0` | Different class with no `plants` overlap, or no class information |

---

### Wiring 3 — Plant→Image

Pass a **mixed corpus** of full-field image embeddings and instance crop embeddings together with an `instance_to_image` mapping that declares which crops were taken from which parent image.

```json
{
  "embeddings": {
    "images/A1/field001.png":   [...],
    "images/A1/field001-0.png": [...],
    "images/A1/field001-1.png": [...],
    "images/D1/field002.png":   [...],
    "images/D1/field002-0.png": [...]
  },
  "instance_to_image": {
    "images/A1/field001.png": [
      "images/A1/field001-0.png",
      "images/A1/field001-1.png"
    ],
    "images/D1/field002.png": [
      "images/D1/field002-0.png"
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
    "images/D1/inst-2.png": [...]
  },
  "instance_labels": {
    "images/A1/inst-0.png": "A1",
    "images/A1/inst-1.png": "A1",
    "images/D1/inst-2.png": "D1"
  },
  "k_values": [5, 10]
}
```

Every embedding key must have a label and every label key must exist in `embeddings` (complete labeling is enforced).

---

## Running the API server

```bash
# Unified server — embedding + segmentation on one port (default 8000)
precisionai-agrieval-api --dataset-root /path/to/dataset

# Embedding-only server (legacy entry point)
precisionai-agrieval

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
    "images/D2/210625-img.JPG": [-0.45, 0.18, -0.09, "..."]
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
    "images/D2/210625-img.JPG": [-0.45, 0.18, "..."]
  },
  "metadata": {
    "A1": { "class_name": "A1", "images": ["images/A1/220622-img.png"], "attributes": { "plants": ["Crop | Corn"], "time_period": "ss", "camera_model": "HB-25000SBC", "camera_source": "drone", "collecting_date": null } },
    "D2": { "class_name": "D2", "images": ["images/D2/210625-img.JPG"], "attributes": { "plants": ["Crop | Soybean"], "time_period": "ss", "camera_model": "Anafi", "camera_source": "drone", "collecting_date": null } }
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
| `sample_pairs` | `1 000 000` | Pair budget for pairwise stats; `null` = exact. Capped at `5,000,000` — a sampling budget sized independently of the embeddings ceiling above, since further precision beyond that isn't worth the added compute |

---

## Python / CLI usage

```bash
# Image→Image — embeddings only
python examples/emb/example.py

# Image→Image — with metadata (graded nDCG, per-attribute KPIs)
python examples/emb/example.py --mode image2image_meta

# Plant→Image — mixed corpus with instance_to_image mapping
python examples/emb/example.py --mode plant2image

# Plant→Plant — instance crops with class labels
python examples/emb/example.py --mode plant2plant

# With options
python examples/emb/example.py \
  --mode image2image_meta \
  --k-values 5 10 20 \
  --sample-pairs 500000 \
  --output-dir output \
  --tsne-dimensions 3
```

```python
from precisionai.agrieval.emb import (
    MetadataGroup,
    print_result,
    run_image2image_eval,
    run_plant2image_eval,
    run_plant2plant_eval,
)

embeddings = {
    "images/A1/img1.png":  [...],  # L2-normalized float32
    "images/D2/img2.JPG":  [...],
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
        "attributes": {"plants": ["Crop | Corn"], "time_period": "ss", "camera_model": "HB-25000SBC", "camera_source": "drone", "collecting_date": None},
    }),
    "D2": MetadataGroup.model_validate({
        "class_name": "D2",
        "images": ["images/D2/img2.JPG"],
        "attributes": {"plants": ["Crop | Soybean"], "time_period": "ss", "camera_model": "Anafi", "camera_source": "drone", "collecting_date": None},
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
    "images/D1/field002.png":   [...],
    "images/D1/field002-0.png": [...],
}
result = run_plant2image_eval(
    embeddings=mixed_embeddings,
    instance_to_image={
        "images/A1/field001.png": ["images/A1/field001-0.png", "images/A1/field001-1.png"],
        "images/D1/field002.png": ["images/D1/field002-0.png"],
    },
    k_values=[5, 10],
    dataset_root=None,
    sample_pairs=None,
)

# Wiring 4 — Plant→Plant (instance crops with explicit class labels)
instance_embeddings = {
    "images/A1/inst-0.png": [...],
    "images/A1/inst-1.png": [...],
    "images/D1/inst-2.png": [...],
}
result = run_plant2plant_eval(
    embeddings=instance_embeddings,
    instance_labels={
        "images/A1/inst-0.png": "A1",
        "images/A1/inst-1.png": "A1",
        "images/D1/inst-2.png": "D1",
    },
    k_values=[5, 10],
    sample_pairs=None,
)

print_result(result)
```

---

## Metrics

When metadata is provided, retrieval KPIs use explicit-positive ground truth from your declared similarity groups. Without metadata, class labels inferred from the path structure are used — useful for a quick sanity check but less precise. Geometry and neighbor diagnostics are always computed regardless of whether metadata is supplied.

### Quick reference

| Metric | Description | Better when |
|---|---|---|
| `pairwise_similarity_stats` | Mean / std / percentiles of all pairwise cosine similarities. | Context-dependent |
| `centroid_similarity_stats` | How close each embedding sits to the dataset's center of mass. | Lower (more spread) |
| `intra_inter_similarity_gap` | Same-class similarity minus different-class similarity. | **Higher** |
| `effective_rank` | How many embedding dimensions are meaningfully used. | **Higher** |
| `uniformity` | How evenly embeddings are spread across the space (Wang & Isola 2020). | **More negative** |
| `hubness` | Whether a small number of images are disproportionately the nearest neighbor of everyone else. | **Lower** |
| `knn_radius@k` | How far you need to reach to find k neighbors. | Context-dependent |
| `mean_top_k_sim@k` | Average cosine similarity to the k nearest neighbors. | **Higher** |
| `outlier_score@k` | How isolated each image is from its neighbors. | **Lower** |
| `knn_label_purity@k` | Fraction of k nearest neighbors sharing the same crop class. | **Higher** |
| `knn_label_ndcg@k` | Are same-class images ranked near the top? | **Higher** |
| `knn_map@k` | Can the model consistently find all images of the same class? | **Higher** |
| `knn_label_mrr@k` | How far down before the first correct match? | **Higher** |
| `knn_label_r_precision` | Precision at R, where R = class size. | **Higher** |
| `alignment` | How close explicit positives are in embedding space (metadata only). | **Lower** |
| `knn_metadata_precision@k` | Fraction of top-k that are declared explicit positives. | **Higher** |
| `knn_metadata_ndcg@k` | Graded ranking quality using the 0–3 relevance scale. | **Higher** |
| `knn_metadata_map@k` | How consistently are all explicit positives surfaced early? | **Higher** |
| `knn_metadata_mrr@k` | How far down before the first explicit positive? | **Higher** |
| `knn_metadata_r_precision` | Precision at R, where R = number of explicit positives. | **Higher** |
| `knn_attribute_ndcg@k` | One score per metadata attribute (e.g. `plants`, `camera_source`). | **Higher** per attribute |

### Always present

| Metric | Description |
|---|---|
| `pairwise_similarity_stats` | Mean / std / percentiles of all pairwise cosine similarities |
| `centroid_similarity_stats` | Cosine similarity of each embedding to the dataset centroid (anisotropy signal) |
| `intra_inter_similarity_gap` | Mean intra-class cosine − mean inter-class cosine |
| `effective_rank` | Participation-ratio effective dimensionality of the embedding space |
| `uniformity` | Wang & Isola (2020) uniformity — `log E[exp(-t‖u−v‖²)]`; more negative = better spread |
| `hubness` | K-occurrence statistics: mean, std, Gini coefficient, top hubs |
| `knn_radius@k` | Mean distance to the k-th nearest neighbor (search radius) |
| `mean_top_k_sim@k` | Mean cosine similarity to the k nearest neighbors |
| `outlier_score@k` | Per-item outlier signal based on neighbor distances |

### Without metadata

| Metric | Description |
|---|---|
| `knn_label_purity@k` | Fraction of each item's k nearest neighbors sharing its path-inferred class label |
| `knn_label_ndcg@k` | nDCG with binary relevance — rewards same-class neighbors ranked at the top |
| `knn_map@k` | MAP — rewards finding all same-class neighbors early |
| `knn_label_mrr@k` | MRR — reciprocal rank of the first same-class neighbor |
| `knn_label_r_precision` | R-Precision — precision at R where R = class size − 1 |

### With metadata

| Metric | Description |
|---|---|
| `alignment` | Wang & Isola (2020) alignment — mean `‖u−v‖²` over explicit positive pairs; lower = better |
| `knn_metadata_precision@k` | P@K — fraction of top-k retrieved items that are explicit positives |
| `knn_metadata_ndcg@k` | nDCG@K — graded relevance (0–3) by explicit group, class, and attributes |
| `knn_metadata_map@k` | MAP@K — average precision at finding all explicit positives early |
| `knn_metadata_mrr@k` | MRR@K — reciprocal rank of the first explicit positive |
| `knn_metadata_r_precision` | R-Precision — precision at R where R = number of explicit positives |
| `knn_attribute_ndcg@k` | One binary nDCG entry per attribute key — only present when at least one metadata group has a non-empty `attributes` dict |

> **Graded relevance for nDCG** — grade 3: same L2 group (explicit positive); grade 2: same L1 class; grade 1: different L1 but overlapping `plants`; grade 0: no relationship.

### Group analysis (with metadata only)

`group_analysis` is a top-level key containing one entry per declared metadata group. It is a diagnostic, not a KPI — it flags groups whose embeddings may be too coarse for reliable retrieval benchmarking.

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

Each class reports geometry metrics (`pairwise_similarity_stats`, `centroid_similarity_stats`, `effective_rank`) plus the same retrieval KPI tier sliced to that class — label-based without metadata, metadata-based with metadata.

---

## Visualizations

Generated by `examples/emb/example.py --output-dir output` (or called directly from Python):

| File | Function | Description |
|---|---|---|
| `class_confusion_matrix.html` | `plot_knn_confusion` | KNN confusion matrix — which classes get misidentified as which |
| `cosine_similarity.html` | `plot_cosine_similarity` | Full N×N pairwise cosine similarity heatmap, sorted by class |
| `tsne.html` | `plot_tsne` | Rotatable 3D t-SNE scatter colored by class |
| `lle.html` | `plot_lle` | Rotatable 3D LLE scatter — preserves local neighborhood structure |

All outputs are interactive Plotly HTML. Static PNG export requires `pip install kaleido`.

```python
from precisionai.agrieval.emb import (
    plot_cosine_similarity,
    plot_knn_confusion,
    plot_lle,
    plot_tsne,
)

plot_knn_confusion(result, k=10, output_path="output/confusion.html")
plot_cosine_similarity(embeddings, result, output_path="output/cosine.html")
plot_tsne(embeddings, result, output_path="output/tsne.html", dimensions=3)
plot_lle(embeddings, result, output_path="output/lle.html")
```

---

## Example notebooks

Two notebooks are provided in `examples/emb/`.

**[`examples/emb/example.ipynb`](../examples/emb/example.ipynb)** — Image→Image, two parts.
- **Part 1 — Basic** (`emb_image2image.json`): class labels inferred from the L2 path hierarchy. Covers KNN purity, nDCG, MAP, pairwise similarity, and all visualizations.
- **Part 2 — With Metadata** (`emb_image2image_meta.json`): explicit `MetadataGroup` clusters (A1, A2, D1, D2). Adds graded `knn_metadata_ndcg` and per-attribute `knn_attribute_ndcg`.

**[`examples/emb/example_plant.ipynb`](../examples/emb/example_plant.ipynb)** — Plant wirings.
Covers Plant→Image with `instance_to_image` mapping and Plant→Plant with `instance_labels`.

```bash
jupyter notebook examples/emb/example.ipynb
jupyter notebook examples/emb/example_plant.ipynb
```
