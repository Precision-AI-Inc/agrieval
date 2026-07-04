# Dense Patch Token Evaluation — `precisionai.agrieval.dpt`

Evaluates dense per-patch feature-map "tiles" produced by tiling an image through a vision backbone (e.g. DINOv2). Always computes unsupervised embedding-space geometry and per-tile spatial-health diagnostics; optionally layers on label-aware separation metrics when ground-truth masks are supplied.

| Wiring | Input | Output |
|---|---|---|
| **Tile evaluation** | `tile_id → [C, H, W]` feature maps (+ optional ground-truth masks + class map) | Geometry, per-tile diagnostics, optional per-class/kNN metrics |

`dpt/` does not tile images or run a vision backbone — that step happens upstream (e.g. in `pai-vision-feature-map-eval`). This subpackage only ingests the resulting tiles and scores them. See [Scope and alignment](#scope-and-alignment-with-pai-vision-feature-map-eval) for the precise division of responsibility.

---

## Data format

### Tiles

Tiles can be supplied **two ways — exactly one per request**:

1. **`tiles_path` (recommended for real workloads)** — a path, resolved against `dataset_root`, to either a directory of `.npy` files (one per tile, file stem = tile ID, searched recursively) or a single `.npz` archive (member name = tile ID). Binary transport is 5–10× smaller and much faster to parse than the same arrays as JSON text. Arrays are loaded with `allow_pickle=False`, so archives containing pickled objects are rejected. Tiles produced as torch tensors should be saved via `tensor.numpy()` → `np.save` — `.pt` files are not supported (loading them would require a torch dependency).

2. **`tiles` (inline JSON, for small/demo payloads)** — a dict mapping each tile ID to a channels-first feature map:

```json
{
  "tiles": {
    "tile_0001": [[[0.12, 0.05, ...], ...], ...],
    "tile_0002": [[[0.09, 0.11, ...], ...], ...]
  }
}
```

Each tile is shaped `[C, H, W]` — embedding dimension, patch-grid height, patch-grid width (e.g. `[384, 38, 57]` for a DINOv2-small extractor tiling a 798×532px image into 14px patches). **All tiles in one request must share the same `C`, `H`, and `W`** — a single evaluation run is always against one backbone/tiling configuration. Values must be finite (no `NaN`/`inf`, including values that overflow at float32 precision); unlike `emb`, tiles are **not** required to be L2-normalised — raw patch features aren't pre-normalised the way whole-image embeddings are. **Submit raw backbone features**, not normalised or PCA-projected ones: the metrics apply L2 normalisation internally exactly where the upstream benchmark does (cosine/kNN metrics) and use raw centered features everywhere else (spectrum metrics, norm statistics).

Tile IDs must be non-empty and must not contain path separators (`/`, `\`) — a tile ID names a mask file *stem*, never a path. IDs containing glob characters (`*`, `?`, `[`) are matched literally.

### Ground-truth masks (optional)

When supplied, `masks_dir` and `classes_path` follow **exactly the same format as `precisionai.agrieval.seg`** — see [SEGMENTATION.md](SEGMENTATION.md) for the full colour-mask and `class_map.json` spec. The only `dpt`-specific rule is the file-matching convention: each ground-truth mask's filename stem must equal a tile ID (e.g. `masks_dir/tile_0001.png` for tile `"tile_0001"`; subdirectories are searched recursively).

Ground-truth masks are almost always a different resolution than the patch grid (e.g. a 798×532px mask vs. a 38×57 patch grid). `dpt` **majority-vote downsamples** each mask to its tile's `(H, W)` grid before scoring — each patch is assigned the class covering the most pixels in its mask region, matching the label-assignment convention of the upstream benchmark. Cells receiving no pixels (only possible when a mask is *smaller* than the patch grid) fall back to nearest-neighbor sampling. The background class participates fully in all label-aware metrics, also matching the upstream convention.

**Supported mask extensions (case-sensitive):** `.png`, `.PNG`, `.jpg`, `.JPG`, `.jpeg`, `.JPEG`.

---

## The evaluation wiring

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `tiles` | `dict[str, array]` \| `None` | Inline map of tile ID → `[C, H, W]` feature map. Exactly one of `tiles` / `tiles_path`. |
| `tiles_path` | path \| `None` | Directory of `.npy` files or a `.npz` archive, resolved against `dataset_root`. Exactly one of `tiles` / `tiles_path`. |
| `masks_dir` | directory \| `None` | Ground-truth colour-coded masks, one per tile. Requires `classes_path`. |
| `classes_path` | file \| `None` | Class-definition JSON, same format as `seg`. Requires `masks_dir`. |
| `dataset_root` | str \| `None` | Base path for relative `masks_dir`/`classes_path` (API layer only — the `run_dpt_eval()` function itself expects already-resolved paths, like `run_seg_eval`). |
| `k_values` | `list[int]` | K cutoffs for kNN label metrics. Default `[5, 10, 20]`. |
| `sample_pairs` | `int \| None` | Max random pairs for global pairwise similarity stats. `None` computes exactly. |
| `max_patches` | `int` | Max patches used for O(N²) label-aware kNN computation; larger corpora are randomly subsampled (seeded) and the drop reported in `warnings`. Default `20 000`. |

### Outputs

Always present: `n_tiles`, `embed_dim`, `grid_height`, `grid_width`, `n_patches`, `tile_ids`, `k_values`, `global_metrics`, `per_tile`.

Present only when `masks_dir`/`classes_path` are supplied: `classes`, `per_class`, `knn_confusion`.

---

## Output format

```json
{
  "n_tiles": 12,
  "embed_dim": 384,
  "grid_height": 38,
  "grid_width": 57,
  "n_patches": 259920,
  "tile_ids": ["tile_0001", "tile_0002", "..."],
  "k_values": [5, 10, 20],
  "global_metrics": {
    "effective_rank": { "embedding_dim": 384, "effective_rank": 41.2, "effective_rank_ratio": 0.107 },
    "pca_explained_variance": { "pc1": 0.31, "top_10": 0.68, "...": "..." },
    "pairwise_similarity_stats": { "mean": 0.42, "p50": 0.44, "...": "..." },
    "centroid_similarity_stats": { "mean_cosine_to_centroid": 0.61, "...": "..." },
    "uniformity": -1.42,
    "mean_patch_smoothness": 0.87,
    "mean_outlier_fraction": 0.01
  },
  "per_tile": {
    "tile_0001": {
      "patch_norm_stats": { "mean": 12.1, "std": 2.3, "p05": 8.9, "p50": 12.0, "p95": 15.8 },
      "patch_smoothness": 0.91,
      "outlier_fraction": 0.02
    }
  },
  "classes": ["background", "Crop | Soybean"],
  "per_class": {
    "background": { "n_patches": 18000, "effective_rank": { "...": "..." }, "pca_explained_variance": { "...": "..." } },
    "Crop | Soybean": { "n_patches": 4200, "...": "..." }
  },
  "knn_confusion": {
    "confusion": { "5": { "background": { "background": 0.97, "Crop | Soybean": 0.03 }, "...": "..." } },
    "purity": { "5": { "mean": 0.94, "std": 0.11 } }
  },
  "warnings": null
}
```

---

## Metrics

### Unsupervised (always computed)

Computed by flattening all tiles into a single `[N·H·W, C]` patch-token matrix and reusing `precisionai.agrieval.emb.metrics` — the same math already used for whole-image embedding evaluation, since it operates generically on any `[N, D]` matrix. Spectrum metrics run on **raw centered** features (no L2 normalisation), matching the upstream benchmark; cosine metrics are inherently computed on L2-normalised tokens.

| Metric | Input normalisation | Description |
|---|---|---|
| `effective_rank` | raw, centered | Participation-ratio estimate of the patch-token space's effective dimensionality. |
| `pca_explained_variance` | raw, centered | Cumulative variance explained by the leading principal components. |
| `pairwise_similarity_stats` | L2-normalised | Distribution of pairwise cosine similarities (sampled via `sample_pairs`). |
| `centroid_similarity_stats` | L2-normalised | Cosine similarity of each patch to the corpus centroid (anisotropy/collapse indicator). |
| `uniformity` | L2-normalised | Wang & Isola (2020) uniformity of the patch-token distribution on the unit hypersphere. |

### Per-tile diagnostics (always computed)

Aligned with the upstream benchmark's per-tile health checks:

| Metric | Description |
|---|---|
| `patch_norm_stats` | Mean/std/percentiles of per-patch L2 norm on raw features — doubles as a collapse/dead-patch health check. |
| `patch_smoothness` | Average of the horizontal-neighbor mean and vertical-neighbor mean cosine similarity (L2-normalised patches) — the upstream smoothness definition, weighting both directions equally regardless of grid aspect ratio. |
| `outlier_fraction` | Fraction of a tile's patches whose norm is strictly above **mean + 3σ of all patch norms in the request** — the upstream artifact-patch definition. The threshold is corpus-wide, so per-tile fractions are directly comparable. |

### Label-aware (only with ground truth)

Reuses `precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` and `label_aware.knn_confusion_matrix` / `knn_label_purity_at_k` directly against the per-patch class labels — no separate kNN implementation. Patch tokens are L2-normalised before cosine kNN, matching the upstream benchmark:

| Metric | Description |
|---|---|
| `knn_confusion.confusion` | Per-K confusion matrix: fraction of each class's patches whose k-NN neighbors belong to each class. |
| `knn_confusion.purity` | Per-K mean/std fraction of a patch's k nearest neighbors sharing its class. |
| `per_class[...].effective_rank` / `pca_explained_variance` | Spectrum metrics (raw, centered) computed on the subset of patch tokens belonging to that class. |

---

## Scope and alignment with pai-vision-feature-map-eval

The upstream repo owns everything **before** the tiles exist; `dpt` owns everything **after**:

| Responsibility | Owner |
|---|---|
| Image tiling, patch-grid geometry, backbone inference, feature extraction | `pai-vision-feature-map-eval` (out of scope here) |
| PCA visualisation, RGB rendering, MLflow publishing, provenance manifests | `pai-vision-feature-map-eval` (out of scope here) |
| Model benchmarking with train/eval image splits and classifier probes | `pai-vision-feature-map-eval` (out of scope here) |
| Scoring already-extracted tiles as a reusable service (API + Python) | `precisionai.agrieval.dpt` |

Metric-by-metric alignment status against the upstream separation benchmark (`bin/eval_separation.py` and `verification.py`):

| Upstream metric | Status in `dpt` | Notes |
|---|---|---|
| Adjacent-patch smoothness | **Aligned** | Same definition: L2-normalised patches, mean of horizontal-pair mean and vertical-pair mean, per tile, averaged over tiles. |
| Outlier/artifact fraction | **Aligned** | Same definition: per-patch L2 norm strictly above corpus-wide mean + 3σ. |
| Per-patch norm statistics | **Aligned** | Raw-feature L2 norms. |
| Mask → patch-grid labels | **Aligned** | Majority-vote per cell; background class included in metrics. |
| kNN input normalisation | **Aligned** | L2-normalised before cosine kNN; upstream's default k=20 is in our default `k_values`. |
| PCA explained variance | **Aligned** | Raw centered features, same as upstream's spectrum (EVR) computation. |
| Effective rank | **Intentional divergence** | Upstream uses Σ(Sᵢ/S₁)² on centered singular values; `dpt` reuses `emb`'s participation ratio (Σλ)²/Σλ² for cross-subpackage consistency. Both are collapse indicators; values are not directly comparable. |
| kNN evaluation protocol | **Intentional divergence** | Upstream fits kNN on a train split and predicts a held-out eval split (kNN-mIoU); `dpt` computes self-retrieval purity/confusion (leave-one-out within the corpus) — a retrieval-quality view, not a classifier benchmark. |
| Patch subsampling | **Intentional divergence** | Upstream water-fills a class/size-balanced 2000-patch budget per image; `dpt` uses all patches up to `max_patches` with a seeded uniform subsample above it. |
| Unlabeled-patch exclusion | **Not applicable** | Upstream marks unmapped patches −1 and excludes them; in `dpt` every mask pixel must map to a class-map colour, so no unlabeled patches exist. |
| kNN-mIoU, PC1-AUROC, linear probe | **Out of scope** | Require train/eval splits and classifier fitting — model benchmarking, which stays upstream. |
| Silhouette, Calinski-Harabasz, ARI/NMI | **Out of scope (this version)** | sklearn-dependent clustering scores already computed by the upstream benchmark; may be added later behind the `analysis` extra. |

---

## API server

```bash
precisionai-agrieval-api --dataset-root /path/to/dataset
open http://localhost:8000/docs
```

### `POST /v1/dense-patch-tokens/evaluate`

```bash
# Binary tiles from disk (recommended): a .npz archive or a directory of .npy files
curl -X POST http://localhost:8000/v1/dense-patch-tokens/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "tiles_path": "tiles/batch_01.npz",
    "masks_dir": "masks/batch_01",
    "classes_path": "class_map.json"
  }'

# Inline JSON tiles (small/demo payloads)
curl -X POST http://localhost:8000/v1/dense-patch-tokens/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "tiles": { "tile_0001": [[[0.1, 0.2]]], "tile_0002": [[[0.3, 0.1]]] }
  }'
```

HTTP 422 is returned for request-schema validation errors (mismatched tile shapes, non-finite values, both or neither of `tiles`/`tiles_path` set, `masks_dir` set without `classes_path` or vice versa). HTTP 400 is returned for service-level errors (missing `tiles_path` or `masks_dir`, a tile with no matching ground-truth mask, unknown mask colours, invalid tile files).

---

## Python usage

```python
from precisionai.agrieval.dpt import load_tiles, run_dpt_eval

# From disk — a directory of .npy files or a .npz archive
tiles = load_tiles("tiles/batch_01.npz")

result = run_dpt_eval(
    tiles=tiles,
    masks_dir="masks/batch_01",
    classes_path="class_map.json",
)

print(result["global_metrics"]["effective_rank"])
print(result["knn_confusion"]["purity"]["5"]["mean"])
```

`run_dpt_eval` accepts any array-like tile values — nested lists, numpy arrays, or CPU torch tensors (converted via `np.asarray`; no torch dependency needed):

```python
result = run_dpt_eval(tiles={"tile_0001": my_tensor})  # torch.Tensor works as-is
assert result["classes"] is None  # unsupervised-only when masks are omitted
```

---

## Validation errors

| Condition | Exception |
|---|---|
| Fewer than 1 tile, or tiles with mismatched `(C, H, W)` | `ValueError` (schema and loader layers) |
| Ragged (non-rectangular) tile nesting | `ValueError` (schema layer) |
| Non-finite value in a tile (including float32 overflow) | `ValueError` (schema and loader layers) |
| Empty tile ID, or tile ID containing a path separator | `ValueError` (schema, loader, and service layers) |
| Both or neither of `tiles` / `tiles_path` provided | `ValueError` (schema layer) |
| `tiles_path` is neither a directory nor a `.npz` file | `FileNotFoundError: tiles_path must be an existing directory of .npy files or a .npz archive` |
| No `.npy` files under a `tiles_path` directory, or duplicate stems | `ValueError` |
| Pickled object arrays in a tile file | `ValueError` (numpy refuses with `allow_pickle=False`) |
| Only one of `masks_dir` / `classes_path` set | `ValueError` (schema layer) |
| `masks_dir` does not exist | `FileNotFoundError: masks_dir does not exist or is not a directory: '...'` |
| A tile has no matching ground-truth mask file | `FileNotFoundError: No ground-truth mask found for tile '...' under '...'` |
| Colour in a mask not defined in the class-definition file | `ValueError: Mask '...' contains N unknown color(s) not in classes.json: ...` |
