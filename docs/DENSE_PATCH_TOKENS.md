# Dense Patch Token Evaluation — `precisionai.agrieval.dpt`

Evaluates dense per-patch feature-map "tiles" produced by tiling an image through a vision backbone. Always computes unsupervised embedding-space geometry and per-tile spatial-health diagnostics; optionally layers on label-aware separation metrics when ground-truth masks are supplied.

| Wiring | Endpoint | Input | Output |
|---|---|---|---|
| **Tile evaluation** | `POST /evaluate/tiles` | `tile_id → [P, H, W]` feature maps (+ optional ground-truth masks + class map) | Geometry, per-tile diagnostics, optional per-class/kNN/separation metrics |
| **Image evaluation** | `POST /evaluate/image` | `image_id → [P, H, W]` feature maps (+ optional ground-truth masks + class map) | Geometry, per-image diagnostics, optional per-class/kNN/separation metrics |

The two wirings apply the exact same rules and compute the exact same metrics — the only difference is field naming (`tiles`/`per_tile` vs. `images`/`per_image`) so that requests and responses read naturally for whichever unit each entry represents. Use `/evaluate/tiles` when each entry is a crop of a larger image; use `/evaluate/image` when each entry is one whole (untiled) image. Both require every entry in a request to share the same `(P, H, W)` shape — a single evaluation run is always against one backbone/tiling configuration, so a "whole image" and a "tile" are the same kind of array.

`dpt/` does not tile images or run a vision backbone — that step happens before `dpt` is involved. This subpackage only ingests the resulting feature maps and scores them.

---

## Data format

### Tiles / images

Both wirings accept their entries **two ways — exactly one per request**. The tiles wiring uses `tiles`/`tiles_path`; the image wiring uses `images`/`images_path`.

1. **`tiles_path` / `images_path` (recommended for real workloads)** — a path, resolved against `dataset_root`, to a single batched `.npz` archive. Binary transport is 5–10× smaller and much faster to parse than the same arrays as JSON text. Arrays are loaded with `allow_pickle=False`, so archives containing pickled objects are rejected. The tiles and images schemas differ, since only tiles have a position within a larger source image:

   **`tiles_path`** — the archive must contain:

   | Key | Shape | Dtype | Meaning |
   |---|---|---|---|
   | `feature_maps` | `[T, P, H, W]` | float | One `[P, H, W]` patch-token grid per tile. |
   | `tile_image_id` | `[T]` | int | Which source image each tile belongs to (index into `filenames`). |
   | `tile_index` | `[T]` | int | 0-based sequence number of each tile within its source image. |
   | `tile_y0` / `tile_x0` | `[T]` | int | Pixel offset of each tile's top-left corner within its source image. |
   | `filenames` | `[N]` | str | Source image filename per `tile_image_id` index. |
   | `meta` | scalar | str | JSON string; required only for **label-aware** evaluation (`load_tile_placement()`), where it must include `"tile": "<width>x<height>"` (e.g. `"798x532"`) — the only source of each tile's pixel size, needed to crop ground-truth masks. Unsupervised runs (`load_tiles()` alone, or the API without `masks_dir`) work without it. Otherwise informational (model name, embed dim, ...) and unused. |

   Tiles are identified by their exact pixel rectangle (`tile_y0`, `tile_x0`, and the pixel size from `meta.tile`), **not** a row/col grid position — real extractors can emit overlapping tiles (e.g. corner-anchored crops of an image only slightly larger than one tile), so `dpt` never assumes tiles partition an image into a clean non-overlapping grid.

   Each tile is keyed by a synthesized ID, `f"{image_stem}_tile_{tile_index:02d}"` (`image_stem` = the matching `filenames` entry with its extension stripped, `tile_index` zero-padded to 2 digits) — used for `per_tile` response keys, and matching the naming convention of a per-tile `tile_00.npz`-style export. Use `load_tile_placement()` to recover each tile's exact placement (as a `TilePlacement(image_stem, y0, x0, height, width)`) for label-aware evaluation (see [Ground-truth masks](#ground-truth-masks-optional) below).

   **`images_path`** — simpler, since a whole image has no position within something bigger:

   | Key | Shape | Dtype | Meaning |
   |---|---|---|---|
   | `features` | `[N, P, H, W]` | float | One `[P, H, W]` feature map per whole image. |
   | `filenames` | `[N]` | str | Source image filename per entry. |

   Each image is keyed directly by its `filenames` entry's stem.

2. **`tiles` / `images` (inline JSON, for small/demo payloads)** — a dict mapping each entry ID to a patch-first feature map:

```json
{
  "tiles": {
    "tile_0001": [[[0.12, 0.05, ...], ...], ...],
    "tile_0002": [[[0.09, 0.11, ...], ...], ...]
  }
}
```

Each entry is shaped `[P, H, W]` — embedding dimension, patch-grid height, patch-grid width (e.g. `[384, 38, 57]` for a backbone tiling a 798×532px image into 14px patches). **All entries in one request must share the same `P`, `H`, and `W`** — a single evaluation run is always against one backbone/tiling configuration, whether entries are tiles or whole images. Values must be finite (no `NaN`/`inf`, including values that overflow at float32 precision); unlike `emb`, entries are **not** required to be L2-normalised — raw patch features aren't pre-normalised the way whole-image embeddings are. **Submit raw backbone features**, not normalised or PCA-projected ones: the metrics apply L2 normalisation internally where noted (cosine/kNN metrics) and use raw centered features everywhere else (spectrum metrics, norm statistics).

Entry IDs (tile IDs or image IDs) must be non-empty and must not contain path separators (`/`, `\`) — an entry ID names a mask file *stem*, never a path. IDs containing glob characters (`*`, `?`, `[`) are matched literally.

### Ground-truth masks (optional)

When supplied, `masks_dir` and `classes_path` follow **exactly the same format as `precisionai.agrieval.seg`** — see [SEGMENTATION.md](SEGMENTATION.md) for the full colour-mask and `class_map.json` spec. Mask alignment works differently depending on where entries came from:

- **Inline `tiles`/`images`, or images loaded from `images_path`** — one mask per entry, filename stem must equal the entry ID (e.g. `masks_dir/tile_0001.png` for tile `"tile_0001"`, or `masks_dir/field_001.png` for image ID `"field_001"`; subdirectories are searched recursively). The whole mask is majority-vote downsampled directly to that entry's `(H, W)` grid.
- **Tiles loaded from `tiles_path`** — one mask per *source image* instead, filename stem must equal the tile's `image_stem` (from `load_tile_placement()`), not the synthesized tile ID. Each tile's exact pixel rectangle — `[tile_y0 : tile_y0+height, tile_x0 : tile_x0+width]`, using the pixel size from `meta.tile` — is cropped out of the mask before that crop is majority-vote downsampled to the tile's own `(H, W)` grid. This is exact, not an inferred/proportional grid split, so it works correctly even when tiles overlap. A crop rectangle partially extending past the mask's edge is clipped to what's available; a rectangle lying *entirely* outside the mask raises a clear `ValueError` (it would otherwise have no pixels to vote on).

In both cases, the final per-entry downsample step is the same: `dpt` **majority-vote downsamples** the relevant mask region to the target `(H, W)` grid — each patch is assigned the class covering the most pixels in its mask region. Cells receiving no pixels (only possible when the source region is *smaller* than the target grid) fall back to nearest-neighbor sampling. The background class participates fully in all label-aware metrics.

**Supported mask extensions (case-sensitive):** `.png`, `.PNG`, `.jpg`, `.JPG`, `.jpeg`, `.JPEG`.

---

## The evaluation wirings

### Inputs — tile evaluation (`POST /evaluate/tiles`)

| Parameter | Type | Description |
|---|---|---|
| `tiles` | `dict[str, array]` \| `None` | Inline map of tile ID → `[P, H, W]` feature map. Exactly one of `tiles` / `tiles_path`. |
| `tiles_path` | path \| `None` | A batched `.npz` archive, resolved against `dataset_root` — see [Tiles / images](#tiles--images). Exactly one of `tiles` / `tiles_path`. |
| `masks_dir` | directory \| `None` | Ground-truth colour-coded masks — one per tile for inline `tiles`, one per source image (cropped per tile) for `tiles_path`. Requires `classes_path`. |
| `classes_path` | file \| `None` | Class-definition JSON, same format as `seg`. Requires `masks_dir`. |
| `dataset_root` | str \| `None` | Base path for relative `masks_dir`/`classes_path` (API layer only — the `run_dpt_eval()` function itself expects already-resolved paths, like `run_seg_eval`). |
| `k_values` | `list[int]` | K cutoffs for kNN label metrics. Default `[5, 10, 20]`. |
| `sample_pairs` | `int \| None` | Max random pairs for global pairwise similarity stats. `None` computes exactly. |
| `max_patches` | `int` | Max patches used for O(N²) label-aware kNN computation; larger corpora are randomly subsampled (seeded) and the drop reported in `warnings`. Default `20 000`. |

Always present in the response: `n_tiles`, `embed_dim`, `grid_height`, `grid_width`, `n_patches`, `tile_ids`, `k_values`, `global_metrics`, `per_tile`. Present only when `masks_dir`/`classes_path` are supplied: `classes`, `per_class`, `knn_confusion`, `separation`.

### Inputs — image evaluation (`POST /evaluate/image`)

Identical parameters and response shape to the tile wiring, with `tiles`/`tiles_path` replaced by `images`/`images_path` and the response keyed by `image` instead of `tile`:

| Parameter | Type | Description |
|---|---|---|
| `images` | `dict[str, array]` \| `None` | Inline map of image ID → `[P, H, W]` feature map. Exactly one of `images` / `images_path`. |
| `images_path` | path \| `None` | A batched `.npz` archive, resolved against `dataset_root` — see [Tiles / images](#tiles--images). Exactly one of `images` / `images_path`. |
| `masks_dir`, `classes_path`, `dataset_root`, `k_values`, `sample_pairs`, `max_patches` | — | Same as the tile wiring, applied per image instead of per tile. |

Always present in the response: `n_images`, `embed_dim`, `grid_height`, `grid_width`, `n_patches`, `image_ids`, `k_values`, `global_metrics`, `per_image`. Present only when `masks_dir`/`classes_path` are supplied: `classes`, `per_class`, `knn_confusion`, `separation`.

---

## Output format

Tile-evaluation response (`POST /evaluate/tiles`); the image-evaluation response (`POST /evaluate/image`) is identical except `n_tiles`→`n_images`, `tile_ids`→`image_ids`, and `per_tile`→`per_image`:

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
  "separation": {
    "silhouette": 0.38,
    "calinski_harabasz": 2141.7,
    "ari": 0.71,
    "nmi": 0.63,
    "pc1_auroc": 0.92
  },
  "warnings": null
}
```

---

## Metrics

The metrics below are described in terms of "tiles" for brevity — the image wiring computes the exact same metrics per image instead of per tile.

### Unsupervised (always computed)

Computed by flattening all tiles into a single `[N·H·W, P]` patch-token matrix and reusing `precisionai.agrieval.emb.metrics` — the same math already used for whole-image embedding evaluation, since it operates generically on any `[N, D]` matrix. Spectrum metrics run on **raw centered** features (no L2 normalisation); cosine metrics are inherently computed on L2-normalised tokens.

| Metric | Input normalisation | Description |
|---|---|---|
| `effective_rank` | raw, centered | Participation-ratio estimate of the patch-token space's effective dimensionality. |
| `pca_explained_variance` | raw, centered | Cumulative variance explained by the leading principal components. |
| `pairwise_similarity_stats` | L2-normalised | Distribution of pairwise cosine similarities (sampled via `sample_pairs`). |
| `centroid_similarity_stats` | L2-normalised | Cosine similarity of each patch to the corpus centroid (anisotropy/collapse indicator). |
| `uniformity` | L2-normalised | Wang & Isola (2020) uniformity of the patch-token distribution on the unit hypersphere. |

### Per-tile diagnostics (always computed)

| Metric | Description |
|---|---|
| `patch_norm_stats` | Mean/std/percentiles of per-patch L2 norm on raw features — doubles as a collapse/dead-patch health check. |
| `patch_smoothness` | Average of the horizontal-neighbor mean and vertical-neighbor mean cosine similarity (L2-normalised patches), weighting both directions equally regardless of grid aspect ratio. |
| `outlier_fraction` | Fraction of a tile's patches whose norm is strictly above **mean + 3σ of all patch norms in the request**. The threshold is corpus-wide, so per-tile fractions are directly comparable. |

### Label-aware (only with ground truth)

Reuses `precisionai.agrieval.emb.metrics.similarity.top_k_neighbors` and `label_aware.knn_confusion_matrix` / `knn_label_purity_at_k` directly against the per-patch class labels — no separate kNN implementation. Patch tokens are L2-normalised before cosine kNN:

| Metric | Description |
|---|---|
| `knn_confusion.confusion` | Per-K confusion matrix: fraction of each class's patches whose k-NN neighbors belong to each class. |
| `knn_confusion.purity` | Per-K mean/std fraction of a patch's k nearest neighbors sharing its class. |
| `per_class[...].effective_rank` / `pca_explained_variance` | Spectrum metrics (raw, centered) computed on the subset of patch tokens belonging to that class. |

### Separation (only with ground truth)

Split-free label separation metrics from `precisionai.agrieval.emb.metrics.separation` — all non-parametric or closed-form, with no train/eval split anywhere. Calinski-Harabasz and PC1-AUROC are O(N·D) and run on the **full patch corpus**; silhouette and ARI/NMI are O(N²) and iterative respectively, so they run on the same seeded `max_patches` subsample as the kNN metrics.

| Metric | Input normalisation | Description |
|---|---|---|
| `separation.silhouette` | L2-normalised | Mean silhouette coefficient of the class partition under cosine distance, in `[-1, 1]`. |
| `separation.calinski_harabasz` | raw | Between-class over within-class dispersion ratio (Euclidean); higher is better separated. |
| `separation.ari` / `separation.nmi` | L2-normalised | Adjusted Rand index / normalized mutual information between the true classes and a deterministic k-means clustering (k = number of present classes) — are the classes recoverable as unsupervised clusters? Requires scikit-learn. |
| `separation.pc1_auroc` | raw, centered | Threshold-free AUROC of foreground (every class except `background`) vs. the `background` class along the corpus's first principal component. Sign-free — reported as `max(auc, 1 − auc)`, so `0.5` = no signal, `1.0` = perfect linear separation. |

Each metric degrades gracefully: when its preconditions fail — only one class present, no class named `background` in the class map (PC1-AUROC), or scikit-learn not installed (ARI/NMI) — it is reported as `null` with an explanatory entry in `warnings` instead of failing the run.

---

## API server

```bash
precisionai-agrieval-api --dataset-root /path/to/dataset
open http://localhost:8000/docs
```

### `POST /v1/dense-patch-tokens/evaluate/tiles`

```bash
# Binary tiles from disk (recommended): a batched .npz archive
curl -X POST http://localhost:8000/v1/dense-patch-tokens/evaluate/tiles \
  -H "Content-Type: application/json" \
  -d '{
    "tiles_path": "tiles/batch_01.npz",
    "masks_dir": "masks/batch_01",
    "classes_path": "class_map.json"
  }'

# Inline JSON tiles (small/demo payloads)
curl -X POST http://localhost:8000/v1/dense-patch-tokens/evaluate/tiles \
  -H "Content-Type: application/json" \
  -d '{
    "tiles": { "tile_0001": [[[0.1, 0.2]]], "tile_0002": [[[0.3, 0.1]]] }
  }'
```

### `POST /v1/dense-patch-tokens/evaluate/image`

Identical rules, just with `images`/`images_path` in place of `tiles`/`tiles_path` — use this endpoint when each entry is one whole (untiled) image rather than an arbitrary tile crop:

```bash
curl -X POST http://localhost:8000/v1/dense-patch-tokens/evaluate/image \
  -H "Content-Type: application/json" \
  -d '{
    "images_path": "images/batch_01.npz",
    "masks_dir": "masks/batch_01",
    "classes_path": "class_map.json"
  }'
```

HTTP 422 is returned for request-schema validation errors (mismatched shapes, non-finite values, both or neither of the source fields set, `masks_dir` set without `classes_path` or vice versa). HTTP 400 is returned for service-level errors (missing path, an entry with no matching ground-truth mask, unknown mask colours, invalid feature-map files).

---

## Python usage

```python
from precisionai.agrieval.dpt import load_tile_placement, load_tiles, print_result, run_dpt_eval

# From disk — a batched .npz archive
# (feature_maps/tile_image_id/tile_index/tile_y0/tile_x0/filenames/meta)
tiles = load_tiles("tiles/batch_01.npz")
tile_placement = load_tile_placement("tiles/batch_01.npz")  # {tile_id: TilePlacement(image_stem, y0, x0, height, width)}

result = run_dpt_eval(
    tiles=tiles,
    tile_placement=tile_placement,
    masks_dir="masks/batch_01",  # one whole-image mask per image_stem, cropped per tile
    classes_path="class_map.json",
)

print_result(result)  # human-readable summary of every metric
print(result["knn_confusion"]["purity"]["5"]["mean"])  # or index the raw dict directly
print(result["separation"]["silhouette"])  # split-free separation metrics (None when a precondition fails)
```

`run_dpt_eval` accepts any array-like tile values — nested lists, numpy arrays, or CPU torch tensors (converted via `np.asarray`; no torch dependency needed). `tile_placement` is optional — omit it (or pass inline `tiles` built by hand) to fall back to matching each tile directly to its own mask by tile ID, exactly like the image wiring:

```python
result = run_dpt_eval(tiles={"tile_0001": my_tensor})  # torch.Tensor works as-is
assert result["classes"] is None  # unsupervised-only when masks are omitted
```

The image wiring is simpler — `load_images`/`run_dpt_image_eval` in place of `load_tiles`/`run_dpt_eval`, keyed by `image_id` instead of `tile_id`, with no `tile_placement` equivalent (a whole image already maps 1:1 to a whole mask, no cropping needed):

```python
from precisionai.agrieval.dpt import load_images, run_dpt_image_eval

images = load_images("images/batch_01.npz")

result = run_dpt_image_eval(
    images=images,
    masks_dir="masks/batch_01",
    classes_path="class_map.json",
)

print(result["global_metrics"]["effective_rank"])
print(result["per_image"].keys())
```

---

## Example notebook

**[`examples/dpt/example.ipynb`](../examples/dpt/example.ipynb)** — runs both wirings side by side against real (non-synthetic) feature maps shipped in `tests/data/dpt_tiles/`, with real ground-truth masks, and asserts the two wirings agree exactly modulo field naming.

```bash
jupyter notebook examples/dpt/example.ipynb
```

A plain-Python equivalent without Jupyter is available at [`examples/dpt/example.py`](../examples/dpt/example.py):

```bash
python examples/dpt/example.py
```

---

## Validation errors

**Inline `tiles`/`images` (schema layer):**

| Condition | Exception |
|---|---|
| Fewer than 1 entry, or entries with mismatched `(P, H, W)` | `ValueError` |
| Ragged (non-rectangular) entry nesting | `ValueError` |
| Non-finite value in an entry (including float32 overflow) | `ValueError` |
| Empty entry ID, or entry ID containing a path separator | `ValueError` |
| Both or neither of the source fields (`tiles`/`tiles_path` or `images`/`images_path`) provided | `ValueError` |
| Only one of `masks_dir` / `classes_path` set | `ValueError` |

**`tiles_path`/`images_path` batched `.npz` archives (loader layer):**

| Condition | Exception |
|---|---|
| `tiles_path`/`images_path` is not an existing `.npz` file | `FileNotFoundError: {kind}s_path must be an existing .npz archive: '...'` |
| A required key is missing (`feature_maps`/`tile_image_id`/`tile_index`/`tile_y0`/`tile_x0`/`filenames` for tiles; `features`/`filenames` for images) | `ValueError: Archive '...' is missing required key(s): [...]` |
| `feature_maps`/`features` is not 4-D, or has an empty axis | `ValueError` |
| A companion array's length doesn't match the feature array's first axis (tiles: `tile_image_id`/`tile_index`/`tile_y0`/`tile_x0`; images: `filenames`) | `ValueError` |
| A `tile_image_id` value is out of range for `filenames` (tiles only) | `ValueError` |
| A negative value in `tile_index`/`tile_y0`/`tile_x0` (tiles only) | `ValueError` |
| Two tiles share the same `(tile_image_id, tile_index)` (tiles only) | `ValueError: Duplicate tile position: ...` |
| Two `filenames` entries share the same stem (tiles: would collide in synthesized tile IDs; images: would collide in image IDs) | `ValueError` |
| Non-finite value in `feature_maps`/`features` | `ValueError` |
| Pickled object arrays in `feature_maps`/`features` | `ValueError` (numpy refuses with `allow_pickle=False`) |
| `meta` missing, malformed JSON, or `meta.tile` not a `"<width>x<height>"` string (`load_tile_placement` only) | `ValueError` |

**Ground-truth masks (service layer):**

| Condition | Exception |
|---|---|
| `masks_dir` does not exist | `FileNotFoundError: masks_dir does not exist or is not a directory: '...'` |
| An entry (or its source image, for `tiles_path`-loaded tiles) has no matching ground-truth mask file | `FileNotFoundError: No ground-truth mask found for tile '...' under '...'` (wording says "tile" regardless of whether the match is by tile ID or image stem) |
| A tile ID in the evaluation has no entry in `tile_placement` | `ValueError: tile_placement is missing entries for tile ID(s): ...` |
| A tile's pixel rectangle lies entirely outside its ground-truth mask | `ValueError: Tile '...' pixel rectangle ... lies entirely outside its ...` |
| Colour in a mask not defined in the class-definition file | `ValueError: Mask '...' contains N unknown color(s) not in classes.json: ...` |
