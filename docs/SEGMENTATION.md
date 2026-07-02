# Segmentation Evaluation — `precisionai.agrieval.seg`

Pixel-level evaluation of semantic segmentation models against colour-coded ground-truth masks. Produces per-class and dataset-level KPIs — IoU, Dice/F1, per-class accuracy, mIoU, mAcc, and FWIoU — from a pair of mask directories and a class-definition file.

| Wiring | Input | Output |
|---|---|---|
| **Mask evaluation** | Predicted masks dir + ground-truth masks dir + class map | `output_summary.json` + `image_summary.json` |

---

## Data format

### Canonical directory layout

```
masks/                      ← ground-truth colour-coded masks
  A1/
    pai-abc123.png
    pai-def456.png
  A2/
    pai-ghi789.png
  D1/
    pai-jkl012.png

predictions/                ← predicted colour-coded masks (mirrors masks/)
  A1/
    pai-abc123.png
    pai-def456.png
  A2/
    pai-ghi789.png
  D1/
    pai-jkl012.png

class_map.json              ← class definitions: name, RGB colour, integer ID
```

Subdirectory structure is supported and traversed recursively. Files are paired between `pred` and `masks` by their relative path and stem — cross-extension matching is supported (e.g. a `.png` prediction paired with a `.jpg` ground-truth mask of the same name).

**Supported extensions (case-sensitive):** `.png`, `.PNG`, `.jpg`, `.JPG`, `.jpeg`, `.JPEG`.

---

### Colour-coded masks

Each mask is an RGB PNG (or JPEG) where every pixel's colour identifies its class. Background is conventionally black `(0, 0, 0)`. All colours present in any mask must be defined in the class-definition file — unknown colours raise a `ValueError` listing the offending colours and the file path.

```
pixel (R, G, B) → look up in class_map.json → class name + integer ID
```

The predicted mask and its corresponding ground-truth mask must have identical spatial dimensions `(H, W)`. A `ValueError` is raised on any size mismatch.

---

### Class definition file

`load_classes()` expects the AgriBench format (the `"classes"` key):

```json
{
  "description": "AgriBench v1.0.0 segmentation mask class map.",
  "classes": [
    { "id": 0, "name": "background",     "color": [0,   0,   0  ], "hex": "#000000" },
    { "id": 9, "name": "Crop | Soybean", "color": [49,  140, 101], "hex": "#318c65" },
    { "id": 14,"name": "Weed | Grass",   "color": [242, 73,  76 ], "hex": "#f2494c" }
  ],
  "by_color": {
    "#000000": "background",
    "#318c65": "Crop | Soybean",
    "#f2494c": "Weed | Grass"
  }
}
```

Entries are sorted by class ID internally before use, so the order in the file does not matter. Class IDs must form a contiguous range `0 .. n_classes − 1`.

---

## The evaluation wiring

One wiring is defined for segmentation evaluation. It accepts a directory of predicted masks, a directory of ground-truth masks, and a class-definition file. It walks both directories recursively, validates every mask against the class palette, computes confusion-matrix-derived KPIs for each image, aggregates them across the dataset, and writes two JSON files.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `pred_dir` | directory | Predicted colour-coded masks. |
| `masks_dir` | directory | Ground-truth colour-coded masks. |
| `classes_path` | file | Class-definition JSON (AgriBench or legacy format). |
| `output_dir` | directory | Where to write JSON output. `None` skips file output. |
| `output_summary_name` | str | Dataset-level output filename. Default: `output_summary.json`. |
| `image_summary_name` | str | Per-image output filename. Default: `image_summary.json`. |
| `verbose` | bool | Print detailed run context plus per-image, dataset, and per-class metrics. Default: `False`. |
| `show_progress` | bool | Display a tqdm progress bar while image pairs are evaluated. Default: `True`. |
| `num_workers` | int \| None | Worker threads for mask-pair processing. `None` selects auto, up to 4. Use `1` for sequential processing. |

### Outputs

Two JSON files are written to `output_dir`. Both are also returned as Python dicts by `run_seg_eval()`.

---

## Output format

### `output_summary.json` — dataset-level

```json
{
  "n_images": 20,
  "classes": {
    "background": {
      "iou":      0.9412,
      "dice":     0.9698,
      "accuracy": 0.9601
    },
    "Crop | Soybean": {
      "iou":      0.7834,
      "dice":     0.8786,
      "accuracy": 0.8901
    },
    "Weed | Grass": {
      "iou":      null,
      "dice":     null,
      "accuracy": null
    }
  },
  "summary": {
    "mIoU":  0.8623,
    "mAcc":  0.9251,
    "FWIoU": 0.9117
  }
}
```

Per-class metrics are `null` for classes whose denominator is zero (absent from both prediction and ground truth across the full dataset). Such classes are excluded from the macro-average summary metrics.

Dataset-level KPIs are computed from the confusion matrix accumulated over all images — not as a mean of per-image KPIs.

---

### `image_summary.json` — per-image

```json
{
  "A1/pai-abc123.png": {
    "classes": {
      "background":     { "iou": 0.9501, "dice": 0.9745, "accuracy": 0.9712 },
      "Crop | Soybean": { "iou": 0.8123, "dice": 0.8966, "accuracy": 0.9104 },
      "Weed | Grass":   { "iou": null,   "dice": null,   "accuracy": null   }
    },
    "summary": {
      "mIoU":  0.8812,
      "mAcc":  0.9408,
      "FWIoU": 0.9322
    }
  },
  "A1/pai-def456.png": {
    "...": "..."
  }
}
```

Image keys are the predicted mask paths relative to `pred_dir`, using forward slashes on all platforms.

Per-image summary metrics are computed from that image's confusion matrix alone. A class with zero pixels in both prediction and ground truth for a given image gets `null` for that image.

---

## Metrics

All metrics are derived from the pixel-level confusion matrix `cm[i, j]` — the count of pixels truly belonging to class *i* that were predicted as class *j*.

| Symbol | Description | Formula | Better when |
|---|---|---|---|
| **IoU** | Intersection over Union (Jaccard index) per class | `TP / (TP + FP + FN)` | **Higher** |
| **Dice** | Dice coefficient / F1 score per class | `2·TP / (2·TP + FP + FN)` | **Higher** |
| **Accuracy** | Per-class recall — fraction of ground-truth pixels correctly predicted | `TP / (TP + FN)` | **Higher** |
| **mIoU** | Macro-average IoU over valid classes | `mean(IoU_c)` | **Higher** |
| **mAcc** | Macro-average per-class accuracy over valid classes | `mean(Accuracy_c)` | **Higher** |
| **FWIoU** | Frequency-weighted IoU — classes weighted by ground-truth pixel frequency | `Σ freq_c · IoU_c` | **Higher** |

**NaN / null handling:** a class with `TP = FP = FN = 0` (completely absent from both prediction and ground truth) has an undefined IoU. It is reported as `null` in JSON and excluded from all macro averages. Classes that are predicted but never present in the ground truth (`FP > 0`, `FN = 0`) receive an IoU of `0.0`, not `null`.

**Dice vs IoU:** Dice is monotonically related to IoU via `Dice = 2·IoU / (1 + IoU)` and thus produces identical model rankings. Dice values are more lenient (higher) than the corresponding IoU. It is reported alongside IoU for compatibility with medical-segmentation conventions.

**FWIoU vs mIoU:** FWIoU weights classes by their pixel frequency in the ground truth, so dominant background classes pull the score upward. mIoU treats all classes equally — use mIoU when class imbalance should not inflate the headline number.

---

## CLI usage

```bash
# Evaluate predictions against ground truth (writes output_summary.json and image_summary.json)
precisionai-agrieval-seg \
  --pred    predictions/ \
  --masks   ground_truth/ \
  --classes class_map.json \
  --output-dir results/ \
  --num-workers 4 \
  --verbose

# Custom output filenames
precisionai-agrieval-seg \
  --pred    predictions/ \
  --masks   ground_truth/ \
  --classes class_map.json \
  --output-dir results/ \
  --output-summary dataset_kpis.json \
  --image-summary  per_image_kpis.json
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--pred DIR` | — | *(required)* Directory of predicted masks. |
| `--masks DIR` | — | *(required)* Directory of ground-truth masks. |
| `--classes FILE` | — | *(required)* Class-definition JSON. |
| `--output-dir DIR` | `.` | Output directory for JSON files. |
| `--output-summary NAME` | `output_summary.json` | Filename for dataset-level summary. |
| `--image-summary NAME` | `image_summary.json` | Filename for per-image summary. |
| `--num-workers N` | auto, up to 4 | Number of worker threads. Use `1` for sequential processing. |
| `--no-progress` | off | Disable the tqdm progress bar. |
| `--verbose` | off | Print detailed run context plus per-image, dataset, and per-class metrics. |

---

## API server

The unified `precisionai-agrieval-api` server exposes segmentation evaluation alongside embedding evaluation on a single FastAPI instance.

```bash
# Start the unified API server (default port 8000)
precisionai-agrieval-api --dataset-root /path/to/dataset

# Interactive API docs
open http://localhost:8000/docs
```

The `--dataset-root` flag (or the `PAI_DATASET_ROOT` environment variable) sets the base directory against which relative paths in request bodies are resolved. Absolute paths bypass it entirely.

### `POST /v1/segmentation/evaluate`

```bash
curl -X POST http://localhost:8000/v1/segmentation/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "pred_dir":    "predictions/batch_01",
    "masks_dir":   "ground_truth/batch_01",
    "classes_path": "class_map.json",
    "output_dir":  "results/batch_01"
  }'
```

**Request body fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `pred_dir` | string | — | *(required)* Path to predicted masks directory. |
| `masks_dir` | string | — | *(required)* Path to ground-truth masks directory. |
| `classes_path` | string | — | *(required)* Path to AgriBench class-definition JSON. |
| `output_dir` | string \| null | `null` | Output directory for JSON files. `null` skips file output. |
| `output_summary_name` | string | `"output_summary.json"` | Filename for dataset-level summary. |
| `image_summary_name` | string | `"image_summary.json"` | Filename for per-image summary. |
| `num_workers` | integer \| null | `null` | Worker threads for mask-pair processing. `null` selects auto, up to 4. |
| `dataset_root` | string \| null | `null` | Base path for relative path fields. Falls back to `PAI_DATASET_ROOT`. |

**Response body** — mirrors `output_summary.json` with `image_summary` appended:

```json
{
  "n_images": 20,
  "classes": {
    "background":     { "iou": 0.9412, "dice": 0.9698, "accuracy": 0.9601 },
    "Crop | Soybean": { "iou": 0.7834, "dice": 0.8786, "accuracy": 0.8901 },
    "Weed | Grass":   { "iou": null,   "dice": null,   "accuracy": null   }
  },
  "summary": { "mIoU": 0.8623, "mAcc": 0.9251, "FWIoU": 0.9117 },
  "image_summary": {
    "A1/pai-abc123.png": {
      "classes": { "...": "..." },
      "summary": { "mIoU": 0.8812, "mAcc": 0.9408, "FWIoU": 0.9322 }
    }
  }
}
```

HTTP 400 is returned for any input validation error (unknown colours, size mismatch, missing ground-truth mask, non-contiguous class IDs).

---

## Python usage

```python
from precisionai.agrieval.seg import run_seg_eval

dataset_summary, image_summary = run_seg_eval(
    pred_dir="predictions/",
    masks_dir="ground_truth/",
    classes_path="class_map.json",
    output_dir="results/",
    num_workers=4,
    verbose=True,
)

print(dataset_summary["summary"]["mIoU"])    # dataset-level mIoU
print(image_summary["A1/pai-abc123.png"]["summary"]["FWIoU"])  # per-image FWIoU
```

Calling `run_seg_eval` with `output_dir=None` skips JSON file writing and returns the result dicts only:

```python
dataset_summary, image_summary = run_seg_eval(
    pred_dir="predictions/",
    masks_dir="ground_truth/",
    classes_path="class_map.json",
    output_dir=None,
)
```

Individual metrics are also available as pure functions:

```python
import numpy as np
from precisionai.agrieval.seg.metrics import (
    confusion_matrix,
    per_class_iou,
    per_class_dice,
    per_class_accuracy,
    mean_iou,
    mean_accuracy,
    frequency_weighted_iou,
)

# Build a confusion matrix from flat class-ID arrays
pred_ids = np.array([0, 0, 1, 2], dtype=np.int32)
gt_ids   = np.array([0, 1, 1, 2], dtype=np.int32)
cm = confusion_matrix(pred_ids, gt_ids, n_classes=3)

iou  = per_class_iou(cm)   # shape (n_classes,) — NaN for absent classes
dice = per_class_dice(cm)
acc  = per_class_accuracy(cm)

print(f"mIoU:  {mean_iou(iou):.4f}")
print(f"mAcc:  {mean_accuracy(acc):.4f}")
print(f"FWIoU: {frequency_weighted_iou(cm, iou):.4f}")
```

---

## Validation errors

| Condition | Exception |
|---|---|
| Colour in a mask not defined in the class-definition file | `ValueError: Mask '...' contains N unknown color(s) not in classes.json: (R, G, B), …` |
| Predicted mask spatial dimensions differ from ground-truth | `ValueError: Size mismatch for '...': prediction (H, W) vs ground truth (H, W)` |
| A predicted mask has no corresponding ground-truth mask | `FileNotFoundError: Prediction '...' has no corresponding ground-truth mask in '...'` |
| No supported image files found in `pred_dir` | `ValueError: No supported image files found in '...'` |
| Class IDs in class-definition file are not contiguous from 0 | `ValueError: Class IDs must be contiguous 0..N-1; got: [...]` |
| Unrecognised class-definition JSON schema | `ValueError: Unrecognised class-definition format in '...'. Expected a 'classes' key` |
