# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-07-08

### Changed

- Split `precisionai.agrieval.emb.services.evaluate` (previously ~1,100 lines) into focused modules: `labels` (path parsing, metadata wiring, dataset loading), `knn_metrics` (KNN metric bundles, per-class breakdowns, group analysis), and `serialization` (JSON conversion). The evaluation entry points (`run_image2image_eval`, `run_plant2image_eval`, `run_plant2plant_eval`) remain importable from `services.evaluate` and the top-level `precisionai.agrieval.emb` unchanged.

### Added

- `precisionai.agrieval.dpt` — dense patch token evaluation: unsupervised geometry diagnostics (effective rank, PCA explained variance, anisotropy, uniformity), per-entry spatial health (patch norm statistics, neighbor smoothness, outlier fraction), and optional label-aware metrics (kNN confusion and purity, per-class geometry) against color-coded ground-truth masks downsampled to the patch grid.
- Two sibling wirings on the unified FastAPI server: `POST /v1/dense-patch-tokens/evaluate/tiles` for tile crops and `POST /v1/dense-patch-tokens/evaluate/image` for whole (untiled) images — identical rules and metrics, differing only in field naming — plus the `run_dpt_eval`/`load_tiles`/`load_tile_placement`/`TilePlacement`, `run_dpt_image_eval`/`load_images`, and `print_result` top-level re-exports.
- Binary ingestion: `tiles_path`/`images_path` accept a batched `.npz` archive (loaded with `allow_pickle=False`) as an alternative to inline JSON `tiles`/`images` — the recommended transport for realistically sized feature maps. `tiles_path` archives carry `feature_maps`/`tile_image_id`/`tile_index`/`tile_y0`/`tile_x0`/`filenames`/`meta`, enabling crop-aware label alignment (`load_tile_placement`, `run_dpt_eval(tile_placement=...)`) against one whole-image ground-truth mask per source image — cropped to each tile's exact pixel rectangle, so overlapping tiles are handled correctly — instead of one mask per tile; `images_path` archives carry `features`/`filenames`.
- `docs/DENSE_PATCH_TOKENS.md` data-format and metrics documentation, `examples/dpt/example.py` runnable script, `examples/dpt/example.ipynb` notebook demonstrating both wirings side by side, and `tests/dpt/` test suite.
- `precisionai.agrieval.emb.metrics.separation` — split-free label separation metrics operating on any `[N, D]` feature matrix plus per-row labels, with no train/eval split anywhere: `silhouette_cosine` and `calinski_harabasz` (cluster geometry), `kmeans_label_agreement` (ARI/NMI cluster recoverability; requires scikit-learn), and `pc1_auroc` (sign-free linear separability of a foreground/background partition along the corpus's first principal component). Silhouette and Calinski-Harabasz match scikit-learn's reference implementations without requiring it.
- `dpt` label-aware evaluation reports a `separation` block (`silhouette`, `calinski_harabasz`, `ari`, `nmi`, `pc1_auroc`) alongside `knn_confusion` in both wirings. Calinski-Harabasz and PC1-AUROC run on the full patch corpus; silhouette and ARI/NMI run on the same seeded `max_patches` subsample as the kNN metrics. PC1-AUROC scores the class named `background` against all other classes. A metric whose preconditions fail (a single class present, no `background` class in the class map, scikit-learn absent) is reported as `null` with an explanatory entry in `warnings` instead of failing the run. `print_result` prints the new block.

## [0.1.0] - Initial public release

- `precisionai.agrieval.emb` — embedding evaluation: KNN retrieval benchmarking (nDCG, MAP, MRR, purity), geometry diagnostics, and interactive visualizations.
- `precisionai.agrieval.seg` — semantic segmentation evaluation: per-class IoU, Dice/F1, accuracy, mIoU, mAcc, FWIoU from color-coded masks.
- Unified FastAPI server exposing both subpackages, plus a standalone segmentation CLI (`precisionai-agrieval-seg`).
- Top-level re-exports on both subpackages so the primary entry points can be imported directly, e.g. `from precisionai.agrieval.emb import run_image2image_eval` and `from precisionai.agrieval.seg import run_seg_eval`.

[Unreleased]: https://github.com/Precision-AI-Inc/agrieval/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Precision-AI-Inc/agrieval/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Precision-AI-Inc/agrieval/releases/tag/v0.1.0
