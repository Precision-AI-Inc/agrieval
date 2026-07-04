# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-07-04

### Changed

- Split `precisionai.agrieval.emb.services.evaluate` (previously ~1,100 lines) into focused modules: `labels` (path parsing, metadata wiring, dataset loading), `knn_metrics` (KNN metric bundles, per-class breakdowns, group analysis), and `serialization` (JSON conversion). The evaluation entry points (`run_image2image_eval`, `run_plant2image_eval`, `run_plant2plant_eval`) remain importable from `services.evaluate` and the top-level `precisionai.agrieval.emb` unchanged.

### Added

- `precisionai.agrieval.dpt` — dense patch token evaluation: unsupervised geometry diagnostics (effective rank, PCA explained variance, anisotropy, uniformity), per-tile spatial health (patch norm statistics, neighbor smoothness, outlier fraction), and optional label-aware metrics (kNN confusion and purity, per-class geometry) against colour-coded ground-truth masks downsampled to the patch grid.
- `POST /v1/dense-patch-tokens/evaluate` endpoint on the unified FastAPI server, plus the `run_dpt_eval` and `load_tiles` top-level re-exports.
- Binary tile ingestion: `tiles_path` accepts a directory of `.npy` files or a single `.npz` archive (loaded with `allow_pickle=False`) as an alternative to inline JSON `tiles` — the recommended transport for realistically sized feature maps.
- `docs/DENSE_PATCH_TOKENS.md` data-format and metrics documentation, `examples/dpt/example.py` runnable script, and `tests/dpt/` test suite.

## [0.1.0] - Initial public release

- `precisionai.agrieval.emb` — embedding evaluation: KNN retrieval benchmarking (nDCG, MAP, MRR, purity), geometry diagnostics, and interactive visualizations.
- `precisionai.agrieval.seg` — semantic segmentation evaluation: per-class IoU, Dice/F1, accuracy, mIoU, mAcc, FWIoU from colour-coded masks.
- Unified FastAPI server exposing both subpackages, plus a standalone segmentation CLI (`precisionai-agrieval-seg`).
- Top-level re-exports on both subpackages so the primary entry points can be imported directly, e.g. `from precisionai.agrieval.emb import run_image2image_eval` and `from precisionai.agrieval.seg import run_seg_eval`.

[Unreleased]: https://github.com/Precision-AI-Inc/agrieval/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Precision-AI-Inc/agrieval/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Precision-AI-Inc/agrieval/releases/tag/v0.1.0
