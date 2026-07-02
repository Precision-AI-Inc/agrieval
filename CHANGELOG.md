# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Initial public release

- `precisionai.agrieval.emb` — embedding evaluation: KNN retrieval benchmarking (nDCG, MAP, MRR, purity), geometry diagnostics, and interactive visualizations.
- `precisionai.agrieval.seg` — semantic segmentation evaluation: per-class IoU, Dice/F1, accuracy, mIoU, mAcc, FWIoU from colour-coded masks.
- Unified FastAPI server exposing both subpackages, plus a standalone segmentation CLI (`precisionai-agrieval-seg`).

[Unreleased]: https://github.com/Precision-AI-Inc/agrieval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Precision-AI-Inc/agrieval/releases/tag/v0.1.0
