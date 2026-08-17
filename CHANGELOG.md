# Changelog

## Unreleased

### Changed

- Pixel statistics reported in stored units (`normalize_pixel_values=False`) — a 12-bit mean of `0.4944` reads `2024.59`
- Outlier flags unmoved by the pixel rescale; every threshold method offered is location-scale equivariant
- Visual statistics unmoved as well, now read against the 0–255 display range in any encoding
- Image statistic factors renamed for their level — `brightness` to `unit_brightness`, `target_*` to `instance_*`
- Image statistic factors binned over their own population rather than all over the detection rows
- `Balance` values corrected for chance, so factors of differing cardinality rank differently
- Workflow cache bumped to `v3` for the pixel rescale; reclaim old entries with `rm -rf <cache_dir>/v2`
- Stored image statistics and `Balance` results need recomputing before comparison against fresh ones

### Fixed

- Binning configuration silently discarded by the data analysis, data cleaning, and OOD detection workflows
- OOD detection factors misaligned against `class_labels` on object detection datasets, read from the interleaved frame
- Per-factor metadata summaries read off image-level rows, hiding detection-level factors behind replicated copies

### Added

- Per-factor `level` and `is_binned` in metadata summaries, the only record that a factor reached evaluators as codes
- `dropped_factors` in metadata summaries — vector statistics (`histogram`, `percentiles`, `center`) with no column form
- `invalid_box` carried through as a factor, previously dropped alongside the hash columns
- DataEval binning and `value_range` diagnostics pinned at `WARNING` so raising `lib_level` no longer suppresses them

### Removed

- Unused `per_channel` parameter from `get_or_compute_stats`, `scope_key`, and `DatasetCache.load_or_compute_stats`

## v0.1.0

### Features

- Workflow orchestration framework with registry, task runner, and pipeline configuration
- Data cleaning workflow (outlier + duplicate detection) with text reports
- Data analysis workflow with statistical summaries
- Dataset splitting workflow with stratified and random strategies
- Drift detection workflow with classwise drift support
- Out-of-distribution (OOD) detection workflow
- Prioritization workflow for dataset sample ranking
- Interactive TUI application for config editing, task execution, and result viewing
- Simple CLI config builder for environments without TUI
- Disk-backed and in-memory caching layer for workflow results
- Support for HuggingFace, MAITE, TorchVision, ImageFolder, COCO, and YOLO datasets
- Embedding extraction with ONNX inference support
- Multi-variant Docker containers (CPU, CUDA 11.8, 12.4, 12.8)
- Cosign-signed container images published to Harbor registry
- Sphinx documentation with tutorial notebooks

### Infrastructure

- GitLab CI/CD pipeline with lint, type check, test, security scanning, and container publishing
- GitHub Actions workflow for PyPI publishing via trusted publisher
- Nox automation for lint, type, test, schema validation, and Dockerfile generation
- JATIC-compliant security scanning (SAST, dependency scanning, secret detection, SBOM)
- Trivy container vulnerability scanning
- 90%+ test coverage enforcement
- 100% type completeness score
