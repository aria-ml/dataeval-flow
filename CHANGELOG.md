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
- Dataset loaders ported from `maite-datasets` to `datamaite`, including HuggingFace class-label support
- Coverage gap analysis now computes factor-to-class mutual information the way `Balance` does. It previously
  passed each factor's original continuous/discrete nature as `discrete_features`, but `factor_data` holds integer
  codes for every factor, so a binned continuous factor was marked continuous and routed to the neighbor-based
  estimator — moving its score away from the one `Balance` reports for the same data, and making it depend on the
  seed. Gap findings could therefore differ depending on whether `balance` was enabled, since the two paths feed the
  same `gap_mi_threshold`
- Container mounts follow SDP IR conventions
- Text report **METADATA FACTORS** lists per-bin and per-category detail only for factors with 12 or fewer buckets.
  Above that it reports the count and how the buckets were populated, and labels a factor holding one category per
  sample `(one per sample)` — an identifier column such as `file_name` previously printed one line per sample. The
  result envelope is unchanged and still carries the full map
- Classwise outlier pivots report `count_basis` (`image` / `annotation`) where they previously reported `level`
  (`image` / `target`). The old key collided with DataEval's `level`, which names metadata levels (`unit`,
  `instance`, `track`, `sequence`) and duplicate levels (`item`, `target`) — three different meanings under one key.
  Consumers reading `level` off a classwise finding need updating.

### Fixed

- Binning configuration silently discarded by the data analysis, data cleaning, and OOD detection workflows
- OOD detection factors misaligned against `class_labels` on object detection datasets, read from the interleaved frame
- Per-factor metadata summaries read off image-level rows, hiding detection-level factors behind replicated copies

### Added

- Per-factor `level` and `is_binned` in metadata summaries, the only record that a factor reached evaluators as codes
- `dropped_factors` in metadata summaries — vector statistics (`histogram`, `percentiles`, `center`) with no column form
- `invalid_box` carried through as a factor, previously dropped alongside the hash columns
- DataEval binning and `value_range` diagnostics pinned at `WARNING` so raising `lib_level` no longer suppresses them
- Data coverage workflow (`data-coverage`) — class balance, metadata factor gaps, ontology findings, and per-class
  embedding variety, with an `ontology` extra for loading a label space from an RDF artifact
- `metadata_auto_bin_method`, `metadata_exclude`, and `metadata_continuous_factor_bins` on the data analysis, data
  cleaning, data coverage, and OOD detection workflows
- `task` field on HuggingFace datasets, selecting the loader explicitly rather than inferring it
- MAITE entry points for the dataset adapters and task runner, so both are discoverable by other MAITE-aware tools
- Python 3.13 and 3.14 support; CI now runs the full 3.10 through 3.14 matrix
- `metadata_binning` in the result envelope — per-factor type, level, and what discretizing did: observed range and
  population of every bin, the ordinal-to-value map of every category, what was excluded, and what was dropped.
  Binning decides what evaluators read, and was previously reported only in logs no envelope referenced
- `diagnostics` in the result envelope, capturing the library warnings a run raised rather than leaving them to a
  log file the envelope does not point at
- Metadata factor records now emitted by the data cleaning and OOD detection workflows, which accepted the binning
  configuration but reported none of its effect
- Factor and binning detail rendered in the text report, alongside any diagnostics
- `value_range` on the data analysis, data cleaning, data coverage, data prioritization, and OOD detection
  workflows. Float image data has no inherent range, and the statistics that need one — the whole visual family,
  pixel histogram and entropy, and dimension depth — answer `NaN` without it. It participates in the stats cache
  key, so two runs declaring different ranges do not share an entry

### Removed

- Unused `per_channel` parameter from `get_or_compute_stats`, `scope_key`, and `DatasetCache.load_or_compute_stats`
- COCO and YOLO config fields that were never supported by the underlying loaders

## v0.1.2

### Added

- Custom preprocessors module and the `ToRGB` preprocessor, coercing mixed-channel datasets to three channels

### Infrastructure

- Documentation caching for the publish job

## v0.1.1

### Added

- Parameter sweep workflow (`parameter-sweep`), running a workflow across a grid of parameters and comparing results
- Poetry and conda packaging support alongside pip and uv
- Dynamic versioning via hatch-vcs

### Changed

- DataEval bumped to v1.0.6
- Logging patterns normalized against DataEval's
- HuggingFace dataset paths must now include the namespace

### Fixed

- `DataSplitting` workflow was not exported and could not be referenced from a config

### Removed

- CUDA 12.4 container variant

### Infrastructure

- Container hardening for SDP 1.2 — pinned Trivy, per-image SBOM attestation, GitLab Container-Scanning template,
  PEP/OCI-compliant versioned tags with promotion gated on image scans
- Markdown lint and link-check jobs; governance and DSOR documentation
- FR/NFR verification tests and metarepo artifacts

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
