# Changelog

## Unreleased

### Added

- Top-level `ontologies:` key defining named label spaces referenced by workflows
- `concepts:` on ontology entries to add or override concepts by id without altering source artifacts
- Workflow `ontology:` resolves pool names first, falling back to file paths for backward compatibility
- `merge:` on sources to concatenate multiple inputs into one corpus, unified via `Relabel` views
- Top-level `exports:` key exporting sources to COCO, YOLO, Hugging Face, or VisDrone format with `provenance.json`
- `ontology:` support across all workflows, attaching the vocabulary audit digest to results
- `label_space` on result envelopes, recording conformed vocabulary and matching audit digest
- `channel_groups:` on datasets, measuring band groups separately as `<group>_<statistic>` columns
- Top-level `stats:` key defining policies for measured statistics, background inclusion, and outlier/factor views
- `format: demo` dataset loader resolving tutorial datasets from a fixed table without arbitrary imports
- `crop_padding` and `crop_min_size` on `data-coverage`, with `dropped_detections` reporting omitted annotations

### Changed

- Metadata cache key now includes the stats policy `factor_identity()`, recomputing metadata archives on upgrade
- Stats cache keys remain unaffected, preserving cached stats for policies without band groups or background

### Fixed

- Hash dataset elements lacking `__repr__` by type and contents rather than memory address, enabling cache reuse
- Include source views in cache keys, so editing one invalidates the embeddings, metadata, and statistics it produced
- Relative `ontology:` paths now resolve against the run's data root rather than the process root
- Restrict outlier detection to configured `outlier_flags`, preventing shared caches from widening column checks
- Restrict duplicate detection to configured `duplicate_flags`, preventing shared cache widening
- Restrict injected metadata factors to configured `intrinsic_factors` instead of cache state
- Record band-group statistics in `injected_factors` rather than misattributing them as dataset columns
- Restrict cross-split label parity to shared classes, preventing chi-square errors on gapped label spaces
- Exclude single-split classes from the parity test and report them in `label_overlap` instead
- Restrict cross-split duplicate detection to common columns, preventing crashes on divergent stat caches

### Removed

- Poetry packaging support; install with uv, pip, or conda instead

## v0.2.2

### Added

- `--task NAME` on the headless CLI to select tasks by name, running in the given order regardless of `enabled`
- `--fail-on-warning` on the headless CLI, exiting non-zero when health thresholds are breached
- `--version` flag reporting the installed build
- `dataeval-flow workflows` command listing workflow types and printing parameter JSON Schemas
- Result envelopes carry a `health` block (`status`, `warnings`, `findings`) beside `metadata`
- `WorkflowResult.health`, `.warning_count`, and `.findings` exposing report health programmatically
- `select_tasks` helper resolving executed tasks so callers can pair them with results

### Fixed

- Runner crash when pairing results against tasks marked `enabled: false`

## v0.2.1

### Added

- `split` on YOLO dataset configs, selecting `train`, `val`, or `test` from the dataset root
- `yaml_file` and `ann_dir` on YOLO dataset configs for non-standard layout paths

### Changed

- Bumped CUDA variants from `cu118` / `cu128` to `cu126` / `cu130`, tracking PyTorch builds
- Split `onnx-gpu` extra into `onnx-cu126` and `onnx-cu130` to match the selected CUDA runtime
- Capped `opencv` extra at `4.12.0.88` for compatibility with FIPS-enabled systems
- Empty-load errors now name the requested split and expected YOLO directory layout

### Fixed

- Removed obsolete `images_dir`, `labels_dir`, and `classes_file` fields from example configs

## v0.2.0

### Changed

- Pixel statistics reported in stored units (`normalize_pixel_values=False`)
- Stored image statistics and `Balance` results need recomputing before comparison against fresh ones
- `Balance` values corrected for chance, so factors of differing cardinality rank differently
- Outlier flags and visual statistics unmoved by the pixel rescale
- Image statistic factors renamed for their level — `brightness` to `unit_brightness`, `target_*` to `instance_*`
- Image statistic factors binned over their own population rather than the detection rows
- `data-analysis` gives every split one encoding from the reference split, moving per-split `Balance` and `Diversity`
- Metadata factor records report `encoding` and `fit` in place of the previous bin and category fields
- Classwise outlier pivots report `count_basis` (`image` / `annotation`) instead of `level`
- Coverage gap analysis computes factor-to-class mutual information through `Balance`
- Text report **METADATA FACTORS** names bins from their edges and states each factor's provenance
- Text report **METADATA FACTORS** limits per-bucket detail to factors with 12 or fewer buckets
- Workflow cache bumped to `v1`; the version tracks releases rather than individual format changes
- Metadata archives serve only the binning configuration that wrote them
- Dataset loaders ported from `maite-datasets` to `datamaite`, including HuggingFace class-label support
- Container mounts follow SDP IR conventions

### Fixed

- Binning configuration silently discarded by the data analysis, data cleaning, and OOD detection workflows
- OOD detection factors misaligned against `class_labels` on object detection datasets
- Per-factor metadata summaries read off image-level rows, hiding detection-level factors
- DataEval warnings dropped by a handler-only collector, and lost to the once-per-location registry
- Observed spans in the text report truncated to four significant figures

### Added

- Data coverage workflow (`data-coverage`) — class balance, metadata factor gaps, ontology and embedding findings
- `ontology` extra for loading a label space from an RDF artifact
- Top-level `metadata:` key defining named metadata policies, referenced by workflows by name
- Metadata policy fields `encoding`, `factor_levels`, `strict`, and `reference_split` for declaring the encoding
- Metadata policy fields `auto_bin_method`, `exclude`, `continuous_factor_bins`, and `factor_source`
- Naming a metadata policy and a per-workflow `metadata_*` field on the same workflow is an error
- Metadata policies resolved and checked before the dataset is read
- `dataeval-flow encoding` writes the encoding descriptor a result was computed under, ready to review and commit
- A run with `-o` writes `results/encoding.json` beside its results
- `metadata_factor_source` (`coded` / `values` / `auto`) on every workflow that reads metadata
- `value_range` on the data analysis, cleaning, coverage, prioritization, and OOD workflows; keyed into the stats cache
- Result envelopes record `metadata_binning`, `encoding_digest`, and `diagnostics`
- Result envelopes carry the warnings DataEval raises, not only its log records
- `metadata_binning` records `descriptor_version`, read from what DataEval wrote rather than assumed
- `encoding_digest` is `None` where a multi-split workflow's splits do not share one encoding
- Text report names the factors nobody reviewed, and says whether a run's splits are comparable
- Metadata cache entries write a sidecar naming the encoding the archive was built under
- Metadata factor records emitted by the data cleaning and OOD detection workflows
- `invalid_box` carried through as a factor
- `dropped_factors` naming the vector statistics that have no column form
- DataEval binning and `value_range` diagnostics pinned at `WARNING`
- `task` field on HuggingFace datasets, selecting the loader explicitly
- MAITE entry points for the dataset adapters and task runner
- Python 3.14 support; CI runs the full 3.10 through 3.14 matrix

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

- Container hardening for SDP 1.2 (pinned Trivy, SBOM attestation, PEP/OCI tags gated on scans)
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
