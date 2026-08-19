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
- Metadata factor records now carry the encoding DataEval applied rather than one reconstructed from the data. Each
  factor reports `encoding` — the edges or the vocabulary, `provenance` (`edges` / `count` / `declared` / `accepted` /
  `derived`), and the `method` that placed them — beside `fit`, which is what this run's rows did against it. The two
  were previously collapsed into observed per-bin ranges, which described the draw rather than the decision: the same
  cut over a different sample recorded different numbers, a `continuous_factor_bins` count discarded where its
  interior cuts actually landed, and a declared cutoff never survived into its own label, so
  `{"temp_c": [-inf, 0.0, inf]}` was reported and printed as `[-40, -0.3]`. Consumers reading `bins`, `bin_count`,
  `bins_requested`, `binned_by`, `categories`, `category_count`, `is_binned` or `is_digitized` off a factor record
  need updating; `encoding` and `fit` replace them
- Text report **METADATA FACTORS** names bins from their edges rather than their contents (`< 0`, `[0, 10)`,
  `>= 10`), states each factor's provenance, and reports declared bins nothing reached as `empty`. Levels are listed
  in value order rather than code order, since a vocabulary grows append-only and a late level carries an
  out-of-order code. The names come from `Metadata.code_names`, so they are the strings DataEval's own outputs use —
  `ParityOutput.insufficient_data` keys and `label=` axis groups — rather than a second rendering that would disagree
  with them, and they travel in the envelope so an archived result re-renders identically
- Observed spans in the text report write large magnitudes out in full. Four significant figures printed every
  epoch-millisecond span as `[1.787e+15, 1.787e+15]` however wide it was
- Coverage gap analysis now runs `Balance` for its factor-to-class mutual information rather than calling
  `mutual_info` with flags chosen to imitate it. Imitating it is no longer possible: `factor_source` decides per
  factor whether the codes or the measured values are read, consulting the encoding record's provenance, and a
  per-column boolean cannot express a per-column choice between two estimators. Gap findings could otherwise differ
  depending on whether `balance` was enabled, since both paths feed the same `gap_mi_threshold`
- Workflow cache bumped to `v4`; reclaim old entries with `rm -rf <cache_dir>/v3`. Metadata archives are now named
  `metadata_{policy_hash}.dem` and serve only the binning configuration that wrote them. DataEval v1.1 persists the
  encoding record inside the archive and restores it *underneath* whatever the reader declares, so a shared archive
  handed a run configured for one auto-bin method the edges another run had derived — silently, and for every factor
  the reader did not name. Reading at a configuration the cache has not seen now costs a rebuild rather than a
  wrong answer
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
- Top-level `metadata:` key defining named metadata policies, referenced by workflows the way `datasets`, `views`,
  `sources` and `extractors` already are. A policy carries `encoding`, `factor_levels`, `strict`, `auto_bin_method`,
  `exclude`, `continuous_factor_bins`, `factor_source` and `reference_split`. Defined once and shared, because the
  encoding is a decision rather than a per-workflow setting: two workflows over one dataset that cut it differently
  produce numbers that land in one result file and cannot be compared. The per-workflow `metadata_*` fields still
  work and mean the same things; naming a policy and setting one of them on the same workflow is an error rather than
  a merge
- `encoding` on a metadata policy — a path, under the data root, to a committed descriptor. Applies the recorded cuts
  and vocabularies instead of deriving them from this run's draw, which is what lets two runs over different data be
  compared. A vocabulary it names grows by appending, so codes already assigned keep meaning what they meant
- `factor_levels` and `strict` on a metadata policy, for declaring a vocabulary ahead of the data and for closing it.
  `strict` is refused over a descriptor whose vocabularies still read `provenance: "derived"` — the state a
  descriptor exported from an exploratory run is in — because `strict` does not consult provenance and would
  otherwise enforce a taxonomy nobody decided on, failing the run on the first new category
- Metadata policies are resolved and checked before the dataset is read, so a missing descriptor, a factor declared
  through two channels, or a reference naming no policy costs a config error rather than a run
- `metadata.encoding_digest` on every result envelope — a fingerprint of the encoding every factor was read under.
  Comparing two runs is only sound if each can say which cuts produced it; without it a bias score that moved is
  unattributable between *the override worked* and *the data changed*. It covers the policy rather than the rows, so
  it holds still when only the data changes. For a multi-split workflow it is set only where every split shares one
  encoding and is `None` otherwise, and the text report says which case it is — splits binned automatically routinely
  land on different edges, and their per-factor statistics are not comparable when they do
- `metadata_factor_source` (`coded` / `values` / `auto`) on every workflow that reads metadata, and recorded in the
  result envelope. It selects which representation the bias statistics read, so it moves every number they report —
  two workflows meant to be compared want the same one
- Result envelopes now carry the warnings DataEval raises, not only its log records. The advice a caller is meant to
  act on — factors binned with nobody's say-so, a declared cut that has stopped fitting the data, an encoding naming
  a factor that is not one — is raised with `warnings.warn` rather than logged, so a handler-only collector archived
  the per-factor footnotes and dropped every finding. Warnings are also no longer lost to the once-per-location
  registry, which had attributed several workflows to one call site in this package and told only the first
- Metadata cache entries write a `metadata_{policy_hash}.json` sidecar naming the encoding the archive was built
  under, so a cache directory can be read without loading an archive
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
