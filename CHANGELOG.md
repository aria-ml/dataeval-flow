# Changelog

## Unreleased

v0.3 breaks more than v0.2 did. Every workflow type is now a preset, a chain of steps, and several types are renamed
or removed. Read [What's new in v0.3](https://dataeval-flow.readthedocs.io/en/latest/home/whats_new/v0.3.html) before
you upgrade. It covers each change that a v0.2 config or script needs. "Where old keys went", at the end of this
section, maps every old key to its replacement.

### Added

#### Audit, presets and custom workflows

- The `audit` preset: one chain over train and each evaluation split that gives a verdict before training
  - The verdict is "Not ready", "Ready with caveats" or "Ready", with findings, next steps and a record of what it read
  - `blocking:` names the check types whose warnings block readiness, and `accepted:` records why a warning is accepted
  - `ChainResult.verdict` (`verdict` in the JSON) holds the level and the unaccepted warnings behind it
  - It can run as a step of a custom workflow; only one step per workflow may give a verdict
- `result: require`, `--require` and `DATAEVAL_REQUIRE`: exit 4 when a task's verdict is worse than the level given
- The `bias` preset (class balance and how metadata factors relate to the class, without an extractor) and the
  `taxonomy` preset (a dataset's labels judged against a declared ontology)
- Presets: workflow types whose settings expand into a chain of steps that runs like a custom workflow
  - As a task, a preset returns a `ChainResult`; as a step, its chain runs inside yours as `<step>/<name>`
- Custom workflows: a `workflows:` entry with `inputs:` and `steps:` chaining evaluators, presets and transforms
  - Each step reads an input or an earlier step by address (`clean`, `split.train`, `kfold.train[0]`)
  - A step runs once per element of a list; a failed step skips only the steps that read it
  - `empty:`, `pairs: true`, `groups:` and `scopes:` shape which lists a step reads and how its findings are grouped
  - `CustomWorkflowConfig` and `StepEntry` build one in Python, and `save()` writes it into a config file
- Transforms: `view`, `merge`, `split`, `kfold`, `wrap`, `select`, `remove`, `conform`, `export` and `collect`; plug-in
  transforms register under `dataeval_flow.transforms`
- `matrix:` on a task runs its entry once per combination of the values it lists, returning one `MatrixResult`
- `by: class` on an evaluate step runs its evaluator once per class or group of classes; `by: predicted` keys a step by
  the class a model predicts
- `dataeval-flow steps [NAME] [--json]` and `list_steps()` list every chainable step with its ports and settings schema
- `ChainResult` keeps each step's outcome in `steps`, readable after a failure, and each Dataset's `lineage`

#### Evaluators and checks

- `sufficiency` evaluator and `performance-sufficiency` check, both experimental: how much training data a model needs
  - `sufficiency` runs DataEval's `Sufficiency` on two sources, by position: the training set, then the test set
  - Your training code is a `Trainer` subclass in a Python file under the data root's `trainers/` folder, named by the
    entry's `trainer: {path, class, params}` and loaded only when the run starts
  - The result records the trainer's `path`, `class` and `sha256`, and the fitted curve's `params`
- `feasibility` evaluator and `performance-feasibility` check, both experimental: is a performance target reachable
  - `feasibility` runs DataEval's `Feasibility` on one classification source, with embeddings from the task's
    extractor, which must not have seen the labels
  - Each `target:` gets one answer: demonstrated feasible, ruled out under recorded assumptions, not ruled out,
    insufficient evidence or an evidence conflict
  - `alpha` is one error budget for every bound, and `allowances: {beta, gamma}` are your assertions, recorded as
    assumed
- Evaluators: `balance`, `diversity`, `parity`, `representation`, `label-alignment`, `factor-summary`, `factor-triage`,
  `factor-leakage`, `label-reconciliation`, `ontology-validation`, `coverage`, `prioritization`, `completeness`,
  `divergence`, `drift-univariate`, `drift-mmd`, `drift-kneighbors`, `drift-wasserstein`, `drift-domain-classifier`,
  `ood-kneighbors`, `ood-domain-classifier`, `profile` and `content-digest`
- Checks: `leakage`, `class-sufficiency`, `untrained-classes`, `shortcut-risk`, `eval-coverage`, `factor-parity`,
  `embedding-divergence`, `ood`, `ood-agreement`, `drift`, `class-stratification`, `uncovered-items`, `class-coverage`,
  `dimensional-completeness`, `factor-coverage-gaps`, `class-shortfall`, `factor-issues`, `leaf-coverage`,
  `label-conformance`, `label-mergeability` and `ontology-structure`
- Combines: `ood-union`, `factor-predictors`, `factor-deviation` and `factor-gaps`
- `uncertainty` extractor: normalized entropy per prediction from an ONNX classifier or detector, for unlabeled drift
  and OOD
- `select:` on `prioritization` (`n` or `fraction`) keeps the top items of each pool

#### Datasets and provenance

- Top-level `ontologies:`, `exports:` (COCO, YOLO, Hugging Face or VisDrone, with `provenance.json`) and `stats:` keys;
  `merge:` on sources, `channel_groups:` on datasets, and a `format: demo` loader for the tutorial datasets
- `dataset_digest()` and `dataset_manifest()`: order-independent SHA-256 digests of a dataset, and per-item hashes that
  show which items changed; the `dataeval-flow` command writes each `content-digest` run's manifest under
  `results/manifests/`
- `dataeval-flow verify` checks that a source still holds a manifest's items, naming changed, missing and added ones
- `load_source()` loads a configured source the way a run reads it, with its view and merge applied
- `provenance:` on `datasets:` entries records facts Flow can't measure, such as owner, license and collection date
- Exports record the digest of what they wrote, and every result's metadata carries `library_versions` and `device`
- Every tool runs on CUDA when PyTorch sees a GPU, and on the CPU otherwise; `dataeval_flow.set_device` picks the device

#### Reports and results

- `Result.to_html()` renders a self-contained, printable page with a card per finding, and `--output` writes it as
  `result.html`; it shows thumbnails of the items a finding names (`Result.assets`), which `--no-report-images`
  turns off
- The `result:` block names the result files and picks their formats, detail and text width, limits item tables with
  `max_rows:` and `preview_rows:`, and sets the exit-code gate with `fail_on:` (`failure`, `warning` or `never`);
  `--fail-on-warning` still overrides it
- `junit` and `markdown` result formats, for CI test views and summaries
- `--report-width` and `DATAEVAL_REPORT_WIDTH` set the text report's width
- A short chain report, from `report(detailed=False)` and the console without `-v`

#### Service preview

- `dataeval-flow serve` (new `service` extra, in every image): an HTTP service that queues pipelines and runs them
  - A preview: its API may change or be removed in any release without a deprecation cycle
  - `POST /v1/runs` takes a pipeline and its tasks; each run is a separate headless process, one at a time, with its
    snapshot, logs and result files under `<output>/runs/<id>/`
  - Runs can be cancelled, the history survives restarts, and a run's results, activity and events are available as it
    goes
  - `/healthz`, `/livez` and `/readyz` (IR-2.3-H-2, IR-2.3-S-1), and `/openapi.json` (IR-2.4-S-1)
  - `/v1/runs/{id}/items/{source}` serves the items a run read, each checked against the run's manifest first
  - Builder endpoints for editors that compose pipelines: `/v1/schema`, `/v1/capabilities`, `POST /v1/resolve` and
    `POST /v1/presets/decompose`

#### Python API and plugins

- Plug-in workflows, evaluators, extractors and image transforms through entry points in the
  `dataeval_flow.workflows`, `.evaluators`, `.extractors` and `.image_transforms` groups
- `run(config, data)` runs one workflow or evaluator on in-memory datasets, typed to the config's result
- `run_tasks(..., on_result=...)` calls back with each task's result as it finishes
- `evaluator:` on a task, or a top-level `evaluators:` key, runs one evaluator; `dataeval-flow evaluators` lists them
- Plugin helpers: `Preset.structure` names the settings that decide a preset's steps, `EvaluatorConfig.matrix_refusal`
  lets an evaluator refuse a `matrix:`, and `dataeval_flow.config.pool_ref` marks a config's reference fields
- `ExperimentalWarning`: a run warns with it once per process, the first time it uses an experimental step type
  (`parity` and `factor-parity` are experimental, as DataEval's `Parity` is)

#### Documentation

- Find the Right Step, What's new in v0.3, a Config Reference, evaluator and check catalogs, a Transform Catalog,
  "Workflows as Chains of Steps", a custom-workflow how-to and six evaluator how-tos
- The how-to guides follow the documentation style guide, and the long ones are split so that each covers one task; the
  numbered anchors of those guides, such as `#11-check-a-set-of-splits`, are gone

### Changed

#### Platform and DataEval 1.2

- Requires DataEval 1.2.0 or newer; `environment.yml` installs it from conda-forge
- Pillow (`>=13.0.0`) is a core dependency, used to encode thumbnails
- PyPI releases carry a signed PEP 740 attestation for each file
- A run that fails only on health warnings exits `3` (was `1`), so CI can tell it from a crash or a mistyped flag (`2`)
- The text report is 80 columns wide by default (was 90)
- The `dataeval-flow` command writes each task's result files as soon as the task finishes, so a killed run keeps them
- Metadata keeps missing values null, and drops factors mixing booleans with numbers or text as `mixed_types`
- Outliers fits a statistical threshold only to populations of at least 20; below that, `image-outliers`,
  `target-outliers` and `class-outliers` are not assessed, with DataEval's reason, and `extras.populations` records
  which populations were judged
- `class-coverage` flags the classes DataEval's Coverage flags, by its `is_clustered`, `is_one_dimensional` and
  `has_near_duplicates` columns (breaking)
  - The check's own `dispersion`, `isotropy` and `near_duplicates`, and `checks.class-coverage` on `scope` and `audit`,
    are removed and fail as unknown keys
  - The cutoffs are DataEval's, set on the `coverage` evaluator and on `scope`'s and `audit`'s `coverage:` block:
    `dispersion_factor`, `isotropy_factor` and `near_duplicate_threshold` (unset: 0.5, 0.5 and 0.1)
  - `isotropy_factor` is a share, in (0, 1), of the median isotropy of the classes that report one, where the old
    cutoff was absolute; DataEval 1.2 measures `near_duplicate_fraction` anew, so earlier fractions don't compare
- `drift-wasserstein` reports "Mean ratio" (was "Distance"), the statistic DataEval 1.2 decides on; a drift report's
  "Features drifted" is "Features flagged (diagnostic)"
- The metadata cache key includes the stats policy's `factor_identity()`, so metadata archives recompute on upgrade

#### Thresholds and defaults

- A check now warns only past its bound, not on reaching it
  - This changes `ood`, `ood-agreement`, `eval-coverage`, `drift`'s `chunk_percent` and `consecutive_chunks`, and
    `factor-coverage-gaps`; `consecutive_chunks` and `factor-coverage-gaps` now default to 2 (were 3)
- `eval-coverage` measures a split's flagged share against a baseline: what a split drawn like train flags anyway
  - The baseline is 100 - `threshold_perc` under `ood-kneighbors`, and 0 under other detectors, where the bands move
    from 2/10 to 1/9 percent flagged; `audit`'s `ood-kneighbors` takes DataEval's `threshold_perc` of 95
- Presets use DataEval's default for every setting, except that `prioritization` sorts `hard_first`, and `split` and
  `kfold` hold out `test_frac: 0.2` (and `split` `val_frac: 0.1`), stratified
- `class-imbalance` makes its finding on any Dataset with classes, so an unlabeled Dataset that declares classes now
  warns; `bias` no longer defaults `info` to 2.0, and `audit` defaults `empty` to `true`
- `label-health` lists every declared class, at 0 where it has no labels

#### Renames

- Presets are named for the DataEval module or question they answer; the old ids fail as unknown workflows
  - `data-bias` → `bias`, `data-cleaning` → `quality`, `data-coverage` → `scope`, `data-splitting` → `splits`,
    `data-prioritization` → `prioritization`, `metadata-triage` → `triage`, `label-space` → `taxonomy`
  - `drift-monitoring` and `ood-detection` merge into `shift`, and `data-analysis` becomes `audit`
  - Packages, classes (`QualityConfig`, …) and titles follow
- Five checks are renamed; the old names fail as unknown checks: `classwise-outliers` → `class-outliers`,
  `stratification` → `class-stratification`, `metadata-issues` → `factor-issues`, `mergeability` → `label-mergeability`
  and `distribution-shift` → `embedding-divergence`
- Evaluators drop their family prefix (`bias.balance` → `balance`, `quality.duplicates` → `duplicates`); a prefixed
  name fails to load as an unknown evaluator
- Findings are retitled; anything that matches on a title needs updating (see "Where old keys went")

#### Presets

- Every workflow type is a preset that returns a `ChainResult`, whose `steps` and `findings` replace `raw` and `report`
- Each preset but `audit` owns its checks, so presets run side by side report each finding once
  - Class balance and the metadata factors move to `bias`; `audit`'s chain is unchanged
  - `scope` drops `label-health`, `class-imbalance`, `factor-summary`, `balance`, `diversity`, `factor-gaps` and
    `factor-coverage-gaps`; `quality` and `splits` drop `class-imbalance`; each refuses their settings
- `quality`: `outliers`, `label-health`, `outliers-by-class` and `duplicates`, judged by `image-outliers`,
  `target-outliers`, `class-outliers` and `image-duplicates`
  - Its `clean` step passes on the dataset minus flagged images and boxes, keeping one item per duplicate group; an
    `export` step on it replaces `mode: preparatory`
- `prioritization`: ranks each pool against the reference, and `selected` keeps the top of each ranking; it makes no
  findings
- `triage`: `factor-triage` reads the metadata, and `factor-issues` makes its findings, with a recommended policy
- `shift`: one `detectors:` list takes drift and OOD detectors, each judged by its family's check, `drift` or `ood`
  - With no `detectors:`, it runs `drift-univariate` (KS, Bonferroni) and `ood-kneighbors`
  - Each test source is tested against the reference on its own; `merge` them in a custom workflow to test them as one
  - `classwise:` maps a drift detector to its `by:` (`{drift-mmd: class}`) and takes class groups; lists are refused
- `splits`: a `split` or `kfold`, with each train rebalanced where `rebalance:` is set; the finding is Class
  Stratification per fold, and each part's indices are in `result.steps["split"].details["indices"]`
- `representation` records the `expected` names it ignored, under `extras.ignored_expected`
- `scope`: a `crops` step (`wrap`) crops detection data to one item per box, and `coverage` and `completeness` embed the
  crops; without an extractor the embedding steps are skipped, and their findings say "not assessed"
  - On detection data, uncovered items index the `crops` Dataset, not each one's image and box
  - It no longer judges an ontology; a `taxonomy` entry on the same source does

#### Python API

- Internal modules (`cache`, `stats`, `policy`, …) are private; config types import from `dataeval_flow.config`, and
  `Finding` (was `Reportable`) from `dataeval_flow.steps`
- `dataeval_flow.workflow` and `dataeval_flow.evaluator` merge into `dataeval_flow.workflows` and `.evaluators`
- Each workflow and evaluator has one `<X>Config` in its own package (was `<X>Parameters` plus `<X>WorkflowConfig`);
  `WorkflowParametersBase` is `WorkflowConfig`
- `result.output` replaces `result.data` and an evaluator's `raw`, and `Result.type` replaces `Result.name`; reading
  `output` on a failed run raises
- A failed result's `to_dict()` is `{kind, metadata, errors}`, and a failed workflow's `health.status` is `failed`
- `run_tasks` returns results keyed by task name; `run_task` and `run_tasks` take `data_dir` and `cache_dir` by keyword
  only; `list_workflows()` and `list_evaluators()` return classes
- The `maite.tasks` entry point is `dataeval_flow:run_tasks`

### Fixed

- Config loading
  - COCO's `annotations_file` and `images_dir`, and YOLO's `yaml_file` and `ann_dir`, can no longer leave the data root
  - A task naming one source twice is refused at load, and so is a key no config section defines
  - Relative `ontology:` paths resolve against the run's data root rather than the process root
  - The config builder keeps an explicit `null` and the `result:`, `logging:`, `seed:` and `deterministic:` keys
- Caching
  - A dataset whose class names change but whose items don't gets its own cache entry, so old names aren't served
  - Source views are part of cache keys, so editing a view invalidates cached embeddings, metadata and statistics
  - Every step of a task reads a source through one draw of its view, so random views give all steps the same order
- Checks and statistics
  - Outlier and duplicate detection are limited to the configured flags, and injected metadata factors to the
    configured `intrinsic_factors`, so a shared cache can't widen them
  - Cross-split label parity is limited to shared classes, and cross-split duplicate detection to common columns,
    avoiding errors on gapped label spaces and divergent stat caches
  - `prioritization` ranks a labeled pool under `policy: class_balanced`, and succeeds when cleaning empties a pool
  - A chunked drift finding with `chunk_percent: 0` no longer warns when no chunk drifted
- Reports
  - The TUI shows a name with brackets in it as written; `[/x]` no longer crashes it
  - A chain refused before any step ran says why, and a report's health line says `failed` when a required step failed
- A GitHub Release attaches only the wheel and the sdist, not the `.gitignore` that `uv build` writes into `dist/`

### Removed

- Workflow types that run their own code: every workflow type is now a preset expanding to a chain of steps
  - `Workflow.run`, `WorkflowOutput`, `WorkflowRawOutput`, `WorkflowReport`, `WorkflowResult` and `workflow_result` are
    gone; a workflow type mixes in `Preset` and declares `slots` and `chain`, or its class raises `TypeError`
  - A plugin with an algorithm of its own registers it as an evaluator and chains it in its preset
- `parameter-sweep` and `data-analysis`, with their classes: write a `quality` entry with a `matrix:` on its task, and
  an `audit` entry
- `splits`'s coverage steps, `coverage:`, `extractor:` and `checks.uncovered-items`; judge the parts with an `audit`
  step after the split
- `prioritization`'s `cleaning:` and `stats:`; run the preset after a `quality` step on the reference and one on the
  pools
- Settings that no workflow takes now: the `metadata_*` binning fields, `value_range` on a workflow entry,
  `include_image_stats`, `health_thresholds`, `mode`, and `update_strategy` on `shift`
- The per-key migration messages for keys earlier versions took; each such key now fails as an unknown key or workflow
- The `selections:` and `selection:` aliases for `views:` and `view:`, deprecated since v0.2.0
- The `torch` and `uncertainty` extractors' `device`; Flow chooses the device for every tool
- Poetry packaging support (install with uv, pip or conda) and Python 3.10 support (the minimum is now 3.11)
- Public API: the typed task configs (use `TaskConfig`), `load_config_folder`, `export_params_schema`, `select_tasks`,
  a workflow's `output_schema`, the `DriftDetectorConfig` and `OODDetectorConfig` unions,
  `dataeval_flow.config.schemas`, and the per-type result, output and threshold classes such as `DataCleaningResult`
  and `DataCoverageResult`; each workflow result is a `ChainResult`

### Where old keys went

A v0.2 config that uses an old key fails to load and names it. Each old key and its replacement follow.

#### Finding titles

- "Duplicates" and "Redundancy" → "Image Duplicates"; "Image Quality" → "Image Outliers"
- "Label Distribution", "Label/Directory_Name Distribution" and "Label Balance" → "Class Imbalance"
- "Embedding Coverage" → "Class Coverage"; "Metadata Coverage Gaps" → "Factor Coverage Gaps"
- "Label Space Coverage" → "Leaf Coverage"; "Class Balance Worklist" → "Class Shortfall"
- "Aggregate OOD (all detectors agree)" and "Unique OOD Samples (single-detector only)" → "OOD Agreement"
- "Bias" → "Shortcut Risk", with diversity as evidence; "Label Parity" → "Class Stratification"
- "Label Overlap" → "Untrained Classes" and "Class Sufficiency"; "Distribution Shift" → "Embedding Divergence"

#### quality (was data-cleaning)

- `health_thresholds` → `checks`: `image_outliers`, `target_outliers` and `classwise_outliers` →
  `image-outliers.warning`, `target-outliers.warning` and `class-outliers.warning`; `exact_duplicates` and
  `near_duplicates` → `image-duplicates.exact` and `.near`; `class_label_imbalance` → `bias`'s
  `class-imbalance.warning`
- `outlier_method` and `outlier_threshold` → `outliers.outlier_threshold` (`zscore`, or `[zscore, 3.0]`);
  `outlier_flags` → `outliers.flags`; `outlier_cluster_threshold`, `outlier_cluster_algorithm` and `outlier_n_clusters`
  drop `outlier_` under `outliers:`
- `duplicate_flags` → `duplicates.flags`; `duplicate_merge_near` → `duplicates.merge_near_duplicates`;
  `duplicate_cluster_sensitivity`, `duplicate_cluster_algorithm` and `duplicate_n_clusters` drop `duplicate_` under
  `duplicates:`
- `value_range` is refused: set it on the dataset
- `metadata_auto_bin_method`, `metadata_exclude`, `metadata_continuous_factor_bins` and `metadata_factor_source` are
  refused: name a `metadata:` policy with `auto_bin_method`, `exclude`, `continuous_factor_bins` and `factor_source`;
  every workflow below treats its `metadata_*` fields the same way

#### prioritization (was data-prioritization)

- `method`, `k`, `c`, `n_init`, `max_cluster_size`, `order`, `policy` and `num_bins` move under `prioritization:`
- `n` and `fraction` → `select.n` and `select.fraction`; `health_thresholds` and `value_range` are refused

#### triage (was metadata-triage)

- `max_examples` → `checks.factor-issues.max_examples`; the `metadata_*` binning fields are refused

#### shift (was drift-monitoring and ood-detection)

- A drift detector
  - `method: <name>` → `type: drift-<name>` (`domain_classifier` → `drift-domain-classifier`); the univariate `test` →
    `method`
  - A detector's `classwise: true` → its name in `classwise:`
  - `any_drift_is_warning` and `classwise_any_drift_is_warning` → `checks.drift.warn_on_drift`
  - `chunk_drift_pct_warning` → `chunk_percent`; `consecutive_chunks_warning: N` → `consecutive_chunks: N-1`
  - `chunking.threshold_multiplier: k` → `chunking.threshold: [zscore, k]`; an unset `threshold` now uses DataEval's
    default (legacy used z-score 3), and `[zscore, 3.0]` restores legacy
- An OOD detector
  - `method: kneighbors|domain_classifier` → `type: ood-kneighbors|ood-domain-classifier`; two detectors of one type
    need distinct `name`s
  - A domain classifier thresholds on `n_std` unless `threshold_perc` is set; `threshold_perc: 95` keeps old verdicts
  - `health_thresholds.ood_pct_warning` and `ood_pct_info` → `checks.ood.warning` and `info`; `checks["ood-agreement"]`
    judges the detectors' agreement
  - `max_ood_insights` → `factor-deviation.max_items`; `metadata_insights: false` → `factor-predictors: false` and
    `factor-deviation: false`
  - `value_range` and the `metadata_*` fields are gone: set the range on the dataset, and name a `metadata:` policy

#### splits (was data-splitting)

- `num_folds` → `folds`; `rebalance_method` → `rebalance`; `coverage_percent` and `num_observations` are gone with the
  coverage steps
- `val_frac` with `folds` of 2 or more is refused (each fold's val is its 1/k); unset, it is 0.1 with `folds: 1`
- `metadata`'s `split_sizes` and `stratified`, and `output.raw`, are in `result.steps` and `lineage`

#### scope (was data-coverage)

- `coverage_method`, `coverage_percent` and `num_observations` → `coverage.method`, `.percent` and `.num_observations`;
  `min_class_samples`, `isotropy_min_samples` and `near_duplicate_factor` keep their names under `coverage:`
- `crop_padding` and `crop_min_size` → `wrap.params.padding` and `wrap.params.min_size`; `run_completeness` →
  `completeness`
- `ontology`, `ontology_label_pattern` and `ontology_expected` → a `taxonomy` entry's `ontology`,
  `ontology-validation.label_pattern` and `representation.expected`, or `representation.expected` on `scope`
- `diversity_method`, `balance`, `run_gap_analysis`, `gap_mi_threshold` and `gap_min_representation` are refused; they
  moved to `bias` as `diversity.method`, `factor-gaps.mi_threshold` and `factor-gaps.min_representation`, where balance
  always runs and `factor-gaps: false` replaces `run_gap_analysis: false`
- `health_thresholds` → `checks`
  - `min_dispersion` → `coverage.dispersion_factor`, which must be under 1
  - `min_isotropy` → `coverage.isotropy_factor`, which is a share, in (0, 1), of the median isotropy of the classes
    that report one, where legacy's was absolute
  - `max_near_duplicate_fraction` → `coverage.near_duplicate_threshold`, over a fraction DataEval 1.2 measures anew
  - `uncovered_rate` → `uncovered-items.warning`; `completeness_score` → `dimensional-completeness.warning`, with
    legacy's band at 0.8 → `.info`; two written bounds that cross are refused
  - `class_imbalance_ratio` → `bias`'s `class-imbalance.warning`, with legacy's fixed band at 2.0 → `.info`
  - `gap_count: N` → `bias`'s `factor-coverage-gaps.warning: N-1` (legacy warned at N; this warns past the bound)
  - `leaf_coverage` → `taxonomy`'s `checks.leaf-coverage.coverage`; `dark_branch_count` → `.empty_branches`;
    `unmatched_class_count` → `checks.label-conformance.warning`
- `value_range` and `stats` are refused, since no step of `scope` reads statistics
- `output.raw` and `metadata.has_extractor` are gone; read the steps' outputs instead: `coverage` and `completeness`
  → their outputs, `label_distribution` → `label-health`'s, `metadata_distribution` → `factor-summary`'s, and
  `coverage.dropped_detections` → `result.steps["crops"].details["dropped"]`

#### data-analysis (now audit)

- `outlier_method` and `outlier_threshold` → `outliers.outlier_threshold`, which takes the method, or
  `[method, threshold]`; `outlier_flags` → `outliers.flags`
- `diversity_method` → `diversity.method`, and `divergence_method` → `divergence.method`; `diversity_method: null` has
  no replacement, because `audit` always runs diversity; `balance` is refused, because balance always runs
- `include_image_stats` → the metadata policy's `intrinsic_factors`; `value_range` and the `metadata_*` binning fields
  are refused, so set the range on the dataset and name a `metadata:` policy
- `health_thresholds` → `checks`: `image_outliers` → `image-outliers.warning`; `exact_duplicates` and `near_duplicates`
  → `image-duplicates.exact` and `.near`; `class_label_imbalance` → `class-imbalance.warning`; `distribution_shift` →
  `embedding-divergence.warning`
- `DataAnalysisResult`'s raw fields are steps in an audit's `result.steps`, named per split, as `outliers-train` and
  `outliers-evals`
  - Each split's `image_quality`, `redundancy` and `label_health` → its `outliers`, `duplicates` and `label-health`
  - Train's `bias` → `factor-summary`, `balance` and `diversity`
  - `cross_split`'s duplicate leakage → `duplicates-cross` and `duplicates-pairs`; its label comparisons →
    `class-sufficiency`, `untrained-classes` and `class-stratification`; its divergence → `divergence`
- Gone with it: chi-square label parity, divergence between evaluation splits, bias judged per split, the warning on low
  diversity, and Label Balance's warning on unlabeled images

## v0.2.5

### Added

- `max_processes` config setting, `--max-processes` CLI option, and `DATAEVAL_MAX_PROCESSES` environment variable bound
  the worker processes and the value is recorded in each result's `resolved_config`
- `nox -s verify` writes `output/metarepo/test-results.log`, a per-test-case log for metarepo assessments

### Changed

- Requires `dataeval>=1.1.4`; the lock files and container images resolve DataEval 1.1.4

### Fixed

- Container images no longer ship the standalone interpreter's bundled `pip`, which nothing used

## v0.2.4

### Added

- Published images carry their own vulnerability scan report and CycloneDX SBOM at `/usr/share/dataeval-flow/security/`

### Changed

- Containers now publish to `harbor.jatic.net/aria/dataeval-flow` instead of `harbor.jatic.net/aria/dataeval`
- `main-<variant>` tracks the default branch; `latest-<variant>` is a retag of the newest stable release
- Images are scanned before publication, and HIGH or CRITICAL findings fail the build instead of being reported after
- All variants build on `ubuntu:24.04` and the CUDA runtime and cuDNN install from Python wheels

### Fixed

- Prerelease tags (`vX.Y.Z-rcN`) publish as GitHub prereleases instead of replacing the latest stable release

### Removed

- Floating `<variant>` and `<major>.<minor>-<variant>` image tags; pull `latest-<variant>` or pin `<version>-<variant>`

## v0.2.3

### Added

- `DATAEVAL_*` environment variables to configure CLI parameters; CLI arguments take precedence
- `--no-fail-on-warning` flag to disable `DATAEVAL_FAIL_ON_WARNING` for a single run
- `--log-format {structured,plain}` flag selecting structured or plain log output
- `scripts/release.py` script to cut and tag releases from the current branch

### Changed

- Console logs now include ISO-8601 UTC timestamps and levels; use `--log-format plain` for bare messages
- Requires DataEval 1.1.1 or newer

### Fixed

- Container entrypoint honors `DATAEVAL_OUTPUT` and `DATAEVAL_CACHE` instead of hardcoded paths
- Container image creates `/cache/.not_mounted`; unmounted `/cache` is no longer exported as `DATAEVAL_CACHE`
- Consistently recognize object-detection datasets on Python 3.10 and 3.11, preventing misclassification or crashes
- GitHub release body now carries the changelog section instead of falling back to `Release vX.Y.Z`

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
