# Changelog

## Unreleased

### Added

- `library_versions` on every result's metadata: the installed version of DataEval, NumPy, Pillow, datamaite and the
  task's extractor runtime (PyTorch, ONNX Runtime or OpenCV)
- `provenance:` on `datasets:` entries: facts Flow can't measure, such as owner, licence, origin and collection date,
  as names and plain values recorded in `resolved_config` as written; it is no part of the cache key
- `dataset_digest()`: SHA-256 digests over every item of a dataset, of its images, labels and class names
  (`content`) and of its metadata (`metadata`), whatever the items' order, so a training job can check its data is
  the data that was audited
- `content-digest` evaluator: `dataset_digest()`'s digests of a source, read from every item with no cache between,
  through a new `dataset` input kind that hands an evaluator the Dataset itself
- `label-space` preset: a dataset's labels judged against a declared ontology, as legacy data-coverage judged them
  with `ontology:` set: leaf coverage and the worklist, conformance, alignment with the `Relabel` stanza, and the
  ontology's structure. Its `health_thresholds` are keyed by check type: `leaf-coverage` (`coverage`,
  `empty_branches`) and `label-conformance` (`unmatched`)
- `label-reconciliation` evaluator: which class names resolve to exactly one ontology concept
- `ontology-validation` evaluator: an ontology's structural and naming facts
- `leaf-coverage`, `label-conformance`, `mergeability` and `ontology-structure` checks, which make `label-space`'s
  findings
- `uncertainty` extractor: an ONNX classifier's or detector's normalized entropy per prediction, for drift on unlabelled
  data, one row per detection for a detector; only drift and OOD evaluators read it
- `by: predicted` keys a step by the class a model predicts
- A drift-monitoring detector may name its own `extractor:`, which its steps embed with instead of the task's
- Every tool computes on CUDA when PyTorch sees a GPU, else the CPU; `dataeval_flow.set_device` chooses from Python,
  and `CUDA_VISIBLE_DEVICES` hides GPUs. A config names no device
- `device` on every result's metadata: the device its task computed on, such as `cuda:0 (NVIDIA L4)`
- Top-level `evaluators:` key running a single DataEval evaluator, one of the Evaluator Catalog's twenty-three types
- `evaluator:` on tasks, as the alternative to `workflow:`, checked against the evaluator when the config loads
- `kind` on `TaskConfig`: a loaded task holds either name in `workflow`, and `kind` records which key named it
- `dataeval-flow evaluators` command listing evaluator types, what each consumes, and their parameter schemas
- `kind` on every result envelope, `workflow` or `evaluator`; evaluator results carry no `health`
- `Result`, the base both result kinds share: `report()`, `to_dict()` and `export()`, with no health on it
- Top-level `ontologies:` key defining named label spaces referenced by workflows
- `concepts:` on ontology entries to add or override concepts by id without altering source artifacts
- Workflow `ontology:` resolves pool names first, falling back to file paths for backward compatibility
- `merge:` on sources to concatenate multiple inputs into one dataset, unified via `Relabel` views
- Top-level `exports:` key exporting sources to COCO, YOLO, Hugging Face, or VisDrone format with `provenance.json`
- `ontology:` support across all workflows but `data-coverage`, attaching the vocabulary audit digest to results
- `label_space` on result envelopes, recording conformed vocabulary and matching audit digest
- `channel_groups:` on datasets, measuring band groups separately as `<group>_<statistic>` columns
- Top-level `stats:` key defining policies for measured statistics, background inclusion, and outlier/factor views
- `format: demo` dataset loader resolving tutorial datasets from a fixed table without arbitrary imports
- `crops:` on `data-coverage`, `DetectionCrops`' `padding` and `min_size`, with the `crops` step's `details`
  counting the detections dropped
- `DATAEVAL_*` environment variables to configure CLI parameters; CLI arguments take precedence
- `--no-fail-on-warning` flag to disable `DATAEVAL_FAIL_ON_WARNING` for a single run
- `--log-format {structured,plain}` flag selecting structured or plain log output
- `scripts/release.py` script to cut and tag releases from the current branch
- Published images carry their own vulnerability scan report and CycloneDX SBOM at `/usr/share/dataeval-flow/security/`
- `nox -s verify` writes `output/metarepo/test-results.log`, a per-test-case verification log for metarepo assessments
- Plug-in workflows and evaluators, registered as `dataeval_flow.workflows` / `dataeval_flow.evaluators` entry points
- Every workflow declares its inputs; each task is checked against its workflow or evaluator when the config loads
- `WorkflowContext` methods for a workflow's inputs: `dataset`, `stats`, `embeddings`, `clusters`, `metadata`, `labels`
- Plug-in extractors and image transforms, under the `dataeval_flow.extractors` / `dataeval_flow.image_transforms`
  entry points
- `run(config, data)` runs one workflow or evaluator on datasets in memory, typed to the config's result
- An extractor config without a `name` is named after its `model`, as workflow and evaluator configs are
- Extension bases document how to subclass and register a plugin; a test pins the public API
- Each `<X>Result`'s API page lists the `output.raw` and `metadata` fields its type adds, pinned by a test
- `LoggingConfig`, the type of `PipelineConfig.logging`, is exported from `dataeval_flow.config`
- `--report-width` and `DATAEVAL_REPORT_WIDTH` set the text report's width; `Result.report()` takes `width=`
- `Result.to_html()` renders the report as one self-contained, printable page; `--output` writes `result.html`
- `drift-univariate`, `-mmd`, `-kneighbors`, `-wasserstein` and `-domain-classifier` evaluators, each taking
  `chunking:`; Wasserstein takes a validation source between the reference and the data to test
- `SourceCount.THREE`, for an entry that takes exactly three sources
- `flags` table columns, each cell a list of measurements against their population, and a table's row `preview`
- The HTML report shows a card per finding, warnings open, with sortable and filterable tables and dark mode
- Reports carry a thumbnail of each item their findings name, in `Result.assets`; the HTML report shows them
- `--no-report-images`, `DATAEVAL_REPORT_IMAGES=0` or `report_images=False` turn a run's thumbnails off
- `image` table columns, each cell an item reference or a group of them
- Data analysis, coverage, prioritization, splitting and metadata triage picture the items their findings name
- `uncovered_classes` in `coverage`'s `extras`: each uncovered item's class. Its report section, "Uncovered items",
  gives each item's class and distance
- A pipeline's `result: max_images:` sets how many thumbnails each result embeds (200), shared evenly between findings
- The `result:` block also names the result files and picks their formats, detail, per-task split and text width
- `result: max_rows:` and `preview_rows:` set a table of items' rows (500) and text preview (10); `-1` lifts a limit
- `result: fail_on:` gates the exit code on `failure`, `warning` or `never`; `--fail-on-warning` still overrides it
- `junit` and `markdown` formats: a JUnit report for CI test views and a Markdown summary, both naming failed tasks
- Evaluators can read `metadata`, `labels` and `embeddings` inputs, and the task's resolved `ontology`, on
  `EvaluatorInputs`
- `output_extras` on `Evaluator`: results DataEval keeps outside `data()`, written under `extras` by `to_dict()`
  and `export()` and shown in reports
- `duplicates` takes DataEval's video parameters (`redundancy_radius`, `min_segment_frames`, `max_segment_gap`,
  `segment_offset_tolerance`, `verify_alignment`, `min_track_frames`, `frame_sample`) and writes
  `annotation_divergences` and `factor_cardinality` under `extras`
- `balance`, `diversity` and `parity` evaluators, reading one source's metadata under its
  `metadata:` policy
- `representation` evaluator, counting a source's labels against an ontology, or one synthesized from its
  `index2label`
- `coverage` and `prioritize` evaluators; a second `prioritize` source is the reference its ranking
  is relative to
- `ood-kneighbors` and `ood-domain-classifier` evaluators
- An "Evaluator recipes" how-to with one worked example per evaluator family, from the config entry to reading its
  output
- Custom workflows: a `workflows:` entry with `inputs:` and `steps:` chains evaluators, workflow types and transforms,
  each step reading an input or an earlier step by address (`clean`, `split.train`, `kfold.train[0]`) and running once
  per element of a list; a task runs one with `workflow:`, and a failed step skips only what reads it
- `CustomWorkflowConfig` and `StepEntry` build a custom workflow in Python, and `save()` writes it into a config file;
  loading and saving a config, from the TUI or the config builder too, keeps custom workflows as written
- `save()` and `to_yaml()` on a custom workflow take `definitions`, as `run()` does: the evaluator entries, views,
  policies and other named entries its steps refer to are written beside it, each with its name, its type and the
  settings that differ from their defaults, so a saved block loads and runs on new data on its own
- Nine transforms for custom workflows: `view`, `merge`, `split`, `kfold`, `wrap`, `select`, `remove` (DataEval's
  removal plans from Duplicates and Outliers), `conform` (relabelling onto an ontology, refusing loss beyond `allow:`)
  and `export` (writing to `<output>/datasets/<task>.<step>/`); plug-in transforms register under
  `dataeval_flow.transforms`
- `label-alignment` evaluator, aligning a source's class names to an ontology through
  `dataeval.core.label_alignment`, as `data-coverage` aligns them
- `ChainResult`, a custom workflow's result: each step's outcome in `steps`, readable when a step failed, and each
  Dataset's lineage in `metadata.lineage`, whose digests tell whether two results read the same data
- `output_dir` on `run_tasks`, `run_task` and `run`, where export steps write; the CLI passes `--output`
- `dataeval-flow steps [NAME] [--json]` and `list_steps()` list every step a workflow can chain, with its ports and
  settings schema
- A "Workflows as Chains of Steps" explanation, a how-to that writes a custom workflow, and a Transform Catalog with
  one reference entry per built-in transform
- Presets: a workflow type can be a preset, whose settings expand to a chain of steps that runs as a custom
  workflow's. As a task, a preset returns a `ChainResult` under its own type id; as a step of a custom workflow, its
  chain runs inside yours as `<step>/<name>`, and `<step>.<output>` reads each Dataset it declares as an output
- A chain computes each dataset's statistics in one pass over every family its steps read
- `data-cleaning`'s `clean` step hands on the dataset without each image and box it flagged and each duplicate but the
  first of its group, counting what it removed; a custom workflow that runs data-cleaning as a step reads it as
  `<step>.clean`, and an `export` step on it replaces `mode: preparatory`
- A friendly title on every step type, such as `K-Fold Split`, as `title` in `dataeval-flow steps NAME`, `--json`
  and `list_steps()`
- A chain's short report, `report(detailed=False)` and what the console prints without `-v`: its step count, its
  summary and health, and a Steps table giving each step's status and why it made nothing where it did not
- `by_plan` in a `remove` step's `details`, beside `removed`: what each plan named, at each level it named something
- Report blocks' `in_text` on a table column, `failed` on a summary and `group` on a summary item, each left out
  of the JSON at its default
- `n` and `fraction` on `data-prioritization`: its `selected` step keeps each pool's top `n`, or that share rounded up;
  unset, it keeps every item in ranked order
- A `prioritize` step's report pictures its ranking's 25 highest and 25 lowest items, with rank and score
- `factor-triage`, an evaluator: what a Dataset's metadata failed to read, the policy stanza that repairs it, and, with
  `verify`, what the repair recovers
- `metadata-issues`, a check: metadata-triage's findings, made from a `factor-triage` Output
- `factor-triage` recommends a policy: its suggestions, each value triage could not read dropped to missing, plus
  explicit edges or levels for every factor the policy left unpinned, read back from this data. A dominant value
  such as a speed of zero is left in, with a note to decide. `metadata-issues` shows it as a "Recommended policy"
  finding that opens with a caveat: a policy read from unrepresentative data can give invalid or misleading results
- `by: class` on an evaluate step runs its evaluator once per class, or per named group of classes, on the classes
  with `min_items` items in every input, in one Output that names each class it left out and why. A class whose run
  raises is left out with its error, and the step fails only when every class raises. On a check, it judges each
  class and rolls the findings into one, briefed `2/8 classes warn`
- `ood` and `ood-agreement` checks, and `ood-union`, `factor-predictors` and `factor-deviation` combines, the steps the
  `ood-detection` preset runs
- OOD evaluators read the `uncertainty` extractor: a classifier's rows, one per image, or a detector's, one per
  detection, judged per image, where any detection flagged flags its image
- A combine may read a Dataset's statistics and draw its own report section, and require the Outputs it reads to share
  their Datasets, or to have been computed on its own
- `drift`, a check: a warning when a drift evaluator finds drift or, chunked, when `chunk_percent` of the chunks
  drift or `consecutive_chunks` drift in a row
- The drift evaluators' results have a report section: the verdict's fields, or one row per chunk
- `matrix:` on a task runs its entry once per combination of the values it lists, as lists, inclusive ranges
  `{from, to, step}` or several grids, and returns one `MatrixResult` whose report opens with a table comparing every
  run's findings. It varies the entry's settings, the task's `sources` and `extractor`, and the evaluators, workflows,
  extractors and steps the task reads. The runs share one draw of each source and the cache; an export writes under
  `run-<n>/`
- `stratification` check: each part's class shares against the whole's, warning past `warning` percentage points of
  deviation and informing past `info`
- `uncovered-rate` check: the share of a Dataset a coverage run left uncovered, warning past `rate` percent
- `split`, `kfold` and `view` steps record each output's indices, into the dataset at the bottom of its views, in their
  `details`, and `split` and `kfold` have a report section of each part's size
- A preset's declared outputs read any address in its chain, lists included, so `data-splitting`'s `<step>.train` is
  the rebalanced train where it rebalances
- A subset of a Dataset, such as a split's part, slices the embeddings the run already computed for its parent, where
  its view changes no pixels, instead of extracting them again
- An optional step that needs an extractor is skipped, with the reason, when neither it nor the task names one, where
  load refused it
- `completeness` evaluator: how much of the embedding space's dimensions the data fills, as legacy data-coverage's
  Dimensional Completeness measured it; it refuses fewer than two embeddings
- `metadata-summary` evaluator: each metadata factor's type, binning, nulls, and range or top values
- `class-coverage`, `completeness-score`, `coverage-gaps` and `class-shortfall` checks, which make `data-coverage`'s
  Embedding Coverage, Dimensional Completeness, Metadata Coverage Gaps and Class Balance Worklist findings
- `factor-gaps` combine: each factor's mutual information with the class, read from a `balance` Output, and the
  class-factor-value combinations under-represented among the factors at or over `mi_threshold`
- `other_kinds: pass` on `wrap` hands a Dataset of another kind on unchanged, reading its source's cached embeddings

### Changed

- A chain whose `label-alignment` steps agree, and whose sources and `conform` steps record no label space, stamps
  its result's `label_space_digest` with the alignment's: the join key to a dataset conformed by its stanza
- `representation`'s output records the `expected` names it ignored, under `extras.ignored_expected`, and its report
  section and `label-alignment`'s are short summaries
- `representation`, `coverage`, `prioritize` and `label-alignment` read class labels and no factor, so their reads
  leave a chain's binning record: data-splitting with an extractor records the whole set once, and data-prioritization,
  whose steps read no factor, records no binning record or `encoding_digest`
- `uncertainty` extractor entries need `metadata_path` and `preds_type`, and the TUI no longer offers them
- drift-monitoring's `classwise:` maps each detector to its `by:` (`{drift-mmd: class}`), and takes class groups; the
  list form is refused
- An `unbinned` finding says a declared bin count fixes how many bins there are, not their edges; the recommended
  policy pins the edges
- `MetadataConfigMixin` holds only `metadata:`, the policy name, as `StatsConfigMixin` holds only `stats:`; the
  older `metadata_*` fields stay on the workflows that took them
- The text report is 80 columns wide by default (was 90), and wraps long prose, labels and values to fit
- A run that fails only on health warnings exits `3` (was `1`), so CI can tell a data-quality gate from a crash or a
  mistyped flag, which exits `2`
- Data analysis lists each split's unlabelled images in a table naming up to eight, where it wrote a sentence
- `PipelineConfig.tasks` and `run_tasks` now carry evaluator tasks and results as well as workflow ones
- `run_task` returns a `Result`, a workflow's or an evaluator's; `isinstance` narrows it to the type's `<X>Result`
- A failed workflow's report shows `FAILED` and its errors, as a failed evaluator's does
- `WorkflowResult` takes every argument by keyword only: `type`, `success`, `output`, `metadata`, …
- Containers now publish to `harbor.jatic.net/aria/dataeval-flow` instead of `harbor.jatic.net/aria/dataeval`
- Metadata cache key includes stats policy `factor_identity()`, recomputing metadata archives on upgrade
- Stats cache keys remain unaffected, preserving cached stats for policies without band groups or background
- Console logs now include ISO-8601 UTC timestamps and levels; use `--log-format plain` for bare messages
- `main-<variant>` tracks the default branch; `latest-<variant>` is a retag of the newest stable release
- Images are scanned before publication; a HIGH or CRITICAL finding fails the build before anything is pushed
- Internal modules (`cache`, `stats`, `policy`, …) are private; import config types from `dataeval_flow.config`
- `dataeval_flow.workflow` and `dataeval_flow.evaluator` merge into `dataeval_flow.workflows` / `.evaluators`
- Workflow packages are named after their type: `workflows.data_cleaning`, `workflows.drift_monitoring`, …
- Each workflow and evaluator has one `<X>Config` (was `<X>Parameters` plus `<X>WorkflowConfig`) in its own package
- Each type has a real `<X>Result` class; `isinstance` narrows, and the `is_*_result` guards are gone
- `result.output` replaces `result.data` and an evaluator's `raw`; reading it on a failed run raises
- `Result.type` replaces `Result.name`; the framework, not each workflow, turns exceptions into failed results
- `WorkflowProtocol` is the `Workflow` base class: subclass `Workflow[Config, Result]`, as its docstring shows
- A workflow's `run(config, context)` replaces `execute(context, params)`; `config_type` replaces `params_schema`
- A workflow declares `name` and `description` as class variables, not properties
- `WorkflowParametersBase` becomes `WorkflowConfig`, the base of every workflow config, no longer their union
- `Reportable` is `Finding`; `WorkflowOutputsBase` / `WorkflowReportBase` are `WorkflowRawOutput` / `WorkflowReport`
- `DriftHealthThresholds` / `OODHealthThresholds` are `DriftMonitoringHealthThresholds` / `OODDetectionHealthThresholds`
- Data-prioritization's `CleaningConfig` is `DataPrioritizationCleaningConfig`
- A workflow package's modules (`params`, `outputs`, `workflow`, `report`) are private; import from the package
- `list_workflows()` / `list_evaluators()` return the classes; `get_*` return the class, not an instance
- Extractor configs are imported from `dataeval_flow.config.extractors`; `ToRGB` from
  `dataeval_flow.config.image_transforms`
- `run_tasks` returns results keyed by task name; `load_config` reads a file or a folder
- `run_task` and `run_tasks` take `data_dir` and `cache_dir` by keyword only; a task named twice runs once
- `PipelineConfig` and `load_config` are imported from `dataeval_flow` only, not `dataeval_flow.config`
- `WorkflowResult`, `list_workflows` and `get_workflow` are imported from `dataeval_flow.workflows`, not the top level
- Dataset, source, view, preprocessor and task configs are imported from `dataeval_flow.config`, not the top level
- `run_task` is imported from `dataeval_flow`; the `maite.tasks` entry point is `dataeval_flow:run_tasks`
- `ResultMetadata` is imported from `dataeval_flow`; the config mixins from `dataeval_flow.config`
- A failed result's `to_dict()` is `{kind, metadata, errors}`; a failed workflow's `health.status` is `failed`
- `Finding` drops `report_type` and `data`, and rejects unknown fields; `brief` and typed report `blocks` hold the evidence
- Data cleaning lists each flagged image and box with every metric that flagged it, then each metric's limits
- Data analysis keeps its flagged values in `image_quality.outliers` and lists every flagged image by split
- The HTML report draws a bar chart's thresholds across its bars, labelled on a scale, instead of in a caption
- The TUI draws each finding's evidence natively: data tables as tables, the rest as text at the window's width
- Data cleaning lists its duplicate groups with their items, largest first, and duplicate boxes on their own
- OOD detection lists its samples in tables, naming each in its own test source, rather than as bullets
- Pillow (`>=12.2.0`) is a core dependency, to encode thumbnails
- An evaluator report's console form cuts a list of more than ten values inside its output to the first ten and a count;
  `-v` and `result.txt` show it whole
- `data-coverage` hands `Coverage` its embeddings as extracted, since DataEval rescales them itself; its own
  per-dimension rescale had shifted `dispersion` and the coverage radius
- Evaluators are named for what they compute, without a family prefix, and a prefixed name fails to load as an
  unknown evaluator: `balance`, `diversity` and `parity` (were `bias.*`); `duplicates`, `label-health` and
  `outliers` (were `quality.*`); `coverage`, `label-alignment`, `prioritize` and `representation` (were `scope.*`);
  `drift-domain-classifier`, `drift-kneighbors`, `drift-mmd`, `drift-univariate`, `drift-wasserstein`,
  `ood-domain-classifier` and `ood-kneighbors` (were `shift.*`)
- `data-cleaning` is a preset: its evaluators find outliers, duplicates and label counts, and its checks judge them
  against `health_thresholds`. It returns a `ChainResult`, whose `steps` and `findings` replace `raw` and `report`,
  and `run()` on a `DataCleaningConfig` is typed to `ChainResult`. Its steps are named in kebab case, as ids are:
  `outliers`, `labels`, `by-class`, `dupes`, `image-outliers`, `target-outliers`, `classwise`, `duplicates`,
  `imbalance` and `clean`
- `data-prioritization` is a preset: `cleaning:` runs as `outliers`, `duplicates` and `remove` steps on the reference
  and each pool, `rank` (`prioritize`) ranks each pool against the reference, and `selected` (`select`) keeps the top
  of each ranking. It returns a `ChainResult`, whose `steps` replace `raw` and `report`, and it makes no findings: the
  Pruning warning and each pool's info finding are gone
- `metadata-triage` is a preset: `factor-triage` reads the metadata, and `metadata-issues` makes its findings. It
  returns a `ChainResult`: the issues, the stanza and the verification are its `triage` step's output
- `drift-monitoring` is a preset: each detector is a step judged by a `drift` check, and each detector `classwise`
  names also runs by class. Each test source is tested on its own against the reference, where they were merged;
  `merge` them in a custom workflow to test them as one. `detectors:` takes drift evaluator entries
  (`drift-univariate`, `drift-mmd`, `drift-kneighbors`, `drift-domain-classifier`), `classwise:` lists detector
  names, and `health_thresholds` is keyed by check type, `drift: {warn_on_drift, chunk_percent, consecutive_chunks}`.
  It returns a `ChainResult`; a detector that raises fails its step and the task. A classwise detector makes a
  whole-set finding and a by-class finding, so it can add two warnings where legacy added one. Classwise reads each
  item's label through DataEval's `Metadata`, so it needs a dataset with `.metadata`; without it the by-class run is
  skipped, where legacy read the labels from the targets. To upgrade:
  - `method: univariate|mmd|kneighbors|domain_classifier` is
    `type: drift-univariate|drift-mmd|drift-kneighbors|drift-domain-classifier`, and the univariate `test` is `method`
  - a detector's `classwise: true` is its name in `classwise: [...]`
  - `any_drift_is_warning` and `classwise_any_drift_is_warning` are `health_thresholds.drift.warn_on_drift`;
    `chunk_drift_pct_warning` is `chunk_percent`, and `consecutive_chunks_warning` is `consecutive_chunks`
  - `chunking.threshold_multiplier: k` is `chunking.threshold: [zscore, k]`. Legacy chunked every detector with a
    z-score threshold of 3, while an unset `threshold` now uses DataEval's default for the detector, a constant AUROC
    band for `drift-domain-classifier`, so `threshold: [zscore, 3.0]` restores legacy's judgment
- `ood-detection` is a preset: each detector is a step judged by an `ood` check, an `agreement` step groups each
  flagged image as mutual, partial or unique and is judged by `ood-agreement`, and two optional steps explain the
  flagged images by their metadata, as report sections where they were findings. Each test source is tested on its own
  against the reference, where they were joined; `merge` them in a custom workflow to test them as one. It returns a
  `ChainResult`; a detector that raises fails its step and the task. To upgrade:
  - `method: kneighbors|domain_classifier` is `type: ood-kneighbors|ood-domain-classifier`, and two detectors of one
    type need distinct `name`s
  - a domain-classifier detector thresholds on `n_std` unless `threshold_perc` is written, where legacy always used
    the 95th percentile: write `threshold_perc: 95` to keep legacy's verdicts
  - `health_thresholds.ood_pct_warning` and `ood_pct_info` are `health_thresholds.ood.warning` and `info`, and
    `health_thresholds["ood-agreement"]` judges the agreement
  - `max_ood_insights` is `factor_deviation.max_items`
  - `value_range` and the `metadata_*` fields are gone: set `value_range` on the dataset, and name a `metadata:` policy
- A result's JSON writes NaN and infinities as `null`, which strict JSON parsers require
- `drift-kneighbors` on the `uncertainty` extractor refuses a written `distance_metric: cosine`, which cannot rank one
  number
- The TUI offers workflow, evaluator and extractor types registered after import, plugins included
- `data-cleaning`'s `health_thresholds` take `None`, which judges nothing: the finding is still made, as `info`
- A custom workflow's or preset's result records the encodings its steps read, as `metadata_binning` and
  `encoding_digest`: one record where they read one Dataset one way, and `per_split`, keyed by the Dataset's address,
  where they read several. `dataeval-flow encoding` reads it
- A report's banner is the friendly title of what ran, as `Data Cleaning`; a custom workflow's is its name. The
  text report prints it in capitals; HTML keeps its case
- A report's envelope opens with a line naming what ran, `Workflow: clean (data-cleaning)` or, for an evaluator task,
  `Evaluator: dupes (duplicates)`: the id alone where the entry is unnamed or is the id, and
  `Workflow: name (custom workflow)` for a custom workflow. In HTML it is the provenance list's first row, and the
  page title adds the entry where it differs from the id, as `Data Cleaning — clean`
- A chain's report gives each finding a section, holding the evidence it judged: each step it read, headed *From* and
  the step's title, as `From Outliers`, or a line naming the finding it is shown under already. The steps no finding
  shows follow, then a Steps table of every step's title, type, status, reads and note, where the report gave each
  step a section headed `name (type)` and a line naming what it read. In HTML each finding is a card, and the Steps
  table a folded panel; the text table leaves out the title
- A step's heading is its type's title, with its name beside it where the name is not the type id: `Outliers`,
  `Duplicates · dupes`
- A chain whose checks ran once per element of a list groups their findings by the element's key, such as `train`
  and `val`, in its summary and below it; in HTML each finding stays a card
- `label-health`'s report lists each class's labels and images in a table
- `remove`'s report says what it kept and what each plan named: "Kept 22 of 24 images. Removed 2 images: 1 named by
  `dupes`, 1 by `outliers`."
- The report's configuration leaves out settings left unset, and keeps a setting written as `null`; an evaluator's
  report leaves out extras that hold nothing
- The Image Outliers, Target Outliers, Classwise Outliers and Label Distribution findings have no `description`, which
  repeated their brief
- A text table too wide for the report wraps its text cells, with a blank line between its rows
- A chain's binning record leaves out `label-health`'s reads, which read labels and no factor, so each Dataset it
  alone reads, such as each part of a split, no longer has a Metadata Factors block or binning diagnostics
- `data-splitting` is a preset: the whole set's labels, balance, diversity and coverage, a `split` or `kfold`, each
  train rebalanced where `rebalance:` is set, and each part's labels, stratification and coverage. Run as a step of a
  custom workflow, it exposes `train` (rebalanced where set), `val` and `test`, as lists keyed by fold under
  `folds` of 2 or more. It returns a `ChainResult`: each part's indices are in
  `result.steps["split"].details["indices"]`. Under `folds` of 2 or more, each fold's rebalanced train is in
  `result.steps["rebalance"].elements["<k>"].details["indices"]`; where rebalancing kept the train as it was,
  `details` is `None` and the train's indices from `split` apply.
  Its findings are Label Distribution, Stratification for each fold, and Uncovered Rate under `naive` coverage;
  balance and diversity are report sections, and the split's sizes are in the `split` step's section and the
  `lineage`. Only object-detection Datasets can be exported, so a classification split's parts can be read but not yet
  exported. `health_thresholds` is keyed by check type: `class-imbalance`, `stratification`, `uncovered-rate`. To
  upgrade:
  - `num_folds` is `folds`
  - `rebalance_method` is `rebalance`
  - `coverage_percent` and `num_observations` are `coverage.percent` and `coverage.num_observations`, beside
    `coverage.method`
  - `val_frac` with `folds` of 2 or more is refused, where legacy ignored it: remove it, since each fold's val is its
    1/k. `val_frac` unset is 0.1 with `folds: 1`
  - the `split_sizes` and `stratified` of `metadata`, and `output.raw`, are in `result.steps` and `lineage`
- `data-coverage` is a preset. A `crops` step (`wrap`) crops detection data into one item per box and hands other data
  on unchanged. `coverage` and `completeness` embed the crops, judged by `class-coverage`, by `uncovered-rate` under
  `naive` coverage, and by `completeness-score`. `labels` (`label-health`) is judged by `class-imbalance`; `summary`
  (`metadata-summary`), `balance` and `diversity` read the metadata, and `gaps` (`factor-gaps`), judged by
  `coverage-gaps`, reads balance; `worklist` (`representation`) is judged by `class-shortfall`. Without an extractor
  the embedding steps are skipped, and their findings say "not assessed". It returns a `ChainResult`, whose numbers
  are each step's output in `result.steps`, such as `result.steps["coverage"].output`. Metadata Distribution, balance
  and diversity are report sections, where Metadata Distribution was a finding. Under `naive` coverage, legacy's
  Embedding Coverage finding is two, Embedding Coverage and Uncovered Rate; under `adaptive` the uncovered rate is not
  judged. Naive coverage that overflows is skipped, where legacy re-ran it as adaptive. On detection data, coverage's
  uncovered items index the `crops` Dataset, one item per box, where legacy named each one's image and box. It no
  longer judges an ontology: a `label-space` entry on the same source does, and a data-coverage run on a conformed
  source records no label space of its own, so `label-space` carries the join key. `health_thresholds` is keyed by
  check type: `class-imbalance`, `coverage-gaps`, `class-coverage`, `uncovered-rate` and `completeness-score`. Every
  legacy field is refused by name, with its replacement. To upgrade:
  - `coverage_method`, `coverage_percent`, `num_observations`, `min_class_samples`, `isotropy_min_samples` and
    `near_duplicate_factor` are `coverage.method`, `.percent`, `.num_observations`, `.min_class_samples`,
    `.isotropy_min_samples` and `.near_duplicate_factor`
  - `crop_padding` and `crop_min_size` are `crops.padding` and `crops.min_size`
  - `run_completeness` is `completeness`
  - `diversity_method` is refused: diversity always runs, as a report section, and `diversity` picks its method, so
    legacy's `null` (skip diversity) has no replacement
  - `balance` is refused: balance always runs, as a report section
  - `run_gap_analysis` is refused whatever its value: `gaps: null` replaces `false`. `gap_mi_threshold` and
    `gap_min_representation` are `gaps.mi_threshold` and `gaps.min_representation`
  - `ontology` and `ontology_label_pattern` are refused: write them on a `label-space` entry on the same source, as
    `ontology` and `label_pattern`
  - `ontology_expected` is `expected`, or `label-space`'s `expected` where an ontology is set
  - `metadata_auto_bin_method`, `metadata_exclude`, `metadata_continuous_factor_bins` and `metadata_factor_source`
    are refused: name a policy under `metadata:`
  - `value_range` is refused: set it on the dataset. `stats` is refused, since no step of data-coverage reads statistics
  - in `health_thresholds`, `class_imbalance_ratio` is `class-imbalance.ratio`, and legacy's fixed band at 2.0 is
    `class-imbalance.info`; `gap_count` is `coverage-gaps.count`; `min_dispersion`, `min_isotropy` and
    `max_near_duplicate_fraction` are `class-coverage.dispersion`, `.isotropy` and `.near_duplicates`;
    `uncovered_rate` is `uncovered-rate.rate`; and `completeness_score` is `completeness-score.warning`, and legacy's
    fixed band at 0.8 is `completeness-score.info`. An unset `info` follows `ratio` or `warning` as legacy's band did,
    so `ratio: 1.5` or `warning: 0.9` alone loads; two written bounds that cross are refused
  - an ImageFolder source's label finding is titled "Label/Directory_Name Distribution", where it was "Label
    Distribution"
  - `health_thresholds.leaf_coverage`, `dark_branch_count` and `unmatched_class_count` are `label-space`'s
    `health_thresholds.leaf-coverage.coverage`, `leaf-coverage.empty_branches` and `label-conformance.unmatched`
  - `output.raw` and `metadata.has_extractor` are gone: `coverage`, `completeness` and `metadata_gaps` are the
    `coverage`, `completeness` and `gaps` steps' outputs, `label_distribution` is `labels`', `metadata_distribution`
    is `summary`'s, a skipped step's reason is its `reason`, and `coverage.dropped_detections` is
    `result.steps["crops"].details["dropped"]`
- `label-health` lists every class the Dataset declares, at 0 where it has no labels, and the items with no label as
  `empty_image_indices`. So a declared class with no labels now shows at 0 in data-cleaning's and data-splitting's
  label tables and in stratification's "Classes checked" and its table across the parts, and makes their Label
  Distribution finding warn
- `class-imbalance` makes its finding on any Dataset with classes, declared or observed, so an unlabelled Dataset that
  declares classes now warns in data-cleaning and data-splitting, where it made no finding. Its ratio is taken over
  the classes with labels, and each class with none is named. It takes `info`, a ratio at or under which the finding
  is `ok`, and its evidence gains each class's share and the images with no labels

### Fixed

- Every step of a task reads a source through its Dataset node's one draw of its view, so an unseeded `Shuffle` or any
  random view gives all of a task's steps the same order, where each step drew its own. A result's `dataset` and
  `sources`, and the report's thumbnails, hold that same draw, where they drew the view again
- An exactly declared `continuous_factor_bins` name beats a bare statistic's expansion, whatever the key order
- `metadata-triage` no longer calls a factor its policy's descriptor pins unpinned, so it suggests no bin count the
  policy refuses as named by both `encoding` and `continuous_factor_bins`
- Data prioritization ranks a labeled pool under `policy: class_balanced`, where it always
  raised "Cannot apply class_balanced policy: class_labels not provided"
- Data prioritization succeeds when cleaning empties a pool, ranking it as empty, where the whole task failed
- Classwise drift names each class from the datasets' `index2label`, where it showed the bare class index
- The config builder keeps a pipeline's `result:`, `logging:`, `seed:` and `deterministic:` when it saves a
  config, where it dropped them, and the TUI runs with them
- The TUI shows a task, source, class or split name with brackets in it as written; `[/x]` no longer crashes it
- Data prioritization's `sources` and data splitting's `dataset` are the views they ran on, not views drawn anew
- Classwise drift prints a small p-value as itself (`0.0003`), not `0.00`, and `results.json` keeps it unrounded
- A chunked drift finding with `chunk_percent: 0`, legacy's `chunk_drift_pct_warning: 0`, no longer warns when no
  chunk drifted
- `run_tasks`, the CLI and the TUI share one BoVW fit per task; its embeddings and clusters are cached only with `seed`
- Data-cleaning and parameter-sweep key clusters by their extractor; cached stateless cleaning clusters miss once
- Data-cleaning's cluster-mode duplicate merge now passes `merge_near_duplicates`, agreeing with the `duplicates`
  evaluator
- Hash dataset elements lacking `__repr__` by type and contents rather than memory address, enabling cache reuse
- Include source views in cache keys, invalidating cached embeddings, metadata, and statistics on edit
- Relative `ontology:` paths now resolve against the run's data root rather than the process root
- Restrict outlier detection to configured `outlier_flags`, preventing shared caches from widening column checks
- Restrict duplicate detection to configured `duplicate_flags`, preventing shared cache widening
- Restrict injected metadata factors to configured `intrinsic_factors` instead of cache state
- Record band-group statistics in `injected_factors` rather than misattributing them as dataset columns
- Restrict cross-split label parity to shared classes, preventing chi-square errors on gapped label spaces
- Exclude single-split classes from the parity test and report them in `label_overlap` instead
- Restrict cross-split duplicate detection to common columns, preventing crashes on divergent stat caches
- Container entrypoint honors `DATAEVAL_OUTPUT` and `DATAEVAL_CACHE` instead of hardcoded paths
- Container image creates `/cache/.not_mounted`; unmounted `/cache` is no longer exported as `DATAEVAL_CACHE`
- Consistently recognize object-detection datasets on Python 3.10 and 3.11, preventing misclassification or crashes
- GitHub release body now carries the changelog section instead of falling back to `Release vX.Y.Z`
- Container images no longer ship the standalone interpreter's bundled `pip`, which nothing in the image used
- A task naming one source twice is refused when the config loads; the repeat was dropped, leaving the task a
  source short
- A key no config section defines is refused when the config loads, where it was dropped; a misspelled top-level key
  is named with the section it most resembles
- The HTML report renders inline code in a table cell as it does in prose, where the cell showed the backticks
- A chain refused before any step ran says why in its report, where it said only `Steps: 0 ran`
- A report's health line and its HTML badge say `failed` where a required step failed, where they could say every
  check passed
- A "not assessed" finding's description ends in one full stop where its cause already ends in one, where it ended in
  two

### Removed

- `parameter-sweep`, with `ParameterSweepConfig`, `ParameterSweepResult` and `ParameterSweepWorkflow`: write a
  data-cleaning entry and a `matrix:` on its task (see Sweep settings with a matrix)
- The `torch` and `uncertainty` extractors' `device`, and drift-monitoring MMD's; Flow chooses the device for every tool
- Poetry packaging support; install with uv, pip, or conda instead
- Floating `<variant>` and `<major>.<minor>-<variant>` image tags; pull `latest-<variant>` or pin `<version>-<variant>`
- Python 3.10 support; the minimum supported version is now 3.11
- `dataeval_flow.config.schemas`; most of its types are imported from `dataeval_flow.config`
- Deprecated `SelectionConfig`, `SelectionStep`, `build_selection` and `DatasetContext(selection_steps=)`
- Typed task configs (`DataCleaningTaskConfig`, `EvaluatorTaskConfig`, …); use `TaskConfig`
- `load_config_folder` and `export_params_schema`
- Public per-type output, report and metadata models (`DataCleaningOutputs`, …); narrow to `<X>Result` instead
- The nested output models and TypedDicts those models held (`CoverageAssessment`, `OutlierIssuesDict`, …)
- `select_tasks`; `run_tasks` keys its results by task name, so nothing needs pairing
- A workflow's `output_schema`; its `<X>Result` type argument names the output
- The `DriftDetectorConfig` and `OODDetectorConfig` unions; annotate with the detector classes
- The `AutoBinMethod` and `FactorSource` aliases; their fields take the same strings
- `DataCleaningResult`, with its metadata's `evaluators`, `flagged_indices`, `clean_indices` and `removed_count`;
  a data-cleaning result is a `ChainResult`
- `DataPrioritizationResult`, `DataPrioritizationHealthThresholds` and the `CleaningSummaryDict` and
  `PerDatasetPrioritizationDict` output types; a data-prioritization result is a `ChainResult`
- `health_thresholds` and `value_range` on `data-prioritization`, which refuses them: declare the range on the dataset
- `value_range`, `metadata_auto_bin_method`, `metadata_exclude`, `metadata_continuous_factor_bins` and
  `metadata_factor_source` on `data-cleaning`, which refuses them: declare the range on the dataset, and the binning in
  a `metadata:` policy
- `mode`, from every workflow config and from every result's metadata; a config that still writes it fails to load,
  naming it
- Data-prioritization's `per_source_clean_indices` and `per_source_prioritized_indices`
- The "Preparatory Mode" findings that data-analysis and data-cleaning made
- `MetadataTriageResult`, with its metadata's `blocking` and `verified`; a metadata-triage result is a `ChainResult`
- `metadata_auto_bin_method`, `metadata_exclude`, `metadata_continuous_factor_bins` and `metadata_factor_source` on
  `metadata-triage`, which refuses them: declare the binning in a `metadata:` policy
- `update_strategy` on `drift-monitoring`, which was never applied and is now refused
- The `DriftDetectorUnivariate`, `DriftDetectorMMD`, `DriftDetectorKNeighbors` and `DriftDetectorDomainClassifier`,
  `ChunkingConfig`, `UpdateStrategyConfig` and `DriftMonitoringHealthThresholds` classes; a detector is a drift
  evaluator config, and its `chunking:` a `ChunkedDriftConfig`
- `DriftMonitoringResult` and its parts; a drift-monitoring result is a `ChainResult`
- `OODDetectionResult`, `OODDetectorKNeighbors`, `OODDetectorDomainClassifier` and `OODDetectionHealthThresholds`;
  ood-detection returns a `ChainResult`, and its detectors are OOD evaluator entries
- `DataSplittingResult` and its output and metadata types; a data-splitting result is a `ChainResult`
- `DataCoverageResult` and its output and metadata types, and `DataCoverageHealthThresholds`; a data-coverage result is
  a `ChainResult`, and its thresholds are `DataCoverageThresholds`

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
