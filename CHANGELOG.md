# Changelog

## Unreleased

### Added

- `dataeval-flow serve` (new `service` extra, in every image): an HTTP service that queues pipelines and runs them
  - `POST /v1/runs` takes a pipeline and its tasks; each run is a separate headless `dataeval-flow` process
  - Runs execute one at a time, each keeping its snapshot, logs and result files under `<output>/runs/<id>/`
  - `/v1/runs/{id}/results` shows each task's result as soon as the task finishes
  - Runs can be cancelled, and the run history survives restarts
  - Stopping the service interrupts the running run and keeps the queued ones
  - `/healthz`, `/livez` and `/readyz` (IR-2.3-H-2, IR-2.3-S-1), and `/openapi.json` (IR-2.4-S-1)
  - Run output is logged to the service console, tagged with the run's ID
  - Images set `DATAEVAL_SERVICE_HOST=0.0.0.0` and expose port 8001
- `run_tasks(..., on_result=...)`, called with each task's name and result as soon as the task finishes
- `judges` and `judged_by` in the step catalog link each check to the Outputs it judges
  - `dataeval-flow steps <type>` shows both
- `collect` transform (`CollectConfig`, `CollectTransform`): gathers Datasets into one list keyed by name
- The `audit` preset: one chain over train and each evaluation split that gives a verdict before training
  - The report gives a verdict ("Not ready", "Ready with caveats" or "Ready"), findings and next steps
  - Findings fall under five questions; next steps cover each warning and each check not assessed
  - A record lists each split's items, classes, factors and digests, and the criteria applied
  - `blocking:` names the check types whose warnings block readiness; `accepted:` records why a warning is accepted
  - `accepted:` also takes a check step (`image-outliers-evals`) or one of its runs (`image-outliers-evals[test]`)
  - `ChainResult.verdict` (`verdict` in the JSON) holds the level and the unaccepted warnings behind it
  - It also lists each acceptance and each check not assessed
  - `audit` can run as a step of a custom workflow, on splits an earlier step made
  - As a step it keeps its preflight, and encodes every split like its reference
  - Only one step per workflow may give a verdict, and it can't be `optional:`
  - The verdict names the step's inner steps `audit/...`, and `accepted:` keys cover them without the prefix
  - `no_verdict` explains a missing verdict when the step never started or failed as it started
- `result: require`, `--require` and `DATAEVAL_REQUIRE`: exit 4 when a task's verdict is worse than the level given
  - Levels: `ready-with-caveats`, `ready-with-accepted-risks` and `ready`
  - A run in which no task gives a verdict is refused
- The `bias` preset: one source's class balance and how its metadata factors relate to the class, without an extractor
  - Runs `class-imbalance`, `shortcut-risk` on `balance`, and `factor-parity` on `parity`
  - Runs `factor-gaps` with `factor-coverage-gaps`, and shows `diversity` and `factor-summary` as report sections
- `factor-parity` check: metadata factors tied to the class, judged from a `parity` Output
  - Warns where a factor's Cramér's V is over `warning` (0.3) and its chi-square p-value at most `p_value` (0.05)
  - Names the factors whose sparse tables make their p-values unreliable
- `groups:` on a custom workflow: report headings collecting the findings of the check types they name
- `dataset_digest()`: SHA-256 digests of a dataset that don't depend on the order of its items
  - `content` covers its images, labels and class names, and `metadata` its metadata
- `dataset_manifest()` and `DatasetManifest`: per-item hashes, saved as JSON, that compare to show which items changed
  - The `dataeval-flow` command writes each `content-digest` run's manifest under `results/manifests/`
- `dataeval-flow verify`: checks a source still holds a manifest's items, naming changed, missing and added ones
- `content-digest` evaluator: a source's `dataset_digest()`, read uncached through a new `dataset` input kind
- `load_source()` loads a configured source the way a run reads it, with its view and merge applied
  - It refuses a view that would draw different items on each load
- Exports record the digest of what they wrote, read back as a training job would load it
  - The digest goes in `provenance.json` and the `export` step's record, with each source's `provenance:` facts
- `provenance:` on `datasets:` entries: facts Flow can't measure, such as owner, licence, origin and collection date
  - Recorded in `resolved_config` as written, and not part of the cache key
- `library_versions` on every result's metadata
  - DataEval, PyTorch, NumPy, Pillow, datamaite, OpenCV (where installed) and the extractor runtime (e.g. ONNX Runtime)
- `device` on every result's metadata: the device its task ran on, such as `cuda:0 (NVIDIA L4)`
- Every tool runs on CUDA when PyTorch sees a GPU, and on the CPU otherwise
  - `dataeval_flow.set_device` picks the device from Python, and `CUDA_VISIBLE_DEVICES` hides GPUs
  - Configs don't name a device
- Find the Right Step: a docs index from a question to the presets and steps that answer it
- Config Reference: a docs page per top-level config key, its every key's type and default, from the JSON Schema
  - Preset, combine and naming-convention reference pages
  - Evaluator and check catalogs grouped by question, each entry with ports, settings, checks, presets and an example
- `empty:` on a custom workflow's list input lets a task bind no source to it; its checks then report that reason
  - A step over an empty list leaves one record; each step record's `not_assessed` says why a check judged nothing
- A check port that reads several lists is skipped only when all of them are empty; `may_be_empty` ports never are
- `pairs: true` on a step reading two Datasets through one input runs it once per unordered pair, keyed `a_vs_b`
- Presets may name a reference split whose cuts encode every other split's metadata, so factor statistics line up
  - A policy's `reference_split` names a different reference
- `divergence` evaluator: how far apart two sources' embeddings sit, by `divergence_mst` or `divergence_fnn`
  - `embedding-divergence` check bands it high, moderate or low
- `factor-leakage` evaluator: the raw values of named factors two sources hold, whatever the policy excludes
- `leakage` check: warns on duplicate groups spanning two splits and on group values two splits share
- `class-sufficiency` and `untrained-classes` checks: classes with too few labels, and evaluation classes train lacks
  - `empty: false` on `class-imbalance` leaves classes with no labels to these checks
- `shortcut-risk` check: metadata factors tied to the class, judged from a `balance` Output
- `eval-coverage` check: the share of an evaluation split lying farther from train than most of train does
- The `taxonomy` preset: a dataset's labels judged against a declared ontology
  - Leaf coverage and its worklist, conformance, alignment with the `Relabel` stanza, and the ontology's structure
  - `checks` keyed by type: `leaf-coverage` (`coverage`, `empty_branches`) and `label-conformance` (`warning`)
  - `label-reconciliation` evaluator: which class names resolve to exactly one ontology concept
  - `ontology-validation` evaluator: the ontology's structural and naming facts
  - `leaf-coverage`, `label-conformance`, `label-mergeability` and `ontology-structure` checks
- `uncertainty` extractor: normalized entropy per prediction from an ONNX classifier or detector, for unlabelled drift
  - One row per detection for a detector; only drift and OOD evaluators read it
- `by: predicted` keys a step by the class a model predicts
- A shift detector may name its own `extractor:`, used instead of the task's
- Top-level `evaluators:` key running one DataEval evaluator, any of the catalog's twenty-five types
- `evaluator:` on tasks as an alternative to `workflow:`, checked against the evaluator when the config loads
- `kind` on `TaskConfig`, recording whether `workflow:` or `evaluator:` named the task's target
- `dataeval-flow evaluators` lists evaluator types, what each consumes, and their parameter schemas
- `kind` on every result envelope (`workflow` or `evaluator`); evaluator results carry no `health`
- `Result`, the base both result kinds share, with `report()`, `to_dict()` and `export()` and no health
- Top-level `ontologies:` key for named label spaces that workflows reference
- `concepts:` on ontology entries adds or overrides concepts by id without editing the source artifact
- Workflow `ontology:` resolves pool names first, falling back to file paths
- `merge:` on sources concatenates several inputs into one dataset, unified through `Relabel` views
- Top-level `exports:` key writing sources as COCO, YOLO, Hugging Face or VisDrone, with `provenance.json`
- `ontology:` on every workflow but `scope`, attaching the label-space digest to results
- `label_space` on result envelopes: the conformed vocabulary and its label-space digest
- `channel_groups:` on datasets measures band groups separately, as `<group>_<statistic>` columns
  - A group written `{bands: ..., value_range: [low, high]}` is read against its own range
- Top-level `stats:` key for policies on measured statistics, background inclusion, and outlier and factor views
- Statistic sub-groups wherever a family is named: a stats policy's `measure`, `outliers.flags` and `intrinsic_factors`
  - Pixel and visual: `pixel_basic`, `pixel_distribution` and `visual_basic`
  - Dimension: `dimension_basic`, `dimension_box`, `dimension_offset` and `dimension_position`
  - In `measure` only: `hash_basic` and `hash_d4`
- `format: demo` dataset loader resolving tutorial datasets from a fixed table, with no arbitrary imports
- `wrap:` on `scope`, with `padding` and `min_size` on `DetectionCrops`; the `crops` step counts dropped detections
- `DATAEVAL_*` environment variables for CLI parameters; CLI arguments take precedence
- `--no-fail-on-warning` overrides `DATAEVAL_FAIL_ON_WARNING` for one run
- `--log-format {structured,plain}` picks structured or plain log output
- `scripts/release.py` cuts and tags releases from the current branch
- Published images carry their vulnerability scan report and CycloneDX SBOM in `/usr/share/dataeval-flow/security/`
- `nox -s verify` writes `output/metarepo/test-results.log`, a per-test-case log for metarepo assessments
- Plug-in workflows and evaluators through the `dataeval_flow.workflows` and `dataeval_flow.evaluators` entry points
- Plug-in extractors and image transforms through `dataeval_flow.extractors` and `dataeval_flow.image_transforms`
- Every workflow declares its inputs, and each task is checked against its workflow or evaluator at load
- `WorkflowContext` methods for workflow inputs: `dataset`, `stats`, `embeddings`, `clusters`, `metadata`, `labels`
- `run(config, data)` runs one workflow or evaluator on in-memory datasets, typed to the config's result
- An extractor config without a `name` is named after its `model`, as workflow and evaluator configs are
- Extension bases document how to subclass and register a plugin, and a test pins the public API
- Each `<X>Result`'s API page lists the `output.raw` and `metadata` fields its type adds, pinned by a test
- `LoggingConfig`, the type of `PipelineConfig.logging`, exported from `dataeval_flow.config`
- `--report-width` and `DATAEVAL_REPORT_WIDTH` set the text report's width; `Result.report()` takes `width=`
- `Result.to_html()` renders a self-contained, printable page; `--output` writes it as `result.html`
- The HTML report shows a card per finding, with warnings expanded, sortable and filterable tables, and dark mode
- `drift-univariate`, `drift-mmd`, `drift-kneighbors`, `drift-wasserstein` and `drift-domain-classifier` evaluators
  - Each takes `chunking:`; Wasserstein takes a validation source between the reference and the tested data
- `SourceCount.THREE`, for entries that take exactly three sources
- `flags` table columns, each cell a list of measurements against their population, and a row `preview` on tables
- `image` table columns, each cell an item reference or a group of them
- Thumbnails of each item a finding names, in `Result.assets` and the HTML report
  - `--no-report-images`, `DATAEVAL_REPORT_IMAGES=0` or `report_images=False` on `run()` turn them off
  - `result: max_images:` sets how many each result embeds (200), shared evenly between findings
- Coverage, prioritization, splitting and metadata triage reports show thumbnails of the items they name
- `uncovered_classes` in `coverage`'s `extras`, and an "Uncovered items" section giving each item's class and distance
- The `result:` block names the result files and picks their formats, detail, per-task split and text width
- `result: max_rows:` (500) and `preview_rows:` (10) limit item tables and text previews; `-1` lifts a limit
- `result: fail_on:` sets the exit-code gate (`failure`, `warning` or `never`); `--fail-on-warning` still overrides it
- `junit` and `markdown` formats: a JUnit report for CI test views and a Markdown summary, both naming failed tasks
- Evaluators can read `metadata`, `labels`, `embeddings` and the task's resolved `ontology` on `EvaluatorInputs`
- `output_extras` on `Evaluator`: results DataEval keeps outside `data()`, written under `extras` and shown in reports
- `duplicates` takes DataEval's video parameters and writes `annotation_divergences` and `factor_cardinality` extras
- `balance`, `diversity` and `parity` evaluators, reading one source's metadata under its `metadata:` policy
- `representation` evaluator: a source's labels counted against an ontology, or one built from its `index2label`
- `coverage` and `prioritization` evaluators; a second `prioritization` source is the reference for its ranking
- `ood-kneighbors` and `ood-domain-classifier` evaluators
- An "Evaluator recipes" how-to with a worked example per evaluator family
- Custom workflows: a `workflows:` entry with `inputs:` and `steps:` chaining evaluators, workflow types and transforms
  - Each step reads an input or an earlier step by address (`clean`, `split.train`, `kfold.train[0]`)
  - A step runs once per element of a list; a failed step skips only the steps that read it
  - A task runs a custom workflow with `workflow:`
- `CustomWorkflowConfig` and `StepEntry` build a custom workflow in Python; `save()` writes it into a config file
  - Loading and saving a config, from the TUI and the config builder too, keeps custom workflows as written
  - `save()` and `to_yaml()` take `definitions` and write out the named entries the steps use, so the block stands alone
- Nine transforms for custom workflows
  - `view`, `merge`, `split`, `kfold`, `wrap` and `select`
  - `remove`, applying DataEval's removal plans from Duplicates and Outliers
  - `conform`, relabelling onto an ontology and refusing loss beyond `allow:`
  - `export`, writing to `<output>/datasets/<task>.<step>/`
  - Plug-in transforms register under `dataeval_flow.transforms`
- `label-alignment` evaluator, aligning a source's class names to an ontology as `scope` does
- `ChainResult`: each step's outcome in `steps`, readable after a failure, and each Dataset's lineage in its metadata
  - `metadata.lineage` digests tell whether two results read the same data
- `output_dir` on `run_tasks`, `run_task` and `run`, where export steps write; the CLI passes `--output`
- `dataeval-flow steps [NAME] [--json]` and `list_steps()` list every chainable step with its ports and settings schema
- A "Workflows as Chains of Steps" explanation, a custom-workflow how-to, and a Transform Catalog
- Presets: workflow types whose settings expand into a chain of steps that runs like a custom workflow
  - As a task, a preset returns a `ChainResult` under its own type id
  - As a step, its chain runs inside yours as `<step>/<name>`, and `<step>.<output>` reads each Dataset it declares
- A chain computes each dataset's statistics in one pass over every family its steps read
- `quality`'s `clean` step passes on the dataset minus flagged images and boxes, keeping one item per duplicate group
  - It counts what it removed, and a custom workflow reads it as `<step>.clean`
  - An `export` step on it replaces `mode: preparatory`
- A title on every step type, such as `K-Fold`, as `title` in `dataeval-flow steps` and `list_steps()`
- A short chain report, from `report(detailed=False)` and the console without `-v`
  - Step count, summary, health, and a Steps table with each step's status and why any step made nothing
- `by_plan` in a `remove` step's `details`: what each plan named, at each level
- Report blocks gain `in_text` on table columns, `failed` on summaries and `group` on summary items
  - Each is left out of the JSON at its default
- `select:` on `prioritization` (`n` or `fraction`): its `selected` step keeps each pool's top items
  - A fraction is rounded up; unset, it keeps every item in ranked order
- `prioritization` reports show the 25 highest- and 25 lowest-ranked items, with rank and score
- `factor-triage` evaluator: metadata a Dataset failed to read, and the policy stanza that repairs it
  - With `verify`, also what the repair recovers
- `factor-issues` check: triage's findings, made from a `factor-triage` Output
- `factor-triage` recommends a policy: unreadable values dropped to missing, and edges or levels for unpinned factors
  - A dominant value, such as a speed of zero, stays in, with a note asking you to decide
  - `factor-issues` shows it as a "Recommended policy" finding, cautioning that unrepresentative data can mislead
- `by: class` on an evaluate step runs its evaluator once per class, or per named group of classes
  - Only classes with `min_items` items in every input run; the Output names each class left out and why
  - A class whose run raises is left out with its error; the step fails only when every class raises
  - On a check, it judges each class and rolls the findings into one, with a brief such as `2/8 classes warn`
- `ood` and `ood-agreement` checks, and `ood-union`, `factor-predictors` and `factor-deviation` combines, run by `shift`
- OOD evaluators read the `uncertainty` extractor, per image for a classifier and per detection for a detector
  - Any flagged detection flags its image
- Combines may read a Dataset's statistics and draw their own report section
  - A combine may require the Outputs it reads to share their Datasets, or to be computed on its own Dataset
- `drift` check: warns on drift; on chunked runs, when `chunk_percent` of chunks drift or `consecutive_chunks` in a row
- Drift evaluator results have a report section: the verdict's fields, or one row per chunk
- `matrix:` on a task runs its entry once per combination of the values it lists, returning one `MatrixResult`
  - Values as lists, inclusive `{from, to, step}` ranges or several grids
  - It varies the entry's settings, the task's `sources` and `extractor`, and the entries and steps the task reads
  - The report opens with a table comparing every run's findings
  - Runs share one draw of each source and the cache; exports write under `run-<n>/`
- `class-stratification` check: how far each part's class shares deviate from the whole's, in percentage points
  - Warns past `warning` and informs past `info`
- `uncovered-items` check: warns when a coverage run leaves over `warning` percent of a Dataset uncovered
- `split`, `kfold` and `view` steps record each output's indices in their `details`; `split` and `kfold` report sizes
- A preset's declared outputs read any chain address, lists included; `splits`'s `<step>.train` is the rebalanced train
- A subset whose view changes no pixels, such as a split's part, slices its parent's embeddings instead of re-extracting
- An optional step that needs an extractor is skipped with a reason when neither it nor the task names one
- `completeness` evaluator: how much of the embedding space's dimensions the data fills; it needs two embeddings or more
- `factor-summary` evaluator: each metadata factor's type, binning, nulls, and range or top values
- `class-coverage`, `dimensional-completeness`, `factor-coverage-gaps` and `class-shortfall` checks, for `scope`
- `factor-gaps` combine: each factor's mutual information with the class, read from a `balance` Output
  - Names under-represented class-factor combinations among the factors at or over `mi_threshold`
- `other_kinds: pass` on `wrap` passes other kinds of Dataset through unchanged, reading their cached embeddings

### Changed

- The `dataeval-flow` command writes each task's result files as soon as the task finishes, so a killed run keeps them
  - Each file is replaced whole, so a reader never sees a half-written one
- Every check's description opens with what it judges, such as "Judges `balance`'s output: …"
- `drift-monitoring` and `ood-detection` merge into one `shift` preset; the old ids fail as unknown workflows
  - One `detectors:` list takes drift and OOD detectors, each judged by its family's check
  - `classwise` takes drift detectors; the OOD union, agreement and factor steps run only with OOD detectors
  - With no `detectors:`, it runs `drift-univariate` (KS, Bonferroni) and `ood-kneighbors`
  - `ShiftConfig`, `ShiftChecks` and `ShiftWorkflow` replace the two presets' classes
- Presets are named for the DataEval module or question they answer; the old ids fail as unknown workflows
  - `data-bias` → `bias`, `data-cleaning` → `quality`, `data-coverage` → `scope`, `data-splitting` → `splits`
  - `data-prioritization` → `prioritization`, `metadata-triage` → `triage`, `label-space` → `taxonomy`
  - `triage` covers metadata and annotations only
  - Packages, classes (`QualityConfig`, …) and titles follow
  - The `prioritization` preset's config is `PrioritizationWorkflowConfig`; the evaluator owns `PrioritizationConfig`
- A type id may repeat across kinds: `prioritization` is a preset and an evaluator
- Five checks are renamed for their subject or the evaluator they judge; the old names fail as unknown checks
  - `classwise-outliers` → `class-outliers`, `stratification` → `class-stratification`
  - `metadata-issues` → `factor-issues`, `mergeability` → `label-mergeability`
  - `distribution-shift` → `embedding-divergence`
  - Their classes, titles, preset chain step names and `checks:` keys follow
- `eval-coverage` measures a split's flagged share against a baseline: what a split drawn like train flags anyway
  - The baseline is 100 - `threshold_perc` under `ood-kneighbors`, and 0 under other detectors
  - `info` (1.0) and `warning` (9.0) are points above it, which keeps the current bands at `threshold_perc: 99`
  - Under other detectors the bands move from 2/10 to 1/9 percent flagged
  - `audit`'s `ood-kneighbors` takes DataEval's `threshold_perc` of 95
- Presets use DataEval's default for every setting and no longer restate them
  - `scope` and `audit` no longer default `coverage.num_observations` to 50; DataEval's 20 applies
- Flow departs from DataEval's defaults in two places
  - `prioritization` sorts `hard_first` (DataEval: `easy_first`)
  - `split` and `kfold` hold out `test_frac: 0.2`, and `split` `val_frac: 0.1`, stratified (DataEval holds nothing out)
- One settings model per step type, shared by every preset that offers it
  - `ClassImbalanceSettings` replaces `DataBiasClassImbalanceSettings` and `AuditClassImbalanceSettings`
  - `RepresentationSettings` replaces `DataCoverageRepresentationSettings` and `LabelSpaceRepresentationSettings`
  - `CoverageSettings` and `UncoveredItemsSettings` are renamed from `DataCoverage…`
- `class-imbalance` uses its own defaults in every preset
  - `bias` no longer defaults `info` to 2.0: ratios up to 2.0 are `info` unless `info: 2` is set
  - `audit` defaults `empty` to `true`, so a declared class with no labels warns there too
- Each preset but `audit` owns its checks, so presets run side by side report each finding once
  - Class balance and the metadata factors move to `bias`; `audit`'s chain is unchanged
  - `scope` drops `label-health`, `class-imbalance`, `factor-summary`, `balance` and `diversity`
  - It also drops `factor-gaps` and `factor-coverage-gaps`
  - Its `metadata`, `diversity`, `factor-gaps`, `checks.class-imbalance` and `checks.factor-coverage-gaps` are refused
  - `quality` drops `class-imbalance`, keeping `label-health` for `target-outliers`
  - `splits` drops the whole set's `class-imbalance`, `balance` and `diversity`; `label-health` stays for stratification
  - `quality` and `splits` refuse `checks.class-imbalance`
  - `DiversitySettings`, `FactorGapsSettings` and `ShortcutRiskSettings` import only from `dataeval_flow.workflows.bias`
  - `DataCleaningClassImbalanceSettings` is gone
- `prioritization`'s ranking settings sit under a `prioritization:` block, modelled by `PrioritizationSettings`
  - `method`, `k`, `c`, `n_init`, `max_cluster_size`, `order`, `policy` and `num_bins`
- Findings from quality, scope and shift are retitled; anything that matches on a title needs updating
  - "Duplicates" → "Image Duplicates"
  - "Label Distribution", and "Label/Directory_Name Distribution" on ImageFolder, → "Class Imbalance"
  - "Embedding Coverage" → "Class Coverage"; "Metadata Coverage Gaps" → "Factor Coverage Gaps"
  - "Label Space Coverage" → "Leaf Coverage"; "Class Balance Worklist" → "Class Shortfall"
  - "Aggregate OOD (all detectors agree)" and "Unique OOD Samples (single-detector only)" → "OOD Agreement"
- Names follow one rule per kind (see Naming conventions)
  - A step type and its title name one thing, as `class-imbalance` and "Class Imbalance"
  - Checks are named for what they judge (`image-outliers`), and a check's one bound is `warning`
  - A preset holds a step's settings under its type, in the step's words (`outliers: {flags, outlier_threshold}`)
  - A preset's checks sit under `checks:` keyed by check type, and its steps are named for their types
- `Finding` is exported from `dataeval_flow.steps`, beside `Check`, and no longer from `dataeval_flow.workflows`
- A check now warns only past its bound, not on reaching it
  - This changes `ood`, `ood-agreement` and `eval-coverage`, which warned and informed at their bounds
  - It also changes `drift`'s `chunk_percent` and `consecutive_chunks`, and `factor-coverage-gaps`
  - `consecutive_chunks` and `factor-coverage-gaps` now default to 2 (were 3), so three in a row, or three gaps, warn
- A chain's `label_space_digest` comes from its `label-alignment` steps when they agree and nothing else records one
  - It joins the result to a dataset conformed by that alignment's stanza
- `representation` records the `expected` names it ignored, under `extras.ignored_expected`
  - Its report section and `label-alignment`'s are short summaries
- `representation`, `coverage`, `prioritization` and `label-alignment` read only labels, and add no binning record
  - `splits` with an extractor records the whole set once; `prioritization` records no binning or `encoding_digest`
- `uncertainty` extractor entries need `metadata_path` and `preds_type`, and the TUI no longer offers them
- shift's `classwise:` maps each detector to its `by:` (`{drift-mmd: class}`) and takes class groups; lists are refused
- `unbinned` findings explain that a bin count sets how many bins there are, and the recommended policy pins the edges
- `MetadataConfigMixin` holds only `metadata:`, as `StatsConfigMixin` holds only `stats:`
  - No workflow takes the older `metadata_*` fields
- The text report is 80 columns wide by default (was 90), wrapping long prose, labels and values
- A run that fails only on health warnings exits `3` (was `1`), so CI can tell it from a crash or a mistyped flag (`2`)
- `PipelineConfig.tasks` and `run_tasks` carry evaluator tasks and results as well as workflow ones
- `run_task` returns a `Result`, a workflow's or an evaluator's; `isinstance` narrows it to the type's `<X>Result`
- A failed workflow's report shows `FAILED` and its errors, as a failed evaluator's does
- `WorkflowResult` takes every argument by keyword only
- Containers publish to `harbor.jatic.net/aria/dataeval-flow` instead of `harbor.jatic.net/aria/dataeval`
- The metadata cache key includes the stats policy's `factor_identity()`, so metadata archives recompute on upgrade
- Stats cache keys are unchanged; each dataset keeps one entry that grows with every stats policy asked of it
  - The entry covers band groups and background, records each group's bands and range, and remeasures changed groups
- Console logs include ISO-8601 UTC timestamps and levels; `--log-format plain` gives bare messages
- `main-<variant>` tracks the default branch; `latest-<variant>` retags the newest stable release
- Images are scanned before publication; a HIGH or CRITICAL finding fails the build before anything is pushed
- Internal modules (`cache`, `stats`, `policy`, …) are private; import config types from `dataeval_flow.config`
- `dataeval_flow.workflow` and `dataeval_flow.evaluator` merge into `dataeval_flow.workflows` and `.evaluators`
- Workflow packages are named after their type: `workflows.data_cleaning`, `workflows.drift_monitoring`, …
- Each workflow and evaluator has one `<X>Config` in its own package (was `<X>Parameters` plus `<X>WorkflowConfig`)
- Each type has a real `<X>Result` class for `isinstance` to narrow to; the `is_*_result` guards are gone
- `result.output` replaces `result.data` and an evaluator's `raw`; reading it on a failed run raises
- `Result.type` replaces `Result.name`; the framework, not each workflow, turns exceptions into failed results
- `WorkflowProtocol` is the `Workflow` base class: subclass `Workflow[Config, Result]`, as its docstring shows
- A workflow's `run(config, context)` replaces `execute(context, params)`; `config_type` replaces `params_schema`
- A workflow declares `name` and `description` as class variables, not properties
- `WorkflowParametersBase` → `WorkflowConfig`, the base of every workflow config (was their union)
- `Reportable` → `Finding`; `WorkflowOutputsBase` and `WorkflowReportBase` → `WorkflowRawOutput` and `WorkflowReport`
- `DriftHealthThresholds` and `OODHealthThresholds` → `DriftMonitoringChecks` and `OODDetectionChecks`
- Data-prioritization's `CleaningConfig` → `CleaningSettings`, with `outliers`, `duplicates` and `dup_types`
- A workflow package's modules (`params`, `outputs`, `workflow`, `report`) are private; import from the package
- `list_workflows()` and `list_evaluators()` return classes; `get_*` return the class, not an instance
- Extractor configs import from `dataeval_flow.config.extractors`
  - `ToRGB` imports from `dataeval_flow.config.image_transforms`
- `run_tasks` returns results keyed by task name; `load_config` reads a file or a folder
- `run_task` and `run_tasks` take `data_dir` and `cache_dir` by keyword only; a task named twice runs once
  - `run_task` takes the config first, then a task config or a task's name, as `run_tasks` does
- `PipelineConfig` and its section types import from `dataeval_flow.config`; `load_config` only from `dataeval_flow`
- `WorkflowResult`, `list_workflows` and `get_workflow` import from `dataeval_flow.workflows`, not the top level
- Dataset, source, view, preprocessor and task configs import from `dataeval_flow.config`, not the top level
- `run_task` imports from `dataeval_flow`; the `maite.tasks` entry point is `dataeval_flow:run_tasks`
- `ResultMetadata` imports from `dataeval_flow`, and the config mixins from `dataeval_flow.config`
- A failed result's `to_dict()` is `{kind, metadata, errors}`; a failed workflow's `health.status` is `failed`
- `Finding` drops `report_type` and `data` and rejects unknown fields; `brief` and typed `blocks` hold the evidence
- Data cleaning lists each flagged image and box with every metric that flagged it, then each metric's limits
- The HTML report draws a bar chart's thresholds across its bars, on a labelled scale, instead of in a caption
- The TUI draws each finding's evidence natively: data tables as tables, the rest as text at the window's width
- Data cleaning lists duplicate groups with their items, largest first, and duplicate boxes separately
- OOD detection lists its samples in tables, each named in its own test source
- Pillow (`>=12.2.0`) is a core dependency, used to encode thumbnails
- In the console, an evaluator report cuts lists of over ten values to ten and a count; `-v` and `result.txt` show all
- `scope` passes its embeddings to `Coverage` unscaled, since DataEval rescales them itself
  - Flow's own per-dimension rescale had shifted `dispersion` and the coverage radius
- Evaluators drop their family prefix; a prefixed name fails to load as an unknown evaluator
  - `bias.*` → `balance`, `diversity` and `parity`
  - `quality.*` → `duplicates`, `label-health` and `outliers`
  - `scope.*` → `coverage`, `label-alignment`, `prioritization` and `representation`
  - `shift.*` → `drift-domain-classifier`, `drift-kneighbors`, `drift-mmd`, `drift-univariate`, `drift-wasserstein`
  - `shift.*` → `ood-domain-classifier` and `ood-kneighbors`
- `quality` is a preset: evaluators find outliers, duplicates and label counts, and checks judge them under `checks`
  - It returns a `ChainResult`, whose `steps` and `findings` replace `raw` and `report`
  - `run()` on its config is typed to `ChainResult`
  - Steps are named in kebab case, as ids are: `outliers`, `label-health`, `outliers-by-class`, `duplicates`, `clean`
  - Check steps: `image-outliers`, `target-outliers`, `class-outliers`, `image-duplicates` and `class-imbalance`
  - `health_thresholds` is refused:
    - `image_outliers` → `checks.image-outliers.warning`; `target_outliers` → `checks.target-outliers.warning`
    - `classwise_outliers` → `checks.class-outliers.warning`
    - `exact_duplicates` and `near_duplicates` → `checks.image-duplicates.exact` and `.near`
    - `class_label_imbalance` → `checks.class-imbalance.warning`
  - The `outlier_*` and `duplicate_*` settings move into `outliers:` and `duplicates:` blocks:
    - `outlier_method` and `outlier_threshold` → `outliers.outlier_threshold` (`zscore`, or `[zscore, 3.0]`)
    - `outlier_flags` → `outliers.flags`
    - `outlier_cluster_threshold`, `outlier_cluster_algorithm` and `outlier_n_clusters` drop `outlier_`
    - `duplicate_flags` → `duplicates.flags`; `duplicate_merge_near` → `duplicates.merge_near_duplicates`
    - `duplicate_cluster_sensitivity`, `duplicate_cluster_algorithm` and `duplicate_n_clusters` drop `duplicate_`
- `prioritization` is a preset: `cleaning:` runs `outliers`, `duplicates` and `remove` on the reference and each pool
  - `prioritization` ranks each pool against the reference; `selected` (`select`) keeps the top of each ranking
  - It returns a `ChainResult`, whose `steps` replace `raw` and `report`, and makes no findings
  - The Pruning warning and each pool's info finding are gone
  - `cleaning:` takes quality's `outliers:` and `duplicates:` blocks
  - `duplicate_exact_only: true` → `dup_types: [exact]`
  - `n` and `fraction` → `select.n` and `select.fraction`
- `triage` is a preset: `factor-triage` reads the metadata, and `factor-issues` makes its findings
  - `max_examples` sits under `checks.factor-issues`
  - It returns a `ChainResult`; the issues, stanza and verification are its `factor-triage` step's output
- `shift` is a preset: each detector is a step judged by its family's check, `drift` or `ood`
  - Each test source is tested against the reference on its own; `merge` them in a custom workflow to test them as one
  - It returns a `ChainResult`; a detector that raises fails its step and the task
  - Each drift detector named in `classwise:` also runs by class, adding a by-class finding to the whole-set one
  - `detectors:` takes drift evaluator entries, `classwise:` lists detector names, and `checks.drift` holds the bounds
  - Classwise drift reads labels through DataEval's `Metadata`; without `.metadata` on the dataset, it skips by-class
  - An `ood-union` step groups OOD-flagged images as mutual, partial or unique, and `ood-agreement` judges it
  - Two optional steps explain flagged images by their metadata, now as report sections (they were findings)
  - To upgrade a drift detector:
    - `method: <name>` → `type: drift-<name>` (`domain_classifier` → `drift-domain-classifier`)
    - The univariate `test` → `method`
    - A detector's `classwise: true` → its name in `classwise: [...]`
    - `any_drift_is_warning` and `classwise_any_drift_is_warning` → `checks.drift.warn_on_drift`
    - `chunk_drift_pct_warning` → `chunk_percent`; `consecutive_chunks_warning: N` → `consecutive_chunks: N-1`
    - `chunking.threshold_multiplier: k` → `chunking.threshold: [zscore, k]`
    - An unset `threshold` now uses DataEval's default (legacy used z-score 3); `[zscore, 3.0]` restores legacy
  - To upgrade an OOD detector:
    - `method: kneighbors|domain_classifier` → `type: ood-kneighbors|ood-domain-classifier`
    - Two detectors of one type need distinct `name`s
    - A domain classifier thresholds on `n_std` unless `threshold_perc` is set; `threshold_perc: 95` keeps old verdicts
    - `health_thresholds.ood_pct_warning` and `ood_pct_info` → `checks.ood.warning` and `info`
    - `checks["ood-agreement"]` judges the detectors' agreement
    - `max_ood_insights` → `factor-deviation.max_items`
    - `metadata_insights: false` → `factor-predictors: false` and `factor-deviation: false`
    - `value_range` and the `metadata_*` fields are gone: set the range on the dataset, and name a `metadata:` policy
- A result's JSON writes NaN and infinities as `null`, as strict JSON parsers require
- `drift-kneighbors` on the `uncertainty` extractor refuses `distance_metric: cosine`, which can't rank one number
- The TUI offers workflow, evaluator and extractor types registered after import, plugins included
- `quality`'s `checks` accept `None`, which judges nothing; the finding is still made, as `info`
- Custom workflow and preset results record the encodings their steps read, as `metadata_binning` and `encoding_digest`
  - One record where the steps read one Dataset one way; `per_split`, keyed by address, where they read several
  - `dataeval-flow encoding` reads it
- A report's banner is the title of what ran (`Quality`) or a custom workflow's name, in capitals in the text report
- A report's envelope opens with what ran: `Workflow: clean (quality)` or `Evaluator: dupes (duplicates)`
  - The id alone where the entry is unnamed or named for its id; `Workflow: name (custom workflow)` for custom ones
  - In HTML it is the provenance list's first row, and the page title adds the entry's name, as `Quality — clean`
- A chain's report gives each finding a section with the evidence it judged, headed *From* and the step's title
  - A step already shown under another finding gets a line naming that finding instead
  - The steps no finding shows follow, then a Steps table of every step's title, type, status, reads and note
  - In HTML each finding is a card and the Steps table a folded panel; the text table leaves out the title
- A step's heading is its type's title, with its name beside it where the two differ: `Outliers`, `Duplicates · dupes`
- A chain whose checks ran once per list element groups their findings by the element's key, such as `train` and `val`
- `label-health`'s report lists each class's labels and images in a table
- `remove`'s report says what it kept and what each plan named: "Kept 22 of 24 images. Removed 2 images: …"
- Report configurations leave out unset settings but keep ones written as `null`; evaluator reports skip empty extras
- Image Outliers, Target Outliers, Class Outliers and Class Imbalance findings drop a `description` repeating the brief
- A text table too wide for the report wraps its text cells, with a blank line between rows
- A chain's binning record leaves out `label-health`'s reads, which take labels and no factor
  - A Dataset only it reads, such as a split's part, no longer gets a Metadata Factors block or binning diagnostics
- `splits` is a preset: a `split` or `kfold`, with each train rebalanced where `rebalance:` is set
  - It reports the whole set's labels, balance, diversity and coverage, and each part's labels, stratification, coverage
  - As a step, it exposes `train` (rebalanced where set), `val` and `test`, as lists keyed by fold with `folds` of 2+
  - It returns a `ChainResult`; each part's indices are in `result.steps["split"].details["indices"]`
  - Each fold's rebalanced train is in `result.steps["rebalanced"].elements["<k>"].details["indices"]`
  - Where rebalancing kept a train unchanged, `details` is `None` and `split`'s indices apply
  - Findings: Class Imbalance, Class Stratification per fold, and Uncovered Items under `naive` coverage
  - Balance and diversity are report sections; split sizes are in the `split` step's section and the `lineage`
  - Only object-detection Datasets can be exported, so classification parts can be read but not yet exported
  - `checks` is keyed by check type: `class-imbalance`, `class-stratification` and `uncovered-items`
  - To upgrade:
    - `num_folds` → `folds`; `rebalance_method` → `rebalance`
    - `coverage_percent` and `num_observations` → `coverage.percent` and `coverage.num_observations`
    - `val_frac` with `folds` of 2 or more is refused (each fold's val is its 1/k); unset, it is 0.1 with `folds: 1`
    - `metadata`'s `split_sizes` and `stratified`, and `output.raw`, are in `result.steps` and `lineage`
- `scope` is a preset: a `crops` step (`wrap`) crops detection data to one item per box and passes other data through
  - `coverage` and `completeness` embed the crops; `class-coverage` and `dimensional-completeness` judge them
  - `uncovered-items` judges `naive` coverage; under `adaptive` the uncovered rate isn't judged
  - `class-imbalance` judges `label-health`, `factor-coverage-gaps` judges `factor-gaps`
  - `class-shortfall` judges `representation`
  - `factor-summary`, `balance` and `diversity` read the metadata, and `factor-gaps` reads balance
  - Without an extractor, the embedding steps are skipped and their findings say "not assessed"
  - It returns a `ChainResult`; each step's output is in `result.steps`, such as `result.steps["coverage"].output`
  - Metadata Distribution, balance and diversity are report sections; Metadata Distribution was a finding
  - Under `naive` coverage, legacy's Embedding Coverage finding becomes Class Coverage and Uncovered Items
  - Naive coverage that overflows is skipped, where legacy re-ran it as adaptive
  - On detection data, uncovered items index the `crops` Dataset (one item per box), not each one's image and box
  - It no longer judges an ontology; a `taxonomy` entry on the same source does, and carries the join key
  - `checks` is keyed by check type: `class-imbalance`, `factor-coverage-gaps` and `class-coverage`
  - The `uncovered-items` and `dimensional-completeness` checks take keys there too
  - Every legacy field is refused by name, with its replacement. To upgrade:
    - `coverage_method`, `coverage_percent` and `num_observations` → `coverage.method`, `.percent`, `.num_observations`
    - `min_class_samples`, `isotropy_min_samples` and `near_duplicate_factor` → the same names under `coverage.`
    - `crop_padding` and `crop_min_size` → `wrap.params.padding` and `wrap.params.min_size`
    - `run_completeness` → `completeness`
    - `diversity_method` is refused; `diversity.method` picks the method, and diversity can no longer be skipped
    - `balance` is refused: balance always runs, as a report section
    - `run_gap_analysis` is refused; `factor-gaps: false` replaces `run_gap_analysis: false`
    - `gap_mi_threshold` and `gap_min_representation` → `factor-gaps.mi_threshold` and `.min_representation`
    - `ontology` and `ontology_label_pattern` → a `taxonomy` entry's `ontology` and `ontology-validation.label_pattern`
    - `ontology_expected` → `representation.expected`, or `taxonomy`'s `representation.expected` with an ontology
    - The `metadata_*` binning fields are refused: name a policy under `metadata:`
    - `value_range` is refused (set it on the dataset), and so is `stats`, since no step of scope reads statistics
    - `health_thresholds` → `checks`:
      - `class_imbalance_ratio` → `class-imbalance.warning`; legacy's fixed band at 2.0 → `class-imbalance.info`
      - `gap_count: N` → `factor-coverage-gaps.warning: N-1` (legacy warned at N; this warns past the bound)
      - `min_dispersion` and `min_isotropy` → `class-coverage.dispersion` and `.isotropy`
      - `max_near_duplicate_fraction` → `class-coverage.near_duplicates`
      - `uncovered_rate` → `uncovered-items.warning`
      - `completeness_score` → `dimensional-completeness.warning`; legacy's band at 0.8 → `.info`
      - An unset `info` follows `warning` as legacy's band did; two written bounds that cross are refused
    - The label finding is "Class Imbalance" (was "Label Distribution", or "Label/Directory_Name Distribution")
    - `health_thresholds.leaf_coverage` → `taxonomy`'s `checks.leaf-coverage.coverage`
    - `dark_branch_count` → `taxonomy`'s `checks.leaf-coverage.empty_branches`
    - `unmatched_class_count` → `taxonomy`'s `checks.label-conformance.warning`
    - `output.raw` and `metadata.has_extractor` are gone; read the steps' outputs instead:
      - `coverage`, `completeness` and `metadata_gaps` → the `coverage`, `completeness` and `factor-gaps` outputs
      - `label_distribution` → `label-health`'s output; `metadata_distribution` → `factor-summary`'s
      - A skipped step's reason is its `reason`
      - `coverage.dropped_detections` → `result.steps["crops"].details["dropped"]`
- `label-health` lists every declared class, at 0 where it has no labels, and unlabelled items as `empty_image_indices`
  - A declared class with no labels now shows at 0 in `quality`'s and `splits`'s label and stratification tables
  - It also makes their Class Imbalance finding warn
- `class-imbalance` makes its finding on any Dataset with classes, declared or observed
  - An unlabelled Dataset that declares classes now warns in quality and splits (it made no finding before)
  - Its ratio is over the classes with labels, and each class without labels is named
  - `info` is a ratio at or under which the finding is `ok`; evidence adds each class's share and unlabelled images

### Fixed

- A dataset whose class names change but whose items don't gets its own cache entry, so old names aren't served
  - A dataset that declares class names is computed once more, under its new key
- The config builder keeps an explicit `null`, such as `checks.leakage.near: null`, instead of restoring the default
- Every step of a task reads a source through one draw of its view, so random views give all steps the same order
  - A result's `dataset` and `sources`, and the report's thumbnails, use that same draw
- An exactly declared `continuous_factor_bins` name takes precedence over a bare statistic's expansion, in any order
- `triage` no longer reports a factor as unpinned when the policy's descriptor pins it
  - It no longer suggests a bin count the policy refuses as named by both `encoding` and `continuous_factor_bins`
- Data prioritization ranks a labeled pool under `policy: class_balanced` instead of raising "class_labels not provided"
- Data prioritization succeeds when cleaning empties a pool, ranking it as empty, instead of failing the task
- Classwise drift names each class from the datasets' `index2label` instead of showing its bare index
- The config builder keeps `result:`, `logging:`, `seed:` and `deterministic:` when it saves, and the TUI runs with them
- The TUI shows a name with brackets in it as written; `[/x]` no longer crashes it
- Data prioritization's `sources` and data splitting's `dataset` are the views they ran on, not fresh draws
- Classwise drift prints small p-values in full (`0.0003`, not `0.00`), and `results.json` keeps them unrounded
- A chunked drift finding with `chunk_percent: 0` no longer warns when no chunk drifted
- `run_tasks`, the CLI and the TUI share one BoVW fit per task; its embeddings and clusters are cached only with `seed`
- Data-cleaning and parameter-sweep key clusters by their extractor; existing stateless cleaning clusters miss once
- Data-cleaning's cluster-mode duplicate merge passes `merge_near_duplicates`, matching the `duplicates` evaluator
- Dataset elements without `__repr__` hash by type and contents rather than memory address, so caches are reused
- Source views are part of cache keys, so editing a view invalidates cached embeddings, metadata and statistics
- Relative `ontology:` paths resolve against the run's data root rather than the process root
- Outlier detection is limited to the configured `outlier_flags`, so a shared cache can't widen its column checks
- Duplicate detection is limited to the configured `duplicate_flags`, so a shared cache can't widen it
- Injected metadata factors are limited to the configured `intrinsic_factors` instead of what the cache holds
- Band-group statistics are recorded in `injected_factors` instead of being mistaken for dataset columns
- Cross-split label parity is limited to shared classes, avoiding chi-square errors on gapped label spaces
- Single-split classes are left out of the parity test and reported in `label_overlap` instead
- Cross-split duplicate detection is limited to common columns, avoiding crashes on divergent stat caches
- The container entrypoint honors `DATAEVAL_OUTPUT` and `DATAEVAL_CACHE` instead of hardcoded paths
- The container image creates `/cache/.not_mounted`, so an unmounted `/cache` is no longer exported as `DATAEVAL_CACHE`
- Object-detection datasets are recognized consistently on Python 3.10 and 3.11, avoiding misclassification or crashes
- The GitHub release body carries the changelog section instead of falling back to `Release vX.Y.Z`
- Container images no longer ship the standalone interpreter's bundled `pip`, which nothing used
- A task naming one source twice is refused at load; the repeat used to be dropped, leaving the task a source short
- A key no config section defines is refused at load instead of dropped
  - A misspelled top-level key is named with the section it most resembles
- The HTML report renders inline code in a table cell as in prose, instead of showing the backticks
- A chain refused before any step ran says why in its report, instead of only `Steps: 0 ran`
- A report's health line and HTML badge say `failed` when a required step failed, instead of claiming checks passed
- A "not assessed" finding's description ends in one full stop when its cause already ends in one

### Removed

- The per-key migration messages for keys earlier versions took, and the `type: data-analysis` message
  - They covered `audit`'s data-analysis fields, `scope`'s legacy fields and `health_thresholds`, and `method:`
  - They also covered `checks.class-imbalance` on `quality` and `splits`
  - Each of these keys now fails as an unknown key or workflow
- Workflow types that run their own code: every workflow type is now a preset expanding to a chain of steps
  - Gone: `Workflow.run`, `WorkflowOutput`, `WorkflowRawOutput`, `WorkflowReport`, `WorkflowResult`, `workflow_result`
  - A workflow type mixes in `Preset` and declares `slots` and `chain`; otherwise its class raises `TypeError`
  - `Preset` and `PresetChain` are exported from `dataeval_flow.workflows`
  - Every workflow returns a `ChainResult`
  - A plugin with an algorithm of its own registers it as an evaluator and chains it in its preset
- `splits`'s coverage: its `coverage` steps and `uncovered-items` checks, `coverage:` and `checks.uncovered-items`
  - `extractor:` on a splits task or step goes too, with `DataSplittingCoverageSettings`
  - Judge the parts with an `audit` step after the split (see Check a set of splits)
- prioritization's `cleaning:` and `stats:`, with `CleaningSettings`
  - Run the preset in a custom workflow after a `quality` step on the reference and one on the pools
  - To keep near duplicates, run it after `outliers`, `duplicates` and `remove` steps instead (see the Preset Catalog)
- The `selections:` and `selection:` aliases for `views:` and `view:`, deprecated since v0.2.0
  - Also a `views:` entry's `steps:` alias for `operations:`
- `parameter-sweep`, with `ParameterSweepConfig`, `ParameterSweepResult` and `ParameterSweepWorkflow`
  - Write a quality entry with a `matrix:` on its task (see Sweep settings with a matrix)
- `data-analysis` and its classes
  - `DataAnalysisConfig`, `DataAnalysisHealthThresholds`, `DataAnalysisResult` and `DataAnalysisWorkflow`
  - Write an `audit` entry; `type: data-analysis` fails to load, naming `audit` and where each setting went
  - `outlier_method` and `outlier_threshold` → `outliers.outlier_threshold`: the method, or `[method, threshold]`
  - `outlier_flags` → `outliers.flags`
  - `diversity_method` → `diversity.method`; `divergence_method` → `divergence.method`
  - `diversity_method: null` (skip diversity) has no replacement: audit always runs diversity
  - `balance` is refused: balance always runs, and is skipped on metadata with no factors
  - `include_image_stats` → the metadata policy's `intrinsic_factors`
  - `value_range` is refused: set it on the dataset
  - The `metadata_*` binning fields are refused: name a policy under `metadata:`
  - `health_thresholds` → `checks:`:
    - `image_outliers` → `checks.image-outliers.warning`
    - `exact_duplicates` and `near_duplicates` → `checks.image-duplicates.exact` and `.near`
    - `class_label_imbalance` → `checks.class-imbalance.warning`
    - `distribution_shift` → `checks.embedding-divergence.warning`
  - `DataAnalysisResult`'s raw fields are steps in an audit's `result.steps`:
    - Each split's `image_quality`, `redundancy` and `label_health` → its `outliers`, `duplicates`, `label-health` steps
    - Those steps are named per split, as `outliers-train` and `outliers-evals`
    - Train's `bias` → `factor-summary`, `balance` and `diversity`
    - `cross_split`'s duplicate leakage → `duplicates-cross` and `duplicates-pairs`
    - `cross_split`'s label comparisons → `class-sufficiency`, `untrained-classes` and `class-stratification`
    - `cross_split`'s divergence → `divergence`
  - Finding titles:
    - Image Quality → Image Outliers; Redundancy → Image Duplicates; Label Balance → Class Imbalance
    - Bias → Shortcut Risk, with diversity as evidence
    - Label Overlap → Untrained Classes and Class Sufficiency; Label Parity → Class Stratification
    - Leakage keeps its title; Distribution Shift → the `embedding-divergence` check's Embedding Divergence
  - Gone with it: chi-square label parity, divergence between evaluation splits, and bias judged per split
  - Also gone: the warning on low diversity, and Label Balance's warning on unlabelled images
- The settings only `data-analysis` still took, which no workflow takes now:
  - `metadata_auto_bin_method`, `metadata_exclude`, `metadata_continuous_factor_bins`, `metadata_factor_source`
  - Name a `metadata:` policy instead, with `auto_bin_method`, `exclude`, `continuous_factor_bins`, `factor_source`
  - `value_range` on a workflow entry: set it on the dataset
  - `include_image_stats`: set the metadata policy's `intrinsic_factors`
  - `attach_binning`, and the outlier report's `limits_sentence` and `warn_if_unrecorded`, which had no callers
- The `torch` and `uncertainty` extractors' `device`, and shift MMD's; Flow chooses the device for every tool
- Poetry packaging support; install with uv, pip or conda
- Floating `<variant>` and `<major>.<minor>-<variant>` image tags; pull `latest-<variant>` or pin `<version>-<variant>`
- Python 3.10 support; the minimum is now 3.11
- `dataeval_flow.config.schemas`; most of its types import from `dataeval_flow.config`
- Deprecated `SelectionConfig`, `SelectionStep`, `build_selection` and `DatasetContext(selection_steps=)`
- Typed task configs (`DataCleaningTaskConfig`, `EvaluatorTaskConfig`, …); use `TaskConfig`
- `load_config_folder` and `export_params_schema`
- Public per-type output, report and metadata models (`DataCleaningOutputs`, …); narrow to `<X>Result` instead
- The nested output models and TypedDicts those models held (`CoverageAssessment`, `OutlierIssuesDict`, …)
- `select_tasks`; `run_tasks` keys its results by task name, so nothing needs pairing
- A workflow's `output_schema`; its `<X>Result` type argument names the output
- The `DriftDetectorConfig` and `OODDetectorConfig` unions; annotate with the detector classes
- The `AutoBinMethod` and `FactorSource` aliases; their fields take the same strings
- `DataCleaningResult`, with its metadata's `evaluators`, `flagged_indices`, `clean_indices` and `removed_count`
  - A quality result is a `ChainResult`
- `DataPrioritizationResult`, `DataPrioritizationHealthThresholds` and their output types
  - `CleaningSummaryDict` and `PerDatasetPrioritizationDict`
  - A prioritization result is a `ChainResult`
- `health_thresholds` and `value_range` on `prioritization`, which refuses them; declare the range on the dataset
- `value_range` and the `metadata_*` binning fields on `quality`, which refuses them
  - Declare the range on the dataset, and the binning in a `metadata:` policy
- `mode`, from every workflow config and result's metadata; a config that still writes it fails to load, naming it
- Data-prioritization's `per_source_clean_indices` and `per_source_prioritized_indices`
- The "Preparatory Mode" findings that quality made
- `MetadataTriageResult`, with its metadata's `blocking` and `verified`; a triage result is a `ChainResult`
- The `metadata_*` binning fields on `triage`, which refuses them; declare the binning in a `metadata:` policy
- `update_strategy` on `shift`, which was never applied and is now refused
- `DriftDetectorUnivariate`, `DriftDetectorMMD`, `DriftDetectorKNeighbors` and `DriftDetectorDomainClassifier`
  - Also `ChunkingConfig`, `UpdateStrategyConfig` and `DriftMonitoringHealthThresholds`
  - A detector is a drift evaluator config, and its `chunking:` a `ChunkedDriftConfig`
- `DriftMonitoringResult` and its parts; a shift result is a `ChainResult`
- `OODDetectionResult`, `OODDetectorKNeighbors`, `OODDetectorDomainClassifier` and `OODDetectionHealthThresholds`
  - shift returns a `ChainResult`, and its detectors are OOD evaluator entries
- `DataSplittingResult` and its output and metadata types; a splits result is a `ChainResult`
- `DataCoverageResult`, its output and metadata types, and `DataCoverageHealthThresholds`
  - A scope result is a `ChainResult`, and its thresholds are `DataCoverageChecks`

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
