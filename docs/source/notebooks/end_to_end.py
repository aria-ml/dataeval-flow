# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dataeval-flow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Run a full evaluation pipeline end to end
#
# Load a dataset, configure three evaluation workflows in one YAML file, run the
# pipeline in a single call, inspect results, and execute the identical pipeline
# inside a container.

# %% [markdown]
# **Target audience**: You are a model developer, data scientist, or T&E engineer
# who wants to run a complete evaluation pipeline from configuration to containerized
# execution.
#
# **Workflow role**: This guide demonstrates an end-to-end evaluation pipeline:
# dataset staging, configuration definition, pipeline execution via `run_tasks()`
# or CLI, result inspection, and containerized execution.

# %% [markdown]
# ## What you will do
#
# 1. **Load a dataset**: Fetch SkySeaLand and export it in a layout supported by DataEval Flow.
# 2. **Analyze the metadata**: Run `triage` to decide which factors to drop and how to bin the rest.
# 3. **Write the configuration**: Define datasets, sources, extractor, workflows, and tasks in `end_to_end.yaml`.
# 4. **Run the pipeline**: Execute all tasks in a single call with `run_tasks()`.
# 5. **Display results**: Inspect text reports, structured findings, and flagged images.
# 6. **Export results**: Generate machine-readable result envelopes for downstream tools.
# 7. **Run in Docker**: Execute the identical pipeline inside a container without Python code.

# %% [markdown]
# The three tasks demonstrate different evaluation workflows:
#
# | Task | Workflow | Sources | Extractor | Answers |
# | --- | --- | --- | --- | --- |
# | `clean_train` | `quality` | train | BoVW | Are training samples free of severe outliers and duplicates? |
# | `audit_splits` | `audit` | train + test | BoVW | Is the data ready to train on, with no leakage between train and test? |
# | `split_train` | `splits` | train | (none) | How should you partition training data into cross-validation folds? |

# %% [markdown]
# ## What you will learn
#
# - How to run `triage` to decide which factors to exclude and how to bin the rest.
# - How to declare multiple evaluation workflows in a single YAML configuration file.
# - How to run end-to-end pipelines using `run_tasks()` in Python.
# - How to inspect text reports and query structured findings programmatically.
# - How to export auditable result envelopes for CI/CD gates.
# - How to execute pipelines inside Docker without Python code.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `dataeval-plots` to visualize flagged images.
# - Install `maite-datasets[datamaite]` to download and export SkySeaLand.
# - Ensure network access for the initial download; subsequent runs read from disk.
# - Install Docker if you plan to run the containerized execution section.

# %% [markdown]
# ## Step 1: Load the dataset
#
# [SkySeaLand](https://www.kaggle.com/datasets/mdzahidhasanriad/skysealand) is an overhead
# object-detection dataset containing 1,307 frames from four sites, with 19,102 annotations
# across `airplane`, `boat`, `car`, and `ship`. The dataset provides `train` (1,048 frames),
# `val` (132), and `test` (127) splits. In this pipeline, you will evaluate `train` and `test`.

# %% [markdown]
# You can use `maite-datasets` with `as_datamaite=True` to export the dataset into standard
# COCO format for DataEval Flow. Each export directory is named by dataset and `image_set`
# to keep splits isolated on disk.

# %% tags=["remove_output"]
from pathlib import Path

from maite_datasets.object_detection import SkySeaLand

data_root = Path("./data")

# Download once into ./data/skysealand; exports reuse the cached download.
SkySeaLand(root=data_root, image_set="base", download=True)

split_paths = {name: data_root / f"skysealand_datamaite_{name}" for name in ("train", "test")}
for image_set in split_paths:
    SkySeaLand(root=data_root, image_set=image_set, as_datamaite=True)

print("\n".join(f"{name}: {path}" for name, path in split_paths.items()))

# %% [markdown]
# ### Confirm it loads
#
# Before writing configuration, verify that DataEval Flow can read the exported files.
# `load_dataset()` uses the same loader as the pipeline runtime.

# %%
from dataeval_flow import load_dataset

train_ds = load_dataset(split_paths["train"], dataset_format="coco")
image, target, _ = train_ds[0]

print(f"Images:      {len(train_ds)}")
print(f"Image shape: {image.shape}")
print(f"Boxes:       {len(target.boxes)}")
print(f"Classes:     {train_ds.metadata['index2label']}")

# %% [markdown]
# ## Step 2: Analyze the metadata
#
# Before you commit to a `metadata:` policy, run the `triage` preset. It flags factors
# that carry no information and factors whose bin counts would otherwise be derived silently
# (and unstably) at run time.
#
# ```{seealso}
# This section only covers the operational checkpoint: run triage, act on its findings, move on.
# For a full walkthrough of reading triage findings, distribution charts, and remediation
# policies, see {doc}`Triage a dataset's metadata <metadata_triage>`.
# ```

# %% tags=["remove_output"]
from dataeval_flow import run_task
from dataeval_flow.config import (
    CocoDatasetConfig,
    MetadataPolicyConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.workflows.triage import TriageConfig

triage_config = PipelineConfig(
    metadata=[MetadataPolicyConfig(name="skysealand_factors", intrinsic_factors=["visual", "pixel"])],
    datasets=[CocoDatasetConfig(name="skysealand_train", path=str(split_paths["train"]))],
    views=[
        ViewConfig(
            name="sample300",
            operations=[
                ViewOperation(type="Shuffle", params={"seed": 0}),
                ViewOperation(type="Limit", params={"size": 300}),
            ],
        )
    ],
    sources=[SourceConfig(name="train_src", dataset="skysealand_train", view="sample300")],
    workflows=[TriageConfig(name="triage", metadata="skysealand_factors")],
    tasks=[TaskConfig(name="triage_train", workflow="triage", sources="train_src")],
)

triage_result = run_task(triage_config, "triage_train", data_dir=Path("."), cache_dir=Path("./cache"))

# %%
triage = triage_result.steps["factor-triage"].output.data()
findings = triage["findings"]
degenerate = sorted({f.factor for f in findings if f.category == "degenerate"})
unbinned = sorted({f.factor for f in findings if f.category == "unbinned"})

print(f"Factors:              {triage['factor_count']}")
print(f"Findings:             {len(findings)} ({triage['counts'].get('blocking', 0)} blocking)")
print(f"Degenerate (exclude): {degenerate}")
print(f"Need explicit bins:   {len(unbinned)} factors")

# %% [markdown]
# `triage` flags `label_file_exists`, `instance_missing`, and `unit_missing` as
# degenerate: every frame holds the same value, so the factor separates nothing. It also flags
# `instance_zeros` for a sentinel-value remap before it can be binned — that judgment call is
# exactly what the {doc}`metadata triage tutorial <metadata_triage>` covers, so this pipeline excludes
# it instead. The remaining continuous pixel and visual factors need explicit bin counts. Without
# them, `audit` derives the counts from whatever sample happens to run, and warns that it did.
#
# `end_to_end.yaml`'s `metadata:` section already applies these findings: the four degenerate
# and unresolved factors are excluded, and every remaining continuous factor has a pinned
# `continuous_factor_bins` count.

# %% [markdown]
# ## Step 3: Set up the configuration
#
# You can define pipeline execution in a single YAML file using modular sections:
#
# | Section | What it declares |
# | --- | --- |
# | `datasets` | Where data lives and how to read it |
# | `views` | How to narrow a dataset (limit, index range, class filter) |
# | `sources` | A dataset plus an optional view: the input consumed by tasks |
# | `extractors` | How embeddings are produced |
# | `workflows` | Named parameter sets for evaluations |
# | `tasks` | Binds a workflow to sources (and an extractor when needed) |
#
# You can inspect the configuration file `end_to_end.yaml`:

# %%
config_path = Path("end_to_end.yaml")
print(config_path.read_text())

# %% [markdown]
# ### Load and validate it
#
# You can call `load_config()` to parse the YAML file and validate it against the
# pipeline schema. Validation catches invalid parameters, unknown workflow types,
# and missing references before execution.

# %%
from dataeval_flow import load_config

config = load_config(config_path)

print(f"Datasets:   {[d.name for d in config.datasets]}")
print(f"Sources:    {[s.name for s in config.sources]}")
print(f"Extractors: {[e.name for e in config.extractors]}")
print(f"Workflows:  {[(w.name, w.type) for w in config.workflows]}")
print(f"Tasks:      {[(t.name, t.workflow, t.sources) for t in config.tasks]}")

# %% [markdown]
# ## Step 4: Run the pipeline
#
# You can execute all enabled tasks using `run_tasks()`. It returns each task's result keyed by
# task name, in execution order. Two arguments matter here:
#
# - `data_dir`: Base directory for resolving relative paths in configuration.
# - `cache_dir`: Directory for caching embeddings, image statistics, and metadata
#   across tasks and runs.
#
# Tasks sharing datasets reuse intermediate calculations stored in the cache.

# %% tags=["remove_output"]
from dataeval_flow import run_tasks

results = run_tasks(config, data_dir=Path("."), cache_dir=Path("./cache"))

# %%
for name, result in results.items():
    status = "OK " if result.success else "FAIL"
    warnings = result.warning_count
    elapsed = result.metadata.execution_time_s or 0.0
    print(f"[{status}] {name:<16} {result.type:<16} {elapsed:>6.1f}s  {warnings} warning(s)")
    for error in result.errors:
        print(f"         {error}")

# %% tags=["remove_cell"]
assert all(r.success for r in results.values()), [r.errors for r in results.values() if not r.success]

# %%
clean_result, audit_result, split_result = results["clean_train"], results["audit_splits"], results["split_train"]

# %% [markdown]
# ## Step 5: Display the results
#
# Each task result exposes three primary interfaces:
#
# - `result.report()`: Formatted text summary.
# - `result.findings`: Structured finding objects.
# - The numbers behind the findings. All three workflows run as chains of steps, so their results hold each step's
#   output in `result.steps`, by step name. The split's part indices are in `result.steps["split"].details["indices"]`,
#   and the audit's verdict is `result.verdict`.
#
# You can call `report(detailed=False)` for high-level summaries, or `report(detailed=True)`
# for per-finding breakdowns.

# %% [markdown]
# ### 5a. Summary reports

# %%
for result in results.values():
    print(result.report(detailed=False))

# %% [markdown]
# ### 5b. Structured findings
#
# Each finding includes a `title`, a `severity` (`ok`, `info`, or `warning`), a short `brief`, a
# `description` (`None` where the brief says it all), and its evidence as report `blocks`. You can
# query these programmatically in automated CI/CD gates.

# %%
for result in results.values():
    print(f"\n{result.type}")
    for finding in result.findings:
        marker = {"warning": "[!!]", "ok": "[ok]"}.get(finding.severity, "[..]")
        print(f"  {marker} {finding.title:<34} {finding.brief or ''}")

# %% [markdown]
# ### 5c. Data cleaning: Inspect flagged images
#
# The cleaning report summarizes the count of flagged images. You can retrieve specific
# sample indices from the `outliers` and `duplicates` steps, whose outputs are DataEval's own, and slice
# `result.sources` directly without reloading data. It holds each source the task read, after its view.

# %%
issues = clean_result.steps["outliers"].output.data()
image_issues = issues.filter(issues["target_index"].is_null())  # the rest flag single boxes
outlier_indices = sorted(set(image_issues["item_index"].to_list()))
near_groups = clean_result.steps["duplicates"].output.items.near

print(f"Image outliers:        {len(outlier_indices)} images, {image_issues.height} flags")
print(f"Near-duplicate groups: {len(near_groups)}")

# %%
from dataeval_plots import plot

if outlier_indices:
    _ = plot(
        clean_result.sources["train_src"],
        indices=outlier_indices[:6],
        images_per_row=3,
        figsize=(12, 8),
        show_labels=True,
    )

# %% [markdown]
# ### 5d. Audit: The verdict
#
# The audit task judges train and test together. Its verdict says whether the data is ready to
# train on: a warning from a blocking check, such as leakage between the splits, makes it not
# ready, and any other warning, or a check it could not assess, is a caveat. The task names the
# BoVW extractor, so every check is assessed. Train and test share no image, and test has no
# class train lacks, so nothing blocks. The verdict is ready with caveats: the outlier, metadata,
# coverage and shortcut warnings.

# %%
verdict = audit_result.verdict
assert verdict is not None

print(verdict.line())
for item in verdict.blocking + verdict.warnings:
    print(f"  [!!] {item.step:<32} {item.brief}")
for item in verdict.not_assessed:
    print(f"  [..] {item.step:<32} not assessed: {item.reason}")

# %% [markdown]
# ### 5e. Dataset splitting: Partition indices
#
# The splitting task generates explicit index lists for each part, in the split step's
# details. With `folds: 1`, as here, `train`, `val` and `test` are each one list; with
# more folds, `train` and `val` are keyed by fold. You can use these indices to
# construct PyTorch `Subset` or `DataLoader` instances.

# %%
indices = split_result.steps["split"].details["indices"]

print(f"Split sizes: { {part: len(idx) for part, idx in indices.items()} }")
print(f"Train indices (first 10): {indices['train'][:10]}")
print(f"Val   indices (first 10): {indices['val'][:10]}")
print(f"Test  indices (first 10): {indices['test'][:10]}")

# %% [markdown]
# ## Step 6: Export the results
#
# You can call `export()` to write the result envelope to disk. Result envelopes
# contain findings alongside execution metadata: timestamps, tool versions,
# dataset identifiers, and fully resolved configurations. `to_html()` renders the
# same report as one self-contained page that you can open in a browser, attach
# to a ticket, or print to PDF.

# %%
output_dir = Path("./output/end_to_end")

for name, result in results.items():
    written = result.export(output_dir / f"{name}.json")
    page = output_dir / f"{name}.html"
    page.write_text(result.to_html(), encoding="utf-8")
    print(f"{written}  ({written.stat().st_size:,} bytes)")
    print(f"{page}  ({page.stat().st_size:,} bytes)")

# %%
import json

envelope = json.loads((output_dir / "split_train.json").read_text())

print(f"Top-level keys: {list(envelope)}")
print(f"Tool:           {envelope['metadata']['tool']} {envelope['metadata']['tool_version']}")
print(f"Timestamp:      {envelope['metadata']['timestamp']}")
print(f"Sources:        {envelope['metadata']['source_descriptions']}")

# %% [markdown]
# ## Step 7: Run the same pipeline in Docker
#
# You can execute the same YAML pipeline in Docker without Python dependencies.
# The container consumes `end_to_end.yaml`, reads datasets from `/dataeval`, and
# writes result envelopes to `/output`.
#
# The container uses three mount paths:
#
# | Mount | Mode | Holds |
# | --- | --- | --- |
# | `/dataeval` | read-only | Data root: datasets, models, configuration files |
# | `/output` | read-write | Reports and result envelopes |
# | `/cache` | read-write | Caches embeddings and statistics across runs |

# %% [markdown]
# ### Lay out the workspace
#
# Create a root directory containing `end_to_end.yaml` and the exported dataset splits:
#
# ```bash
# mkdir -p dataeval-run/data dataeval-run/output dataeval-run/cache
# cp -r docs/source/notebooks/data/skysealand_datamaite_train dataeval-run/data/
# cp -r docs/source/notebooks/data/skysealand_datamaite_test dataeval-run/data/
# cp docs/source/notebooks/end_to_end.yaml dataeval-run/
#
# tree -L 3 dataeval-run
# ```

# %% [markdown]
# ### Get the image
#
# Pull the pre-built container image or build it locally:
#
# ```bash
# # Pre-built (cpu / cu126 / cu130)
# docker pull harbor.jatic.net/aria/dataeval-flow:latest-cpu
#
# # Optional: verify the signature
# cosign verify --key docker/cosign.pub harbor.jatic.net/aria/dataeval-flow:latest-cpu
#
# # Or build locally from a checkout
# docker build -f docker/Dockerfile.cpu -t dataeval-flow:cpu .
# ```

# %% [markdown]
# ### Run it
#
# Use `--user` so `/output` and `/cache` remain writable by your host account.
# The console prints each report's short form; `-v` prints the full reports instead.
#
# ```bash
# cd dataeval-run
#
# docker run --rm \
#     --user "$(id -u):$(id -g)" \
#     --mount type=bind,source="$PWD",target=/dataeval,readonly \
#     --mount type=bind,source="$PWD/output",target=/output \
#     --mount type=bind,source="$PWD/cache",target=/cache \
#     harbor.jatic.net/aria/dataeval-flow:latest-cpu \
#     python -m dataeval_flow --config end_to_end.yaml -v
# ```
#
# On GPU hosts, include `--gpus all` and select a CUDA image:
#
# ```bash
# docker run --rm --gpus all \
#     --user "$(id -u):$(id -g)" \
#     --mount type=bind,source="$PWD",target=/dataeval,readonly \
#     --mount type=bind,source="$PWD/output",target=/output \
#     --mount type=bind,source="$PWD/cache",target=/cache \
#     harbor.jatic.net/aria/dataeval-flow:latest-cu130 \
#     python -m dataeval_flow --config end_to_end.yaml -v
# ```
#
# :::{note}
# Specify `--config` explicitly when running with multiple YAML files in your data root.
# Without `--config`, DataEval Flow merges all configuration files in the root directory.
# :::

# %% [markdown]
# ### Read the output
#
# The container produces `result.json`, `result.txt`, and run logs in `/output`:
#
# ```bash
# find output -type f
# # output/result.log
# # output/results/result.json
# # output/results/result.txt
# ```
#
# You can query the generated results using `jq`:
#
# ```bash
# # Inspect executed tasks
# jq -r 'keys[]' output/results/result.json
#
# # Print findings and severities: each workflow lists its findings at the top of its entry
# jq -r 'to_entries[] | .key as $task | .value.findings[]
#        | "\($task)\t\(.severity)\t\(.title)"' output/results/result.json
#
# # Gate CI/CD pipelines on warnings
# jq -e '[.[].health.warnings] | add == 0' \
#     output/results/result.json > /dev/null \
#     && echo "PASS: no warnings" || echo "FAIL: warnings present"
#
# # Inspect split partition sizes
# jq -c '.split_train.steps.split.details.indices | map_values(length)' output/results/result.json
# ```

# %% [markdown]
# ### Running offline
#
# Once container images and datasets are staged locally, you can run entirely offline:
#
# ```bash
# docker run --rm --network none \
#     --user "$(id -u):$(id -g)" \
#     -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
#     --mount type=bind,source="$PWD",target=/dataeval,readonly \
#     --mount type=bind,source="$PWD/output",target=/output \
#     --mount type=bind,source="$PWD/cache",target=/cache \
#     harbor.jatic.net/aria/dataeval-flow:latest-cpu \
#     python -m dataeval_flow --config end_to_end.yaml -v
# ```

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Stage dataset splits in isolated disk directories.
# - Run `triage` to decide which factors to exclude and how to bin the rest.
# - Define multi-workflow pipelines in a single YAML configuration file.
# - Execute pipelines using `run_tasks()` and the container CLI.
# - Inspect formatted reports and extract structured findings.
# - Export auditable result envelopes for CI/CD gates and downstream tools.
# - Execute the complete evaluation pipeline inside Docker.

# %% [markdown]
# ## Next steps
#
# - {doc}`Triage a dataset's metadata <metadata_triage>`: Reading triage findings,
#   distribution charts, and remediation policies.
# - {doc}`Clean a dataset <data_cleaning>`: Outlier and duplicate detection.
# - {doc}`Audit a set of splits before training <audit>`: A verdict on train and
#   evaluation splits, and a record of what was audited.
# - [Split a dataset](dataset_splitting): Stratification, cross-validation folds, and group-aware splitting.

# %% [markdown]
# ## Related guides
#
# - **How-to**: [Configure metadata binning](../how_to/configure_metadata_binning.md) explains declaring cuts and vocabularies.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) covers Docker execution options and flags.
# - **How-to**: [Reuse results with the disk cache](../how_to/reuse_results_with_cache.md) explains caching behaviors and invalidation.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) details result envelope structure and querying.
