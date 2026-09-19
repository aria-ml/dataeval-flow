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
# 2. **Write the configuration**: Define datasets, sources, extractor, workflows, and tasks in `end_to_end.yaml`.
# 3. **Run the pipeline**: Execute all tasks in a single call with `run_tasks()`.
# 4. **Display results**: Inspect text reports, structured findings, and flagged images.
# 5. **Export results**: Generate machine-readable result envelopes for downstream tools.
# 6. **Run in Docker**: Execute the identical pipeline inside a container without Python code.

# %% [markdown]
# The three tasks demonstrate different evaluation workflows:
#
# | Task | Workflow | Sources | Extractor | Answers |
# | --- | --- | --- | --- | --- |
# | `clean_train` | `data-cleaning` | train | BoVW | Are training samples free of severe outliers and duplicates? |
# | `profile_splits` | `data-analysis` | train + test | (none) | Do train and test splits share label distributions without leakage? |
# | `split_train` | `data-splitting` | train | (none) | How should you partition training data into cross-validation folds? |

# %% [markdown]
# ## What you will learn
#
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
# ## Step 2: Set up the configuration
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
# ## Step 3: Run the pipeline
#
# You can execute all enabled tasks using `run_tasks()`:
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
for task, result in zip(config.tasks, results, strict=True):
    status = "OK " if result.success else "FAIL"
    warnings = sum(1 for f in result.data.report.findings if f.severity == "warning")
    elapsed = result.metadata.execution_time_s or 0.0
    print(f"[{status}] {task.name:<16} {result.name:<16} {elapsed:>6.1f}s  {warnings} warning(s)")
    for error in result.errors:
        print(f"         {error}")

# %% tags=["remove_cell"]
assert all(r.success for r in results), [r.errors for r in results if not r.success]

# %%
clean_result, profile_result, split_result = results

# %% [markdown]
# ## Step 4: Display the results
#
# Each task result exposes three primary interfaces:
#
# - `result.report()`: Formatted text summary.
# - `result.data.report.findings`: Structured finding objects.
# - `result.data.raw`: Workflow-specific raw numerical metrics.
#
# You can call `report(detailed=False)` for high-level summaries, or `report(detailed=True)`
# for per-finding breakdowns.

# %% [markdown]
# ### 4a. Summary reports

# %%
for result in results:
    print(result.report(detailed=False))

# %% [markdown]
# ### 4b. Structured findings
#
# Each finding includes a `title`, `severity` (`ok`, `info`, or `warning`), and `description`.
# You can query these programmatically in automated CI/CD gates.

# %%
for result in results:
    print(f"\n{result.name}")
    for finding in result.data.report.findings:
        marker = {"warning": "[!!]", "ok": "[ok]"}.get(finding.severity, "[..]")
        headline = (finding.description or "").splitlines()
        print(f"  {marker} {finding.title:<34} {headline[0][:60] if headline else ''}")

# %% [markdown]
# ### 4c. Data cleaning: Inspect flagged images
#
# The cleaning report summarizes the count of flagged images. You can retrieve specific
# sample indices from `result.data.raw` and slice `result.dataset` directly without reloading data.

# %%
raw = clean_result.data.raw

outlier_issues = raw.img_outliers["issues"]
outlier_indices = sorted({issue["item_index"] for issue in outlier_issues})
near_groups = raw.duplicates["items"].get("near", [])

print(f"Image outliers:        {len(outlier_indices)} images, {len(outlier_issues)} flags")
print(f"Near-duplicate groups: {len(near_groups)}")

# %%
from dataeval_plots import plot

assert clean_result.dataset is not None

if outlier_indices:
    _ = plot(
        clean_result.dataset,
        indices=outlier_indices[:6],
        images_per_row=3,
        figsize=(12, 8),
        show_labels=True,
    )

# %% [markdown]
# ### 4d. Data analysis: Cross-split comparisons
#
# The analysis task evaluates both train and test splits. It reports cross-split
# overlap, class parity, and duplicate leakage across splits.

# %%
print(f"Splits analyzed: {profile_result.metadata.split_names}")

for pair, section in profile_result.data.raw.cross_split.items():
    cs = section.model_dump()
    leakage = cs["redundancy"]["duplicate_leakage"]
    overlap = cs["label_health"]["label_overlap"]
    parity = cs["label_health"]["label_parity"]

    only = {key: value for key, value in overlap.items() if key.endswith("_only")}

    print(f"\n{pair}")
    print(f"  Duplicate leakage:  {leakage['exact_count']} exact, {leakage['near_count']} near")
    print(f"  Shared classes:     {overlap['shared_classes']}")
    print(f"  Classes in one split only: {only}")
    print(
        f"  Label parity:       chi2={parity['chi_squared']:.2f}, "
        f"p={parity['p_value']:.4f}, significant={parity['significant']}"
    )

# %% [markdown]
# ### 4e. Dataset splitting: Partition indices
#
# The splitting task generates explicit index lists for each fold. You can use
# these indices to construct PyTorch `Subset` or `DataLoader` instances.

# %%
fold = split_result.data.raw.folds[0]

print(f"Split sizes: {split_result.metadata.split_sizes}")
print(f"Stratified:  {split_result.metadata.stratified}")
print(f"Train indices (first 10): {fold.train_indices[:10]}")
print(f"Val   indices (first 10): {fold.val_indices[:10]}")
print(f"Test  indices (first 10): {split_result.data.raw.test_indices[:10]}")

# %% [markdown]
# ## Step 5: Export the results
#
# You can call `export()` to write the result envelope to disk. Result envelopes
# contain findings alongside execution metadata: timestamps, tool versions,
# dataset identifiers, and fully resolved configurations.

# %%
output_dir = Path("./output/end_to_end")

for task, result in zip(config.tasks, results, strict=True):
    written = result.export(output_dir / f"{task.name}.json")
    print(f"{written}  ({written.stat().st_size:,} bytes)")

# %%
import json

envelope = json.loads((output_dir / "split_train.json").read_text())

print(f"Top-level keys: {list(envelope)}")
print(f"Tool:           {envelope['metadata']['tool']} {envelope['metadata']['tool_version']}")
print(f"Timestamp:      {envelope['metadata']['timestamp']}")
print(f"Sources:        {envelope['metadata']['source_descriptions']}")

# %% [markdown]
# ## Step 6: Run the same pipeline in Docker
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
# docker pull harbor.jatic.net/aria/dataeval:cpu
#
# # Optional: verify the signature
# cosign verify --key docker/cosign.pub harbor.jatic.net/aria/dataeval:cpu
#
# # Or build locally from checkout
# docker build -f docker/Dockerfile.cpu -t dataeval:cpu .
# ```

# %% [markdown]
# ### Run it
#
# Use `--user` so `/output` and `/cache` remain writable by your host account.
# Use `-v` to print formatted reports to standard output.
#
# ```bash
# cd dataeval-run
#
# docker run --rm \
#     --user "$(id -u):$(id -g)" \
#     --mount type=bind,source="$PWD",target=/dataeval,readonly \
#     --mount type=bind,source="$PWD/output",target=/output \
#     --mount type=bind,source="$PWD/cache",target=/cache \
#     harbor.jatic.net/aria/dataeval:cpu \
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
#     harbor.jatic.net/aria/dataeval:cu130 \
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
# # Print findings and severities
# jq -r 'to_entries[] | .key as $task | .value.report.findings[]
#        | "\($task)\t\(.severity)\t\(.title)"' output/results/result.json
#
# # Gate CI/CD pipelines on warnings
# jq -e '[.[].report.findings[] | select(.severity == "warning")] | length == 0' \
#     output/results/result.json > /dev/null \
#     && echo "PASS: no warnings" || echo "FAIL: warnings present"
#
# # Inspect split partition sizes
# jq -r '.split_train.metadata.split_sizes' output/results/result.json
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
#     harbor.jatic.net/aria/dataeval:cpu \
#     python -m dataeval_flow --config end_to_end.yaml -v
# ```

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Stage dataset splits in isolated disk directories.
# - Define multi-workflow pipelines in a single YAML configuration file.
# - Execute pipelines using `run_tasks()` and the container CLI.
# - Inspect formatted reports and extract structured findings.
# - Export auditable result envelopes for CI/CD gates and downstream tools.
# - Execute the complete evaluation pipeline inside Docker.

# %% [markdown]
# ## Next steps
#
# - [Clean a dataset](data_cleaning): Deep dive into outlier and duplicate detection.
# - [Analyze dataset quality across splits](data_analysis): Multi-split quality profiling and distribution shift.
# - [Split a dataset](dataset_splitting): Stratification, cross-validation folds, and group-aware splitting.

# %% [markdown]
# ## Related guides
#
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) covers Docker execution options and flags.
# - **How-to**: [Reuse results with the disk cache](../how_to/reuse_results_with_cache.md) explains caching behaviors and invalidation.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) details result envelope structure and querying.
