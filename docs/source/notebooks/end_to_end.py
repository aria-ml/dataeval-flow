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
# Load a dataset, write one config file, run three workflows against it in a single call,
# read the results — then run the identical pipeline in a container.

# %% [markdown]
# **Who this is for** — anyone who has DataEval Flow installed and wants to see the whole
# operating loop once, start to finish, before drilling into any individual workflow.
#
# **Where this fits** — this is the shape of every DataEval Flow run: data on disk → a
# config file → `run_tasks()` or `dataeval-flow` → result envelopes. The other tutorials
# go deep on one workflow each; this one stays shallow and covers the full path.

# %% [markdown]
# ## What you'll do
#
# 1. **Load a dataset** — fetch CPPE-5 and write it to disk in a layout DataEval Flow reads
# 2. **Write the configuration** — one `end_to_end.yaml` describing the datasets, sources,
#    extractor, three workflows, and three tasks
# 3. **Run the pipeline** — `run_tasks()` executes all three tasks against the same data
# 4. **Display the results** — reports, per-workflow detail, and the flagged images themselves
# 5. **Export the results** — machine-readable envelopes for downstream tooling
# 6. **Run it in Docker** — the same config file, no Python

# %% [markdown]
# The three tasks are deliberately different shapes, because that is what a real evaluation
# looks like:
#
# | Task | Workflow | Sources | Extractor | Answers |
# | --- | --- | --- | --- | --- |
# | `clean_train` | `data-cleaning` | train | BoVW | Is the training data fit to use? |
# | `profile_splits` | `data-analysis` | train + test | — | Do the splits agree with each other? |
# | `split_train` | `data-splitting` | train | — | How do I carve out train/val/test? |

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (brings `dataeval`, `datamaite`, `pydantic`)
# - `dataeval-plots` (to look at flagged images)
# - `datasets` (to fetch CPPE-5 from the HuggingFace Hub)
# - Internet access on the first run; everything after that comes from disk
# - Docker, for the last section only

# %% [markdown]
# ## Step 1: Load the dataset
#
# [CPPE-5](https://huggingface.co/datasets/rishitdagli/cppe-5) is a small object-detection
# dataset of medical personal protective equipment — 5 classes, ~1K train images and 29 test
# images. Two splits is what makes the cross-split half of this pipeline meaningful.

# %% tags=["remove_output"]
from datasets import load_dataset as hf_load

cppe5 = hf_load("rishitdagli/cppe-5")
for split_name, split in cppe5.items():
    print(f"{split_name}: {len(split)} images")

# %% [markdown]
# ### Materialize it on disk
#
# DataEval Flow reads datasets from a filesystem layout, so the in-memory HuggingFace
# `Dataset` has to be written out first. We use the HuggingFace **ImageFolder**
# object-detection convention — image files plus a `metadata.parquet` whose `objects` column
# carries parallel `bbox` / `category` lists. CPPE-5 already stores absolute-pixel `xywh`
# boxes, which is what this convention expects, so no coordinate conversion is needed.
#
# :::{note}
# **Write `metadata.parquet`, not `metadata.jsonl`, to keep class names.** Parquet is the only
# ImageFolder metadata format with a schema channel: `Dataset.to_parquet()` embeds the
# HuggingFace features schema in the file header, and datamaite reads the `ClassLabel` name
# table out of it. That is why the reports below name `Coverall` and `Face_Shield` instead of
# `0` and `1`. `metadata.csv` and `metadata.jsonl` carry no schema, so integer categories stay
# integers there — matching HuggingFace's own behavior.
# :::

# %% tags=["remove_output"]
from pathlib import Path

from datasets import Dataset

data_path = Path("./data/cppe5")


def write_imagefolder(hf_dataset: Dataset, root: Path) -> None:
    """Write a HuggingFace object-detection split in the ImageFolder convention."""
    root.mkdir(parents=True, exist_ok=True)
    for stale in root.glob("metadata.*"):  # a leftover metadata file is read alongside the new one
        stale.unlink()

    file_names = []
    for seq, example in enumerate(hf_dataset):
        file_name = f"{seq:05d}.jpg"
        example["image"].convert("RGB").save(root / file_name, quality=95)
        file_names.append(file_name)

    # Carry the source `objects` column through untouched so its `category` ClassLabel — the
    # Coverall/Face_Shield/... name table — lands in the parquet features schema.
    metadata = hf_dataset.remove_columns(["image", "image_id", "width", "height"]).add_column("file_name", file_names)
    metadata.to_parquet(root / "metadata.parquet")


for split_name, split in cppe5.items():
    write_imagefolder(split, data_path / split_name)

print(f"Wrote {data_path}/train and {data_path}/test")

# %% [markdown]
# ### Confirm it loads
#
# Before writing any config, check that DataEval Flow can actually read what you just wrote.
# `load_dataset()` is the same loader the pipeline uses internally, so if this works the
# config will too.

# %%
from dataeval_flow import load_dataset

train_ds = load_dataset(data_path, split="train", dataset_format="huggingface", task="object_detection")
image, target, _ = train_ds[0]

print(f"Images:      {len(train_ds)}")
print(f"Image shape: {image.shape}")
print(f"Boxes:       {len(target.boxes)}")
print(f"Classes:     {train_ds.metadata['index2label']}")

# %% [markdown]
# ## Step 2: Set up the configuration
#
# Everything about the run lives in one YAML file. The structure is **define once, reference
# by name**:
#
# | Section | What it declares |
# | --- | --- |
# | `datasets` | Where data lives and how to read it |
# | `views` | How to narrow a dataset (limit, index range, class filter) |
# | `sources` | A dataset plus an optional view — what tasks actually consume |
# | `extractors` | How embeddings are produced |
# | `workflows` | Named parameter sets, one per evaluation |
# | `tasks` | Binds a workflow to sources (and an extractor when embeddings are used) |
#
# The file below sits next to this notebook. Read it top to bottom — the three `tasks` at
# the end are the whole pipeline.

# %%
config_path = Path("end_to_end.yaml")
print(config_path.read_text())

# %% [markdown]
# ### Load and validate it
#
# `load_config()` parses the YAML and validates it against the pipeline schema. Bad
# parameters, unknown workflow types, and dangling name references all fail here — before
# any data is touched.

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
# `run_tasks()` executes every enabled task in order and returns one result per task.
#
# - `data_dir` is the root that every relative path in the config resolves against
# - `cache_dir` persists embeddings, image statistics, and metadata between runs — the
#   three tasks share one dataset, so the second and third reuse what the first computed
#
# This takes a few minutes on the first run and is much faster afterwards.

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
# Every result carries the same three access points, whatever the workflow:
#
# - `result.report()` — formatted text summary, the thing you read first
# - `result.data.report.findings` — the same findings as structured objects
# - `result.data.raw` — the underlying numbers, per-workflow
#
# `report(detailed=False)` gives the summary alone; the default `detailed=True` adds a
# section per finding.

# %% [markdown]
# ### 4a. The three summaries

# %%
for result in results:
    print(result.report(detailed=False))

# %% [markdown]
# ### 4b. Findings as data
#
# Each finding has a `title`, a `severity` (`ok` / `info` / `warning`), and a `description`.
# This is what you'd assert on in a CI gate.

# %%
for result in results:
    print(f"\n{result.name}")
    for finding in result.data.report.findings:
        marker = {"warning": "[!!]", "ok": "[ok]"}.get(finding.severity, "[..]")
        headline = (finding.description or "").splitlines()
        print(f"  {marker} {finding.title:<34} {headline[0][:60] if headline else ''}")

# %% [markdown]
# ### 4c. Cleaning — look at what was flagged
#
# The cleaning report says *how many* images were flagged. `result.data.raw` says *which
# ones*, and `result.dataset` is the resolved post-view dataset, so the indices line up
# directly — no reloading.

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
# ### 4d. Analysis — where the splits disagree
#
# The analysis task is the only one that saw both splits, so it is the only one that can
# report cross-split findings: shared label space, label parity, and duplicate leakage
# between train and test.

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
# ### 4e. Splitting — the partitions themselves
#
# The splitting result carries the actual index lists. Feed them straight to a
# `Subset`/`DataLoader`, or persist them as the record of how the data was partitioned.

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
# `export()` writes the **result envelope** — the findings plus the provenance that makes
# them auditable: timestamp, tool version, dataset identifiers, and the fully resolved
# configuration. This is the artifact you hand to another JATIC tool, attach to a review,
# or diff against last week's run.

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
# Nothing above is Python-specific — the pipeline is the YAML file. The container reads the
# same `end_to_end.yaml`, against the same `data/cppe5`, and produces the same envelopes.
# This is how the pipeline runs in CI or on a machine that has no Python environment.
#
# The container has three mount points:
#
# | Mount | Mode | Holds |
# | --- | --- | --- |
# | `/dataeval` | read-only | Data root — datasets, models, config files |
# | `/output` | read-write | Reports and result envelopes |
# | `/cache` | read-write | Embedding and statistics cache (optional, speeds up re-runs) |

# %% [markdown]
# ### Lay out the workspace
#
# One directory becomes the data root. Config at its top level, dataset underneath at the
# path the config names (`data/cppe5`).
#
# ```bash
# mkdir -p dataeval-run/data dataeval-run/output dataeval-run/cache
# cp -r docs/source/notebooks/data/cppe5 dataeval-run/data/cppe5
# cp docs/source/notebooks/end_to_end.yaml dataeval-run/
#
# tree -L 3 dataeval-run
# # dataeval-run
# # ├── cache
# # ├── data
# # │   └── cppe5
# # │       ├── test
# # │       └── train
# # ├── end_to_end.yaml
# # └── output
# ```

# %% [markdown]
# ### Get the image
#
# Pull a pre-built image, or build one from a source checkout.
#
# ```bash
# # Pre-built (cpu / cu118 / cu128)
# docker pull harbor.jatic.net/aria/dataeval:cpu
#
# # Optional: verify the signature
# cosign verify --key docker/cosign.pub harbor.jatic.net/aria/dataeval:cpu
#
# # Or build locally from a checkout
# docker build -f docker/Dockerfile.cpu -t dataeval:cpu .
# ```

# %% [markdown]
# ### Run it
#
# `--user` runs the container as your host user so `/output` and `/cache` are writable.
# `-v` prints the full report to the console; `-vv` and `-vvv` add INFO and DEBUG logs.
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
# On a GPU host, add `--gpus all` and use a CUDA variant — the config does not change:
#
# ```bash
# docker run --rm --gpus all \
#     --user "$(id -u):$(id -g)" \
#     --mount type=bind,source="$PWD",target=/dataeval,readonly \
#     --mount type=bind,source="$PWD/output",target=/output \
#     --mount type=bind,source="$PWD/cache",target=/cache \
#     harbor.jatic.net/aria/dataeval:cu128 \
#     python -m dataeval_flow --config end_to_end.yaml -v
# ```
#
# :::{note}
# `--config` is optional. Without it the container auto-discovers and **merges every**
# YAML/JSON file at the data root — convenient for a directory holding one config, wrong
# for a directory holding several. Name the file explicitly when in doubt.
# :::

# %% [markdown]
# ### Read the output
#
# The container writes one merged `result.json` keyed by task name, one `result.txt`
# holding the detailed text reports, and a run log.
#
# ```bash
# find output -type f
# # output/result.log
# # output/results/result.json
# # output/results/result.txt
# ```
#
# The same three findings sets you read in Step 4, now from the shell:
#
# ```bash
# # Which tasks ran?
# jq -r 'keys[]' output/results/result.json
# # clean_train
# # profile_splits
# # split_train
#
# # Every finding, with its severity
# jq -r 'to_entries[] | .key as $task | .value.report.findings[]
#        | "\($task)\t\(.severity)\t\(.title)"' output/results/result.json
#
# # Fail a CI gate on any warning
# jq -e '[.[].report.findings[] | select(.severity == "warning")] | length == 0' \
#     output/results/result.json > /dev/null \
#     && echo "PASS: no warnings" || echo "FAIL: warnings present"
#
# # The split indices, ready for downstream use
# jq -r '.split_train.metadata.split_sizes' output/results/result.json
# ```

# %% [markdown]
# ### Running offline
#
# Once the image and the dataset are staged locally, the run makes no outbound network
# calls. Pin the environment to be sure:
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
# You ran one dataset through three workflows, from raw download to exported envelopes, and
# then ran the identical pipeline in a container with no Python involved.
#
# The pattern to keep:
#
# 1. Materialize the dataset on disk in a supported layout
# 2. Describe the whole run in one config file — datasets, sources, extractors, workflows, tasks
# 3. `run_tasks()` in Python, or `dataeval-flow` / the container in a shell
# 4. Read `result.report()` interactively, `result.data.report.findings` programmatically
# 5. `export()` the envelopes so a finding stays auditable after the session ends
#
# ### Where to go next
#
# - [Clean a dataset](data_cleaning) — outlier and duplicate detection in depth
# - [Analyze dataset quality across splits](data_analysis) — every assessment area the profile task touched
# - [Split a dataset](dataset_splitting) — stratification, folds, and group-aware splitting
# - [Run workflows in containers](../how_to/containerized_workflows.md) — every container option
# - [Reuse results with the cache](../how_to/reuse_results_with_cache.md) — what is cached and when it invalidates
# - [Read evaluation outputs](../how_to/read_evaluation_outputs.md) — the envelope format in detail
