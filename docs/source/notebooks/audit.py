# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: dataeval-flow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Audit a set of splits before training
#
# Judge SkySeaLand's train, validation and test splits with the [`audit`](../reference/presets.md#audit) preset, accept
# a risk in writing, and read the verdict and the record of what was audited.

# %% [markdown]
# You are a T&E engineer or data scientist who has to approve a dataset for training, and keep a record of what you
# approved and why.
#
# You should run an audit on the splits a model will be trained and evaluated on, before training starts. It asks
# whether the data is clean, whether the labels are sound, whether the data covers what the model must handle, whether
# the model could learn a shortcut, and whether the splits are fit to evaluate on. Its next steps say what to do about
# each warning, and often name the preset that goes deeper, such as {doc}`data_cleaning` or {doc}`data_coverage`. Build
# the splits first with {doc}`dataset_splitting`, and judge them here. Once a model trains on them,
# {doc}`measure_data_sufficiency` asks whether it needs more data. See {doc}`../concepts/DataQualityAndCleaning` and
# {doc}`../concepts/DatasetSplitting` for background.

# %% [markdown]
# ## What you will do
#
# - Download SkySeaLand using `maite-datasets` and export its train, validation, and test splits.
# - Configure an `audit` over the three splits, with BoVW embeddings.
# - Run it with `run_task()` and read its verdict.
# - Look at the flagged images, accept the image-outlier warning with a written reason, and run it again.
# - Read the report: the verdict, the record of what was audited, and one line per question.
# - Check a dataset against the record's content digest, as a training job would.

# %% [markdown]
# ## What you will learn
#
# - How to configure and run the `audit` preset over several splits.
# - How the verdict is decided, and what `blocking` and `accepted` change.
# - How to read `result.verdict` and the report's five questions.
# - Where the record keeps each split's content digest, and how `dataset_digest()` recomputes it.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `dataeval-plots` to visualize flagged images.
# - Install `maite-datasets[datamaite]` to download and export SkySeaLand.
# - Ensure network access for the initial dataset download (~262 MB).

# %% [markdown]
# ## Lay out the data root
#
# The data root is the folder you run this notebook from. After a first run it holds:
#
# ```text
# project/
# ├── data/
# │   ├── skysealand/                    # downloaded on the first run (~262 MB); the three exports are made from it
# │   ├── skysealand_datamaite_train/    # exported on the first run; read: the train split, the first source
# │   ├── skysealand_datamaite_val/      # exported on the first run; read: the validation split
# │   └── skysealand_datamaite_test/     # exported on the first run; read: the test split
# └── cache/                             # written: embeddings and image statistics, reused on a re-run
# ```
#
# The config's dataset paths, such as `data/skysealand_datamaite_train`, are relative to the data root. There is no
# pipeline file. The notebook builds the pipeline in Python, in the "Build the audit configuration" section.

# %% [markdown]
# ## Load the splits the dataset ships with
#
# [SkySeaLand](https://www.kaggle.com/datasets/mdzahidhasanriad/skysealand) is an overhead object-detection dataset with
# 1,307 frames from four sites and four classes: `airplane`, `boat`, `car`, and `ship`. The dataset provides `train`
# (1,048 frames), `val` (132), and `test` (127) partitions. This workflow evaluates whether the partitioned splits agree
# with each other.
#
# Use `maite-datasets` with `as_datamaite=True` to export the splits into COCO format. Each split goes into its own
# directory.

# %% tags=["remove_output"]
from pathlib import Path

from maite_datasets.object_detection import SkySeaLand

data_root = Path("./data")

# One ~262 MB download into ./data/skysealand, shared by all three exports below.
# A re-run reads what is already on disk instead of downloading again.
SkySeaLand(root=data_root, image_set="base", download=True)

split_paths = {name: data_root / f"skysealand_datamaite_{name}" for name in ("train", "val", "test")}
for image_set in split_paths:
    SkySeaLand(root=data_root, image_set=image_set, as_datamaite=True)

print("\n".join(f"{name}: {path}" for name, path in split_paths.items()))

# %% [markdown]
# ## Build the audit configuration
#
# An `audit` entry needs one block, `outliers`: which image statistics to judge and how far out an outlier sits. Here it
# judges dimension, pixel, and visual statistics with adaptive thresholding at 4. Every other setting has a default,
# which the [Preset Catalog](../reference/presets.md#audit) lists.
#
# The task names the splits in order. The first source is train, and every later source is an evaluation split.
#
# Note these four configuration decisions:
#
# - Sample `train`, and audit `val` and `test` completely. Sampling 300 training frames reduces memory and computation.
#   Remove the view to audit the full training set. The record names the view, since a training job must read the same
#   300 frames to match it. For the sampling and filtering operations, see {doc}`../how_to/build_dataset_views`.
# - Shuffle before limiting. SkySeaLand is organized by collection site on disk. Applying `Shuffle` before `Limit`
#   ensures that the sample represents all collection sites.
# - Declare a metadata policy. SkySeaLand ships no telemetry, so its factors are statistics measured from the pixels.
#   Every split is encoded like train, so the splits' factors compare.
# - Name an extractor. BoVW embeddings need no model file. Without an extractor, the coverage, shift and
#   evaluation-coverage checks are not assessed, and the verdict says so. `seed=0` makes the BoVW vocabulary the same on
#   every run. To configure a pretrained extractor, see {doc}`onnx_embeddings`.

# %%
from dataeval.config import set_max_processes

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
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.workflows.audit import AuditConfig

# Limit concurrency to 4 processes for memory management during image decoding.
set_max_processes(4)

audit_workflow = AuditConfig(
    name="skysealand_audit",
    outliers={"flags": ["dimension", "pixel", "visual"], "outlier_threshold": ["adaptive", 4.0]},
    metadata="skysealand_factors",
)

task = TaskConfig(
    name="skysealand-audit",
    workflow="skysealand_audit",
    sources=["train", "val", "test"],
    extractor="bovw_ext",
)

config = PipelineConfig(
    max_processes=2,
    seed=0,
    metadata=[
        MetadataPolicyConfig(
            name="skysealand_factors",
            # Image statistics evaluated as factors
            intrinsic_factors=["visual", "pixel"],
            # Exclude constant metadata fields
            exclude=["label_file_exists"],
            # Shared encoding reference for cross-split factor comparability
            reference_split="train",
        )
    ],
    datasets=[CocoDatasetConfig(name=f"skysealand_{name}", path=str(path)) for name, path in split_paths.items()],
    views=[
        ViewConfig(
            name="sample300",
            operations=[
                ViewOperation(type="Shuffle", params={"seed": 0}),
                ViewOperation(type="Limit", params={"size": 300}),
            ],
        ),
    ],
    sources=[
        SourceConfig(name="train", dataset="skysealand_train", view="sample300"),
        SourceConfig(name="val", dataset="skysealand_val"),
        SourceConfig(name="test", dataset="skysealand_test"),
    ],
    extractors=[BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32)],
    workflows=[audit_workflow],
    tasks=[task],
)

print("Configuration ready:")
print(f"  Workflow:   {audit_workflow.name} (type={audit_workflow.type})")
print(f"  Task:       {task.name} -> {task.workflow}")
print(f"  Sources:    {task.sources}")

# %% [markdown]
# ## Run the audit

# %%
result = run_task(config, task, cache_dir=Path("./cache"))

# %% [markdown]
# :::{note}
# The run logs two expected warnings.
#
# The continuous factors were binned automatically because the policy declares no bins, so their bin counts depend on
# the sample. Use {doc}`metadata_triage` to declare them.
#
# "Declared cuts left bins unused" comes from the `factor-triage` step, which reads the metadata back under the bin
# counts it suggests. Its 40 bins for `instance_kurtosis` leave 3 empty on this sample.
# :::

# %% tags=["remove_cell"]
if not result.success:
    print(f"Workflow failed: {result.errors}")
assert result.success

# %% [markdown]
# ## Read the verdict
#
# The verdict is `not-ready` when a blocking check warned unaccepted, `ready-with-caveats` when anything else warned or
# a check was not assessed, and `ready` otherwise. The [Preset Catalog](../reference/presets.md#audit) gives the rule in
# full. The blocking checks are `leakage` and `untrained-classes` unless the entry's `blocking:` says otherwise.
# `result.verdict` lists each warning by check and step, and each check that could not be assessed with its reason.

# %%
verdict = result.verdict
assert verdict is not None

print(verdict.line())
for item in verdict.blocking + verdict.warnings:
    print(f"  {item.step:<28} {item.title}: {item.brief}")
for item in verdict.not_assessed:
    print(f"  {item.step:<28} not assessed: {item.reason}")

# %% [markdown]
# ### What this run found
#
# Nothing blocks. The warnings fall under four of the five questions:
#
# - Is the data clean? Every split flags more images as outliers than the default 3%: 5.7% of train's sample, 7.6% of
#   val, and 7.9% of test. Every split also has unreadable metadata. Its identifier columns hold a different value for
#   every frame, and its `histogram` and `percentiles` statistics hold an array, so none of them can be a factor.
# - Are the labels sound? Yes. Every split holds all four classes, with enough labels in each, and the most common class
#   has about 2.6 times the labels of the rarest at most.
# - Does the data cover what the model must handle? The coverage checks read BoVW embeddings of train's boxes. `ship`
#   and `boat` are one-dimensional there, and the embeddings fill 5% of their dimensions. With 27 automatically binned
#   factors over 300 frames, 1,952 combinations of a class and a factor's value hold too few frames.
#   {doc}`data_coverage` goes deeper.
# - Could the model learn a shortcut? Yes. 22 of the 27 factors carry information about the class, `unit_mean` the most
#   at 0.88. With no bins declared, DataEval reads these factors' measured values, a reading it does not correct for
#   chance, so the figure runs higher than a binned one. With bins declared, `unit_mean` reads 0.27 over the same 300
#   frames (see {doc}`end_to_end`), still past the 0.1 at which the check warns. Statistics such as a frame's mean
#   brightness differ by class, so a model could tell the classes apart by the scene instead of the object.
# - Are the splits fit to evaluate on? Mostly. No image appears in two splits, so `leakage` passes. Leakage blocks by
#   default, so a leak would make the verdict `not-ready`. SkySeaLand's metadata records no site or scene, so the entry
#   sets no `factor-leakage` group factor. Val's boxes are 33% boats against 16% in train, so `class-stratification`
#   warns on val, while test stays within 3.3 points of train. Almost no evaluation frame lies farther from train than
#   train's own frames do, and neither evaluation split has shifted from train.

# %% [markdown]
# ## Accept a risk with a reason
#
# A warning you have reviewed and decided to live with goes under `accepted:`, keyed by check type, with the reason. The
# accepted warning keeps its severity and its evidence, and the report lists it as an accepted risk. It can no longer
# make the verdict `not-ready`, but it is still a caveat. An acceptance keyed by check type covers that check on every
# split, on this run and later ones. Key it by step instead, such as `image-outliers-evals[test]`, to accept one split's
# warning alone.
#
# Before accepting the image outliers, look at them. These are the first eight of train's flagged frames:

# %%
from dataeval_plots import plot

flags = result.steps["outliers-train"].output.data()  # one row per flagged image and statistic
flagged = sorted(set(flags["item_index"].to_list()))
print(f"{len(flagged)} flagged frames in train's sample")

assert result.sources is not None
_ = plot(result.sources["train"], indices=flagged[:8], images_per_row=4, figsize=(16, 8))

# %% [markdown]
# The flagged frames are narrow crops of one vessel or a row of cars, and dark open-water scenes. Val's and test's are
# the same kinds. They are normal for this imagery, and none is corrupt, so accept the warning:

# %%
audit_workflow = AuditConfig(
    name="skysealand_audit",
    outliers={"flags": ["dimension", "pixel", "visual"], "outlier_threshold": ["adaptive", 4.0]},
    metadata="skysealand_factors",
    accepted={"image-outliers": "Narrow crops and dark open-water scenes, normal for this imagery; none is corrupt."},
)
config = config.model_copy(update={"workflows": [audit_workflow]})

result = run_task(config, task, cache_dir=Path("./cache"))
verdict = result.verdict
assert verdict is not None

print(verdict.line())
for acceptance in verdict.accepted:
    print(f"  {acceptance.check} ({acceptance.state}): {acceptance.reason}")

# %% [markdown]
# The verdict is still `ready-with-caveats`. The other warnings remain, and an accepted warning is a caveat too. Before
# training on this data, re-split val, or document why its boat share differs, and decide what to do about the shortcut
# risk. `factor-issues` stays unaccepted, though the "Read the verdict" section found its columns harmless, because an
# acceptance covers its check on later runs too, and would also accept a column that a later version of the data cannot
# be read from. The three coverage warnings stay as caveats. They rest on BoVW embeddings and on 27 automatically binned
# factors over 300 frames, and {doc}`data_coverage` shows how to look into them.

# %% [markdown]
# ## Read the report
#
# The short report gives the verdict, then the record of what was audited, then one line per question. The record has a
# column per split: its source and view, its items, labels and classes, its metadata factors, and the digests of its
# content and its metadata. Below them are the run's library versions and extractor, and the criteria in force: the
# settings of each check the chain ran, the blocking checks, and each acceptance with its reason. Call
# `report(detailed=True)` or `to_html()` for each finding's evidence and the next steps.

# %%
print(result.report(detailed=False))

# %% [markdown]
# ### Read the HTML report
#
# `to_html()` draws the same blocks as the text report, as one self-contained page, with each finding's evidence and the
# next steps. {doc}`view_html_reports` shows what else the page does.

# %% tags=["remove_cell"]
# The page styles its whole document, so the docs show it in an iframe of its own.
import html

from IPython.display import display


def show(page: str, height: int = 560) -> None:
    """*page* in an iframe, so its stylesheet and script stay inside it."""
    frame = (
        f'<iframe srcdoc="{html.escape(page, quote=True)}" title="HTML report" loading="lazy" '
        f'style="width:100%;height:{height}px;border:1px solid #d1d9e0;border-radius:6px"></iframe>'
    )
    display({"text/html": frame}, raw=True)


# %% tags=["remove_input"]
show(result.to_html())

# %% [markdown]
# ## Check data against the record
#
# Each split's content digest covers its images, its labels, and its class names, whatever their order. A training job
# recomputes it with `dataset_digest()` on the data it is about to train on, and refuses a mismatch. The digest of
# train's 300-frame view matches the record. The whole train split does not, because it is not the data that was
# audited.

# %%
from dataeval_flow import dataset_digest, load_dataset

recorded = {"train": result.steps["content-digest-train"].output.data()}
for name, step in (result.steps["content-digest-evals"].elements or {}).items():
    recorded[name] = step.output.data()

for name, record in recorded.items():
    print(f"{name:<5} {record['items']:>4} items  {record['content']}")

sample = dataset_digest(result.sources["train"])  # the 300 frames the view selects
whole = dataset_digest(load_dataset(split_paths["train"], dataset_format="coco"))  # all 1,048 frames

print(f"\ntrain's view matches the record:  {sample.content == recorded['train']['content']}")
print(f"the whole train split matches it: {whole.content == recorded['train']['content']}")

# %% [markdown]
# ## Export the result
#
# The JSON holds the same verdict under `verdict`, and each split's digests in its `content-digest` step. A CI job can
# read the verdict from it without Flow installed. {doc}`../how_to/read_evaluation_outputs` explains result structure
# and finding severity levels, and {doc}`../how_to/containerized_workflows` explains how to execute workflows in Docker.

# %%
import json

envelope = json.loads(result.export(fmt="json"))

print(f"Verdict:        {envelope['verdict']['level']}")
print(f"Accepted:       {[acceptance['check'] for acceptance in envelope['verdict']['accepted']]}")
print(f"Train's digest: {envelope['steps']['content-digest-train']['output']['data']['content']}")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Configure an `audit` over a train split and two evaluation splits.
# - Read the verdict, and the warnings and unassessed checks behind it.
# - Accept a reviewed warning with a written reason, and see it listed as an accepted risk.
# - Read the record of what was audited, with each split's content digest.
# - Recompute a content digest with `dataset_digest()`, and tell audited data from data that was not.
# - Export the verdict and the record to JSON.

# %% [markdown]
# ## Next steps
#
# - {doc}`../how_to/gate_training_on_an_audit` — Refuse data that was not audited, or not approved.
# - {doc}`dataset_splitting` — Generate balanced, stratified partitions when published splits diverge, as val does here.
# - {doc}`data_cleaning` — List and remove flagged outliers and duplicates.
# - {doc}`../concepts/DatasetSplitting` — Learn why leakage and unrepresentative splits make a test score untrustworthy.

# %% [markdown]
# ## On your own
#
# To repeat the audit with your own data and settings, change these entries in the notebook's configuration:
#
# - Point the `CocoDatasetConfig` entries at your own exports, one for each split, and list the sources in the task in
#   order: train first, then each evaluation split.
# - To audit the whole training set, remove the `view` from the `train` source. To change the sample, edit the `size` of
#   the `Limit` operation in the `sample300` view.
# - To change what counts as an outlier, edit `flags` and `outlier_threshold` in the `outliers` block.
# - To change the factors the audit reads, edit `intrinsic_factors`, `exclude`, `reference_split` and
#   `continuous_factor_bins` in the `MetadataPolicyConfig`.
# - If your metadata records a site or scene, set `factor-leakage` with its `factors`, so that each value stays in one
#   split.
# - To accept a warning, add its check type to `accepted` with your reason, or key it by step, such as
#   `image-outliers-evals[test]`, for one split. To change which warnings block, edit `blocking`.
# - To use a pretrained model instead of BoVW, replace the `BoVWExtractorConfig` with another extractor config, such as
#   `OnnxExtractorConfig` with a `model_path`. To change the BoVW vocabulary, edit its `vocab_size`.
# - To change the memory use during image decoding, edit the number of processes in `set_max_processes`.
