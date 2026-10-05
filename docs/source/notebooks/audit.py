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
# Judge a dataset's train, validation and test splits with the `audit` preset, accept a risk in writing, and read the
# verdict and the record of what was audited.

# %% [markdown]
# **Target audience**: You are a T&E engineer or data scientist who has to approve a dataset for training, and keep a
# record of what you approved and why.
#
# **Workflow role**: You should run an audit on the splits a model will be trained and evaluated on, before training
# starts. It asks whether the data is clean, whether the labels are sound, whether the data covers what the model must
# handle, whether the model could learn a shortcut, and whether the splits are fit to evaluate on. Its next steps say
# what to do about each warning, and often name the preset that goes deeper, such as
# {doc}`Clean a dataset <data_cleaning>` or {doc}`Assess dataset coverage <data_coverage>`. See
# [Data quality and cleaning](../concepts/DataQualityAndCleaning.md) and
# [Dataset splitting](../concepts/DatasetSplitting.md) for background.

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
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load the splits the dataset ships with
#
# [SkySeaLand](https://www.kaggle.com/datasets/mdzahidhasanriad/skysealand) is an overhead
# object-detection dataset containing 1,307 frames from four sites across `airplane`, `boat`,
# `car`, and `ship`. The dataset provides `train` (1,048 frames), `val` (132), and `test`
# (127) partitions. You can use this workflow to evaluate whether partitioned splits agree
# with each other.
#
# You can use `maite-datasets` with `as_datamaite=True` to export the splits into COCO format.
# Each split is exported into a distinct directory.

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
# ## Step 1: Build the audit configuration
#
# An `audit` entry needs one block, `outliers`: which image statistics to judge and how far out an outlier sits. Here
# it judges dimension, pixel, and visual statistics with **adaptive** thresholding at 4. Every other setting has a
# default, which the [Preset Catalog](../reference/presets.md#audit) lists.
#
# The task names the splits in order. The first source is train, and every later source is an evaluation split.
#
# Note these four configuration decisions:
#
# - **Sample `train`; audit `val` and `test` completely**: Sampling 300 training frames reduces memory and
#   computation. Remove the view to audit the full training set. The record names the view, since a training job must
#   read the same 300 frames to match it.
# - **Shuffle before limiting**: SkySeaLand is organized by collection site on disk. Applying `Shuffle` before `Limit`
#   ensures that the sample represents all collection sites.
# - **Declare a metadata policy**: SkySeaLand ships no telemetry, so its factors are statistics measured from the
#   pixels. Every split is encoded like train, so the splits' factors compare.
# - **Name an extractor**: BoVW embeddings need no model file. Without an extractor, the coverage, shift and
#   evaluation-coverage checks are not assessed, and the verdict says so. `seed=0` makes the BoVW vocabulary the same
#   on every run.

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
# ## Step 2: Run the audit

# %%
result = run_task(config, task, cache_dir=Path("./cache"))

# %% [markdown]
# :::{note}
# The run logs two expected warnings.
#
# The continuous factors were binned automatically because the policy declares no bins, so their bin counts depend on
# the sample. You can use {doc}`Metadata triage <metadata_triage>` to declare them.
#
# "Declared cuts left bins unused" comes from the `factor-triage` step, which reads the metadata back under the bin
# counts it suggests. Its 40 bins for `instance_kurtosis` leave 3 empty on this sample.
# :::

# %% tags=["remove_cell"]
if not result.success:
    print(f"Workflow failed: {result.errors}")
assert result.success

# %% [markdown]
# ## Step 3: Read the verdict
#
# The verdict is `not-ready` when a blocking check warned unaccepted, `ready-with-caveats` when anything else warned or
# a check was not assessed, and `ready` otherwise. The [Preset Catalog](../reference/presets.md#audit) gives the rule
# in full. The blocking checks are `leakage` and `untrained-classes` unless the entry's `blocking:` says otherwise.
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
# - **Is the data clean?** Every split flags more images as outliers than the default 3%: 5.7% of train's sample,
#   7.6% of val, and 7.9% of test. Every split also has unreadable metadata. Its identifier columns hold a different
#   value for every frame, and its `histogram` and `percentiles` statistics hold an array, so none of them can be a
#   factor.
# - **Are the labels sound?** Yes. Every split holds all four classes, with enough labels in each, and the most common
#   class has about 2.6 times the labels of the rarest at most.
# - **Does the data cover what the model must handle?** The coverage checks read BoVW embeddings of train's boxes.
#   `ship` and `boat` are one-dimensional there, and the embeddings fill 5% of their dimensions. With 27 automatically
#   binned factors over 300 frames, 1,952 combinations of a class and a factor's value hold too few frames.
#   {doc}`Assess dataset coverage <data_coverage>` goes deeper.
# - **Could the model learn a shortcut?** Yes. 22 of the 27 factors carry information about the class, `unit_mean` the
#   most at 0.88. With no bins declared, DataEval reads these factors' measured values, a reading it doesn't correct
#   for chance, so the figure runs higher than a binned one. With bins declared, the
#   {doc}`end-to-end tutorial <end_to_end>` reads 0.27 for `unit_mean` over the same 300 frames, still past the 0.1 at
#   which the check warns. Statistics such as a frame's mean brightness differ by class, so a model could tell the
#   classes apart by the scene instead of the object.
# - **Are the splits fit to evaluate on?** Mostly. No image appears in two splits, so `leakage` passes. Leakage
#   blocks by default, so a leak would make the verdict `not-ready`. SkySeaLand's metadata records no site or scene,
#   so the entry sets no `factor-leakage` group factor. Val's boxes are 33% boats against 16% in train, so
#   `stratification` warns on val, while test stays within 3.3 points of train. Almost no evaluation frame lies
#   farther from train than train's own frames do, and neither evaluation split has shifted from train.

# %% [markdown]
# ## Step 4: Accept a risk with a reason
#
# A warning you have reviewed and decided to live with goes under `accepted:`, keyed by check type, with the reason.
# The accepted warning keeps its severity and its evidence, and the report lists it as an accepted risk. It can no
# longer make the verdict `not-ready`, but it is still a caveat. An acceptance covers its check type on every split,
# on this run and later ones.
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
# The verdict is still `ready-with-caveats`. The other warnings remain, and an accepted warning is a caveat too.
# Before training on this data, re-split val, or document why its boat share differs, and decide what to do about the
# shortcut risk. `metadata-issues` stays unaccepted, though Step 3 found its columns harmless, because an acceptance
# covers its check on later runs too, and would also accept a column that a later version of the data can't be read
# from. The three coverage warnings stay as caveats: they rest on BoVW embeddings and on 27 automatically binned
# factors over 300 frames, and {doc}`Assess dataset coverage <data_coverage>` shows how to look into them.

# %% [markdown]
# ## Step 5: Read the report
#
# The short report gives the verdict, then the record of what was audited, then one line per question. The record has
# a column per split: its source and view, its items, labels and classes, its metadata factors, and the digests of its
# content and its metadata. Below them are the run's library versions and extractor, and the criteria in force: every
# check's settings, the blocking checks, and each acceptance with its reason. Call `report(detailed=True)` or
# `to_html()` for each finding's evidence and the next steps.

# %%
print(result.report(detailed=False))

# %% [markdown]
# ## Step 6: Check data against the record
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
# ## Step 7: Export the result
#
# The JSON holds the same verdict under `verdict`, and each split's digests in its `content-digest` step. A CI job can
# read the verdict from it without Flow installed.

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
# - **Gating training**: Use [Gate training on an audit](../how_to/gate_training_on_an_audit.md) to refuse data that
#   was not audited, or not approved.
# - **Dataset splitting**: Use [Split a dataset](dataset_splitting) to generate balanced, stratified partitions when
#   published splits diverge, as val does here.
# - **Data cleaning**: Use {doc}`Clean a dataset <data_cleaning>` to list and remove flagged outliers and duplicates.
# - **ONNX embeddings**: Configure an ONNX model for higher-fidelity embeddings in the coverage and shift checks.

# %% [markdown]
# ## Related guides
#
# - **Reference**: [Preset Catalog](../reference/presets.md#audit) lists the audit's chain, settings, checks and
#   verdict rule.
# - **Concept**: [Dataset splitting](../concepts/DatasetSplitting.md) explains why leakage and unrepresentative splits
#   make a test score untrustworthy.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) explains result structure and finding
#   severity levels.
# - **How-to**: [Narrow a dataset with views](../how_to/build_dataset_views.md) explains dataset sampling and filtering
#   operations.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) explains how to execute workflows in
#   Docker.
# - **Guide**: [Use an ONNX model for embeddings](onnx_embeddings) shows how to configure pretrained extractors.
# - **Tutorial**: {doc}`Triage a dataset's metadata <metadata_triage>` explains how to define explicit metadata binning
#   policies.
