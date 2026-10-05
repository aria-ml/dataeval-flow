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
# # Assess dataset coverage
#
# Detect class imbalance, metadata gaps, missing label-space regions, and
# embedding blind spots with two config-driven presets: `data-coverage` and `label-space`.

# %% [markdown]
# **Target audience**: You are a T&E engineer or data scientist who needs to verify
# that a dataset covers required operational conditions and label taxonomies before
# model training or certification.
#
# **Workflow role**: You should run coverage assessment during early data preparation
# alongside {doc}`Clean a dataset <data_cleaning>`. While data cleaning checks data quality,
# coverage evaluates completeness across classes, metadata factors, and feature spaces.
# Gaps identified here inform data collection and establish trustworthy baselines for
# {doc}`Monitor incoming data for drift <drift_monitoring>` and
# {doc}`Detect out-of-distribution samples <ood_detection>`. See [Dataset coverage](../concepts/Coverage.md)
# for background.

# %% [markdown]
# ## What you will do
#
# - Load MilitaryVehicles and filter out the Air Defense category using `ClassFilter` to simulate missing collection categories.
# - Run `data-coverage` without an extractor for a fast label and metadata pass.
# - See why class counts alone do not reveal a missing category.
# - Run `label-space` on the same source, with the dataset's taxonomy as its ontology, to name the unsampled concepts.
# - Re-run `data-coverage` with a BoVW extractor to evaluate embedding coverage and dimensional completeness.
# - Read each step's output from the result, and tune the health thresholds.

# %% [markdown]
# ## What you will learn
#
# - How to configure and run the `data-coverage` and `label-space` presets with `run_task()`.
# - How coverage evaluates two axes: taxonomic representation (`label-space`, against an ontology) and visual
#   variation (`data-coverage`, in embedding space).
# - Why count-based distributions fail to detect unsampled classes when loaders drop missing categories.
# - How to distinguish genuine collection gaps from intentional ontology scope boundaries.
# - How to adjust each check's health thresholds for varying domain risk tolerances.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `maite-datasets` to download MilitaryVehicles.
# - Ensure network access for the initial dataset download.

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: a collection with a hole in it
#
# Operational recognition systems require predefined label taxonomies. MilitaryVehicles
# provides a `hierarchy` attribute categorizing tanks, BMPs, BTRs, self-propelled
# artillery, air defense systems, and related vehicle types.
#
# You will drop the Air Defense category using `ClassFilter` to simulate a collection
# cycle that missed a required category. All remaining class counts and imagery remain
# unmodified.

# %% tags=["remove_output"]
from pathlib import Path

import numpy as np
from dataeval.data import ClassFilter, Limit, Shuffle, View
from datamaite import load_ic
from maite_datasets.image_classification import MilitaryVehicles

data_root = Path("./data")
MilitaryVehicles(root=data_root, image_set="base", download=True)
MilitaryVehicles(root=data_root, image_set="train", as_datamaite=True)

vehicles_raw = load_ic(data_root / "militaryvehicles_datamaite_train" / "train", dataset_format="huggingface_vision")
index2label = vehicles_raw.metadata.get("index2label", {})

# The category this collection cycle never captured.
AIR_DEFENSE = {"30N6E", "Iskander", "Pantsir-S1", "Rs-24"}
collected_classes = [i for i, name in index2label.items() if name not in AIR_DEFENSE]

# `ClassFilter` enforces the gap by filtering on sample labels.
# `Shuffle` and `Limit` sample the result for faster execution.
collected = View(vehicles_raw, [ClassFilter(collected_classes), Shuffle(seed=0), Limit(1500)])

print(f"Full collection:  {len(vehicles_raw)} frames, {len(index2label)} types")
print(f"After the gap:    {len(collected)} frames, {len(collected_classes)} types")
print(f"Never collected:  {sorted(AIR_DEFENSE)}")

# %% [markdown]
# ### Look at the class distribution
#
# Before running anything, the obvious first check: are the classes balanced?

# %%
labels = np.array([int(np.argmax(collected[i][1])) for i in range(len(collected))])
class_counts = {index2label[c]: int((labels == c).sum()) for c in sorted(set(labels.tolist()))}
ordered = sorted(class_counts.items(), key=lambda kv: -kv[1])

largest, smallest = ordered[0], ordered[-1]
ratio = largest[1] / smallest[1]
print(f"Largest class:  {largest[0]} ({largest[1]})")
print(f"Smallest class: {smallest[0]} ({smallest[1]})")
print(f"Imbalance ratio: {ratio:.2f}:1")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))
names = [n for n, _ in ordered]
values = [v for _, v in ordered]
ax.bar(names, values, color="#3498db")
ax.axhline(y=float(np.mean(values)), color="gray", linestyle="--", label="mean")
ax.set_ylabel("Frame count")
ax.set_title("Class distribution of the collected data")
ax.tick_params(axis="x", rotation=60)
for tick in ax.get_xticklabels():
    tick.set_horizontalalignment("right")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The sample distribution shows an imbalance ratio of 1.77:1. Count-based checks
# indicate a balanced dataset, but cannot identify missing categories absent an
# external ontology definition.

# %% [markdown]
# ## Step 1: Run coverage without an extractor (metadata only)
#
# You can run `data-coverage` without an extractor for a fast initial pass. `data-coverage`
# is a preset: its settings expand to a chain of steps, each an evaluator, a check that
# judges one, or a transform. Without an extractor, the steps that embed the images,
# `coverage` and `completeness`, are skipped, and the steps that read labels and metadata
# run: `labels`, `summary`, `balance`, `diversity`, `gaps` (the gap analysis) and
# `worklist` (what each class lacks of an even spread).
#
# You will evaluate intrinsic image factors (such as brightness, contrast, and
# sharpness) as metadata conditions. The gap analysis cross-tabulates class labels against
# binned factors to detect whether particular vehicle classes were imaged under
# limited operational conditions.
#
# `checks` is keyed by the type of the check it sets: `class-imbalance` judges
# the label distribution, and `factor-coverage-gaps` the gap analysis.

# %%
from dataeval_flow import PipelineConfig, run_task
from dataeval_flow.config import DatasetProtocolConfig, MetadataPolicyConfig, SourceConfig, TaskConfig
from dataeval_flow.workflows.data_coverage import DataCoverageConfig

vehicle_factors = MetadataPolicyConfig(
    name="vehicle_factors",
    # Nothing in this dataset's own per-sample metadata is a collection condition:
    # `id` is unique per frame (so it looks perfectly class-predictive) and height and
    # width describe the crop, not the scene. Ask for measured image statistics instead.
    intrinsic_factors=["visual", "pixel"],
    exclude=["id", "height", "width"],
    # Explicitly declared bins keep factor cuts consistent across runs.
    continuous_factor_bins={
        "brightness": 5,
        "contrast": 5,
        "darkness": 5,
        "entropy": 5,
        "kurtosis": 5,
        "mean": 5,
        "sharpness": 5,
        "skew": 5,
        "std": 5,
        "var": 5,
        "zeros": 5,
    },
)

metadata_only_workflow = DataCoverageConfig.model_validate(
    {
        "name": "coverage-metadata-only",
        "metadata": "vehicle_factors",
        "factor-gaps": {"min_representation": 5},  # Flag class-factor-value combos with < 5 samples
        "diversity": {"method": "simpson"},
        "checks": {
            "class-imbalance": {"warning": 3.0},  # Catch moderate class imbalance
            "factor-coverage-gaps": {"warning": 2},  # Warn if >= 2 gaps found
        },
    }
)

task_metadata = TaskConfig(
    name="vehicles-coverage-metadata",
    workflow="coverage-metadata-only",
    sources="vehicles_src",
    # No extractor for metadata-only pass
)

config_metadata = PipelineConfig(
    metadata=[vehicle_factors],
    datasets=[
        DatasetProtocolConfig(name="vehicles_collected", format="maite", dataset=collected),
    ],
    sources=[
        SourceConfig(name="vehicles_src", dataset="vehicles_collected"),
    ],
    workflows=[metadata_only_workflow],
    tasks=[task_metadata],
)

# %%
result_metadata = run_task(task_metadata, config_metadata, cache_dir=Path("./cache"))

# %% [markdown]
# ### Coverage report (metadata only)
#
# The report gives each finding a section: Class Imbalance, Factor Coverage Gaps and
# the Class Shortfall. The metadata summary, balance and diversity are report
# sections, not findings. Class Coverage and Dimensional Completeness are reported as
# not assessed, and the Steps table at the end says why each embedding step was skipped.

# %%
print(result_metadata.report())

# %% [markdown]
# ### Drill into each step's output
#
# The result is a `ChainResult`. `result.steps` holds each step's output by step name,
# for programmatic inspection. The `labels` step's output is a `label-health` count of
# every class the dataset declares, at 0 where it has no labels.

# %%
label_health = result_metadata.steps["labels"].output.data()
print(f"Number of classes: {label_health['class_count']}")
print(f"Empty images: {label_health['empty_image_count']}")
print("\nClass distribution (five largest, five smallest):")
by_size = sorted(label_health["label_counts_per_class"].items(), key=lambda x: x[1], reverse=True)
for cls, count in by_size[:5]:
    print(f"  {cls:<26} {count}")
print("  ...")
for cls, count in by_size[-5:]:
    print(f"  {cls:<26} {count}")

# %%
# Metadata gaps: evaluate if a class was only imaged under narrow conditions
gaps = result_metadata.steps["gaps"].output
print(f"Metadata coverage gaps: {len(gaps.gaps)}\n")

print("  Mutual information (class -> factor), five strongest:")
mi = sorted(gaps.mutual_information.items(), key=lambda x: -x[1])
for factor, value in mi[:5]:
    print(f"    {factor:<28} {value:.4f}")

print("\n  Gap details (first ten):")
for gap in gaps.gaps[:10]:
    print(
        f"    {gap.class_name:<14} x {gap.factor_name}={gap.factor_value}: "
        f"count={gap.class_count}, expected={gap.expected_count:.1f}, "
        f"deficit={gap.deficit:.0%}"
    )
if not gaps.gaps:
    print("    No metadata gaps detected.")

# %%
# The class worklist: each class short of an even spread over the declared classes
worklist = result_metadata.steps["worklist"].output
print(f"Total deficit: {worklist.total_deficit} labels\n")
for row in worklist.data().iter_rows(named=True):
    print(f"  {row['label']:>14}  {row['action']:<8} have {row['count']:>4}, want {row['target']:>4}")

# %% [markdown]
# ### Interpreting the findings
#
# In this output, the imbalance ratio (1.8:1) satisfies the configured threshold of 3.0.
# The Class Imbalance finding warns anyway, because four declared classes, the Air
# Defense types, have zero samples. `class-imbalance` takes its ratio over the classes with
# labels, and a declared class with none always warns.
#
# The gap analysis finds nothing. A factor is searched for gaps only when its mutual
# information with the class reaches `mi_threshold`, 0.1 by default, and the strongest
# here, `contrast`, scores 0.0027. The measured image statistics barely vary with the
# vehicle class in this sample, so Factor Coverage Gaps is `ok`.
#
# The worklist cell lists six classes short of an even spread over the 24 declared
# classes, 266 labels in all: the four Air Defense types have 0 of a target of 62, and
# `BTR-70` and `T-90` have 53.
#
# `ClassFilter` removed samples while leaving `index2label` intact in the dataset
# metadata. If a dataset loader drops unused category entries completely, count-based
# workflows cannot detect missing categories. To guarantee detection of missing
# classes regardless of dataset metadata formatting, you should declare an ontology.

# %% [markdown]
# ## Step 2: Judge the labels against the sanctioned label space
#
# The `label-space` preset compares a dataset's labels against a declared ontology, the
# full taxonomy specification. It runs on the same source as `data-coverage`, as a task
# of its own.
#
# You will load the hierarchy attribute from MilitaryVehicles as your ontology:

# %%
import json

print(json.dumps(MilitaryVehicles.hierarchy, indent=2)[:900] + "\n...")

# %% [markdown]
# The hierarchy defines `land vehicle` along with `watercraft` and `aircraft`.
# When evaluated against this taxonomy, unsampled concepts will be identified.

# %%
from dataeval_flow.workflows.label_space import LabelSpaceConfig

vehicle_ontology = MilitaryVehicles.hierarchy

vocab_workflow = LabelSpaceConfig(name="vocab", ontology=vehicle_ontology)

task_vocab = TaskConfig(
    name="vehicles-vocab",
    workflow="vocab",
    sources="vehicles_src",  # the source data-coverage read
)

config_vocab = PipelineConfig(
    datasets=config_metadata.datasets,
    sources=config_metadata.sources,
    workflows=[vocab_workflow],
    tasks=[task_vocab],
)

result_vocab = run_task(task_vocab, config_vocab, cache_dir=Path("./cache"))
print(result_vocab.report())

# %% [markdown]
# The `representation` step's output is DataEval's collection worklist against the
# ontology, one row per leaf concept short of its expected share. Its leaf coverage,
# total deficit and dark branches hang off it as attributes.

# %%
representation = result_vocab.steps["representation"].output
print(f"leaf coverage: {representation.leaf_coverage:.1%}")
print(f"total deficit: {representation.total_deficit} labels\n")

print("Dark branches: whole regions of the label space with zero examples:")
for branch in representation.dark_branches.iter_rows(named=True):
    print(f"  {branch['label']:<16} {branch['leaves']} leaf class(es), zero examples")

print("\nTop of the collection worklist:")
for row in representation.data().head(8).iter_rows(named=True):
    print(f"  {row['label']:>14}  {row['action']:<8} have {row['count']:>4}, want {row['target']:>4}")

# %% [markdown]
# ### What the ontology adds
#
# `label-space` makes four findings:
#
# - **Leaf Coverage**: Identifies leaf coverage (76.9%, 20 of the ontology's 26
#   leaves) and specific missing concepts. In addition to the four held-out Air Defense
#   classes, it surfaces `aircraft` and `watercraft` concepts defined in the taxonomy. It
#   warns: the coverage is under the default 0.9, and Air Defense is a wholly empty branch.
# - **Label Conformance**: Verifies whether all collected dataset classes exist in the
#   ontology. All 24 class names resolve to exactly one concept.
# - **Mergeability**: Shows mapping between dataset class names and ontology concepts,
#   providing relabeling rules where names differ. Here every class carries over one-to-one.
# - **Ontology Structure**: Reports concept hierarchy size, leaf counts, and depth: 34
#   concepts, 26 leaves, depth 4, and one single-child link, which it notes.
#
# The two worklists differ in what they spread the 1,500 labels over. `data-coverage`'s
# Class Shortfall spreads them over the 24 classes the dataset declares, a target of
# 62 each, 266 labels short. `label-space` spreads them over the ontology's 26 leaves, a
# target of 58 each, 358 labels short, and so adds `aircraft` and `watercraft`.

# %% [markdown]
# ### Read the worklist with scope in mind
#
# The report flags `Air Defense` as a dark branch (4 leaves, 0 samples). This represents
# an in-scope gap that requires targeted data collection.
#
# The report also flags `aircraft` and `watercraft`. For a ground-vehicle system,
# these categories represent ontology concepts outside operational scope. You should
# scope your ontology to match operational requirements so health thresholds track valid
# system targets.

# %% [markdown]
# ### When a label does not reconcile
#
# Label conformance performs exact matching against ontology concepts. Any misspellings
# or unsanctioned classes are flagged as unmatched:

# %%
from dataeval import Ontology
from dataeval.core import label_reconciliation

# `label-space` builds this from its `ontology` field. You can also construct the
# object directly to reconcile arbitrary label lists against it.
ontology_obj = Ontology.from_hierarchy(vehicle_ontology)
check = label_reconciliation(["T-72", "BTR-80", "T72", "technical"], ontology_obj)
print("matched:  ", dict(check["matched"]))
print("unmatched:", list(check["unmatched"]))

# %% [markdown]
# `T-72` and `BTR-80` match successfully. `T72` fails due to the missing hyphen,
# and `technical` fails because it is absent from the ontology.

# %% [markdown]
# ### Sharing one ontology across tasks
#
# You can define ontologies centrally under `ontologies:` in YAML and reference them
# by name. The two presets run as two tasks on one source:
#
# ```yaml
# ontologies:
#   - name: vehicles
#     source: config/vehicles.jsonld
#     concepts:
#       - id: http://example.org/vehicles#technical
#         label: technical
#         synonyms: [pickup-mounted, NSV]
#
# workflows:
#   - name: vocab
#     type: label-space
#     ontology: vehicles
#   - name: coverage
#     type: data-coverage
#     metadata: vehicle_factors
#
# tasks:
#   - name: vehicles-vocab
#     workflow: vocab
#     sources: vehicles_src
#   - name: vehicles-coverage
#     workflow: coverage
#     sources: vehicles_src
# ```

# %% [markdown]
# ## Step 3: Run coverage with an extractor (full analysis)
#
# You can add an extractor to evaluate embedding coverage and dimensional completeness.
#
# Embedding coverage identifies low-density or uncovered regions in feature space.
# Dimensional completeness evaluates how uniformly samples span embedding dimensions.
#
# You will configure a BoVW extractor with a 256-word vocabulary. Note that `isotropy`
# requires more samples per class than embedding dimensions. For classes with fewer
# than 256 samples, isotropy reports `null`, while `dispersion` and `near_duplicate_fraction`
# evaluate fully.
#
# The `coverage` settings are the `coverage` step's. `dimensional-completeness` judges the
# completeness step, and `class-coverage` judges each class's coverage.

# %%
from dataeval_flow.config.extractors import BoVWExtractorConfig

full_workflow = DataCoverageConfig.model_validate(
    {
        "name": "coverage-full",
        "metadata": "vehicle_factors",
        "coverage": {
            "method": "adaptive",
            "percent": 0.01,  # adaptive: flag the sparsest 1% of observations
            "num_observations": 50,  # Number of neighbors for coverage analysis
        },
        "factor-gaps": {"min_representation": 5},
        "diversity": {"method": "simpson"},
        "checks": {
            "dimensional-completeness": {"warning": 0.5},  # Warn if completeness < 0.5
            "class-imbalance": {"warning": 3.0},
            "factor-coverage-gaps": {"warning": 2},
        },
    }
)

task_full = TaskConfig(
    name="vehicles-coverage-full",
    workflow="coverage-full",
    sources="vehicles_src",
    extractor="bovw_ext",
)

config_full = PipelineConfig(
    seed=0,
    metadata=config_metadata.metadata,
    datasets=config_metadata.datasets,
    sources=config_metadata.sources,
    extractors=[
        BoVWExtractorConfig(name="bovw_ext", vocab_size=256, batch_size=32),
    ],
    workflows=[full_workflow],
    tasks=[task_full],
)

# %%
result_full = run_task(task_full, config_full, cache_dir=Path("./cache"))

# %% [markdown]
# ### Full coverage report
#
# Now the report includes Class Coverage and Dimensional Completeness in addition
# to the label and metadata findings. The label and metadata evidence repeats Step 1's,
# so this cell prints the short form, `report(detailed=False)`: the summary of findings,
# the health and the Steps table. `report()` gives the full evidence; the cells below read
# it from the steps.

# %%
print(result_full.report(detailed=False))

# %% [markdown]
# ### Inspect embedding results
#
# The `coverage` step's output is DataEval's `Coverage` output: a table with one row per
# class, and the indices of the uncovered items. The method is the entry's setting.

# %%
import polars as pl

coverage = result_full.steps["coverage"].output
per_class = coverage.data()
total = int(per_class["count"].sum())
uncovered = len(coverage.uncovered_indices)
print(f"Method: {full_workflow.coverage.method}")
print(f"Uncovered: {uncovered} of {total} ({uncovered / total:.1%})\n")

columns = ["class", "count", "uncovered", "dispersion", "isotropy", "near_duplicate_fraction"]
with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, float_precision=2):
    print(per_class.select(columns).sort("dispersion"))

# %% [markdown]
# ### Reading the per-class columns
#
# You should evaluate the per-class embedding metrics:
#
# - **dispersion**: Class variance relative to average class variance. Values near 1.0
#   indicate typical spread; low values indicate tight clustering.
# - **isotropy**: Directional variance across embedding dimensions. Low values indicate
#   variation concentrated along few principal axes.
# - **near_duplicate_fraction**: Proportion of samples in near-identical pairs.
#
# Across all 20 classes, dispersion is balanced (0.96 to 1.03) and near-duplicate fractions
# are 0.00, showing healthy feature-space coverage for sampled classes. Isotropy is `null`
# for every class, since none has more than 256 samples.
#
# Adaptive coverage flags its `percent` of the items by construction, so 15 of 1,500
# uncovered (1.0%) is the setting at work rather than a finding. The 15 fall in 12 classes,
# no more than 2 in any one. The Class Coverage finding informs while any item is
# uncovered, and warns only on a clustered, one-dimensional or duplicate-padded class:
# there is none.

# %%
completeness = result_full.steps["completeness"].output.data()
print("Dimensional Completeness:")
print(f"  Score: {completeness['completeness']:.3f}")
print(f"  Nearest neighbor pairs: {len(completeness['nearest_neighbor_pairs'])}")

# %% [markdown]
# Dimensional completeness is 0.573, between the warning band (0.5) and the info band
# (0.8), so its finding informs. On visual variation, this sample shows no weakness the
# default thresholds warn on: its gap is in the taxonomy.

# %% [markdown]
# ## Step 4: Tune health thresholds
#
# Health thresholds control when findings escalate from `info` to
# `warning`. Each preset keys its `checks` by check type. The right
# thresholds depend on your domain:
#
# | Preset | Check | Field | Default | Safety-critical | Web-scraped data |
# |---|---|---|---|---|---|
# | `data-coverage` | `uncovered-items` | `warning` | 10% | 3–5% | 15–20% |
# | `data-coverage` | `dimensional-completeness` | `warning` | 0.5 | 0.7–0.8 | 0.3–0.4 |
# | `data-coverage` | `class-imbalance` | `warning` | 5:1 | 2–3:1 | 10–20:1 |
# | `data-coverage` | `factor-coverage-gaps` | `warning` | 3 | 1 | 5–10 |
# | `data-coverage` | `class-coverage` | `dispersion` | 0.5 | 0.7 | 0.3 |
# | `data-coverage` | `class-coverage` | `isotropy` | 0.5 | 0.7 | 0.3 |
# | `data-coverage` | `class-coverage` | `near_duplicates` | 0.1 | 0.02 | 0.25 |
# | `label-space` | `leaf-coverage` | `coverage` | 0.9 | 0.95 | 0.6 |
# | `label-space` | `leaf-coverage` | `empty_branches` | 0 | 0 | 2–5 |
# | `label-space` | `label-conformance` | `warning` | 0 | 0 | 3–10 |
#
# `class-imbalance` and `dimensional-completeness` also take `info`, the band between `ok` and
# a warning: a ratio over 2.0, or a score under 0.8, informs by default. `null` turns a
# threshold off.
#
# `uncovered-items` judges only `naive` coverage, since adaptive coverage flags its
# `percent` of the items by construction. The label-space thresholds go on the
# `label-space` entry, as `LabelSpaceConfig(..., checks={"leaf-coverage": {"coverage": 0.95}})`.

# %%
from dataeval_flow.workflows.data_coverage import DataCoverageChecks

strict_thresholds = DataCoverageChecks.model_validate(
    {
        "dimensional-completeness": {"warning": 0.6},
        "class-imbalance": {"warning": 2.0},
        "factor-coverage-gaps": {"warning": 1},
        "class-coverage": {"dispersion": 0.7, "isotropy": 0.7, "near_duplicates": 0.02},
    }
)

strict_workflow = full_workflow.model_copy(
    update={"name": "coverage-strict", "checks": strict_thresholds},
)

task_strict = TaskConfig(
    name="vehicles-coverage-strict",
    workflow="coverage-strict",
    sources="vehicles_src",
    extractor="bovw_ext",
)

config_strict = PipelineConfig(
    seed=0,
    metadata=config_full.metadata,
    datasets=config_full.datasets,
    sources=config_full.sources,
    extractors=config_full.extractors,
    workflows=[full_workflow, strict_workflow],
    tasks=[task_strict],
)

result_strict = run_task(task_strict, config_strict, cache_dir=Path("./cache"))
print(result_strict.report(detailed=False))

# %% [markdown]
# Compare each finding's severity under the two sets of thresholds:

# %%
for default, strict in zip(result_full.findings, result_strict.findings, strict=True):
    print(f"{default.title:<28} {default.severity:<8} -> {strict.severity:<8} {strict.brief}")

# %% [markdown]
# Exactly one additional warning triggers: Dimensional Completeness (0.573) falls below
# the strict 0.6 threshold. The other strict limits change nothing: no class's dispersion
# is under 0.7, every class's near-duplicate fraction prints as 0.00, isotropy is not
# measured, there are no gaps to count, and Class Imbalance warns already for its empty
# classes.
#
# You can adjust individual health thresholds to match your domain tolerance without
# altering underlying data calculations.

# %% [markdown]
# ## Results Exploration: Export results

# %%
json_str = result_full.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:500] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Simulate missing categories using `ClassFilter` view operations.
# - Run `data-coverage` without an extractor to assess class balance and cross-tabulated factor gaps.
# - Run `label-space` on the same source to benchmark the labels against an ontology and name missing categories.
# - Reconcile label names against an ontology.
# - Configure feature extractors to evaluate embedding dispersion and dimensional completeness.
# - Read each step's output from `result.steps`.
# - Tune health thresholds to enforce strict domain requirements.
# - Export structured coverage results to JSON.
#
# You should assess both label completeness against an ontology and feature diversity
# in embedding space to ensure comprehensive dataset coverage.

# %% [markdown]
# ## Next steps
#
# - **Data cleaning**: Use {doc}`Clean a dataset <data_cleaning>` to detect outliers and duplicates.
# - **Drift monitoring**: Use {doc}`Monitor incoming data for drift <drift_monitoring>` to track
#   operational distribution shifts.
# - **Targeted prioritization**: Feed coverage gap targets into {doc}`Prioritize unlabeled data <data_prioritization>`
#   to prioritize acquisition of missing concepts.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Dataset coverage](../concepts/Coverage.md) covers label-space
#   and embedding-space coverage theory.
# - **How-to**: [Declare an ontology](../how_to/declare_an_ontology.md) explains
#   defining label spaces via inline dictionaries or SKOS/OWL files, and what `label-space` finds.
# - **How-to**: [Build dataset views](../how_to/build_dataset_views.md) covers
#   view operations such as `ClassFilter`, `Shuffle`, and `Limit`.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) explains
#   running coverage tasks in Docker.
# - **Guide**: [Use an ONNX model for embeddings](onnx_embeddings) covers configuring
#   pretrained ONNX models for coverage feature extraction.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) details
#   how to parse reports and export envelopes.
