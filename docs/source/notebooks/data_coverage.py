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
# embedding blind spots using the config-driven `data-coverage` workflow.

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
# - Execute `data-coverage` without an extractor for rapid label and metadata gap analysis.
# - Evaluate class balance metrics and identify why count-based metrics alone do not detect missing categories.
# - Define an ontology using the dataset taxonomy to expose unsampled concepts.
# - Re-run coverage with a BoVW extractor to evaluate embedding dispersion and dimensional completeness.
# - Inspect coverage reports and configure health thresholds.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute `data-coverage` with `run_task()`.
# - How coverage evaluates two orthogonal axes: taxonomic representation (ontology) and visual variation (embeddings).
# - Why count-based distributions fail to detect unsampled classes when loaders drop missing categories.
# - How to distinguish genuine collection gaps from intentional ontology scope boundaries.
# - How to adjust health thresholds for varying domain risk tolerances.

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
# The sample distribution shows an imbalance ratio under 2:1. Count-based checks
# indicate a balanced dataset, but cannot identify missing categories absent an
# external ontology definition.

# %% [markdown]
# ## Step 1: Run coverage without an extractor (metadata only)
#
# You can execute `data-coverage` without an extractor for a fast initial pass.
# This pass skips embedding coverage and dimensional completeness while evaluating
# label distributions, metadata distributions, and cross-tabulated metadata gaps.
#
# You will evaluate intrinsic image factors (such as brightness, contrast, and
# sharpness) as metadata conditions. Gap analysis cross-tabulates class labels against
# binned factors to detect whether particular vehicle classes were imaged under
# limited operational conditions.

# %%
from dataeval_flow import PipelineConfig, run_task
from dataeval_flow.config import DatasetProtocolConfig, MetadataPolicyConfig, SourceConfig, TaskConfig
from dataeval_flow.workflows.data_coverage import DataCoverageConfig, DataCoverageHealthThresholds

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

metadata_only_workflow = DataCoverageConfig(
    name="coverage-metadata-only",
    metadata="vehicle_factors",
    run_gap_analysis=True,
    gap_min_representation=5,  # Flag class-factor-value combos with < 5 samples
    balance=True,
    diversity_method="simpson",
    health_thresholds=DataCoverageHealthThresholds(
        class_imbalance_ratio=3.0,  # Catch moderate class imbalance
        gap_count=2,  # Warn if >= 2 gaps found
    ),
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
# The report summarizes label distribution, metadata distribution,
# metadata gaps, and a class balance worklist. When you run without
# an extractor, embedding coverage and completeness are skipped.

# %%
print(result_metadata.report())

# %% [markdown]
# ### Drill into raw results
#
# The `result.output.raw` object provides machine-readable access to every
# metric for programmatic inspection.

# %%
raw = result_metadata.output.raw

ld = raw.label_distribution
print(f"Number of classes: {ld.num_classes}")
print(f"Empty images: {len(ld.empty_images)}")
print("\nClass distribution (five largest, five smallest):")
by_size = sorted(ld.class_distribution.items(), key=lambda x: x[1], reverse=True)
for cls, count in by_size[:5]:
    print(f"  {cls:<26} {count}")
print("  ...")
for cls, count in by_size[-5:]:
    print(f"  {cls:<26} {count}")

# %%
# Metadata gaps: evaluate if a class was only imaged under narrow conditions
if raw.metadata_gaps and raw.metadata_gaps.gaps:
    print(f"Metadata coverage gaps: {len(raw.metadata_gaps.gaps)}\n")

    print("  Mutual information (class -> factor), five strongest:")
    mi = sorted(raw.metadata_gaps.mutual_info_class_to_factor.items(), key=lambda x: -x[1])
    for factor, value in mi[:5]:
        print(f"    {factor:<28} {value:.4f}")

    print("\n  Gap details (first ten):")
    for gap in raw.metadata_gaps.gaps[:10]:
        print(
            f"    {gap.class_name:<14} x {gap.factor_name}={gap.factor_value}: "
            f"count={gap.class_count}, expected={gap.expected_count:.1f}, "
            f"deficit={gap.deficit:.0%}"
        )
else:
    print("No metadata gaps detected.")

# %% [markdown]
# ### Interpreting the findings
#
# In this output, the imbalance ratio (1.8:1) satisfies the configured threshold.
# However, a warning appears because four declared classes have zero samples.
#
# `ClassFilter` removed samples while leaving `index2label` intact in the dataset
# metadata. If a dataset loader drops unused category entries completely, count-based
# workflows cannot detect missing categories. To guarantee detection of missing
# classes regardless of dataset metadata formatting, you should declare an ontology.

# %% [markdown]
# ## Step 2: Declare the sanctioned label space
#
# When you supply an ontology, the workflow compares dataset labels against the
# full taxonomy specification.
#
# You will load the hierarchy attribute from MilitaryVehicles as your ontology:

# %%
import json

print(json.dumps(MilitaryVehicles.hierarchy, indent=2)[:900] + "\n...")

# %% [markdown]
# The hierarchy defines `land vehicle` along with `watercraft` and `aircraft`.
# When evaluated against this taxonomy, unsampled concepts will be identified.

# %%
vehicle_ontology = MilitaryVehicles.hierarchy

ontology_workflow = metadata_only_workflow.model_copy(
    update={"name": "coverage-ontology", "ontology": vehicle_ontology},
)

task_ontology = TaskConfig(
    name="vehicles-coverage-ontology",
    workflow="coverage-ontology",
    sources="vehicles_src",
)

config_ontology = PipelineConfig(
    metadata=config_metadata.metadata,
    datasets=config_metadata.datasets,
    sources=config_metadata.sources,
    workflows=[ontology_workflow],
    tasks=[task_ontology],
)

result_ontology = run_task(task_ontology, config_ontology, cache_dir=Path("./cache"))
print(result_ontology.report())

# %%
onto = result_ontology.output.raw.ontology
print(f"source: {onto.source} (synthesized={onto.synthesized})")
print(f"leaf coverage: {onto.representation.leaf_coverage:.0%}")
print(f"total deficit: {onto.representation.total_deficit} labels\n")

print("Dark branches: whole regions of the label space with zero examples:")
for branch in onto.representation.dark_branches:
    print(f"  {branch.label:<16} {branch.leaves} leaf class(es), zero examples")

print("\nTop of the collection worklist:")
for row in onto.representation.worklist[:8]:
    print(f"  {row.label:>14}  {row.action:<8} have {row.count:>4}, want {row.target:>4}")

# %% [markdown]
# ### What the ontology adds
#
# Declaring an ontology provides four additional evaluation sections:
#
# - **Label Space Coverage**: Identifies leaf coverage (76.9%) and specific missing
#   concepts. In addition to the four held-out Air Defense classes, it surfaces
#   `aircraft` and `watercraft` concepts defined in the taxonomy.
# - **Label Conformance**: Verifies whether all collected dataset classes exist in the ontology.
# - **Label Alignment**: Shows mapping between dataset class names and ontology concepts,
#   providing relabeling rules where names differ.
# - **Ontology Structure**: Reports concept hierarchy size, leaf counts, and depth.

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

# The workflow builds this automatically from the `ontology` field. You can also
# construct the object directly to reconcile arbitrary label lists against it.
ontology_obj = Ontology.from_hierarchy(vehicle_ontology)
check = label_reconciliation(["T-72", "BTR-80", "T72", "technical"], ontology_obj)
print("matched:  ", dict(check["matched"]))
print("unmatched:", list(check["unmatched"]))

# %% [markdown]
# `T-72` and `BTR-80` match successfully. `T72` fails due to the missing hyphen,
# and `technical` fails because it is absent from the ontology.

# %% [markdown]
# ### Sharing one ontology across workflows
#
# You can define ontologies centrally under `ontologies:` in YAML and reference them
# by name across tasks:
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
#   - name: coverage-ontology
#     type: data-coverage
#     ontology: vehicles
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

# %%
from dataeval.config import set_max_processes

from dataeval_flow.config.extractors import BoVWExtractorConfig

set_max_processes(8)

full_workflow = DataCoverageConfig(
    name="coverage-full",
    metadata="vehicle_factors",
    ontology=vehicle_ontology,  # carry the label space forward
    coverage_method="adaptive",
    coverage_percent=0.01,  # adaptive: flag the sparsest 1% of observations
    num_observations=50,  # Number of neighbors for coverage analysis
    run_completeness=True,  # Measure dimensional completeness
    run_gap_analysis=True,
    balance=True,
    diversity_method="simpson",
    health_thresholds=DataCoverageHealthThresholds(
        uncovered_rate=10.0,  # Warn if > 10% uncovered in embedding space
        completeness_score=0.5,  # Warn if completeness < 0.5
        class_imbalance_ratio=3.0,
        gap_count=2,
    ),
)

task_full = TaskConfig(
    name="vehicles-coverage-full",
    workflow="coverage-full",
    sources="vehicles_src",
    extractor="bovw_ext",
)

config_full = PipelineConfig(
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
# Now the report includes embedding coverage and dimensional
# completeness in addition to label, ontology and metadata findings.

# %%
print(result_full.report())

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
# Across all 20 classes, dispersion is balanced (~1.0) and near-duplicate fractions are 0.00,
# showing healthy feature-space coverage for sampled classes.

# %% [markdown]
# ### Inspect embedding results

# %%
raw_full = result_full.output.raw

if raw_full.coverage:
    cov = raw_full.coverage
    print(f"Method: {cov.method}")
    print(f"Uncovered: {cov.uncovered_count} ({cov.uncovered_rate:.1%})\n")
    print(f"{'class':>14}  {'count':>5}  {'disp':>6}  {'iso':>6}  {'nearDup':>7}")

    def _fmt(v: float | None) -> str:
        return "  -   " if v is None else f"{v:6.2f}"

    for row in sorted(cov.per_class, key=lambda r: -(r.near_duplicate_fraction or 0.0)):
        print(
            f"{row.class_name:>14}  {row.count:>5}  {_fmt(row.dispersion)}  "
            f"{_fmt(row.isotropy)}  {_fmt(row.near_duplicate_fraction):>7}"
        )
elif raw_full.coverage_skipped_reason:
    print(f"Embedding coverage skipped: {raw_full.coverage_skipped_reason}")
else:
    print("No embedding coverage (extractor not configured)")

print()

if raw_full.completeness:
    comp = raw_full.completeness
    print("Dimensional Completeness:")
    print(f"  Score: {comp.completeness_score:.3f}")
    print(f"  Nearest neighbor pairs: {len(comp.nearest_neighbor_pairs)}")
else:
    print("No completeness data (extractor not configured)")

# %% [markdown]
# ## Step 4: Tune health thresholds
#
# Health thresholds control when findings escalate from `info` to
# `warning`. The right thresholds depend on your domain:
#
# | Metric | Default | Safety-critical | Web-scraped data |
# |---|---|---|---|
# | `uncovered_rate` | 10% | 3–5% | 15–20% |
# | `completeness_score` | 0.5 | 0.7–0.8 | 0.3–0.4 |
# | `class_imbalance_ratio` | 5:1 | 2–3:1 | 10–20:1 |
# | `gap_count` | 3 | 1 | 5–10 |
# | `min_dispersion` | 0.5 | 0.7 | 0.3 |
# | `min_isotropy` | 0.5 | 0.7 | 0.3 |
# | `max_near_duplicate_fraction` | 0.1 | 0.02 | 0.25 |
# | `leaf_coverage` | 0.9 | 0.95 | 0.6 |
# | `dark_branch_count` | 0 | 0 | 2–5 |
# | `unmatched_class_count` | 0 | 0 | 3–10 |
#
# All ten threshold parameters require numeric values. To prevent a metric from
# triggering a warning, set it beyond reachable ranges (`uncovered_rate=100.0`,
# `leaf_coverage=0.0`).
#
# `uncovered_rate` applies when `coverage_method="naive"`. Label-space thresholds
# apply when an explicit ontology is configured.

# %%
strict_thresholds = DataCoverageHealthThresholds(
    uncovered_rate=5.0,
    completeness_score=0.6,
    class_imbalance_ratio=2.0,
    gap_count=1,
    min_dispersion=0.7,
    min_isotropy=0.7,
    max_near_duplicate_fraction=0.02,
    leaf_coverage=0.95,
    dark_branch_count=0,
    unmatched_class_count=0,
)

strict_workflow = full_workflow.model_copy(
    update={"name": "coverage-strict", "health_thresholds": strict_thresholds},
)

task_strict = TaskConfig(
    name="vehicles-coverage-strict",
    workflow="coverage-strict",
    sources="vehicles_src",
    extractor="bovw_ext",
)

config_strict = PipelineConfig(
    metadata=config_full.metadata,
    datasets=config_full.datasets,
    sources=config_full.sources,
    extractors=config_full.extractors,
    workflows=[full_workflow, strict_workflow],
    tasks=[task_strict],
)

result_strict = run_task(task_strict, config_strict, cache_dir=Path("./cache"))
print(result_strict.report())

# %% [markdown]
# Exactly one additional warning triggers: Dimensional Completeness (0.578) falls below
# the strict 0.6 threshold.
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
# - Execute metadata-only coverage to assess class balance and cross-tabulated factor gaps.
# - Benchmark class counts against an ontology to detect missing categories.
# - Define and reconcile label taxonomies using ontologies.
# - Configure feature extractors to evaluate embedding dispersion and dimensional completeness.
# - Tune health thresholds to enforce strict domain requirements.
# - Export structured coverage results to JSON.
#
# You should assess both label completeness against ontologies and feature diversity
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
#   defining label spaces via inline dictionaries or SKOS/OWL files.
# - **How-to**: [Build dataset views](../how_to/build_dataset_views.md) covers
#   view operations such as `ClassFilter`, `Shuffle`, and `Limit`.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) explains
#   running coverage tasks in Docker.
# - **Guide**: [Use an ONNX model for embeddings](onnx_embeddings) covers configuring
#   pretrained ONNX models for coverage feature extraction.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) details
#   how to parse reports and export envelopes.
