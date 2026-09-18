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
# **Who this is for** — T&E engineers and data scientists who need to establish
# whether a dataset actually spans the operational conditions a model will face,
# before that dataset is used to train or to certify.
#
# **Where this fits** — Coverage assessment belongs at the front of the data
# pipeline, alongside [data cleaning](data_cleaning). Cleaning asks whether the
# data you collected is *sound*; coverage asks whether it is *complete* — whether
# any sanctioned class, metadata condition, or region of the embedding space was
# never collected at all. Gaps found here drive targeted collection, and they
# bound how much a downstream [drift](drift_monitoring) or
# [OOD](ood_detection) baseline can be trusted: a reference set with a blind spot
# cannot flag drift into it. See the
# [Dataset coverage](../concepts/Coverage.md) concept page for the ideas behind
# the checks.

# %% [markdown]
# ## What you'll do
#
# - Load the **MilitaryVehicles** ground-vehicle dataset and simulate a
#   collection cycle that missed an entire vehicle category, using a
#   `ClassFilter` view operation
# - Run the `data-coverage` workflow **without** an extractor to surface
#   label and metadata issues quickly
# - Find that the collection looks **healthy by count** — and see why that
#   is not the same as being complete
# - Declare an **ontology** — and use the taxonomy the dataset already ships
#   to name the categories raw counts never could
# - Re-run **with** a BoVW extractor to add per-class embedding variety
#   signals and dimensional completeness analysis
# - Read the built-in **coverage report** and drill into raw results
# - Tune **health thresholds** to control when findings become warnings

# %% [markdown]
# ## What you'll learn
#
# - Why a well-balanced class distribution is **not** evidence of coverage,
#   and what question it actually answers
# - How to configure and run the `data-coverage` workflow via `run_task()`
# - The two orthogonal axes coverage measures: **which categories you
#   have** — your labels checked against an ontology — and **how varied
#   each one is** — your embeddings checked for clustering, low
#   dimensionality, and duplication
# - Why a class with zero examples is invisible to counts but visible to
#   an ontology, and why a class can be "well represented" by count and
#   still be embedding-collapsed
# - That an **empty concept is not automatically a defect** — sometimes it
#   tells you the ontology is broader than the collection's scope, which is
#   a different finding with a different response
# - That whether a count-based check can name a missing class depends on
#   whether your loader still declares it
# - How to adjust health thresholds for different risk tolerances

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `maite-datasets` (to download MilitaryVehicles)
# - Internet connection (first run only — the dataset is cached under `./data`)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: a collection with a hole in it
#
# You are standing up a ground-vehicle recognition capability. The label space
# is not "whatever we happened to photograph" — it is a **taxonomy**, agreed in
# advance, of the vehicle types the system is required to recognize.
#
# MilitaryVehicles ships exactly that: a `hierarchy` attribute describing
# tanks, BMPs, BTRs, self-propelled artillery, air defense systems and two
# one-off types, all nested under `land vehicle`, with `watercraft` and
# `aircraft` declared alongside. That is a real ontology, versioned with the
# dataset, rather than one invented for a tutorial.
#
# To simulate a collection cycle that went wrong, we drop the whole **Air
# Defense** category with a `ClassFilter` view operation. Everything else about
# the data — the class counts, the imagery, the imbalance — stays exactly as
# collected.

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

# `ClassFilter` is what enforces the gap — it reads each frame's own label, so nothing
# here re-derives the label space by hand. `Shuffle` then `Limit` samples the result to
# keep the tutorial quick; coverage findings are about *which categories exist*, which a
# random sample preserves.
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
# The distribution is **fine**. Largest to smallest is under 2:1 — comfortably
# inside any reasonable imbalance threshold. Nothing about these bars is a
# warning, and a collection this even is not what a broken pipeline looks like.
#
# The chart also cannot show what is not there: four sanctioned vehicle types
# have no bar, because a chart plotted from the data has no row for a category
# the data contains none of. Whether anything *else* can name them turns out to
# depend on a detail we come back to in Step 1.

# %% [markdown]
# ## Step 1: Run coverage without an extractor (metadata only)
#
# The `data-coverage` workflow can run **without** an extractor for a
# fast first pass. This skips embedding coverage and completeness but
# still analyzes label distribution, metadata distribution, metadata
# gaps, and — since no `ontology` is configured yet — a class balance
# worklist synthesized from the dataset's own declared classes.
#
# For metadata we use **intrinsic factors**: statistics computed from the
# imagery itself (brightness, contrast, sharpness, and so on). This dataset
# carries no collection metadata of its own, and rather than invent some, we
# measure what is actually in the frames. Gap analysis then cross-tabulates
# class against binned factor values, which answers a real question: *was any
# vehicle type only ever imaged under one set of conditions?*

# %%
from dataeval_flow.config import PipelineConfig, SourceConfig
from dataeval_flow.config.schemas import (
    DataCoverageTaskConfig,
    DataCoverageWorkflowConfig,
    DatasetProtocolConfig,
    MetadataPolicyConfig,
)
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.coverage.params import DataCoverageHealthThresholds

vehicle_factors = MetadataPolicyConfig(
    name="vehicle_factors",
    # Nothing in this dataset's own per-sample metadata is a collection condition:
    # `id` is unique per frame (so it looks perfectly class-predictive) and height and
    # width describe the crop, not the scene. Ask for measured image statistics instead.
    intrinsic_factors=["visual", "pixel"],
    exclude=["id", "height", "width"],
    # Declare the bins rather than letting them be inferred. Auto-binning derives the bin
    # count from the data, so the same factor measured on two samples is not comparable —
    # fine for a one-off look, wrong for anything you intend to track over time.
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

metadata_only_workflow = DataCoverageWorkflowConfig(
    name="coverage-metadata-only",
    metadata="vehicle_factors",
    run_gap_analysis=True,
    gap_min_representation=5,  # Flag class-factor-value combos with < 5 samples
    balance=True,
    diversity_method="simpson",
    health_thresholds=DataCoverageHealthThresholds(
        class_imbalance_ratio=3.0,  # Tight — we want to catch even moderate imbalance
        gap_count=2,  # Warn if >= 2 gaps found
    ),
)

task_metadata = DataCoverageTaskConfig(
    name="vehicles-coverage-metadata",
    workflow="coverage-metadata-only",
    sources="vehicles_src",
    # No extractor — metadata-only pass
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
# metadata gaps, and a class balance worklist. Since we didn't configure
# an extractor, embedding coverage and completeness are skipped.

# %%
print(result_metadata.report())

# %% [markdown]
# ### Drill into raw results
#
# The `result.data.raw` object provides machine-readable access to every
# metric for programmatic inspection.

# %%
raw = result_metadata.data.raw

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
# Metadata gaps — where a class was only ever imaged under some conditions
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
# Two things are worth stopping on.
#
# **The imbalance check passes and the label distribution warns anyway.** The
# ratio is 1.8:1, nowhere near the 3:1 we configured. The warning comes from a
# different clause of the same finding — *4 declared class(es) have zero
# samples* — naming all four Air Defense systems.
#
# That deserves scrutiny rather than applause, because it depends on how the data
# got here. `ClassFilter` removes *samples*; it does not rewrite the dataset's
# `index2label`. This view therefore still declares twenty-four classes and
# reports twenty of them with counts. A collection that genuinely never captured
# those systems — no folder, no label entry — would declare twenty classes, and
# there would be nothing left for this finding to name. **The workflow can report
# an empty class only while something still declares it.** Getting this warning
# for free is a property of your loader, not of your collection.
#
# **No metadata gaps.** Gap analysis cross-references class labels against binned
# intrinsic factors and looks for combinations under-represented relative to an
# even spread. Nothing cleared the default mutual-information threshold, and that
# is the honest answer: measured image statistics are not strongly
# class-predictive in this dataset, so there is no "this vehicle type was only
# ever imaged in bright conditions" story to tell. A null result is a result.
#
# So the classes are balanced, the imagery is not conditioned on class, and the
# one warning we did get arrived by luck of the loader. Step 2 asks the question
# that does not depend on luck.

# %% [markdown]
# ## Step 2: Declare the sanctioned label space
#
# A recognition system is specified against a taxonomy, not against a
# collection. Writing that taxonomy down as an **ontology** is what lets the
# workflow name a class that was never collected at all.
#
# MilitaryVehicles already carries one, so there is nothing to invent:

# %%
import json

print(json.dumps(MilitaryVehicles.hierarchy, indent=2)[:900] + "\n...")

# %% [markdown]
# Note `watercraft` and `aircraft` sitting alongside `land vehicle` with nothing
# under them. Hold on to that — it matters in a moment.

# %%
vehicle_ontology = MilitaryVehicles.hierarchy

ontology_workflow = metadata_only_workflow.model_copy(
    update={"name": "coverage-ontology", "ontology": vehicle_ontology},
)

task_ontology = DataCoverageTaskConfig(
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
onto = result_ontology.data.raw.ontology
print(f"source: {onto.source} (synthesized={onto.synthesized})")
print(f"leaf coverage: {onto.representation.leaf_coverage:.0%}")
print(f"total deficit: {onto.representation.total_deficit} labels\n")

print("Dark branches — whole regions of the label space with nothing in them:")
for branch in onto.representation.dark_branches:
    print(f"  {branch.label:<16} {branch.leaves} leaf class(es), zero examples")

print("\nTop of the collection worklist:")
for row in onto.representation.worklist[:8]:
    print(f"  {row.label:>14}  {row.action:<8} have {row.count:>4}, want {row.target:>4}")

# %% [markdown]
# ### What the ontology adds
#
# Four new sections replace the Class Balance Worklist:
#
# - **Label Space Coverage** — 76.9% leaf coverage, six concepts to acquire, a
#   deficit of 358 frames. Four of the six are the Air Defense systems Step 1
#   also named. **The other two are the point**: `aircraft` and `watercraft` are
#   sanctioned by the taxonomy and were never in this dataset's vocabulary at
#   all — no label index, no folder, nothing for a count to notice. No amount of
#   reading the data surfaces them; only a declared label space does.
# - **Label Conformance** — every collected class name resolves to a concept,
#   so the label set conforms. Nothing was captured that the taxonomy does not
#   sanction.
# - **Label Alignment** — the collected names already *are* the ontology's leaf
#   names, so alignment is lossless. Conformance reports whether a name
#   resolves; alignment reports what it maps to, and provides a `Relabel` stanza
#   you can paste into a view to conform a dataset whose spellings differ.
# - **Ontology Structure** — the artifact itself: 34 concepts, 26 leaves,
#   depth 4, no collisions.
#
# The worklist turns that into a collection order: six `acquire` rows at 58
# frames each, then `augment` rows for the thinnest classes actually collected.

# %% [markdown]
# ### Read the worklist with scope in mind
#
# One branch comes back dark: `Air Defense`, four leaves, zero examples. That is
# a real finding. It sits under `land vehicle` alongside the categories we did
# collect, it was in scope, and it is missing. `dark_branches` rolls missing
# leaves up to the highest wholly-empty concept, so this is reported once rather
# than four times.
#
# `aircraft` and `watercraft` are a different matter. They are leaves in their
# own right rather than branches, so they do not appear in the dark-branch
# rollup — they appear in the worklist, as `acquire` rows wanting 58 frames
# each. Acquiring them would be absurd: this is a *ground-vehicle* collection.
# They are empty because the taxonomy is broader than the collection's scope,
# not because anything went wrong.
#
# The workflow cannot tell those apart for you. Scope is your knowledge, not the
# data's. What it buys you is that the question gets asked at all — six named
# concepts with nothing in them, each one either a gap to fill or a scope
# boundary to write down and stop re-litigating.
#
# It is also the argument for keeping the ontology no broader than what you
# intend to field. Rooted at `land vehicle`, this collection would be scored
# against concepts it is actually responsible for, and `dark_branch_count=0`
# would be a meaningful health threshold rather than a permanent warning.

# %% [markdown]
# ### When a label does not reconcile
#
# Conformance catches the opposite problem: a class name the ontology does not
# sanction. Reconciliation is exact, not fuzzy, so a typo or an unsanctioned
# class shows up as **unmatched**.

# %%
from dataeval import Ontology
from dataeval.core import label_reconciliation

# The workflow builds this for you from the `ontology` field; here we construct
# the same object directly so we can reconcile an arbitrary label list against it.
ontology_obj = Ontology.from_hierarchy(vehicle_ontology)
check = label_reconciliation(["T-72", "BTR-80", "T72", "technical"], ontology_obj)
print("matched:  ", dict(check["matched"]))
print("unmatched:", list(check["unmatched"]))

# %% [markdown]
# `T-72` and `BTR-80` resolve. `T72` does not — the hyphen matters, and exact
# reconciliation is the point: a silent fuzzy match on vehicle designations is
# exactly the kind of convenience that turns a labelling error into a training
# set. `technical` does not resolve either, because the taxonomy does not
# sanction it.

# %% [markdown]
# ### Sharing one ontology across workflows
#
# The example above passes the ontology inline, which is the simplest form. When several
# workflows read the same label space, define it once under `ontologies:` and reference it
# by name so the definitions cannot drift apart:
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
#
# Use `concepts:` to add concepts on top of `source` when you need to extend an artifact you
# do not own. Give each declared concept the dataset's own spelling under `synonyms`:
# alignment matches on labels and synonyms, and a concept without the dataset's spelling
# will not match that class. Flow reads a workflow's `ontology:` value as a name first and
# as a path second, so configurations that name a file keep working.

# %% [markdown]
# ## Step 3: Run coverage with an extractor (full analysis)
#
# Now we add a **BoVW** (Bag of Visual Words) extractor to unlock
# embedding-based analyses: **embedding coverage** and **dimensional
# completeness**. These tell us whether the dataset's feature space has
# blind spots that metadata alone can't reveal.
#
# Embedding coverage checks whether the feature space has uncovered
# regions — areas where the model would encounter inputs unlike anything
# in the training data. Dimensional completeness measures how
# effectively the data explores the embedding dimensions.
#
# We use a 256-word vocabulary — the smallest `vocab_size` the config schema
# allows (`ge=256`). `isotropy` — how many independent directions a class varies
# in — is only defined when a class has more samples than embedding dimensions,
# and our sampled classes sit below 256, so it reports `null` throughout. That
# is a real result, not a failure: the classes are too small relative to BoVW's
# minimum embedding width for shape to be measurable. `dispersion` and
# `near_duplicate_fraction` only require `min_class_samples` (default 20) and
# are unaffected — they are the columns doing the work here.

# %%
from dataeval.config import set_max_processes

from dataeval_flow.config import BoVWExtractorConfig

set_max_processes(8)

full_workflow = DataCoverageWorkflowConfig(
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

task_full = DataCoverageTaskConfig(
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
# The dataset-wide uncovered count is a triage shortlist — with
# `coverage_method="adaptive"` it returns exactly `coverage_percent` of
# observations, so its *rate* restates your config rather than measuring your
# data. 15 of 1,500, at the 1% we asked for. The per-class columns are the part
# that depends on what you collected:
#
# - **dispersion** — how far the class spreads, relative to a typical
#   class. Around 1 is normal; well below means **clustered**.
# - **isotropy** — in how many independent directions it spreads. Low
#   means **one-dimensional** even when dispersion looks fine.
# - **near_duplicate_fraction** — the share sitting in near-identical
#   pairs. High means **padded with repeated frames**.
#
# None of them fire. Dispersion sits between 0.97 and 1.01 across all twenty
# classes — which is what "relative to a typical class" looks like when every
# class is typical. `near_duplicate_fraction` is 0.00 everywhere: nothing here
# is padded with repeated frames. `isotropy` is `null` throughout, for the
# sample-count reason above.
#
# That is a clean bill of health, and it is worth stating plainly rather than
# hunting for something to flag. This collection has no embedding pathology. Its
# problem lies entirely on the other axis — the categories that are not in it.

# %% [markdown]
# ### Inspect embedding results

# %%
raw_full = result_full.data.raw

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
# All ten are required numbers — there is no `None` to switch one off. To
# stop a metric from ever warning, set it past what the data can reach
# (`uncovered_rate=100.0`, `leaf_coverage=0.0`, and so on).
#
# `uncovered_rate` applies only when `coverage_method="naive"`. The three
# label-space thresholds apply only when an `ontology` is configured —
# against a synthesized one they would be scoring their own construction.
# Because `coverage-full` carries the real ontology forward, they bite here.

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

task_strict = DataCoverageTaskConfig(
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
# Exactly one more finding escalates: **Dimensional Completeness**, at 0.578, is
# informational against the default 0.5 and a warning against the strict 0.6.
# Two warnings become three.
#
# One finding, not a cascade — and that is the useful observation. Tightening a
# threshold changes an outcome only when the measured value sits near it.
# `class_imbalance_ratio` went from 3.0 to 2.0 and still does not fire, because
# the data is at 1.8. `max_near_duplicate_fraction` went from 0.1 to 0.02 and
# still does not fire, because the data is at 0.00. Thresholds are how you tune
# sensitivity to a metric you are already measuring; they are not a way to make
# a count-based check notice a category that is missing entirely. Only the
# ontology does that.

# %% [markdown]
# ## Results Exploration: Export results

# %%
json_str = result_full.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:500] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial you learned how to:
#
# - **Simulate a collection gap** with a `ClassFilter` view operation, leaving
#   everything else about the data exactly as collected
# - **Run coverage without an extractor** for a fast metadata-only pass that
#   checks class balance and cross-tabulates classes against measured image
#   statistics
# - **Recognize a clean bill of health for what it is** — a well-balanced class
#   distribution answers "are the classes I have evenly represented?", which is
#   a different question from "did I collect the right ones?"
# - **Understand why an empty class is sometimes visible to a count** — because
#   the label map still declares it — and why that is luck rather than coverage
# - **Declare an ontology** — here, the taxonomy the dataset already ships — and
#   see `Label Space Coverage` name concepts the dataset's own vocabulary never
#   contained
# - **Read the worklist with scope in mind**, distinguishing a genuine
#   collection gap from an ontology broader than the collection's remit
# - **Add an extractor** to unlock per-class embedding signals
#   (dispersion, isotropy, near-duplicate fraction) and dimensional completeness
# - **Tune health thresholds** to match your domain's risk tolerance
#
# The result on this dataset is worth stating bluntly: **every check that looks
# inside the collection passes.** The classes are balanced at 1.8:1. No metadata
# gap clears the threshold. Dispersion is ~1.0 for all twenty classes and the
# near-duplicate fraction is zero. Tightening every threshold at once adds a
# single warning. By any within-data measure this is a healthy dataset.
#
# It is also missing 23% of its sanctioned label space, and that is the finding
# that matters. Coverage has **two orthogonal axes**. *How varied each category
# is* is an embeddings question, and this collection answers it well. *Which
# categories you have at all* is a labels-against-an-ontology question, and no
# amount of measuring the data you collected will answer it — you have to state
# what you were supposed to collect. Running `data-coverage` before training
# catches both kinds of blind spot early, when they're cheapest to fix.

# %% [markdown]
# ## What's next
#
# - **Data cleaning** — Use the `data-cleaning` workflow to flag
#   outliers and duplicates in your dataset before training
# - **Drift monitoring** — After deploying, use `drift-monitoring` to
#   detect when incoming data drifts away from your training distribution
# - **Targeted collection** — Use the worklist rows to drive the next
#   collection cycle: the four Air Defense systems, in the quantities the
#   `acquire` rows name
# - **Ship a real ontology** — Replace the inline hierarchy with a
#   versioned SKOS or OWL file (`ontology: ./taxonomy.ttl`), which also
#   carries synonyms, definitions and stable concept ids. Needs
#   `pip install "dataeval[ontology]"`.
# - **Targeted labeling** — Feed the gap findings into the
#   [prioritization workflow](data_prioritization) to rank unlabeled candidates
#   that fall in the under-covered regions

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Dataset coverage](../concepts/Coverage.md): the label-space
#   and embedding-space completeness ideas behind this workflow.
# - **How-to: Declare an ontology** — [Declare an ontology](../how_to/declare_an_ontology.md)
#   to define the sanctioned label space this workflow checks a dataset against,
#   inline or as a versioned SKOS/OWL file.
# - **How-to: Build dataset views** — [Build dataset views](../how_to/build_dataset_views.md)
#   for the `ClassFilter`, `Shuffle` and `Limit` operations used to shape the data here.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to build a container image, write a YAML config, and run this workflow with `docker run`.
# - **How-to: Use an ONNX model for embeddings** — [ONNX embeddings](onnx_embeddings)
#   to swap the BoVW extractor used here for a pretrained model with higher-fidelity embeddings.
# - **How-to: Read evaluation outputs** — [Read evaluation outputs](../how_to/read_evaluation_outputs.md)
#   to interpret the coverage report, its health severities, and the exported result envelope.
# - **How-to: Reuse results with the disk cache** — [Reuse results with the disk cache](../how_to/reuse_results_with_cache.md)
#   for the `cache_dir` used throughout this tutorial — what it stores and what invalidates it.
