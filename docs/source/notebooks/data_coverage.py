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
# - Load MNIST from HuggingFace and build a **deliberately biased** subset
#   that simulates a flawed data collection pipeline — including two
#   classes that were never collected at all
# - Attach synthetic metadata factors (lighting condition, collection
#   facility) with intentional gaps for certain classes
# - Run the `data-coverage` workflow **without** an extractor to surface
#   label and metadata issues quickly
# - Declare an **ontology** — the sanctioned label space — and see it
#   name classes that raw counts never could
# - Re-run **with** a BoVW extractor to add per-class embedding variety
#   signals and dimensional completeness analysis
# - Read the built-in **coverage report** and drill into raw results
# - Tune **health thresholds** to control when findings become warnings

# %% [markdown]
# ## What you'll learn
#
# - How uneven data collection creates coverage gaps that are invisible
#   until you explicitly measure them
# - How to configure and run the `data-coverage` workflow via `run_task()`
# - The two orthogonal axes coverage measures: **which categories you
#   have** — your labels checked against an ontology — and **how varied
#   each one is** — your embeddings checked for clustering, low
#   dimensionality, and duplication
# - Why a class with zero examples is invisible to counts but visible to
#   an ontology, and why a class can be "well represented" by count and
#   still be embedding-collapsed
# - The difference between running with and without an extractor, and
#   with and without an ontology
# - How to adjust health thresholds for different risk tolerances

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `datasets` (to download MNIST from HuggingFace Hub)
# - Internet connection (to download MNIST from HuggingFace Hub on first run)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Build a biased dataset
#
# Imagine you're building a digit recognition model for a postal sorting
# system. Your data comes from three sorting facilities:
#
# - **Facility A** — handles digits 0–3, well-staffed, collected plenty
#   of images under both bright and dim lighting
# - **Facility B** — handles digits 4–6, moderate collection effort
# - **Facility C** — handles digits 7–9, understaffed, and this
#   collection cycle only ever got digit 7 into the training set — 8
#   and 9 never showed up at all
#
# This kind of uneven collection is common in real-world ML pipelines.
# The result is a dataset with **class imbalance** (digit 7 is thin,
# digits 8 and 9 are absent), **metadata gaps** (dim lighting nearly
# absent for Facility C), and **poor embedding coverage** in the
# under-represented region.
#
# We'll simulate this by subsampling MNIST and attaching synthetic
# metadata.

# %% tags=["remove_output"]
from typing import cast

from datasets import Dataset
from datasets import load_dataset as hf_load

# Download MNIST test split (10 000 images)
mnist_test = cast(Dataset, hf_load("ylecun/mnist", split="test"))

print(f"Downloaded {len(mnist_test)} MNIST test images")

# %% [markdown]
# ### Subsample with collection bias
#
# We take different numbers of samples per digit class to mimic the
# uneven collection across the three facilities.
#
# datamaite has no in-memory constructor — every dataset comes from a
# filesystem loader — so we materialize the biased subset as an
# **ImageFolder** tree (`<root>/<label>/<seq>.png`) and read it back with
# `load_ic(..., dataset_format="huggingface_vision")`. That loader also
# supplies the `index2label` mapping the workflow uses to name classes.

# %%
import tempfile
from pathlib import Path

import numpy as np
from datamaite import load_ic

rng = np.random.default_rng(42)

# Per-class sample budgets — Facility C (digits 7–9) gets far fewer, and the
# collection never captured 8 or 9 at all.
samples_per_class = {
    0: 200,
    1: 200,
    2: 200,
    3: 200,  # Facility A — well-covered
    4: 120,
    5: 120,
    6: 120,  # Facility B — moderate
    7: 30,  # Facility C — under-represented
}

# Collect indices per class, then subsample
selected_indices: list[int] = []
labels_full = np.asarray(mnist_test["label"])

for cls, n in samples_per_class.items():
    cls_indices = np.where(labels_full == cls)[0]
    chosen = rng.choice(cls_indices, size=min(n, len(cls_indices)), replace=False)
    selected_indices.extend(chosen.tolist())

selected_indices.sort()

# Materialize the biased subset as an ImageFolder tree, then load it via datamaite
postal_root = Path(tempfile.mkdtemp(prefix="dataeval_flow_postal_"))
for seq, idx in enumerate(selected_indices):
    example = mnist_test[int(idx)]
    label_dir = postal_root / str(example["label"])
    label_dir.mkdir(parents=True, exist_ok=True)
    example["image"].save(label_dir / f"{seq:05d}.png")


# Two classes are plentiful by count but hollow in embedding space — the defects
# that per-class coverage catches and raw counts cannot.
#
#   digit 7 — every image is the same scanned card with tiny pixel jitter, so
#             almost the whole class sits in near-identical pairs. It also ends
#             up with the lowest dispersion, though (as we'll see) not low
#             enough to trip the default threshold.
#   digit 6 — 120 genuine images plus 60 copies of a single frame, so roughly a
#             third of the class is redundant.
def _jittered(frame: np.ndarray, count: int, rng: np.random.Generator) -> list[np.ndarray]:
    """`count` near-duplicate copies of one frame with tiny pixel jitter."""
    f = frame.astype(np.int16)
    return [np.clip(f + rng.integers(-3, 4, size=f.shape), 0, 255).astype(np.uint8) for _ in range(count)]


from PIL import Image

seven_dir = postal_root / "7"
seven_frame = np.asarray(Image.open(sorted(seven_dir.glob("*.png"))[0]))
for path in sorted(seven_dir.glob("*.png")):
    path.unlink()
for seq, arr in enumerate(_jittered(seven_frame, 30, rng)):
    Image.fromarray(arr).save(seven_dir / f"{seq:05d}.png")

six_dir = postal_root / "6"
six_frame = np.asarray(Image.open(sorted(six_dir.glob("*.png"))[0]))
for seq, arr in enumerate(_jittered(six_frame, 60, rng), start=10_000):
    Image.fromarray(arr).save(six_dir / f"{seq:05d}.png")

postal_maite = load_ic(postal_root, dataset_format="huggingface_vision")

# Computed from the materialized tree, not the collection budget above — that's
# the only way the count reflects digit 6's planted duplicates.
class_counts = {int(d.name): len(list(d.glob("*.png"))) for d in sorted(postal_root.iterdir())}

print(f"Selected {len(postal_maite)} images with biased class distribution")
print(f"Per-class counts: {class_counts}")

# %% [markdown]
# ### Attach synthetic metadata
#
# We add two metadata factors that simulate real collection conditions:
#
# - **lighting** — `"bright"` or `"dim"`. Facility C rarely collected
#   dim lighting samples, creating a metadata gap for digit 7.
# - **facility** — `"A"`, `"B"`, or `"C"`. Encodes where the image was
#   collected. Strongly correlated with class, which gap analysis will
#   detect.


# %%
from collections.abc import Mapping
from typing import Any

from numpy.typing import NDArray


def digit_of(target: Any) -> int:
    """Recover the class index from a datamaite one-hot classification target."""
    t = np.asarray(target)
    return int(np.argmax(t)) if t.ndim == 1 and t.size > 1 else int(t)


class BiasedMNIST:
    """Wraps a MAITE dataset with synthetic collection-condition metadata factors.

    Simulates a postal sorting system with uneven data collection across
    three facilities, each handling different digit classes under different
    lighting conditions.
    """

    def __init__(
        self,
        dataset: Any,
        rng: np.random.Generator,
    ) -> None:
        self._dataset = dataset

        # A MAITE `AnnotatedDataset` must expose a `metadata` mapping — dataeval
        # reads `index2label` from it to name classes in the report. Forward the
        # wrapped dataset's mapping rather than dropping it.
        self.metadata: dict[str, Any] = dict(dataset.metadata)

        # Pre-compute labels for metadata assignment
        self._labels = np.array([digit_of(dataset[i][1]) for i in range(len(dataset))])

        # Assign facility based on digit class
        self.facility = np.array(["A" if lbl <= 3 else "B" if lbl <= 6 else "C" for lbl in self._labels])

        # Assign lighting — dim is common for Facility A/B, rare for C
        self.lighting = np.empty(len(self._labels), dtype=object)
        for i, lbl in enumerate(self._labels):
            if lbl <= 6:
                # Facility A/B: 40% dim, 60% bright
                self.lighting[i] = rng.choice(["bright", "dim"], p=[0.6, 0.4])
            else:
                # Facility C: 95% bright, 5% dim — a metadata gap
                self.lighting[i] = rng.choice(["bright", "dim"], p=[0.95, 0.05])

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[NDArray[Any], Any, Mapping[str, Any]]:
        image, target, sample_metadata = self._dataset[index]

        # Keep the loader's per-sample metadata (`id`, `height`, `width`) and add ours
        metadata: dict[str, Any] = {
            **dict(sample_metadata),
            "lighting": self.lighting[index],
            "facility": self.facility[index],
        }

        return np.asarray(image), target, metadata


biased_dataset = BiasedMNIST(postal_maite, rng)
print(f"BiasedMNIST: {len(biased_dataset)} samples")

# Spot-check metadata
for i in [0, 100, len(biased_dataset) - 1]:
    _, target, meta = biased_dataset[i]
    print(f"  Sample {i}: digit={digit_of(target)}, lighting={meta['lighting']}, facility={meta['facility']}")

# %% [markdown]
# ### Visualize the collection bias
#
# Before running the workflow, let's see what the imbalance looks like.

# %%
import matplotlib.pyplot as plt

# class_counts was computed from the materialized tree above, so digit 6's
# planted duplicates and digit 7's near-duplicate replacement both show up here.
ordered_classes = sorted(class_counts)  # range(8): digits 0-7
colors = ["#2ecc71"] * 4 + ["#f39c12"] * 3 + ["#e74c3c"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Class distribution
ax1.bar([str(c) for c in ordered_classes], [class_counts[c] for c in ordered_classes], color=colors)
ax1.set_xlabel("Digit class")
ax1.set_ylabel("Sample count")
ax1.set_title("Class distribution (biased collection)")
ax1.axhline(y=np.mean(list(class_counts.values())), color="gray", linestyle="--", label="mean")
ax1.legend()

# Lighting distribution per facility
facilities = ["A", "B", "C"]
bright_counts = []
dim_counts = []
for fac in facilities:
    mask = biased_dataset.facility == fac
    bright_counts.append(int(np.sum(biased_dataset.lighting[mask] == "bright")))
    dim_counts.append(int(np.sum(biased_dataset.lighting[mask] == "dim")))

x = np.arange(len(facilities))
ax2.bar(x - 0.15, bright_counts, 0.3, label="bright", color="#f1c40f")
ax2.bar(x + 0.15, dim_counts, 0.3, label="dim", color="#34495e")
ax2.set_xticks(x)
ax2.set_xticklabels(facilities)
ax2.set_xlabel("Facility")
ax2.set_ylabel("Sample count")
ax2.set_title("Lighting distribution per facility")
ax2.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# The charts show exactly the kind of blind spots we want to catch:
# - Facility C (digit 7) has far fewer samples than Facility A or B — and the
#   bar chart cannot show what it doesn't have any of: digits 8 and 9 are
#   simply missing from the x-axis
# - Facility C has almost no dim lighting samples
#
# A model trained on this data might perform well on bright images of
# common digits but fail on dim images of digit 7 — and it would never
# even see digits 8 or 9. Step 1 below can only speak to the classes the
# data declares; Step 2 introduces a way to name the ones it doesn't.

# %% [markdown]
# ## Step 1: Run coverage without an extractor (metadata only)
#
# The `data-coverage` workflow can run **without** an extractor for a
# fast first pass. This skips embedding coverage and completeness but
# still analyzes label distribution, metadata distribution, metadata
# gaps, and — since no `ontology` is configured yet — a class balance
# worklist synthesized from the dataset's own declared classes.

# %%
from pathlib import Path

from dataeval_flow.config import PipelineConfig, SourceConfig
from dataeval_flow.config.schemas import (
    DataCoverageTaskConfig,
    DataCoverageWorkflowConfig,
    DatasetProtocolConfig,
)
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.coverage.params import DataCoverageHealthThresholds

metadata_only_workflow = DataCoverageWorkflowConfig(
    name="coverage-metadata-only",
    # datamaite's loader attaches `id`, `height` and `width` to every sample.
    # `id` is unique per image (so it looks perfectly class-predictive) and the
    # two dimensions are constant here — exclude all three so the analysis sees
    # only the collection factors we care about.
    metadata_exclude=["id", "height", "width"],
    run_gap_analysis=True,
    # `lighting` only deviates for 1 of 8 classes (digit 7), so its class→factor
    # mutual information is small (~0.013). Keep the threshold below that or
    # the dim-lighting gap this tutorial is built around never gets
    # cross-tabulated.
    gap_mi_threshold=0.01,
    gap_min_representation=5,  # Flag class-factor-value combos with < 5 samples
    balance=True,
    diversity_method="simpson",
    health_thresholds=DataCoverageHealthThresholds(
        class_imbalance_ratio=3.0,  # Tight — we want to catch moderate imbalance
        gap_count=2,  # Warn if ≥ 2 gaps found
    ),
)

task_metadata = DataCoverageTaskConfig(
    name="postal-coverage-metadata",
    workflow="coverage-metadata-only",
    sources="postal_src",
    # No extractor — metadata-only pass
)

config_metadata = PipelineConfig(
    datasets=[
        DatasetProtocolConfig(name="postal_biased", format="maite", dataset=biased_dataset),
    ],
    sources=[
        SourceConfig(name="postal_src", dataset="postal_biased"),
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
# ### Interpreting the findings
#
# The report reveals several issues:
#
# - **Label Distribution** — The 200:30 ratio between the largest and
#   smallest classes (≈6.7:1) exceeds our 3:1 threshold, triggering a
#   warning. Digit 7 is severely under-represented — and digits 8 and 9
#   don't show up in this finding at all, because the dataset never
#   declares them as classes in the first place.
# - **Metadata Coverage Gaps** — The gap analysis cross-references class
#   labels with metadata factors. It flags that digit 7 has almost no
#   `dim` lighting samples (≈1 where ≈11 are expected) — a blind spot
#   that could hurt real-world performance.
# - **Metadata Distribution** — Shows the `lighting` and `facility`
#   factors and their value counts.
# - **Class Balance Worklist** — With no `ontology` configured, the
#   workflow synthesizes a flat one from the dataset's own `index2label`
#   and reports how many labels each class is short of an even spread.
#   Three classes fall short of an even 156-per-class split: digit 7 by
#   126 images, digits 4 and 5 by 36 each. Note what this *cannot* say:
#   it only knows about the eight classes the data declares.
#
# One caveat on reading the gap table: `facility` is a *deterministic*
# function of the digit class here (A handles 0–3, B handles 4–6, C
# handles 7 — the only Facility-C digit we actually collected), so every
# class shows a 100% deficit for the two facilities that never see it.
# Those rows are by construction, not collection failures. The
# `lighting` rows are the real signal — the factor varies within every
# class, so a class that is missing one of its values genuinely was
# under-collected.

# %% [markdown]
# ### Drill into raw results
#
# The `result.data.raw` object provides machine-readable access to every
# metric for programmatic inspection.

# %%
raw = result_metadata.data.raw

# Label distribution
ld = raw.label_distribution
print(f"Number of classes: {ld.num_classes}")
print(f"Empty images: {len(ld.empty_images)}")
print("\nClass distribution:")
for cls, count in sorted(ld.class_distribution.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cls}: {count}")

# %%
# Metadata gaps — the most actionable finding
if raw.metadata_gaps and raw.metadata_gaps.gaps:
    print("Metadata coverage gaps:")
    print(f"  Total gaps found: {len(raw.metadata_gaps.gaps)}\n")

    print("  Mutual information (class → factor):")
    for factor, mi in raw.metadata_gaps.mutual_info_class_to_factor.items():
        print(f"    {factor}: {mi:.4f}")

    print("\n  Gap details:")
    for gap in raw.metadata_gaps.gaps:
        print(
            f"    {gap.class_name} × {gap.factor_name}={gap.factor_value}: "
            f"count={gap.class_count}, expected={gap.expected_count:.1f}, "
            f"deficit={gap.deficit:.0%}"
        )
else:
    print("No metadata gaps detected.")

# %% [markdown]
# ## Step 2: Declare the sanctioned label space
#
# A postal sorter reads *alphanumeric* codes — the label space is not
# "whatever digits we happened to collect", it is every character a code
# can contain. Writing that down as an **ontology** is what lets the
# workflow name a class that was never collected at all.

# %%
import string

postal_ontology = {
    "postal_char": {
        "digit": {
            "low": [str(d) for d in range(5)],
            "high": [str(d) for d in range(5, 10)],
        },
        "letter": {
            "vowel": list("AEIOU"),
            "consonant": [c for c in string.ascii_uppercase if c not in "AEIOU"],
        },
    }
}

ontology_workflow = metadata_only_workflow.model_copy(
    update={"name": "coverage-ontology", "ontology": postal_ontology},
)

task_ontology = DataCoverageTaskConfig(
    name="postal-coverage-ontology",
    workflow="coverage-ontology",
    sources="postal_src",
)

config_ontology = PipelineConfig(
    datasets=config_metadata.datasets,
    sources=config_metadata.sources,
    workflows=[ontology_workflow],
    tasks=[task_ontology],
)

result_ontology = run_task(task_ontology, config_ontology, cache_dir=Path("./cache"))
print(result_ontology.report())

# %% [markdown]
# ### What the ontology found that counts could not
#
# Three new sections replace the Class Balance Worklist:
#
# - **Label Space Coverage** — 8 of 36 sanctioned characters have any
#   examples. Digits `8` and `9` show up as `acquire` rows: sanctioned,
#   never collected. Step 1 could not name them, because a class with no
#   images declares nothing to count.
# - **Label Conformance** — every collected class name resolves to a
#   concept, so the label set conforms.
# - **Ontology Structure** — the artifact itself: 36 leaves under two
#   branches, no collisions.

# %%
onto = result_ontology.data.raw.ontology
print(f"source: {onto.source} (synthesized={onto.synthesized})")
print(f"leaf coverage: {onto.representation.leaf_coverage:.0%}")
print(f"total deficit: {onto.representation.total_deficit} labels\n")

print("Dark branches — whole regions of the label space with nothing in them:")
for branch in onto.representation.dark_branches:
    print(f"  {branch.label}: {branch.leaves} leaf classes, zero examples")

print("\nTop of the collection worklist:")
for row in onto.representation.worklist[:6]:
    print(f"  {row.label:>10}  {row.action:<8} have {row.count:>3}, want {row.target:>3}")

# %% [markdown]
# The `letter` branch is entirely dark — 26 sanctioned characters, no
# images. That is one finding, not 26, because `dark_branches` rolls
# missing leaves up to the highest wholly-empty concept.
#
# ### When a label does not reconcile
#
# Conformance catches the opposite problem: a class name the ontology
# does not sanction. Reconciliation is exact, not fuzzy, so a typo or an
# unsanctioned class shows up as **unmatched**.

# %%
from dataeval.core import label_reconciliation

from dataeval_flow.workflows._ontology import load_ontology

ontology_obj, _ = load_ontology(postal_ontology)
check = label_reconciliation(["0", "1", "oh", "seven"], ontology_obj)
print("matched:  ", dict(check["matched"]))
print("unmatched:", list(check["unmatched"]))

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

# %% [markdown]
# We use a 256-word vocabulary — the smallest `vocab_size` the config
# schema allows (`ge=256`) — rather than a larger one. `isotropy` — how
# many independent directions a class varies in — is only defined when a
# class has more samples than embedding dimensions. Our largest class
# has 200 images, which is below even this floor, so `isotropy` reports
# `null` for every class below. That is a real, honest result: our
# classes are simply too small relative to BoVW's minimum embedding
# width for shape to be measurable here. `dispersion` and
# `near_duplicate_fraction` only require `min_class_samples` (default
# 20) — far below any of our class sizes — and are unaffected; they are
# the metrics doing the work in this tutorial.

# %%
from dataeval.config import set_max_processes

from dataeval_flow.config import BoVWExtractorConfig

set_max_processes(8)

full_workflow = DataCoverageWorkflowConfig(
    name="coverage-full",
    metadata_exclude=["id", "height", "width"],
    coverage_method="adaptive",
    coverage_percent=0.01,  # adaptive: flag the sparsest 1% of observations
    num_observations=50,  # Number of neighbors for coverage analysis
    run_completeness=True,  # Measure dimensional completeness
    run_gap_analysis=True,
    gap_mi_threshold=0.01,  # lighting's MI is ~0.013 — see the note in Step 1
    balance=True,
    diversity_method="simpson",
    health_thresholds=DataCoverageHealthThresholds(
        uncovered_rate=10.0,  # Warn if > 10% uncovered in embedding space
        completeness_score=0.5,  # Warn if completeness < 0.5
        class_imbalance_ratio=3.0,  # Keep the tight threshold
        gap_count=2,
    ),
)

task_full = DataCoverageTaskConfig(
    name="postal-coverage-full",
    workflow="coverage-full",
    sources="postal_src",
    extractor="bovw_ext",
)

config_full = PipelineConfig(
    datasets=[
        DatasetProtocolConfig(name="postal_biased", format="maite", dataset=biased_dataset),
    ],
    sources=[
        SourceConfig(name="postal_src", dataset="postal_biased"),
    ],
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
# completeness in addition to label and metadata findings.

# %%
print(result_full.report())

# %% [markdown]
# ### Understanding the new findings
#
# With the extractor enabled, two new sections appear:
#
# - **Embedding Coverage** — Reports how many observations are
#   "uncovered" in embedding space — i.e. sit in a region where the model
#   would have little training support, broken down per class. Read the
#   dataset-wide rate as a *ranking* of which samples are most isolated,
#   not as a health percentage; see the note below on why. The per-class
#   columns are the part that is actually data-driven.
#
# - **Dimensional Completeness** — A score between 0 and 1 measuring how
#   well the data fills the embedding dimensions. Low completeness
#   suggests the data clusters in a narrow subspace, leaving many
#   directions unexplored. A biased dataset like ours tends to have
#   lower completeness because the under-represented classes don't
#   contribute enough variation.

# %% [markdown]
# ### Per-class coverage is the signal
#
# The dataset-wide uncovered count is a triage shortlist — with
# `coverage_method="adaptive"` it always returns exactly
# `coverage_percent` of observations, so its *rate* restates your config.
# The per-class columns are the part that depends on the data:
#
# - **dispersion** — how far the class spreads, relative to a typical
#   class. Around 1 is normal; well below means **clustered**.
# - **isotropy** — in how many independent directions it spreads. Low
#   means **one-dimensional** even when dispersion looks fine.
# - **near_duplicate_fraction** — the share sitting in near-identical
#   pairs. High means **padded with repeated frames**.
#
# In this run `near_duplicate_fraction` does all the work, and it flags
# three classes — one more than we planted:
#
# - **digit 7** (0.69) — the scanned-card class. Every image is one frame
#   plus jitter, so nearly all of it sits in near-identical pairs.
# - **digit 1** (0.45) — *not planted*. These are 200 genuine MNIST
#   images; handwritten 1s simply resemble each other more than other
#   digits do. A real property of the data, found the same way as the
#   synthetic ones.
# - **digit 6** (0.36) — the padded class: 120 real images plus 60 copies
#   of a single frame, so roughly a third is redundant.
#
# Digit 1 is the useful lesson. The metric does not know which
# redundancy you introduced and which the world handed you — it reports
# what is there, and you decide whether 45% near-duplicates in a class is
# a collection failure or just what that class looks like.
#
# Note what does **not** fire. Digit 7 has the lowest `dispersion` (0.56)
# but the default `min_dispersion` is 0.5, so it stays just inside the
# threshold and is never reported as clustered — 30 jittered copies still
# spread further, relative to a typical class, than the cutoff allows
# for. And `isotropy` is `null` for every class, for the sample-count
# reason described above. Three columns, one of them carrying the signal
# here: that is normal, and it is why the workflow reports all three
# rather than collapsing them into a score.

# %% [markdown]
# ### Inspect embedding results

# %%
raw_full = result_full.data.raw

# Embedding coverage
if raw_full.coverage:
    cov = raw_full.coverage
    print(f"Method: {cov.method}")
    print(f"Uncovered: {cov.uncovered_count} ({cov.uncovered_rate:.1%})\n")
    print(f"{'class':>8}  {'count':>5}  {'disp':>6}  {'iso':>6}  {'nearDup':>7}")
    for row in cov.per_class:

        def _fmt(v: float | None) -> str:
            return "  -   " if v is None else f"{v:6.2f}"

        print(
            f"{row.class_name:>8}  {row.count:>5}  {_fmt(row.dispersion)}  {_fmt(row.isotropy)}  {_fmt(row.near_duplicate_fraction):>7}"
        )
elif raw_full.coverage_skipped_reason:
    print(f"Embedding coverage skipped: {raw_full.coverage_skipped_reason}")
else:
    print("No embedding coverage (extractor not configured)")

print()

# Dimensional completeness
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
#
# For a safety-critical application like postal sorting where
# misclassification has real cost, tighten the thresholds:

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
    name="postal-coverage-strict",
    workflow="coverage-strict",
    sources="postal_src",
    extractor="bovw_ext",
)

config_strict = PipelineConfig(
    datasets=config_full.datasets,
    sources=config_full.sources,
    extractors=config_full.extractors,
    workflows=[full_workflow, strict_workflow],
    tasks=[task_strict],
)

result_strict = run_task(task_strict, config_strict, cache_dir=Path("./cache"))
print(result_strict.report())

# %% [markdown]
# With stricter thresholds, more findings escalate to warnings — the
# same data now triggers more alerts. This is the right behavior for
# safety-critical deployments where you'd rather over-flag than miss
# a coverage gap.
#
# Note that `leaf_coverage`, `dark_branch_count`, and
# `unmatched_class_count` are inert here: `coverage-strict` is copied
# from `full_workflow`, which has no `ontology` configured, so its
# ontology is synthesized and those three thresholds never apply. To
# see them bite, copy `ontology_workflow` from Step 2 instead.

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
# - **Simulate a biased dataset** that mirrors real-world uneven data
#   collection across facilities and conditions — including two classes
#   collected zero times
# - **Run coverage without an extractor** for a fast metadata-only pass
#   that catches class imbalance and metadata gaps, and synthesizes a
#   balance worklist from the classes the data happens to declare
# - **Declare an ontology** for the sanctioned label space, and see
#   `Label Space Coverage` name the classes that were never collected —
#   something a balance worklist over declared classes cannot do
# - **Add an extractor** to unlock per-class embedding signals
#   (dispersion, isotropy, near-duplicate fraction) and dimensional
#   completeness — revealing blind spots that counts alone can't show
# - **Read the coverage report** — a single `result.report()` call
#   covering both axes, with health status
# - **Drill into raw results** for programmatic access to gap details,
#   per-class coverage metrics, ontology findings, and distribution data
# - **Tune health thresholds** to match your domain's risk tolerance
#
# The key takeaway: coverage has **two orthogonal axes**, and low
# coverage on either is often invisible until you explicitly measure it.
# *Which categories you have* is a labels-against-an-ontology question —
# a class with zero examples doesn't even appear in a count. *How varied
# each one is* is an embeddings question — a class can be plentiful by
# count and still collapse into one pocket of the space, or turn out to
# be mostly repeated frames. Running the `data-coverage` workflow before
# training helps you catch both kinds of blind spot early, when they're
# cheapest to fix.

# %% [markdown]
# ## What's next
#
# - **Data cleaning** — Use the `data-cleaning` workflow to flag
#   outliers and duplicates in your dataset before training
# - **Drift monitoring** — After deploying, use `drift-monitoring` to
#   detect when incoming data drifts away from your training distribution
# - **Targeted collection** — Use the gap analysis results to guide
#   additional data collection: specifically gather dim lighting samples
#   from Facility C to fill the identified gaps
# - **Ship a real ontology** — Replace the inline hierarchy with a
#   versioned SKOS or OWL file (`ontology: ./taxonomy.ttl`), which also
#   carries synonyms, definitions and stable concept ids. Needs
#   `pip install "dataeval[ontology]"`.
# - **Targeted labeling** — Feed the gap findings into the
#   `prioritization` workflow to rank unlabeled candidates that fall in
#   the under-covered regions

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Dataset coverage](../concepts/Coverage.md): the label-space
#   and embedding-space completeness ideas behind this workflow.
# - **How-to: Declare an ontology** — [Declare an ontology](../how_to/declare_an_ontology.md)
#   to define the sanctioned label space this workflow checks a dataset against,
#   inline or as a versioned SKOS/OWL file.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to build a container image, write a YAML config, and run this workflow with `docker run`.
# - **How-to: Use an ONNX model for embeddings** — [ONNX embeddings](onnx_embeddings)
#   to swap the BoVW extractor used here for a pretrained model with higher-fidelity embeddings.
# - **How-to: Read evaluation outputs** — [Read evaluation outputs](../how_to/read_evaluation_outputs.md)
#   to interpret the coverage report, its health severities, and the exported result envelope.
# - **How-to: Reuse results with the disk cache** — [Reuse results with the disk cache](../how_to/reuse_results_with_cache.md)
#   for the `cache_dir` used throughout this tutorial — what it stores and what invalidates it.
