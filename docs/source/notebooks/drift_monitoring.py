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
# # Monitor incoming data for drift
#
# Detect distribution drift between a reference dataset and incoming data using
# the config-driven `drift-monitoring` workflow.

# %% [markdown]
# **Who this is for** — T&E engineers who operate a deployed model and need to know
# when the data feeding it has shifted away from the data the model was evaluated on.
#
# **Where this fits** — Drift monitoring is an operational, ongoing stage of the T&E
# workflow: after a model is deployed against a validated reference dataset, you watch
# incoming data for distribution shift that could silently degrade performance and
# trigger re-evaluation or retraining. See the
# [Distribution shift](../concepts/DistributionShift.md) concept page for the detectors
# and the relationship to [OOD detection](ood_detection) and [classwise drift](classwise_drift).

# %% [markdown]
# ## What you'll do
#
# - Load **MILCO** side-scan sonar imagery, which ships a reference and an operational
#   split defined by *when the data was collected*
# - Use the 2015/2017/2021 collection as the **reference** the model was evaluated against
# - Monitor the 2010 and 2018 operational archive for drift away from it
# - Configure the `drift-monitoring` workflow with **K-Neighbors**, **MMD**, and
#   **Univariate (CVM)** detectors
# - Enable **per-detector chunking** — chunked for K-Neighbors/MMD, non-chunked for Univariate
# - Use the chunk breakdown to locate *where in the stream* the data changes

# %% [markdown]
# ## What you'll learn
#
# - How to configure and run the `drift-monitoring` workflow via `run_task()`
# - How **per-detector chunking** lets you mix chunked and non-chunked detectors in one run
# - How to read the built-in drift report with per-detector and per-chunk results
# - The difference between **K-Neighbors** (distance-based), **MMD** (distribution-wide),
#   and **Univariate CVM** (per-feature) detectors
# - Why the extractor you choose decides what "drift" can even mean
# - **Why a drift verdict is uninterpretable without a control** — and how to build one
#   out of the reference set you already have

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `maite-datasets` (to download MILCO)
# - Internet connection (first run only — the dataset is cached under `./data`)

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: a reference campaign and an operational archive
#
# MILCO is side-scan sonar imagery collected by an autonomous underwater vehicle,
# annotated for **MILCO** (mine-like contacts) and **NOMBO** (non-mine-like bottom
# objects). What makes it a good drift subject is that it was collected across five
# separate years, and the dataset ships a split along that boundary:
#
# | Split | Years | Frames |
# |---|---|---|
# | `train` | 2015, 2017, 2021 | 261 |
# | `operational` | 2010, 2018 | 909 |
#
# So the scenario needs no synthetic corruption. Take the `train` collection as the
# data a detector was evaluated against, point it at the `operational` archive, and
# ask the question an operator actually has: **is this data still like the data we
# validated on?**
#
# :::{important}
# Both splits are exported in collection order, and that ordering is what makes
# chunked analysis meaningful. In the operational archive the 345 frames from 2010
# come first, followed by the 564 frames from 2018 — so a chunk boundary near index
# 345 is a real change of collection campaign, eight years wide, not an artifact.
# :::

# %% tags=["remove_output"]
from pathlib import Path

from maite_datasets.object_detection import MILCO

data_root = Path("./data")

# `base` fetches the archive once; the two calls below export each split for datamaite.
MILCO(root=data_root, image_set="base", download=True)
MILCO(root=data_root, image_set="train", as_datamaite=True)
MILCO(root=data_root, image_set="operational", as_datamaite=True)

reference_path = data_root / "milco_datamaite_train"
operational_path = data_root / "milco_datamaite_operational"

# %% [markdown]
# ### Confirm the campaign boundary
#
# The year is in every filename, so we can read the structure straight out of the
# exported annotations rather than taking it on trust.

# %%
import json
from collections import Counter

for name, path in (("reference", reference_path), ("operational", operational_path)):
    annotations = json.loads((path / "annotations" / "instances.json").read_text())
    years = [Path(img["file_name"]).stem.split("_")[-1] for img in annotations["images"]]
    first_index = {}
    for idx, year in enumerate(years):
        first_index.setdefault(year, idx)
    print(f"{name:<12} {len(years):>4} frames   {dict(Counter(years))}")
    print(f"{'':<12} first index of each year: {first_index}")

# %% [markdown]
# ### Look at the imagery
#
# Sonar is not photography, and it is worth seeing what the detectors are being asked
# to compare before trusting any number they produce.

# %%
import matplotlib.pyplot as plt
import numpy as np

samples = {
    "reference 2015": MILCO(root=data_root, image_set="train")[0][0],
    "reference 2021": MILCO(root=data_root, image_set="train")[255][0],
    "operational 2010": MILCO(root=data_root, image_set="operational")[0][0],
    "operational 2018": MILCO(root=data_root, image_set="operational")[500][0],
}

fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4))
for ax, (title, image) in zip(axes, samples.items(), strict=True):
    ax.imshow(np.transpose(np.asarray(image), (1, 2, 0)))
    ax.set_title(f"{title}\n{np.asarray(image).shape[1]}x{np.asarray(image).shape[2]}", fontsize=10)
    ax.axis("off")
fig.suptitle("MILCO side-scan sonar — reference campaigns vs operational archive", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# Note the frame sizes. MILCO ships images at both 416x416 and 1024x1024, mixed within
# every collection year, which rules out the simplest possible extractor before we
# start: a `flatten` extractor turns each image into a raw pixel vector, and vectors of
# two different lengths cannot be compared at all. Even where it fits, raw pixels are
# the wrong representation for sonar — see the
# [classwise drift](classwise_drift) tutorial for what that failure looks like when it
# does *not* raise an error.

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# The `drift-monitoring` workflow needs:
#
# 1. **Two datasets** — the first is the reference, the rest are test (incoming) data
# 2. **An extractor** — to compute embeddings that detectors compare
# 3. **Detector configuration** — which statistical tests to run
# 4. **Chunking** — to split incoming data into windows
#
# We use a **BoVW** (Bag of Visual Words) extractor. It detects SIFT keypoints, quantizes
# them against a learned vocabulary, and returns a fixed-length histogram — so it copes
# with the two frame sizes natively, and it describes sonar texture rather than raw
# intensity. A preprocessor resizes every frame to a common resolution first, which keeps
# the descriptor count comparable across frames and removes image size as a confound.

# %%
from dataeval_flow.config import (
    BoVWExtractorConfig,
    CocoDatasetConfig,
    PipelineConfig,
    PreprocessorConfig,
    SourceConfig,
)
from dataeval_flow.preprocessing import PreprocessingStep

preprocessor_config = PreprocessorConfig(
    name="sonar",
    steps=[PreprocessingStep(step="Resize", params={"size": [256, 256], "antialias": True})],
)

# BoVW fits one vocabulary across every source a task compares, so the reference and the
# operational archive are described in the same visual words. A per-source vocabulary
# would report drift between a dataset and itself.
extractor_config = BoVWExtractorConfig(name="bovw", vocab_size=256, batch_size=32, preprocessor="sonar")

reference_dataset = CocoDatasetConfig(name="reference", path=str(reference_path))
operational_dataset = CocoDatasetConfig(name="operational", path=str(operational_path))

# %% [markdown]
# ### Configure drift detectors and chunking
#
# We'll set up three complementary detectors:
#
# | Detector | What it tests | Chunked? | Strengths |
# |---|---|---|---|
# | **kneighbors** | Test points farther from reference neighbors | Yes | Robust in high dimensions, non-parametric |
# | **mmd** | Overall distribution distance via kernel trick | Yes | Sensitive to multivariate shifts |
# | **univariate (CVM)** | Per-feature CDF distance | No | Per-feature breakdown, high power |
#
# **K-Neighbors** drift detection checks whether incoming samples are farther
# from their k-nearest reference neighbors than expected. It operates on pairwise
# distances rather than per-feature statistics, which avoids the multiple-testing
# burden that makes univariate tests noisy across many features.
#
# **CVM (Cramér-von Mises)** is a univariate test that measures the integrated
# squared distance between empirical CDFs for each feature independently. It
# has higher statistical power than the default Kolmogorov-Smirnov test for
# detecting subtle distributional shifts. We run it **without chunking** so
# the report shows the contrast between a single overall verdict and the
# temporal chunk breakdown from the other two detectors.
#
# **Chunking** is configured **per detector**. With `chunk_size=200` over 909
# operational frames we get five windows, and because the archive is in collection
# order we know exactly which campaign each one covers.

# %%
from dataeval_flow.config import DriftMonitoringTaskConfig, DriftMonitoringWorkflowConfig
from dataeval_flow.workflow import run_task
from dataeval_flow.workflows.drift.params import (
    ChunkingConfig,
    DriftDetectorKNeighbors,
    DriftDetectorMMD,
    DriftDetectorUnivariate,
    DriftHealthThresholds,
)

drift_task = DriftMonitoringTaskConfig(
    name="milco-drift-overall",
    workflow="milco-drift",
    sources=["ref_src", "ops_src"],
    extractor="bovw",
)

config = PipelineConfig(
    datasets=[reference_dataset, operational_dataset],
    sources=[
        SourceConfig(name="ref_src", dataset="reference"),
        SourceConfig(name="ops_src", dataset="operational"),
    ],
    preprocessors=[preprocessor_config],
    extractors=[extractor_config],
    workflows=[
        DriftMonitoringWorkflowConfig(
            name="milco-drift",
            detectors=[
                DriftDetectorKNeighbors(k=10, chunking=ChunkingConfig(chunk_size=200, threshold_multiplier=4.0)),
                DriftDetectorMMD(n_permutations=100, chunking=ChunkingConfig(chunk_size=200, threshold_multiplier=4.0)),
                DriftDetectorUnivariate(test="cvm"),  # non-chunked overall test
            ],
            health_thresholds=DriftHealthThresholds(
                chunk_drift_pct_warning=15.0,  # warn if >15% of chunks drift
                consecutive_chunks_warning=2,  # warn on 2+ consecutive drifted chunks
            ),
        ),
    ],
    tasks=[drift_task],
)

# %% [markdown]
# ## Step 2: Run the drift monitoring workflow

# %%
result = run_task(drift_task, config, cache_dir=Path("./cache"))

# %% [markdown]
# ## Results Exploration: Drift report
#
# The workflow produces a text report summarizing each detector's findings. With
# chunking enabled, you'll see a per-chunk breakdown showing exactly which windows
# drifted.

# %%
print(result.report())

# %% [markdown]
# ### What each chunk actually contains
#
# `chunk_size=200` over 909 frames yields **four** windows, not five — the trailing
# remainder is folded into the last chunk rather than left as a short one. They map
# onto the two collection campaigns like this:
#
# | Chunk | Campaign |
# |---|---|
# | `[0:199]` | **2010** throughout |
# | `[200:399]` | **mixed** — 2010 up to index 344, then 2018 |
# | `[400:599]` | **2018** throughout |
# | `[600:908]` | **2018** throughout |
#
# ### Reading the verdict
#
# Every chunk drifts, on both chunked detectors, and CVM reports **256 of 256**
# features drifted at p = 0.000116. The archive does not drift away from the
# reference partway through — it never matched it in the first place. The 2010
# campaign is as far from the 2015/2017/2021 reference as the 2018 campaign is.
#
# That is worth separating from what this page's structure might lead you to expect.
# Chunking answers *when did the stream change?*, and it can only answer that when
# some of the stream still resembles the reference. Here none of it does, so the
# chunk breakdown reports a uniform verdict and the interesting question moves
# elsewhere: not "when did it change" but "was the reference ever representative?"
#
# The detectors do not agree on shape, which is informative in itself. K-Neighbors is
# nearly flat across the archive (0.83 to 0.94), while MMD singles out `[400:599]` at
# 0.5232 against roughly 0.22–0.27 for every other window. Both say "drifted"; only
# one says the middle of the 2018 campaign is unusual *among* the drifted windows.
#
# A 4/4, 256/256 verdict should make you suspicious before it makes you confident.
# A detector that flags everything is indistinguishable from a detector that is
# broken, and the reference here is only 261 frames. Before drawing any conclusion
# from the numbers above, the next cell establishes what these detectors say about
# data we have no reason to call drifted.

# %% [markdown]
# ### Control: do two reference campaigns drift from each other?
#
# The reference is itself three collections — 2015 (frames 0–119), 2017 (120–212) and
# 2021 (213–260). If *those* drift from one another as readily as the operational
# archive does, then "different campaign" is all the detector is measuring and the
# headline verdict says little about the operational data specifically.
#
# So we run the same detectors on a split that should come back clean-ish: 2015 as the
# reference, 2017+2021 as the incoming stream. `ViewConfig` with `Indices` carves both
# out of the dataset we already configured — no second copy on disk.

# %%
from dataeval_flow.config import ViewConfig, ViewOperation

control_task = DriftMonitoringTaskConfig(
    name="milco-drift-control",
    workflow="milco-drift-control",
    sources=["y2015_src", "y2017_2021_src"],
    extractor="bovw",
)

control_config = PipelineConfig(
    datasets=[reference_dataset],
    views=[
        ViewConfig(
            name="y2015", operations=[ViewOperation(type="Indices", params={"indices": {"start": 0, "stop": 120}})]
        ),
        ViewConfig(
            name="y2017_2021",
            operations=[ViewOperation(type="Indices", params={"indices": {"start": 120, "stop": 261}})],
        ),
    ],
    sources=[
        SourceConfig(name="y2015_src", dataset="reference", view="y2015"),
        SourceConfig(name="y2017_2021_src", dataset="reference", view="y2017_2021"),
    ],
    preprocessors=[preprocessor_config],
    extractors=[extractor_config],
    workflows=[
        DriftMonitoringWorkflowConfig(
            name="milco-drift-control",
            # No chunking — 141 incoming frames is one window, and all we want is a
            # single verdict on whether campaign-to-campaign difference alone trips these.
            detectors=[
                DriftDetectorKNeighbors(k=10),
                DriftDetectorMMD(n_permutations=100),
                DriftDetectorUnivariate(test="cvm"),
            ],
        ),
    ],
    tasks=[control_task],
)

control_result = run_task(control_task, control_config, cache_dir=Path("./cache"))
print(control_result.report())

# %% [markdown]
# ### The control fires too — and that changes the answer
#
# Two campaigns *inside the reference* drift from each other on all three detectors,
# at almost exactly the magnitude the operational archive produced:
#
# | | K-Neighbors | MMD | CVM |
# |---|---|---|---|
# | **Control** (2015 vs 2017+2021) | 0.8425 | 0.2271 | 2.92, 218/256 features |
# | **Operational** (per chunk) | 0.83 – 0.94 | 0.22 – 0.27, one at 0.52 | 14.46, 256/256 features |
#
# Read the K-Neighbors row first. The control sits at 0.8425, squarely inside the
# 0.83–0.94 band the operational chunks occupy. By that detector, the 2018 archive is
# no more different from the reference than two reference campaigns are from each
# other. The MMD row is worse: the pure-2010 chunk scores **0.2178**, *below* the
# control's 0.2271 — the 2010 campaign differs from the reference slightly less than
# the reference differs from itself.
#
# So the headline verdict does not say what it appears to say. On this representation
# these detectors are largely measuring **"a different collection campaign"**, which
# is a property every MILCO subset has, rather than anything specific to operational
# data. A bare `drifted: True` was never going to distinguish those two.
#
# What does survive the comparison:
#
# - **MMD on `[400:599]`, at 0.5232** — more than double the control and double every
#   other window. That one is genuinely outside normal campaign variation and worth
#   investigating.
# - **CVM's magnitude** — 14.46 across 256/256 features against the control's 2.92
#   across 218/256. Same verdict, very different strength.
#
# The lesson generalizes past this dataset. **A drift verdict is uninterpretable
# without a baseline for normal variation**, and you usually have the material to
# build one: hold out a slice of the reference and run the identical detectors on it.
# Without that control this page would have reported a confident 4/4 drift result and
# been largely wrong about what it meant.

# %% [markdown]
# ### Inspect chunk-level details programmatically
#
# The raw output gives you structured access to per-detector, per-chunk results
# for custom analysis or visualization.

# %%
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)

raw = result.data.raw
print(f"Reference size: {raw.reference_size}")
print(f"Test size:      {raw.test_size}")
print()

for method, det_result in raw.detectors.items():
    print(f"── {method} ({det_result['metric_name']}) ──")
    print(f"  Overall drifted: {det_result['drifted']}")
    print(f"  Distance:        {det_result['distance']:.6g}")

    chunks = det_result.get("chunks", [])
    if chunks:
        df = pl.DataFrame(chunks).select("key", "value", "lower_threshold", "upper_threshold", "drifted")
        print(df)
    print()

# %% [markdown]
# ### Visualize chunk drift over time
#
# A simple bar chart makes the pattern across the stream immediately visible.

# %%
# Only plot detectors that have chunk results
chunked_methods = [m for m, r in raw.detectors.items() if r.get("chunks")]
fig, axes = plt.subplots(1, len(chunked_methods), figsize=(6 * len(chunked_methods), 4))
if len(chunked_methods) == 1:
    axes = [axes]  # type: ignore[list-item]

for ax, method in zip(axes, chunked_methods, strict=True):
    chunks = raw.detectors[method]["chunks"]  # type: ignore[typeddict-item]

    labels = [c["key"] for c in chunks]
    values = [c["value"] for c in chunks]
    colors = ["#e74c3c" if c["drifted"] else "#2ecc71" for c in chunks]

    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)

    # Draw threshold lines and expand y-axis to include them
    upper_thresh = chunks[0].get("upper_threshold")
    lower_thresh = chunks[0].get("lower_threshold")
    all_y = list(values)
    if upper_thresh is not None:
        ax.axhline(
            y=upper_thresh,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label=f"upper={upper_thresh:.4g}",
        )
        all_y.append(upper_thresh)
    if lower_thresh is not None:
        ax.axhline(
            y=lower_thresh,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"lower={lower_thresh:.4g}",
        )
        all_y.append(lower_thresh)
    y_min, y_max = min(all_y), max(all_y)
    margin = (y_max - y_min) * 0.15 or abs(y_max) * 0.1 or 0.01
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.legend(fontsize=8)
    ax.set_title(method, fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance")
    ax.set_xlabel("Chunk")
    ax.tick_params(axis="x", rotation=30)

fig.suptitle("Chunk-level drift — green = ok, red = drift detected", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Results Exploration: Export results
#
# The JSON output contains all raw detector results, chunk data, and metadata —
# ready for integration with monitoring dashboards or automated pipelines.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:600] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial you learned how to:
#
# - **Use a real collection boundary** as the drift scenario — MILCO's reference and
#   operational splits differ by collection year, so nothing had to be corrupted to
#   create something to detect
# - **Choose an extractor the data supports** — mixed frame sizes rule out `flatten`
#   outright, and BoVW describes sonar texture rather than raw intensity
# - **Configure** the `drift-monitoring` workflow with multiple detectors
#   (K-Neighbors, MMD, Univariate CVM)
# - **Use per-detector chunking** — chunked analysis for some detectors, non-chunked
#   for others
# - **Read the drift report** — per-detector summaries and per-chunk breakdowns
# - **Map chunks back to the stream** so a drifted window names a real event
# - **Run a control before believing the verdict** — holding out a slice of the
#   reference and running the same detectors on it is what tells you whether a
#   `drifted: True` means anything
# - **Export** structured JSON results for downstream automation
#
# The result here is the part worth carrying away. Every detector flagged the
# operational archive, on every chunk, across every feature — and the control showed
# that two campaigns within the reference score almost identically. Most of that
# verdict was campaign-to-campaign variation, not an operational shift. One finding
# survived the comparison (MMD on `[400:599]`, at more than double the control), and
# without the control there would have been no way to tell it apart from the rest.

# %% [markdown]
# ## What's next
#
# - **Classwise drift** — Add `classwise: true` to break the verdict down by MILCO
#   versus NOMBO and see which class carries the shift
# - **Different detectors** — Try `domain_classifier` (trains a binary classifier to
#   distinguish ref from test) or other univariate tests (`ks`, `mwu`, `anderson`, `bws`)
# - **A stronger representation** — Swap BoVW for a pretrained model via
#   [ONNX embeddings](onnx_embeddings) and see whether the verdict survives a change
#   of feature space

# %% [markdown]
# ## Related guides
#
# - **Concept** — [Distribution shift](../concepts/DistributionShift.md):
#   how drift, classwise drift, and OOD detection relate, and what each detector measures.
# - **How-to: Read evaluation outputs** — [Read evaluation outputs](../how_to/read_evaluation_outputs.md)
#   to interpret per-batch drift flags and p-values and export them for a monitoring dashboard.
# - **How-to: Run workflows in containers** — [Containerized workflows](../how_to/containerized_workflows.md)
#   to run drift monitoring on a schedule against live data pipelines from a container.
# - **How-to: Use an ONNX model for embeddings** — [ONNX embeddings](onnx_embeddings)
#   to use a pretrained model (e.g. ResNet) for richer embeddings that capture higher-level features.
