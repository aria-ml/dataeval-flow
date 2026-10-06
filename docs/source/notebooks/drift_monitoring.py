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
# **Target audience**: You are a T&E engineer operating a deployed model who needs
# to detect when incoming data shifts away from the validation baseline.
#
# **Workflow role**: You should run drift monitoring during operational deployment.
# Drift monitoring tracks distribution shifts over time that can degrade model
# performance. See [Distribution shift](../concepts/DistributionShift.md) for
# background on drift detection, {doc}`Detect out-of-distribution samples <ood_detection>`,
# and [Detect classwise drift](classwise_drift).

# %% [markdown]
# ## What you will do
#
# - Load MILCO side-scan sonar imagery partitioned into reference and operational splits by collection year.
# - Use the 2015, 2017, and 2021 campaigns as the baseline reference set.
# - Monitor the 2010 and 2018 operational archive for distribution drift.
# - Configure the `drift-monitoring` workflow with K-Neighbors, MMD, and Univariate CVM detectors.
# - Configure per-detector chunking to evaluate temporal drift progression.
# - Run a control comparison using reference subsets to calibrate baseline campaign variation.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute the `drift-monitoring` workflow with `run_task()`.
# - How to combine chunked and non-chunked detectors in a single pipeline.
# - How the size of the reference limits the chunk size.
# - How to interpret formatted drift reports and chunked metric trends.
# - How to read each detector's chunks from the result's steps.
# - How K-Neighbors, MMD, and Univariate CVM detectors evaluate distribution shift.
# - How feature representations influence drift sensitivity.
# - How to construct baseline controls from reference data to establish expected variance.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `maite-datasets` to download MILCO.
# - Ensure network access for the initial dataset download.

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: a reference campaign and an operational archive
#
# MILCO contains side-scan sonar imagery collected by AUVs,
# annotated for `MILCO` (mine-like contacts) and `NOMBO` (non-mine-like bottom objects).
# The imagery spans five collection years:
#
# | Split | Years | Frames |
# |---|---|---|
# | `train` | 2015, 2017, 2021 | 261 |
# | `operational` | 2010, 2018 | 909 |
#
# Use `train` as the baseline reference dataset and evaluate whether the
# `operational` archive exhibits distribution drift.
#
# :::{important}
# Both splits are exported in chronological collection order. In the operational
# archive, 345 frames from 2010 appear first, followed by 564 frames from 2018, so
# sequential chunks follow the collection campaigns.
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
# You can inspect collection years directly from exported annotation filenames:

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
# Sonar is not photography, so it is worth seeing what the detectors are being asked
# to compare before trusting what they produce.

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
fig.suptitle("MILCO side-scan sonar: reference campaigns vs operational archive", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# MILCO includes frames at both 416x416 and 1024x1024 resolutions within each year.
# A `flatten` extractor cannot process variable input sizes. You should use a
# feature extractor that produces fixed-dimension embeddings across varying image sizes.

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# To configure the `drift-monitoring` workflow, you specify:
#
# 1. **Datasets**: A reference dataset followed by one or more test sources.
# 2. **Extractor**: An extractor to produce embedding vectors.
# 3. **Detectors**: Drift evaluator entries, one per statistical drift detection algorithm.
# 4. **Chunking**: Window parameters for temporal analysis, set on each detector.
#
# You will use a Bag of Visual Words (BoVW) extractor. BoVW quantizes SIFT keypoints
# against a learned visual vocabulary, yielding fixed-length histograms regardless of
# native image resolution. A preprocessing step resizes frames to 256x256 to ensure
# uniform keypoint extraction density.

# %%
from dataeval_flow.config import (
    CocoDatasetConfig,
    PipelineConfig,
    PreprocessingStep,
    PreprocessorConfig,
    SourceConfig,
)
from dataeval_flow.config.extractors import BoVWExtractorConfig

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
# You will configure three complementary detectors:
#
# | Detector | What it tests | Chunked? | Strengths |
# |---|---|---|---|
# | **kneighbors** | Test points farther from reference neighbors | Yes | Robust in high dimensions, non-parametric |
# | **mmd** | Overall distribution distance via kernel trick | Yes | Sensitive to multivariate shifts |
# | **univariate (CVM)** | Per-feature CDF distance | No | Per-feature breakdown, high power |
#
# **K-Neighbors** drift detection checks whether incoming samples are farther
# from their k-nearest reference neighbors than expected. It operates on pairwise
# distances rather than per feature statistics, which avoids the multiple-testing
# burden that makes univariate tests noisy across many features.
#
# **CVM (Cramér-von Mises)** is a univariate test that measures the integrated
# squared distance between empirical CDFs for each feature independently. It
# has higher statistical power than the default Kolmogorov-Smirnov test for
# detecting subtle distributional shifts. You will run it **without chunking**
# to contrast an overall verdict with temporal chunk breakdowns from other detectors.
#
# **Chunking** is configured **per detector**. The chunk size applies to the reference
# as well as the operational data: each reference chunk is scored against the rest of
# the reference, and the drift bounds are the mean of those scores plus or minus
# `k` standard deviations, where `threshold: [zscore, k]` sets the method and `k`. The reference must therefore split into
# at least 3 chunks, and more chunks give a steadier spread. With 261 reference frames,
# a `chunk_size` of 200 would leave only two reference chunks, and the detector refuses
# to fit.
#
# You will use `chunk_size=50`. It splits the reference into five chunks and the
# operational archive into 18 windows. The reference does not divide evenly, so
# `incomplete="append"` merges its last 11 frames into the fifth chunk. Kept as a chunk
# of their own, 11 frames would give a noisy score that widens every bound. The
# operational data always merges its remainder into its last window.

# %%
from dataeval_flow import run_task, set_device
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.shift import (
    ChunkedDriftConfig,
    DriftKNeighborsConfig,
    DriftMMDConfig,
    DriftUnivariateConfig,
)
from dataeval_flow.steps.checks import DriftThresholds
from dataeval_flow.workflows.drift_monitoring import DriftMonitoringChecks, DriftMonitoringConfig

drift_task = TaskConfig(
    name="milco-drift-overall",
    workflow="milco-drift",
    sources=["ref_src", "ops_src"],
    extractor="bovw",
)

chunking = ChunkedDriftConfig(chunk_size=50, incomplete="append", threshold=("zscore", 4.0))

# Computing on the CPU makes the numbers below reproduce on a machine with a GPU too.
set_device("cpu")

config = PipelineConfig(
    seed=0,
    datasets=[reference_dataset, operational_dataset],
    sources=[
        SourceConfig(name="ref_src", dataset="reference"),
        SourceConfig(name="ops_src", dataset="operational"),
    ],
    preprocessors=[preprocessor_config],
    extractors=[extractor_config],
    workflows=[
        DriftMonitoringConfig(
            name="milco-drift",
            detectors=[
                DriftKNeighborsConfig(k=10, chunking=chunking),
                DriftMMDConfig(n_permutations=100, chunking=chunking),
                DriftUnivariateConfig(method="cvm"),  # non-chunked overall test
            ],
            checks=DriftMonitoringChecks(
                drift=DriftThresholds(
                    chunk_percent=15.0,  # warn if >15% of chunks drift
                    consecutive_chunks=2,  # warn on 2+ consecutive drifted chunks
                )
            ),
        ),
    ],
    tasks=[drift_task],
)

# %% [markdown]
# ## Step 2: Run the drift monitoring workflow

# %%
result = run_task(config, drift_task, cache_dir=Path("./cache"))

# %% [markdown]
# ## Results Exploration: Drift report
#
# Each detector is a step of the workflow, and a `drift` check judges it. The report
# summarizes each check's finding. With chunking enabled, you will see a per-chunk
# breakdown showing which windows drifted.

# %%
print(result.report())

# %% [markdown]
# ### What each chunk actually contains
#
# Before you read the verdict, list the collection year and frame resolution in each
# operational window:

# %%
operational_images = json.loads((operational_path / "annotations" / "instances.json").read_text())["images"]
# `result.steps` holds each step by name, a detector's named for its type unless its entry sets `name`. The
# step's `elements` hold one run per test source, and a chunked run's chunks are the output's `details` table.
mmd_chunks = (result.steps["drift-mmd"].elements or {})["ops_src"].output.details
for chunk in mmd_chunks.iter_rows(named=True):
    window = operational_images[chunk["start_index"] : chunk["end_index"] + 1]
    years = Counter(Path(img["file_name"]).stem.split("_")[-1] for img in window)
    sizes = Counter(f"{img['width']}x{img['height']}" for img in window)
    print(f"{chunk['key']:<10} {dict(years)!s:<26} {dict(sorted(sizes.items()))}")

# %% [markdown]
# The 2010 campaign fills the windows through `[250:299]`, `[300:349]` holds its last 45
# frames and the first 5 from 2018, and 2018 fills the rest. Every
# window up to `[750:799]` mixes the two resolutions, with 21 to 32 of its 50 frames at
# 1024x1024. From frame 800 on, every frame is 416x416.
#
# ### Reading the verdict
#
# In this run:
#
# - **K-Neighbors** flags the last two windows, `[800:849]` and `[850:908]`. Two
#   consecutive drifted windows meet `consecutive_chunks=2`, so the finding is
#   a warning.
# - **MMD** flags only `[850:908]`. Its values for the 2018 windows `[350:399]` through
#   `[600:649]` are 0.39 to 0.41, well above the 2010 windows (0.11 to 0.25) but just
#   under the upper bound of 0.42.
# - **CVM** tests the whole archive at once and flags 251 of 256 features.
#
# The windows that drift are the ones made only of 416x416 frames. Every frame is
# resized to 256x256 before BoVW sees it, but a change in the mix of native
# resolutions is still a change in how the data was collected. Confirm a
# finding like this against the collection records before you treat it as a
# change in the seafloor.
#
# The bounds are wide because the reference's own chunks differ from one another. Its
# five chunks span the 2015, 2017, and 2021 campaigns, so the bounds already allow for
# campaign-to-campaign variation. A chunked detector can only flag what falls outside
# the variation its reference already contains.
#
# To determine whether a signal reflects operational change or routine campaign
# variation, you should run a baseline control.

# %% [markdown]
# ### Control: Evaluate baseline variation between reference campaigns
#
# The reference dataset contains three campaigns: 2015 (frames 0 to 119), 2017
# (120 to 212), and 2021 (213 to 260). You can test 2015 as the reference against
# 2017 and 2021 as the incoming data to evaluate expected inter-campaign variance.

# %%
from dataeval_flow.config import ViewConfig, ViewOperation

control_task = TaskConfig(
    name="milco-drift-control",
    workflow="milco-drift-control",
    sources=["y2015_src", "y2017_2021_src"],
    extractor="bovw",
)

control_config = PipelineConfig(
    seed=0,
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
        DriftMonitoringConfig(
            name="milco-drift-control",
            # No chunking: 141 incoming frames forms a single window to test baseline variation.
            detectors=[
                DriftKNeighborsConfig(k=10),
                DriftMMDConfig(n_permutations=100),
                DriftUnivariateConfig(method="cvm"),
            ],
        ),
    ],
    tasks=[control_task],
)

control_result = run_task(control_config, control_task, cache_dir=Path("./cache"))
print(control_result.report())

# %% [markdown]
# ### Comparing operational drift to the control baseline
#
# You should compare operational distances directly against the control:
#
# | Detector | Control (2015 vs 2017+2021) | Operational (per window) |
# |---|---|---|
# | **K-Neighbors** | 0.8585 | 0.52 to 0.83 |
# | **MMD** | 0.2387 | 0.11 to 0.57 |
# | **CVM** | 3.00 (225/256 features) | 9.31 (251/256 features) |
#
# The control flags drift on all three detectors, so the reference campaigns already
# differ from one another by enough for each test to detect. Against that baseline:
#
# - **K-Neighbors**: every operational window stays below the control's 0.8585,
#   including the two flagged windows (0.79 and 0.83). A nearest-neighbor distance
#   also depends on how many reference frames there are to be near, and the control's
#   reference holds 120 frames against the full 261, so treat this comparison as rough.
# - **MMD**: the 2010 windows (0.11 to 0.25) sit at or below the control's 0.2387. The 2018
#   windows `[350:399]` through `[600:649]` (0.39 to 0.41) and the final window (0.57)
#   are well above it.
# - **CVM**: the operational distance is about three times the control's (9.31 against
#   3.00).
#
# Taken together, the 2018 campaign differs from the reference by more than the
# reference campaigns differ from each other, and its final 416x416 stretch differs
# most. The 2010 campaign stays within routine campaign variation.
#
# You should always evaluate drift against baseline controls to distinguish normal
# collection variance from operational drift.

# %% [markdown]
# ### Inspect chunk-level details programmatically
#
# Each detector's step holds one output per test source. It is DataEval's `DriftOutput`: `drifted`,
# `distance`, `threshold`, `metric_name` and, when the detector is chunked, a `details` table with a row per chunk.
# A check's step, such as `drift-mmd-check`, holds the finding that judged it.

# %%
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_rows(-1)

outputs = {
    name: (step.elements or {})["ops_src"].output
    for name, step in result.steps.items()
    if name.startswith("drift-") and not name.endswith("-check")
}

for name, output in outputs.items():
    print(f"── {name} ({output.metric_name}) ──")
    print(f"  Overall drifted: {output.drifted}")
    print(f"  Distance:        {output.distance:.6g}")

    if isinstance(output.details, pl.DataFrame):
        print(output.details.select("key", "value", "lower_threshold", "upper_threshold", "drifted"))
    print()

# %% [markdown]
# ### Visualize chunk drift over time
#
# A simple bar chart makes the pattern across the stream immediately visible.

# %%
# Only plot detectors that have chunk results
chunked_methods = [name for name, output in outputs.items() if isinstance(output.details, pl.DataFrame)]
fig, axes = plt.subplots(1, len(chunked_methods), figsize=(6 * len(chunked_methods), 4))
if len(chunked_methods) == 1:
    axes = [axes]  # type: ignore[list-item]

for ax, method in zip(axes, chunked_methods, strict=True):
    chunks = outputs[method].details.to_dicts()

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
    ax.tick_params(axis="x", rotation=90, labelsize=8)

fig.suptitle("Chunk-level drift: green = ok, red = drift detected", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Results Exploration: Export results
#
# You can export results to JSON for monitoring dashboards and
# pipeline automations.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:600] + "\n...")

# %% [markdown]
# ## Conclusion
#
# This tutorial shows how to:
#
# - Configure the `drift-monitoring` workflow across reference and operational datasets.
# - Use BoVW feature extractors to generate fixed-length descriptors from variable-sized sonar frames.
# - Combine chunked and non-chunked detectors in a single workflow.
# - Choose a chunk size the reference can support, and merge a short final reference chunk
#   with `incomplete="append"`.
# - Interpret formatted drift reports and per-chunk metric trends.
# - Construct reference controls using `ViewConfig` to calibrate expected baseline variation.
# - Evaluate operational drift against control baselines to identify genuine anomalies.
# - Export structured drift findings to JSON.

# %% [markdown]
# ## Next steps
#
# - **Classwise drift**: Use [Detect classwise drift](classwise_drift) with `classwise={detector: "class"}`
#   to identify which target classes drive the drift signal.
# - **Alternative detectors**: Test alternative statistical detectors such as `drift-domain-classifier`
#   or Kolmogorov-Smirnov (`ks`), and see [Monitor drift with steps](../how_to/monitor_drift.md) to merge test sources or drift on crops.
# - **Embedding backbones**: Configure an ONNX model via [Use an ONNX model for embeddings](onnx_embeddings)
#   to test pretrained deep representations.

# %% [markdown]
# ## Related guides
#
# - **Concept**: [Distribution shift](../concepts/DistributionShift.md) covers
#   drift monitoring, classwise drift, and out-of-distribution detection.
# - **How-to**: [Read evaluation outputs](../how_to/read_evaluation_outputs.md) explains
#   drift metric outputs, p-values, and export envelopes.
# - **How-to**: [Containerized workflows](../how_to/containerized_workflows.md) explains
#   how to schedule drift monitoring pipelines in Docker.
# - **Guide**: [Use an ONNX model for embeddings](onnx_embeddings) shows how to configure
#   pretrained extractors for drift monitoring.
