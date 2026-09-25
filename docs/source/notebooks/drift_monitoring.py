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
# - How to interpret formatted drift reports and chunked metric trends.
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
# MILCO contains side-scan sonar imagery collected by autonomous underwater vehicles,
# annotated for `MILCO` (mine-like contacts) and `NOMBO` (non-mine-like bottom objects).
# The imagery spans five collection years:
#
# | Split | Years | Frames |
# |---|---|---|
# | `train` | 2015, 2017, 2021 | 261 |
# | `operational` | 2010, 2018 | 909 |
#
# You will use `train` as your baseline reference dataset and evaluate whether the
# `operational` archive exhibits distribution drift.
#
# :::{important}
# Both splits are exported in chronological collection order. In the operational
# archive, 345 frames from 2010 appear first, followed by 564 frames from 2018.
# Sequential chunk boundaries align with these distinct collection campaigns.
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
# 3. **Detectors**: Statistical drift detection algorithms.
# 4. **Chunking**: Window parameters for temporal analysis.
#
# You will use a Bag of Visual Words (BoVW) extractor. BoVW quantizes SIFT keypoints
# against a learned visual vocabulary, yielding fixed-length histograms regardless of
# native image resolution. A preprocessing step resizes frames to 256x256 to ensure
# uniform keypoint extraction density.

# %%
from dataeval_flow import PipelineConfig
from dataeval_flow.config import (
    CocoDatasetConfig,
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
# distances rather than per-feature statistics, which avoids the multiple-testing
# burden that makes univariate tests noisy across many features.
#
# **CVM (Cramér-von Mises)** is a univariate test that measures the integrated
# squared distance between empirical CDFs for each feature independently. It
# has higher statistical power than the default Kolmogorov-Smirnov test for
# detecting subtle distributional shifts. You will run it **without chunking**
# to contrast an overall verdict with temporal chunk breakdowns from other detectors.
#
# **Chunking** is configured **per detector**. Specifying `chunk_size=200` over 909
# operational frames yields windows that align with chronological collection campaigns.

# %%
from dataeval_flow import run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.workflows.drift_monitoring import (
    ChunkingConfig,
    DriftDetectorKNeighbors,
    DriftDetectorMMD,
    DriftDetectorUnivariate,
    DriftMonitoringConfig,
    DriftMonitoringHealthThresholds,
)

drift_task = TaskConfig(
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
        DriftMonitoringConfig(
            name="milco-drift",
            detectors=[
                DriftDetectorKNeighbors(k=10, chunking=ChunkingConfig(chunk_size=200, threshold_multiplier=4.0)),
                DriftDetectorMMD(n_permutations=100, chunking=ChunkingConfig(chunk_size=200, threshold_multiplier=4.0)),
                DriftDetectorUnivariate(test="cvm"),  # non-chunked overall test
            ],
            health_thresholds=DriftMonitoringHealthThresholds(
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
# chunking enabled, you will see a per-chunk breakdown showing which windows
# drifted.

# %%
print(result.report())

# %% [markdown]
# ### What each chunk actually contains
#
# A `chunk_size=200` setting over 909 frames produces four windows:
#
# | Chunk | Campaign |
# |---|---|
# | `[0:199]` | 2010 campaign |
# | `[200:399]` | 2010 up to index 344, followed by 2018 |
# | `[400:599]` | 2018 campaign |
# | `[600:908]` | 2018 campaign (includes remainder samples) |
#
# ### Reading the verdict
#
# In this run, all chunks trigger drift warnings across K-Neighbors and MMD, and
# Univariate CVM flags 256 of 256 features.
#
# To determine whether this signal reflects genuine operational degradation or
# routine campaign variation, you should run a baseline control.

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
# ### Comparing operational drift to the control baseline
#
# You should compare operational distances directly against the control:
#
# | Detector | Control (2015 vs 2017+2021) | Operational (per chunk) |
# |---|---|---|
# | **K-Neighbors** | 0.8425 | 0.83 to 0.94 |
# | **MMD** | 0.2271 | 0.22 to 0.27 (chunk `[400:599]`: 0.5232) |
# | **CVM** | 2.92 (218/256 features) | 14.46 (256/256 features) |
#
# K-Neighbors distances for operational data match the inter-campaign control baseline
# (0.8425), showing that general collection shifts account for much of the observed
# difference.
#
# However, two metrics indicate notable shifts beyond baseline variation:
#
# - **MMD on chunk `[400:599]` (0.5232)**: More than double the control baseline,
#   indicating a substantial localized shift in the 2018 campaign.
# - **CVM magnitude (14.46)**: Substantially larger than the control magnitude (2.92),
#   reflecting widespread feature-level divergence.
#
# You should always evaluate drift against baseline controls to distinguish normal
# collection variance from severe operational drift.

# %% [markdown]
# ### Inspect chunk-level details programmatically
#
# You can query per-detector and per-chunk metrics directly from `result.output.raw`:

# %%
import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)

raw = result.output.raw
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

fig.suptitle("Chunk-level drift: green = ok, red = drift detected", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Results Exploration: Export results
#
# You can export results to JSON format for integration with monitoring dashboards
# and pipeline automations.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:600] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Configure the `drift-monitoring` workflow across reference and operational datasets.
# - Use BoVW feature extractors to generate fixed-length descriptors from variable-sized sonar frames.
# - Combine chunked and non-chunked detectors in a single workflow.
# - Interpret formatted drift reports and per-chunk metric trends.
# - Construct reference controls using `ViewConfig` to calibrate expected baseline variation.
# - Evaluate operational drift against control baselines to identify genuine anomalies.
# - Export structured drift findings to JSON.

# %% [markdown]
# ## Next steps
#
# - **Classwise drift**: Use [Detect classwise drift](classwise_drift) with `classwise=True`
#   to identify which target classes drive the drift signal.
# - **Alternative detectors**: Test alternative statistical detectors such as `domain_classifier`
#   or Kolmogorov-Smirnov (`ks`).
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
