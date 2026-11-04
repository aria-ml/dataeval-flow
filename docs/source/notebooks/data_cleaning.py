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
# # Clean a dataset
#
# Flag outliers and duplicates in SkySeaLand using the config-driven [`quality`](../reference/presets.md#quality)
# workflow, and take the dataset without them.

# %% [markdown]
# You are a T&E engineer or data scientist who vets an operational dataset for quality issues before downstream use.
#
# Run data cleaning as the initial stage of preparing an operational dataset. You can flag and remove outliers and
# duplicates before {doc}`dataset_splitting`, {doc}`drift_monitoring` or model training. See
# {doc}`../concepts/DataQualityAndCleaning` for the detection concepts.

# %% [markdown]
# ## What you will do
#
# - Load the SkySeaLand object-detection dataset using `maite-datasets`.
# - Configure a `quality` workflow with BoVW (Bag of Visual Words) embeddings.
# - Run `run_task()` to detect statistical outliers and image duplicates.
# - Read the cleaning report and evaluate the health status indicators.
# - Inspect the flagged outlier and duplicate images using `dataeval-plots`.
# - Take the cleaned dataset from the workflow's `clean` step, and see how an `export` step writes it.
# - Export the cleaning results to JSON.

# %% [markdown]
# ## What you will learn
#
# - How to configure and execute the `quality` workflow with `run_task()`.
# - How to use BoVW feature extractors without external pretrained model files.
# - How to configure outlier detection parameters and duplicate sensitivity.
# - How to set check thresholds to trigger warning statuses.
# - How to interpret the formatted cleaning report.
# - How to read the result's findings, and what each of the workflow's steps found.
# - How to inspect flagged samples with `dataeval-plots`.
# - How to get the dataset without what the workflow flagged, and write it to disk.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow`, which includes `dataeval`, `datamaite` and `pydantic`.
# - Install `dataeval-plots` to visualize flagged images.
# - Install `maite-datasets` to access SkySeaLand.
# - Make sure you have network access for the initial dataset download (~262 MB).

# %% [markdown]
# ## Lay out the data root
#
# The data root is the folder you run this notebook from. After a first run it holds:
#
# ```text
# project/
# ├── data_cleaning.yaml                # read by the dataeval-flow command, not by the notebook's code
# ├── data/
# │   ├── skysealand/                   # downloaded on the first run (~262 MB); the export is made from it
# │   └── skysealand_datamaite_base/    # exported on the first run; read: the COCO export the pipeline loads
# ├── cache/                            # written: embeddings and image statistics, reused on a re-run
# └── output/                           # written only by the command: its results and the cleaned dataset
# ```
#
# The notebook builds its pipeline in Python. `data_cleaning.yaml` holds the same pipeline for the command, and its
# dataset path, `data/skysealand_datamaite_base`, is relative to the data root. The notebook does not read
# `data_cleaning.yaml`, and it writes nothing to `output/`.

# %% [markdown]
# ## Load the dataset
#
# [SkySeaLand](https://www.kaggle.com/datasets/mdzahidhasanriad/skysealand) is an overhead object-detection dataset
# containing 1,307 frames across four classes: `airplane`, `boat`, `car`, and `ship`.
#
# Use `maite-datasets` to download the dataset and export it with `as_datamaite=True`, in a format readable by DataEval
# Flow. Later runs load the export from disk. To adapt a torchvision dataset instead, see {doc}`torchvision_datasets`.

# %% tags=["remove_output"]
from pathlib import Path

from maite_datasets.object_detection import SkySeaLand

# Downloads to ./data/skysealand on first run (~262 MB), then writes the datamaite-format
# export beside it. A re-run reuses the export instead of rebuilding it.
SkySeaLand(root="./data", image_set="base", download=True, as_datamaite=True)

data_path = Path("./data/skysealand_datamaite_base")

# %% [markdown]
# ## Build the workflow configuration
#
# You must specify the parameters of `quality` explicitly. This configuration detects outliers using **adaptive
# thresholding** across dimension, pixel, and visual statistics, and detects duplicates by hash and by cluster. The BoVW
# (Bag of Visual Words) extractor learns visual words directly from the dataset images, so it needs no external model
# file. To use a pretrained ONNX model instead, see {doc}`onnx_embeddings`.
#
# Adaptive thresholding selects between the Z-score and the modified Z-score for each metric, based on how the data is
# distributed. A threshold of 3.5 flags outliers conservatively.
#
# :::{note}
# Shuffle before you limit a view. SkySeaLand frames are grouped by collection site on disk. Applying `Shuffle` with a
# fixed seed ensures that the 300-frame sample represents all collection sites. See
# {doc}`../how_to/build_dataset_views`.
# :::

# %%
from dataeval_flow import run_task
from dataeval_flow.config import (
    CocoDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.workflows.quality import (
    ClassOutliersSettings,
    DuplicatesSettings,
    ImageDuplicatesSettings,
    ImageOutliersSettings,
    OutliersSettings,
    QualityChecks,
    QualityConfig,
    TargetOutliersSettings,
)

workflow = QualityConfig(
    name="skysealand_cleaning",
    outliers=OutliersSettings(
        outlier_threshold=("adaptive", 3.5),  # Adaptive thresholding for outliers, bound 3.5
        flags=["dimension", "pixel", "visual"],  # All image stat groups
        cluster_threshold=3.5,  # Cluster-based detection in embedding space (requires extractor).
        cluster_algorithm="hdbscan",
        n_clusters=4,  # SkySeaLand has 4 classes
    ),
    duplicates=DuplicatesSettings(  # Duplicate detection: hash-based plus cluster-based.
        cluster_sensitivity=0.5,
        cluster_algorithm="hdbscan",
        n_clusters=4,
    ),
    checks=QualityChecks(
        image_duplicates=ImageDuplicatesSettings(
            exact=0.0,  # No exact duplicates allowed (default)
            near=5.0,  # Up to 5% near duplicates before warning (default)
        ),
        image_outliers=ImageOutliersSettings(
            warning=5.0  # Relaxed from 3% default for four collection sites and varied sensors
        ),
        target_outliers=TargetOutliersSettings(
            warning=10.0  # Relaxed from 3% default for annotation variance in object detection
        ),
        class_outliers=ClassOutliersSettings(
            warning=12.0  # Relaxed from 3% default for diverse class appearances
        ),
    ),
)

task = TaskConfig(
    name="skysealand_clean",
    workflow="skysealand_cleaning",
    sources="skysealand_src",
    extractor="bovw_ext",
)

# Build the pipeline configuration: datasets, sources, extractors, views, workflows, and tasks
config = PipelineConfig(
    max_processes=2,
    seed=0,
    datasets=[
        CocoDatasetConfig(name="skysealand_base", path=str(data_path)),
    ],
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
        SourceConfig(name="skysealand_src", dataset="skysealand_base", view="sample300"),
    ],
    extractors=[
        BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32),
    ],
    workflows=[workflow],
    tasks=[task],
)

# %% [markdown]
# ## Run the cleaning workflow
#
# Pass the configuration and the task to `run_task()`, with a cache directory for the embeddings and image statistics.

# %%
result = run_task(config, task, cache_dir=Path("./cache"))

# %% [markdown]
# ### Read the cleaning report
#
# Call `result.report()` to display outlier counts, duplicate groups, label statistics, and health statuses in a
# formatted text summary.

# %%
print(result.report())

# %% [markdown]
# ### View the HTML report
#
# `result.to_html()` renders the same report as one HTML page. It shows a card for each finding, with the flagged images
# and boxes and the limits they crossed, and it ends with the table of steps and the configuration. You can sort and
# filter its tables. The page follows your system's dark mode setting, so here it may not match this site's theme. See
# {doc}`view_html_reports` for the rest of the page.

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
# ### Findings and steps
#
# `quality` is a preset: its settings expand to a chain of steps. Evaluators find the outliers and duplicates and count
# the labels, checks judge what they found against `checks`, and `clean` removes what was flagged. The report above
# gives each finding a section, with the evaluators it judged below it. It then gives a section to each step that no
# finding showed, such as `clean`, and ends with a table of every step. `run_task()` returns a `ChainResult` holding
# each step by name, in run order:

# %%
print(list(result.steps))

# %% [markdown]
# `result.findings` holds the checks' findings, the ones the report's summary lists:

# %%
for finding in result.findings:
    print(f"{finding.severity:<8} {finding.title:<20} {finding.brief}")

# %% [markdown]
# ### Read the health status
#
# The `Health` summary line says whether findings exceeded the configured thresholds. Each status has a marker:
#
# | Status | Marker | Meaning |
# |---|---|---|
# | `ok` | `[ok]` | Nothing was flagged. |
# | `info` | `[..]` | The finding is within the allowable threshold. |
# | `warning` | `[!!]` | The finding exceeds the threshold and requires review. |
#
# Configure when findings warn with `checks`, keyed by check type. The default values are:
#
# | Metric | Default | When to adjust |
# |---|---|---|
# | `image-duplicates.exact` | 0% | Raise above 0 only if your pipeline intentionally repeats images |
# | `image-duplicates.near` | 5% | Lower to 1–2% for curated benchmarks; raise to 10–15% for web-scraped data |
# | `image-outliers.warning` | 3% | Lower to 1% for safety-critical data; raise to 5–10% for visually diverse collections |
# | `target-outliers.warning` | 3% | Lower to 1% for annotation reviews; raise to 5–10% for dense object detection |
# | `class-outliers.warning` | 3% | Lower to 1% for label-quality reviews; raise to 5–10% for diverse classes |
#
# The [`bias`](../reference/presets.md#bias) preset judges class imbalance, not this one.
#
# In this tutorial, thresholds are relaxed because SkySeaLand includes four distinct capture sites with differing
# sensors, altitudes, and lighting conditions.
#
# To apply stricter thresholds, specify tighter tolerances:
#
# ```python
# from dataeval_flow.workflows.quality import (
#     QualityChecks,
#     ImageDuplicatesSettings,
#     ImageOutliersSettings,
# )
#
# strict = QualityChecks(
#     image_duplicates=ImageDuplicatesSettings(exact=0.0, near=2.0),
#     image_outliers=ImageOutliersSettings(warning=1.0),
# )
# ```

# %% [markdown]
# ### Inspect the flagged images
#
# Inspect the flagged images with `dataeval-plots` to decide whether the detected anomalies are data quality errors or
# acceptable operational variation.
#
# `result.sources` holds the dataset each source read, after its view: here the 300 frames the workflow ran on, which
# the flagged indices refer to. You can retrieve images from it without reloading from disk.

# %%
ds = result.sources["skysealand_src"]

# %% [markdown]
# #### Outlier images
#
# The `outliers` step's output is DataEval's own Outliers output. Its `data()` gives one row per flag: the image, the
# box where a box was flagged, the metric (such as brightness, entropy, or a dimension), its value, and the limit it
# crossed. Extract the images flagged as a whole and display a sample.

# %%
outliers = result.steps["outliers"].output
issues = outliers.data()

# Image-level flags name no box; the rest flag single boxes.
image_issues = issues.filter(issues["target_index"].is_null())
outlier_grouped: dict[int, list[str]] = {}
for row in image_issues.iter_rows(named=True):
    outlier_grouped.setdefault(row["item_index"], []).append(row["metric_name"])

outlier_indices = sorted(outlier_grouped)
print(f"Image outliers: {len(outlier_indices)} images flagged, {image_issues.height} total flags")

# %%
from dataeval_plots import plot

# Show a sample of outlier images (first 9)
if outlier_indices:
    sample = outlier_indices[:9]
    print("Outlier sample: flagged metrics per image:")
    for idx in sample:
        print(f"  Image {idx:>5d}: {', '.join(outlier_grouped[idx])}")
    _ = plot(ds, indices=sample, images_per_row=3, figsize=(12, 8), show_labels=True)

# %% [markdown]
# #### Duplicate images
#
# Plot the duplicate groups side by side, both exact and near duplicates, to check whether the images are redundant. The
# `duplicates` step's output is DataEval's own Duplicates output, and its `items` holds the groups of whole images.
#
# SkySeaLand contains distinct captures without duplicates in this sample. When a dataset has duplicate images, each
# group renders here for visual inspection.

# %%
duplicates = result.steps["duplicates"].output.items
exact_groups = duplicates.exact
near_groups = duplicates.near

print(f"Exact duplicate groups: {len(exact_groups)}")
print(f"Near  duplicate groups: {len(near_groups)}")

# %%
# Plot exact duplicate groups (if any)
for i, indices in enumerate(exact_groups[:3]):
    print(f"\nExact group {i}: indices {indices}")
    _ = plot(ds, indices=indices, images_per_row=len(indices), figsize=(4 * len(indices), 4), show_labels=True)

# %%
# Plot near duplicate groups (if any)
for i, (indices, methods) in enumerate(near_groups[:3]):
    print(f"\nNear group {i}: indices {indices}  (methods: {methods})")
    _ = plot(ds, indices=indices, images_per_row=len(indices), figsize=(4 * len(indices), 4), show_labels=True)

# %% [markdown]
# ## Take the cleaned dataset
#
# The chain's last step, `clean`, removes each flagged image and box, and each duplicate but the first of its group. Its
# output is a DataEval `View` of the images that survived, which you can go on to train on or evaluate from Python. Its
# `details` count what it removed at each level, images (`items`) and boxes (`detections`), and under `by_plan` what
# each plan named, `duplicates` and `outliers`.

# %%
clean = result.steps["clean"].output
print(f"Images before cleaning: {len(ds)}")
print(f"Images after cleaning:  {len(clean)}")
print(f"Removed: {result.steps['clean'].details['removed']}")

# %% [markdown]
# To write the cleaned dataset to disk, run the `quality` entry as a step of a custom workflow, and add an
# [`export`](../reference/transforms.md#export) step that reads the step's `clean` output, `cleaning.clean`. `export`
# writes object-detection datasets, which SkySeaLand is. `data_cleaning.yaml`, beside this notebook, holds this
# tutorial's pipeline with that workflow and a task to run it:
#
# ```yaml
# workflows:
#   - name: clean_export
#     inputs: [data]
#     steps:
#       - {name: cleaning, workflow: skysealand_cleaning, input: data}
#       - {name: dataset, transform: export, input: cleaning.clean, format: coco}
#
# tasks:
#   - name: skysealand_export
#     workflow: clean_export
#     sources: skysealand_src
#     extractor: bovw_ext
# ```
#
# Run `dataeval-flow --config data_cleaning.yaml --output ./output` from this notebook's directory to run both tasks. In
# `skysealand_export`, the `cleaning` step runs the chain above as `cleaning/outliers`, `cleaning/labels`, and so on to
# `cleaning/clean`, and `dataset` writes the cleaned dataset in COCO format under
# `output/datasets/skysealand_export.dataset/`. Without `--output`, nothing is written, and the export step is skipped.
# See {doc}`../how_to/write_a_custom_workflow`. To run the command in a container, see
# {doc}`../how_to/containerized_workflows`.

# %% [markdown]
# ## Export the results
#
# `result.export()` returns the result as a JSON string. See {doc}`../how_to/read_evaluation_outputs` for how to read
# reports and result envelopes.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")
print(json_str[:500] + "\n...")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Configure the `quality` workflow with outlier and duplicate detection parameters.
# - Use BoVW feature extractors without external model dependencies.
# - Set check thresholds to control warning generation.
# - Run the workflow via `run_task()` on a dataset view.
# - Read the cleaning report and its findings, and evaluate health statuses.
# - Inspect flagged outliers and duplicates visually using `dataeval-plots`.
# - Take the cleaned dataset from the `clean` step, and write it with an `export` step.
# - Export cleaning results to JSON format.

# %% [markdown]
# ## Next steps
#
# - {doc}`tune_data_cleaning` — Compare several outlier thresholds and duplicate sensitivities in one run with a task
#   matrix.
# - {doc}`audit` — Check cross-split leakage, distribution shift, and shortcut risk before training.
# - {doc}`../concepts/DataQualityAndCleaning` — Learn which algorithms detect the outliers and duplicates in this
#   workflow.
# - {doc}`../how_to/configure_outlier_detection` — Choose the statistical methods, visual metrics, and check thresholds
#   for your data.

# %% [markdown]
# ## On your own
#
# Repeat the analysis on your own data by changing these settings in the pipeline config:
#
# - Set `path` on `CocoDatasetConfig` to your own dataset in COCO format.
# - Change the `size` param of the `Limit` operation in the view, or remove the operation to run on every frame. Keep
#   the `Shuffle` operation before it.
# - Change `seed` on the pipeline and the `seed` param of the `Shuffle` operation to draw a different sample.
# - Set `n_clusters` in `outliers` and in `duplicates` to the number of classes in your data. SkySeaLand has 4.
# - Change `outlier_threshold` in `outliers` to another method, such as `iqr`, or to a method and a bound, such as
#   `("zscore", 3.0)`. Remove a group from `flags` to skip those statistics.
# - Change `cluster_sensitivity` in `duplicates` to change how sensitive the cluster-based duplicate detection is.
# - Change the values in `checks` to your own tolerance, using the table in the "Read the health status" section.
# - Replace the `BoVWExtractorConfig` with another extractor config, or change its `vocab_size`, which is 512 here.
