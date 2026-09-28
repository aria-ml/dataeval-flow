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
# # View a report as HTML
#
# You can render any result's report as one self-contained HTML page with `to_html()`. The page holds everything the
# text report holds, laid out for reading on screen: a card per finding, tables you can sort and filter, and each
# flagged item's measurements on hover. It loads nothing, so it opens offline, attaches to a ticket as it is, and
# prints to PDF from the browser.
#
# This guide vets a side-scan sonar survey's reference campaigns and checks its operational archive for drift, then
# shows each report's page below as it renders, so you can try the page here.

# %% [markdown]
# ## Used in these tutorials
#
# You can reference this guide from:
#
# - {doc}`Clean a dataset <data_cleaning>`: read the flagged images and boxes, and their limits, on one page.
# - {doc}`Monitor incoming data for drift <drift_monitoring>`: see each chunk's distance against its thresholds.
# - {doc}`Run a full evaluation pipeline end to end <end_to_end>`: export a page next to each result envelope.

# %% [markdown]
# ## Run two workflows
#
# The page draws whatever a result's findings hold. To show most of what it can draw, you will run two workflows on
# MILCO, side-scan sonar imagery of mine-like objects collected over several survey campaigns:
#
# - `data-cleaning` on the reference campaigns, whose outlier findings list every flagged image and every flagged
#   bounding box with the metrics that flagged it;
# - `drift-monitoring` of the operational archive against the reference, in chunks of 200 frames, so each chunk's
#   distance is drawn against the drift thresholds.
#
# {doc}`Monitor incoming data for drift <drift_monitoring>` walks through the same drift configuration in depth.

# %% tags=["remove_output"]
from pathlib import Path

from maite_datasets.object_detection import MILCO

from dataeval_flow import PipelineConfig, run_tasks
from dataeval_flow.config import (
    CocoDatasetConfig,
    PreprocessingStep,
    PreprocessorConfig,
    SourceConfig,
    TaskConfig,
)
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from dataeval_flow.workflows.drift_monitoring import (
    ChunkingConfig,
    DriftDetectorKNeighbors,
    DriftMonitoringConfig,
)

data_root = Path("./data")

# `base` fetches the archive once; the two calls below export each split for datamaite.
MILCO(root=data_root, image_set="base", download=True)
MILCO(root=data_root, image_set="train", as_datamaite=True)
MILCO(root=data_root, image_set="operational", as_datamaite=True)

# %%
config = PipelineConfig(
    seed=0,
    datasets=[
        CocoDatasetConfig(name="reference", path=str(data_root / "milco_datamaite_train")),
        CocoDatasetConfig(name="operational", path=str(data_root / "milco_datamaite_operational")),
    ],
    sources=[
        SourceConfig(name="ref-src", dataset="reference"),
        SourceConfig(name="ops-src", dataset="operational"),
    ],
    # BoVW reads SIFT keypoints, so every frame is resized to one size first.
    preprocessors=[
        PreprocessorConfig(
            name="sonar",
            steps=[PreprocessingStep(step="Resize", params={"size": [256, 256], "antialias": True})],
        )
    ],
    extractors=[BoVWExtractorConfig(name="bovw", vocab_size=256, batch_size=32, preprocessor="sonar")],
    workflows=[
        DataCleaningConfig(name="clean", outlier_method="zscore", outlier_flags=["pixel", "visual"]),
        DriftMonitoringConfig(
            name="drift",
            detectors=[
                DriftDetectorKNeighbors(k=10, chunking=ChunkingConfig(chunk_size=200, threshold_multiplier=4.0)),
            ],
        ),
    ],
    tasks=[
        TaskConfig(name="clean-reference", workflow="clean", sources="ref-src", extractor="bovw"),
        TaskConfig(name="drift-operational", workflow="drift", sources=["ref-src", "ops-src"], extractor="bovw"),
    ],
)

# %% tags=["remove_output"]
results = run_tasks(config, cache_dir=Path("./cache"))
clean, drift = results["clean-reference"], results["drift-operational"]

# %% tags=["remove_cell"]
assert all(r.success for r in results.values()), [r.errors for r in results.values() if not r.success]

# %% tags=["remove_cell"]
# The page styles its whole document, so the docs show it in an iframe of its own rather than
# letting its stylesheet restyle this site. Readers don't need this to use `to_html()`.
import html

from IPython.display import display


def show(page: str, height: int = 720) -> None:
    """*page* in an iframe, so its stylesheet and script stay inside it."""
    frame = (
        f'<iframe srcdoc="{html.escape(page, quote=True)}" title="HTML report" loading="lazy" '
        f'style="width:100%;height:{height}px;border:1px solid #d1d9e0;border-radius:6px"></iframe>'
    )
    display({"text/html": frame}, raw=True)


# %% [markdown]
# ## The same report, as text and as HTML
#
# `report()` gives the text report, 80 columns wide, for a terminal or a log. With `detailed=False` it prints only
# the summary, followed by the run's metadata and configuration:

# %% tags=["hide-output"]
print(clean.report(detailed=False))

# %% [markdown]
# `to_html()` renders the full report as a page. Write it with `encoding="utf-8"`:

# %%
output_dir = Path("./output/html_reports")
output_dir.mkdir(parents=True, exist_ok=True)

for name, result in results.items():
    page = output_dir / f"{name}.html"
    page.write_text(result.to_html(), encoding="utf-8")
    print(f"{page}  ({page.stat().st_size:,} bytes)")

# %% [markdown]
# Here is `clean-reference.html` as a browser shows it. The page follows your system's dark mode setting, so here it may
# not match this site's theme.

# %% tags=["remove_input"]
show(clean.to_html())

# %% [markdown]
# ## Reading the page
#
# Try each of these on the page above:
#
# - **The verdict.** The header gives the report's verdict: the number of warnings, or `passed`.
# - **The cards.** Each finding is a card headed by its title, its value and its severity, so the cards read as the
#   report's summary. A warning starts open, and the rest start closed as one line each. *Expand all* and
#   *Collapse all*, top right, act on every card at once.
# - **Reference.** The metadata factors and the configuration close the report as panels, closed until opened.
# - **Sorting.** Click a column's header to sort the table by it: ascending, descending, then back to the original
#   order. A column of numbers sorts as numbers. The *Flagged by* column sorts by how many flags each row holds.
# - **Filtering.** A table of more than ten rows gets a box above it. Type in it to keep the rows whose visible
#   text matches, such as `brightness` for the images that metric flagged.
# - **Flags.** Each tag reads a value against the limit it crossed, such as `brightness 0.99 > 0.84`. Hover a tag,
#   or reach it with the keyboard, to see where the value ranks in its population and the population's mean and
#   standard deviation.
# - **Limits.** Under the flagged images, the limits table gives each metric's lower and upper limit and its
#   population, so the page keeps them when printed without hover cards.

# %% [markdown]
# ## Thresholds on a scale
#
# A bar chart with thresholds draws each one as a dashed line across the bars, and labels it on a scale below the
# table. In the drift report, each chunk's distance is a bar, and the two dashed lines are the lower and upper drift
# thresholds: a chunk whose distance falls outside them counts as drifted. No chunk drifted, so the finding is `ok`
# and its card starts closed; open *K-Neighbors* to see the chart.

# %% tags=["remove_input"]
show(drift.to_html(), height=780)

# %% [markdown]
# Labels that would overlap take a second or third line, and any that still don't fit are listed under the scale,
# so no two are drawn on top of each other.

# %% [markdown]
# ## A short page
#
# `to_html(detailed=False)` gives the summary page, as `report(detailed=False)` gives the summary text. It keeps the
# header's verdict and the summary table, without the cards:

# %% tags=["remove_input"]
show(clean.to_html(detailed=False), height=360)

# %% [markdown]
# ## Printing, dark mode, and blocked scripts
#
# - **Printing.** The page prints (or saves as PDF from the browser's print dialog) in its light palette. Before it
#   prints, it opens every card and panel and shows every row a filter hid.
# - **Dark mode.** The page follows the system's setting. It has no toggle of its own.
# - **Scripts blocked.** One inline script adds the sorting, the filter boxes and the *Expand all* buttons. Where
#   scripts are blocked, as some mail viewers and locked-down browsers block them, the page shows the same report
#   without those controls, rather than with buttons that do nothing.

# %% [markdown]
# ## Several tasks on one page
#
# From the command line, `--output` writes `result.html` next to `result.json` and `result.txt`. It holds every
# task's report on one page, with a list of the reports at the top linking each one:
#
# ```bash
# dataeval-flow --config params.yaml --output ./results
# ```

# %% [markdown]
# ## Related
#
# - {doc}`Read evaluation outputs <../how_to/read_evaluation_outputs>`: the text report, its severities, the result
#   envelope, and the JSON form a page is drawn from.
