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
# # Triage a dataset's metadata
#
# In this tutorial, you will identify metadata columns that failed to load or parse, and generate configuration fixes
# using the `triage` preset on SeaDrone telemetry.

# %% [markdown]
# You are an engineer or data scientist who needs to verify dataset metadata before computing coverage, drift, or bias
# metrics.
#
# You should run triage before downstream evaluation workflows, so that metadata factors are correctly typed, binned,
# and accounted for in subsequent evaluations. Downstream workflows such as {doc}`data_coverage` and
# {doc}`dataset_splitting` silently drop unparseable metadata columns, mixed-type fields, or high-cardinality values
# without raising errors. {doc}`audit` flags such columns, and triage gives a policy that repairs those it can. Triage
# surfaces these issues so you can configure remediations. This tutorial follows {doc}`end_to_end`, which runs triage as
# one checkpoint of a whole pipeline.

# %% [markdown]
# ## What you will do
#
# - Load a sample of the SeaDrone object-detection dataset with telemetry metadata.
# - Run the `triage` preset and inspect the generated report.
# - Identify dropped or unparseable columns and review suggested remedies.
# - Review factors that parse cleanly but require remediation, such as unique identifiers and sentinel values.
# - Inspect the suggested policy configuration.
# - Complete required placeholder values in the policy, apply it, and re-run triage.
# - Compare results between the initial and corrected runs.

# %% [markdown]
# ## What you will learn
#
# - How to execute `triage` and interpret finding categories and severity levels.
# - How to interpret factor distribution charts and evaluate bin recommendations.
# - How to configure policies to handle missing values, date parsing, and type conversions.
# - How verification tests proposed policies against your dataset.
# - The operational scope and limitations of automated metadata triage.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `maite-datasets` to access SeaDrone.
# - Make sure you have network access for the initial validation split download (~1.2 GB).
#
# :::{note}
# Unlike other tutorials, this notebook does not have a corresponding `.yaml` file. SeaDrone is loaded in-memory via
# `DatasetProtocolConfig`, which is not serializable. In standard pipelines, you can run the workflow from YAML
# configuration against on-disk datasets.
# :::

# %% [markdown]
# ## Lay out the data root
#
# The data root is the folder you run this notebook from. This notebook touches one folder in it:
#
# ```text
# project/
# └── data/
#     └── seadrone/    # downloaded on the first run (~1.2 GB, validation split); read in memory
# ```
#
# SeaDrone reaches the pipeline in memory, so no configuration file or path points at it. The pipeline is the Python
# this notebook builds, so no YAML file sits next to it. The notebook sets no cache or output folder, so nothing else is
# written.

# %% [markdown]
# ## Load the SeaDrone dataset
#
# You will use SeaDrone, an aerial object-detection dataset with per-frame telemetry including altitude, heading, speed,
# and GPS coordinates.

# %% tags=["remove_output"]
from maite_datasets.object_detection import SeaDrone

# Downloads to ./data/seadrone on first run (~1.2 GB for the validation split).
seadrone = SeaDrone(root="./data", image_set="val", download=True)
print(f"{len(seadrone)} images")

# %% [markdown]
# ## Build the workflow configuration
#
# To run triage, you only need dataset metadata. You do not need model predictions, embeddings, or image statistics.
#
# You should use a shuffled sample of 200 frames to represent multiple capture sequences across flights. The sample
# comes from a view with a `Shuffle` and a `Limit` operation. {doc}`../how_to/build_dataset_views` explains dataset
# sampling and filtering operations.

# %%
from dataeval_flow import run_task
from dataeval_flow.config import (
    DatasetProtocolConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.workflows.triage import TriageConfig

triage_workflow = TriageConfig(
    name="triage",
    checks={"factor-issues": {"max_examples": 6}},  # distinct values shown per column in the report
)

task = TaskConfig(
    name="triage_seadrone",
    workflow="triage",
    sources="seadrone_val",
)

config = PipelineConfig(
    max_processes=2,
    datasets=[DatasetProtocolConfig(name="seadrone", dataset=seadrone)],
    views=[
        ViewConfig(
            name="sample",
            operations=[
                ViewOperation(type="Shuffle", params={"seed": 0}),
                ViewOperation(type="Limit", params={"size": 200}),
            ],
        )
    ],
    sources=[SourceConfig(name="seadrone_val", dataset="seadrone", view="sample")],
    workflows=[triage_workflow],
    tasks=[task],
)

# %% [markdown]
# ## Run the triage preset
#
# You can run the task using `run_task()`, then print the execution status and generated report.

# %%
result = run_task(config, task)
print(f"success={result.success}  health={result.health['status']}")

# %%
print(result.report())

# %% [markdown]
# `result.to_html()` draws the same report as a page. Each finding is a card headed by its title and severity, and you
# can sort and filter its tables. {doc}`view_html_reports` covers the rest of the page.

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
# The preset's `factor-triage` step holds the issues, the suggested policy and the verification, and its `factor-issues`
# step makes the findings the report shows.

# %%
triage = result.steps["factor-triage"].output.data()

# %% [markdown]
# ### Read the report
#
# The report opens with the step count and a summary of the findings. It shows each finding with the triage step's
# section beside it (the factors read and the issues counted by category and severity), and ends with the Metadata
# Factors section: the encoding and how each factor was binned.
#
# You should review `blocking` findings first. These indicate metadata columns that could not be processed and were
# dropped from the factor set:
#
# - `date_time` has unique timestamps per row that exceed cardinality limits. You must truncate or bucket the values to
#   a broader granularity (such as day).
# - `latitude` and `longitude` hold 198 numeric values and 2 string values. Each column has mixed types, so it is
#   excluded until you harmonize the values.
#
# You can inspect the value counts and sample values printed for each mixed type:
#
# ```text
#   latitude — [blocking] mixed_types @ unit
#     ███████████████████░  198 numeric, 2 text
#
#     numeric reads: -1, 47.671928, 47.671942, 47.671971, 47.672015, 47.672055
#                    (+114 more)
#     text reads:    'N'
# ```

# %% [markdown]
# ### Incomplete suggestions and required inputs
#
# `latitude` contains the string `'N'`. Because automated triage cannot determine what `'N'` represents, it generates a
# remap rule with a `null` target and sets `complete=False`.
#
# You can inspect individual findings in the `factor-triage` step's `findings`:

# %%
latitude = next(f for f in triage["findings"] if f.factor == "latitude")

print("category :", latitude.category)
print("severity :", latitude.severity)
print("reasons  :", latitude.reasons)
print("suggested:", latitude.suggestion.corrections)
print("runnable :", latitude.suggestion.complete)

# %% [markdown]
# When `complete=False`, you must complete the template before you can apply it.
#
# Automated suggestions are generated for:
#
# - Common numeric formatting (stripping commas, currency symbols, or units such as `kg`)
# - Standard ISO-8601 timestamps
# - Recognized null and sentinel strings (`""`, `"N/A"`, `"unknown"`), mapped to `.nan`
#
# For unrecognized non-numeric values in numeric columns, you will receive remap templates with `null` placeholders that
# you must fill in.

# %% [markdown]
# ### Continuous factor distributions and bin suggestions
#
# Ten factors used automatic binning. Because automatically generated bin edges vary across samples, you should declare
# explicit bin counts in configuration to ensure consistent binning across runs.
#
# You can evaluate proposed bin counts using the distribution charts in the report:
#
# ```text
#   frame — declare 5 bins
#     ▁▂▂▂▃▁▁▁▁▁▁▂▃▁▁▁▃▂▁▂▅█▇▅▂▁  ▁▁▂▁▁▁                                ▁▁▁▁▁▁▁▁▁▁
#     ├────── 3465 ████████│██ 6240 ─────────────────────────────────────────────┤
#     178                5715                                                19800
# ```
#
# The top line shows the column's histogram, and beneath it is a box plot on the same axis. The 25th and 75th
# percentiles are written into the whiskers on either side of the box, and the line below gives the minimum, the median
# under its mark, and the maximum. When a box is too narrow to label, the quartiles are named on a fourth line instead,
# as for `object_size`: `p25 900 · p50 1806 · p75 5508`.
#
# :::{note}
# You can use distribution plots to inspect raw sample quantiles rather than existing bins, ensuring that clusters and
# skewed extremes remain visible regardless of the current binning strategy.
# :::
#
# In `frame`, you can see values cluster in the lower-middle range, followed by a gap and a smaller cluster at the high
# end. When you see a box reach an edge with no whisker (as in `altitude`), at least 25% of values sit directly on that
# extreme value.
#
# ## Review the suggested policy
#
# Triage merges every suggestion into a single configuration block matching the `metadata:` schema. You can inspect this
# policy directly.

# %%
print(triage["suggested_policy_yaml"])

# %% [markdown]
# You can review the suggested policy sections:
#
# - `parse_datetime` for `date_time` parses ISO-8601 timestamps and groups values by `day`.
# - `remap` templates for `latitude` and `longitude` are remap rules with placeholder values that you must complete.
# - `remap` templates for telemetry fields are remap rules for `-1.0` values across telemetry columns.
# - `exclude` for `object_id` excludes the column from factor analysis because it is an item identifier.
# - `continuous_factor_bins` sets explicit bin counts for stable continuous factors.
#
# Factors with severe skew (floor mass) or identifier properties are omitted from `continuous_factor_bins` until
# cleaned. You can check the report for the rationale behind each omitted factor.

# %% [markdown]
# ### Review the recommended policy
#
# Triage also recommends a policy that pins every factor the run left unpinned: explicit edges for each continuous
# factor and the vocabulary for each categorical one, read from this data. A remap value you have not coded yet is
# dropped to missing by default, and marked so you can decide what it means.
#
# The recommendation is only as good as the data it was read from. If this sample does not represent the data you
# expect, results computed under the recommended policy can be invalid or misleading, so review each factor before you
# commit it.

# %%
print(triage["recommended_policy_yaml"])

# %% [markdown]
# ## Apply the policy and re-run
#
# You should now complete the placeholder values in the suggested policy:
#
# - Map `'N'` and `'E'` coordinate strings to `float("nan")` (`.nan` in YAML) to treat them as missing readings.
# - Map `-1.0` in telemetry columns to `float("nan")` to treat unavailable sensor data as missing readings.
#
# :::{important}
# In DataEval Flow configuration, you must map missing numeric values to `.nan` (or `float("nan")` in Python), not
# `null`. If you map to `null`, the column remains typed as mixed (numeric and None) and will be dropped. Mapping to
# `.nan` preserves the numeric data type and records the entries in the missing data category.
# :::
#
# You can now apply the policy to your pipeline configuration and re-run the task.

# %%
from dataeval_flow.config import MetadataPolicyConfig

policy = MetadataPolicyConfig.model_validate(
    {
        "name": "seadrone",
        "corrections": [
            {"kind": "parse_datetime", "factor": "date_time", "every": "day"},
            # Map hemisphere letters to NaN (missing values)
            {"kind": "remap", "factor": "latitude", "rules": [{"match": "N", "to": float("nan")}]},
            {"kind": "remap", "factor": "longitude", "rules": [{"match": "E", "to": float("nan")}]},
            # Map -1.0 sentinel values to NaN
            *(
                {"kind": "remap", "factor": name, "rules": [{"match": -1.0, "to": float("nan")}]}
                for name in ("altitude", "compass_heading", "gimbal_heading", "gimbal_pitch", "speed")
            ),
        ],
        # Exclude object identifier
        "exclude": ["object_id"],
        "continuous_factor_bins": triage["suggested_policy"]["continuous_factor_bins"],
    }
)

corrected = config.model_copy(
    update={
        "metadata": [policy],
        "workflows": [triage_workflow.model_copy(update={"metadata": "seadrone"})],
    }
)
result2 = run_task(corrected, task)
print(f"success={result2.success}  health={result2.health['status']}")

# %%
after = result2.steps["factor-triage"].output.data()
print(f"factors : {triage['factor_count']} -> {after['factor_count']}")
print(f"findings: {len(triage['findings'])} -> {len(after['findings'])}")
print(f"blocking: {triage['counts'].get('blocking', 0)} -> {after['counts'].get('blocking', 0)}")

# %% [markdown]
# All blocking findings are resolved, the recovered columns are included as factors, and pipeline health is `ok`. The
# factor count is 17 because you excluded `object_id`.
#
# Total findings decreased from 24 to 21. By resolving the blocking type issues, you exposed secondary distribution
# warnings:
#
# ```text
#   floor_mass  latitude   a quarter of the rows or more hold -1.0
#   floor_mass  longitude  a quarter of the rows or more hold -1.0
#   floor_mass  speed      a quarter of the rows or more hold 0.0
# ```
#
# - For `latitude` and `longitude`, once you resolved string values, numeric distribution analysis detected that `-1.0`
#   is also present as a sentinel value in over 25% of rows.
# - For `speed`, remapping `-1.0` removed the sentinel, leaving a concentration at `0.0`. If you have stationary
#   targets, `0.0` is a valid measurement, but you should be aware that high concentration at one value can affect
#   binning.
#
# You can also see five factors reporting high missing rates where you remapped `-1.0`:
#
# ```text
#   altitude         29% missing
#   compass_heading  32% missing
#   gimbal_heading   32% missing
#   gimbal_pitch     29% missing
#   speed            32% missing
# ```
#
# Remapping `-1.0` to `.nan` ensures that downstream workflows track missing values explicitly rather than aggregating
# them as valid negative numbers.
#
# ### Verification results
#
# You can read the Verified finding in the report to see the results of applying suggested corrections in a trial pass:
#
# ```text
#   latitude: not applied; 1 values still need codes
#   date_time: became a factor, 8 levels
#   frame: 4 bins, 1 empty
#   object_size: 9 bins, 3 empty
# ```
#
# - `latitude` is not applied because remap rules contained placeholder values that you must define.
# - `date_time` is parsed into a categorical factor with 8 daily levels.
# - `frame` is applied with 5 bins, resulting in 4 populated bins and 1 empty bin.

# %% [markdown]
# ## Triage scope and limitations
#
# When using triage, you should keep its scope in mind:
#
# - Triage is structural. It detects mixed data types, high cardinality, and extreme values. It does not validate domain
#   semantics (such as whether coordinates fall in expected ranges).
# - Distribution checks are threshold-based. They flag concentrations where a single value comprises 25% or more of the
#   rows. You must verify smaller sentinel clusters or valid skewed distributions manually.
# - Findings are unranked. They are categorized by operational severity (`blocking`, `warning`, `note`), but you should
#   prioritize them based on the factors your analysis requires.
#
# ## Export the results
#
# You can export triage findings, suggested policies, and verification results to JSON or dictionary formats.
# {doc}`../how_to/read_evaluation_outputs` covers result objects and export formats.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Detect unparseable, mixed-type, and high-cardinality metadata columns.
# - Identify and exclude identifier fields.
# - Remap sentinel and missing values to `.nan`.
# - Apply and verify a standardized `metadata` policy configuration.

# %% [markdown]
# ## Next steps
#
# - {doc}`../how_to/configure_metadata_binning` — Declare cuts and vocabularies for the factors triage reports.
# - {doc}`data_coverage` — Measure representation across factor combinations.
# - {doc}`audit` — Judge factors, labels and leakage across dataset splits.

# %% [markdown]
# ## On your own
#
# To repeat this analysis with your own data or settings, change one of these:
#
# - Pass your own dataset to `DatasetProtocolConfig` in place of `seadrone`. Triage reads only the dataset's metadata.
# - Change `size` in the `Limit` operation, or `seed` in the `Shuffle` operation, to triage a different sample. Use a
#   sample that represents the data you expect, because the recommended policy is only as good as the data it was read
#   from.
# - Change `max_examples` under `factor-issues` to show more or fewer distinct values per column in the report.
# - Edit `corrections` and `exclude` in `MetadataPolicyConfig` for your own factors. Add the policy to the pipeline's
#   `metadata` list, and set the workflow's `metadata` to its name.
# - Change `every` in the `parse_datetime` correction to group timestamps over a period other than `day`.
# - Change `continuous_factor_bins` to declare your own bin counts. The tutorial takes them from the suggested policy.
