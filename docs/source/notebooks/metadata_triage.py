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
# In this tutorial, you will identify metadata columns that failed to load or parse, and generate
# configuration fixes using the `metadata-triage` workflow on SeaDrone telemetry.

# %% [markdown]
# **Who this is for** — Engineers and data scientists who need to verify dataset metadata before
# computing coverage, drift, or bias metrics.
#
# **Where this fits** — You should run triage before downstream evaluation workflows. Downstream workflows
# like [analysis](data_analysis), [coverage](data_coverage), and [splitting](dataset_splitting) silently drop
# unparseable metadata columns, mixed-type fields, or high-cardinality values without raising errors.
# Triage surfaces these dropped columns so you can configure remediations.

# %% [markdown]
# ## What you'll do
#
# In this tutorial, you will:
# - Load a sample of the SeaDrone object-detection dataset with telemetry metadata
# - Run the `metadata-triage` workflow and inspect the report
# - Identify dropped or unparseable columns and review suggested remedies
# - Review factors that parse cleanly but need remediation, such as unique identifiers and sentinel values
# - Review the suggested policy configuration
# - Complete required placeholder values in the policy, apply it, and re-run triage
# - Compare results between the initial and corrected runs

# %% [markdown]
# ## What you'll learn
#
# You will learn:
# - How to run `metadata-triage` and interpret finding categories and severity levels
# - How to interpret factor distribution charts and evaluate bin recommendations
# - How you can configure policies to handle missing values, date parsing, and type conversions
# - How verification tests proposed policies against your dataset
# - The scope and limitations of automated metadata triage

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `maite-datasets` (provides SeaDrone)
# - Internet connection — SeaDrone's validation split downloads on first run (about 1.2 GB)
#
# ```{note}
# Unlike other tutorials, this notebook does not have a corresponding `.yaml` file. SeaDrone is
# loaded in-memory via `DatasetProtocolConfig`, which is not serializable. In standard pipelines,
# you can run the workflow from YAML configuration against on-disk datasets.
# ```

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load SeaDrone
#
# You will use SeaDrone, an aerial object-detection dataset with per-frame telemetry including altitude,
# heading, speed, and GPS coordinates.

# %% tags=["remove_output"]
from maite_datasets.object_detection import SeaDrone

# Downloads to ./data/seadrone on first run (~1.2 GB for the validation split).
seadrone = SeaDrone(root="./data", image_set="val", download=True)
print(f"{len(seadrone)} images")

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# To run triage, you only need dataset metadata. You do not need model predictions, embeddings,
# or image statistics.
#
# You should use a shuffled sample of 200 frames to represent multiple capture sequences across flights.

# %%
from dataeval_flow.config import (
    DatasetProtocolConfig,
    PipelineConfig,
    SourceConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.schemas import (
    MetadataTriageTaskConfig,
    MetadataTriageWorkflowConfig,
)
from dataeval_flow.workflow import run_task

triage_workflow = MetadataTriageWorkflowConfig(
    name="triage",
    max_examples=6,  # distinct values shown per column in the report
)

task = MetadataTriageTaskConfig(
    name="triage_seadrone",
    workflow="triage",
    sources="seadrone_val",
)

config = PipelineConfig(
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
# ## Step 2: Run the triage workflow
#
# You can run the task using `run_task()`, then print the execution status and generated report.

# %%
result = run_task(task, config)
print(f"success={result.success}  health={result.health['status']}")

# %%
print(result.report())

# %% [markdown]
# ### Reading the report
#
# When you run triage, the report summarizes the factors, total findings, and blocking issues.
#
# You should review **blocking** findings first. These indicate metadata columns that could not be
# processed and were dropped from the factor set:
#
# - **`date_time`** — Unique timestamps per row exceed cardinality limits. You must truncate or bucket
#   the values to a broader granularity (such as day).
# - **`latitude`** and **`longitude`** — 198 numeric values and 2 string values. Because the column has
#   mixed types, it is excluded until you harmonize the values.
#
# You can inspect the value counts and sample values printed for each mixed type:
#
# ```text
#   [blocking] latitude [mixed_types @ unit]
#     ████████████████████  198 numeric, 2 text
#     numeric reads: -1, 47.671928, 47.671942, 47.671971, 47.672015, 47.672055 (+114 more)
#     text reads: 'N'
# ```

# %% [markdown]
# ### Incomplete suggestions and required inputs
#
# `latitude` contains the string `'N'`. Because automated triage cannot determine what `'N'` represents,
# it generates a remap rule with a `null` target and sets `complete=False`.
#
# You can inspect individual findings in `result.data.raw.findings`:

# %%
latitude = next(f for f in result.data.raw.findings if f.factor == "latitude")

print("category :", latitude.category)
print("severity :", latitude.severity)
print("reasons  :", latitude.reasons)
print("suggested:", latitude.suggestion.corrections)
print("runnable :", latitude.suggestion.complete)

# %% [markdown]
# When `complete=False`, you must complete the template before you can apply it.
#
# Automated suggestions are generated for:
# - Common numeric formatting (stripping commas, currency symbols, or units such as `kg`)
# - Standard ISO-8601 timestamps
# - Recognized null and sentinel strings (`""`, `"N/A"`, `"unknown"`), mapped to `.nan`
#
# For unrecognized non-numeric values in numeric columns, you will receive remap templates with `null`
# placeholders that you must fill in.

# %% [markdown]
# ### Continuous factor distributions and bin suggestions
#
# Ten factors used automatic binning. Because automatically generated bin edges vary across samples,
# you should declare explicit bin counts in configuration to ensure consistent binning across runs.
#
# You can evaluate proposed bin counts using the distribution charts in the report:
#
# ```text
#   frame — declare 5 bins
#     178 ▂▂▂▁▁▁▃▁▂▁▄█▅▁ ▂▂▁                 ▁▁▁▁▁ 1.98e+04
#         ├──────████┃█──────────────────────────┤  p25 3465 · p50 5715 · p75 6240
# ```
#
# The top line shows a fixed-width histogram, and the bottom line shows a box plot with quartiles,
# median, and extreme values.
#
# ```{note}
# You can use distribution plots to inspect raw sample quantiles rather than existing bins, ensuring
# that clusters and skewed extremes remain visible regardless of the current binning strategy.
# ```
#
# In `frame`, you can see values cluster in the lower-middle range, followed by a gap and a smaller cluster
# at the high end. When you see a box reach an edge with no whisker (as in `altitude`), at least 25% of
# values sit directly on that extreme value.

# ## Step 3: Review the suggested policy
#
# Triage merges every suggestion into a single configuration block matching the `metadata:` schema.
# You can inspect this policy directly.

# %%
print(result.data.raw.suggested_policy_yaml)

# %% [markdown]
# You can review the suggested policy sections:
#
# - **`parse_datetime` for `date_time`**: Parses ISO-8601 timestamps and groups values by `day`.
# - **`remap` templates for `latitude` and `longitude`**: Remap rules with placeholder values that you
#   must complete.
# - **`remap` templates for telemetry fields**: Remap rules for `-1.0` values across telemetry columns.
# - **`exclude` for `object_id`**: Excludes the column from factor analysis because it is an item identifier.
# - **`continuous_factor_bins`**: Explicit bin counts for stable continuous factors.
#
# Factors with severe skew (floor mass) or identifier properties are omitted from `continuous_factor_bins`
# until cleaned. You can check the report for the rationale behind each omitted factor.

# %% [markdown]
# ## Step 4: Apply the policy and re-run
#
# You should now complete the placeholder values in the suggested policy:
#
# - Map `'N'` and `'E'` coordinate strings to `float("nan")` (`.nan` in YAML) to treat them as missing readings.
# - Map `-1.0` in telemetry columns to `float("nan")` to treat unavailable sensor data as missing readings.
#
# ```{important}
# In DataEval Flow configuration, you must map missing numeric values to `.nan` (or `float("nan")` in Python),
# not `null`. If you map to `null`, the column remains typed as mixed (numeric and None) and will be dropped.
# Mapping to `.nan` preserves the numeric data type and records the entries in the missing data category.
# ```
#
# You can now apply the policy to your pipeline configuration and re-run the task.

# %%
from dataeval_flow.config.schemas import MetadataPolicyConfig

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
        "continuous_factor_bins": result.data.raw.suggested_policy["continuous_factor_bins"],
    }
)

corrected = config.model_copy(
    update={
        "metadata": [policy],
        "workflows": [triage_workflow.model_copy(update={"metadata": "seadrone"})],
    }
)
result2 = run_task(task, corrected)
print(f"success={result2.success}  health={result2.health['status']}")

# %%
before = result.data.raw
after = result2.data.raw
print(f"factors : {before.factor_count} -> {after.factor_count}")
print(f"findings: {len(before.findings)} -> {len(after.findings)}")
print(f"blocking: {result.metadata.blocking} -> {result2.metadata.blocking}")

# %% [markdown]
# All blocking findings are resolved, the recovered columns are included as factors, and pipeline
# health is `ok`. The factor count is 17 because you excluded `object_id`.
#
# Total findings decreased from 24 to 21. By resolving the blocking type issues, you exposed secondary
# distribution warnings:
#
# ```text
#   floor_mass  latitude   a quarter of the rows or more hold -1.0
#   floor_mass  longitude  a quarter of the rows or more hold -1.0
#   floor_mass  speed      a quarter of the rows or more hold 0.0
# ```
#
# - **`latitude` and `longitude`**: Once you resolved string values, numeric distribution analysis
#   detected that `-1.0` is also present as a sentinel value in over 25% of rows.
# - **`speed`**: Remapping `-1.0` removed the sentinel, leaving a concentration at `0.0`. If you have
#   stationary targets, `0.0` is a valid measurement, but you should be aware that high concentration
#   at one value can affect binning.
#
# Additionally, you can see five factors reporting high missing rates where you remapped `-1.0`:
#
# ```text
#   altitude         29% missing
#   compass_heading  32% missing
#   gimbal_heading   32% missing
#   gimbal_pitch     29% missing
#   speed            32% missing
# ```
#
# Remapping `-1.0` to `.nan` ensures that downstream workflows track missing values explicitly rather than
# aggregating them as valid negative numbers.

# ### Verification results
#
# You can inspect the `VERIFIED` section of the report to see the results of applying suggested
# corrections in a trial pass:
#
# ```text
#   latitude: not applied; 1 values still need codes
#   date_time: became a factor, 8 levels
#   frame: 4 bins, 1 empty
#   object_size: 9 bins, 3 empty
# ```
#
# - **`latitude`**: Not applied because remap rules contained placeholder values that you must define.
# - **`date_time`**: Successfully parsed into a categorical factor with 8 daily levels.
# - **`frame`**: Applied 5 bins, resulting in 4 populated bins and 1 empty bin.

# %% [markdown]
# ## Triage scope and limitations
#
# When using triage, you should keep its scope in mind:
#
# - **Structural, not semantic**: Triage detects mixed data types, high cardinality, and extreme values.
#   It does not validate domain semantics (such as whether coordinates fall in expected ranges).
# - **Threshold-based distributions**: Distribution checks flag concentrations where a single value
#   comprises 25% or more of the rows. You must verify smaller sentinel clusters or valid skewed
#   distributions manually.
# - **Unranked findings**: Findings are categorized by operational severity (`blocking`, `warning`, `note`),
#   but you should prioritize them based on the factors your analysis requires.

# ## Export results
#
# You can export triage findings, suggested policies, and verification results to JSON or dictionary formats.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you:
# - Detected unparseable, mixed-type, and high-cardinality metadata columns
# - Identified and excluded identifier fields
# - Remapped sentinel and missing values to `.nan`
# - Applied and verified a standardized `metadata` policy configuration
#
# You should run metadata triage before downstream workflows to ensure that metadata factors are
# correctly typed, binned, and accounted for in subsequent evaluations.

# ## What's next
#
# - [Analyze dataset quality across splits](data_analysis)
# - [Assess dataset coverage](data_coverage)
# - [Run a full evaluation pipeline end to end](end_to_end)

# %% [markdown]
# ## Related guides
#
# - [Configure metadata binning](../how_to/configure_metadata_binning) — declaring cuts and vocabularies
# - [Build dataset views](../how_to/build_dataset_views) — the `Shuffle` and `Limit` used above
# - [Read evaluation outputs](../how_to/read_evaluation_outputs) — the result envelope and its exports
# - [Reuse results with cache](../how_to/reuse_results_with_cache)
