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
# Find the metadata columns a run failed to read, and get a config change for each, using the
# `metadata-triage` workflow on SeaDrone drone telemetry.

# %% [markdown]
# **Who this is for** — T&E engineers and data scientists who have pointed DataEval Flow at a dataset
# and want to know whether its metadata actually arrived, before trusting any bias or coverage number
# computed from it.
#
# **Where this fits** — Triage runs before everything else. [Analysis](data_analysis),
# [coverage](data_coverage) and [splitting](dataset_splitting) all read whatever factors survived
# preprocessing. A column mixing numbers with text is set aside. A timestamp holding one value per
# row is dropped. A factor nobody cut gets cut from this sample. None of that raises an error or
# shows up in the results. Triage is where you find out.

# %% [markdown]
# ## What you'll do
#
# - Load SeaDrone, an aerial object-detection dataset whose drone telemetry is genuinely messy
# - Run the `metadata-triage` workflow and read its report
# - See which columns were lost, why, and what each one would take to read
# - Meet two problems that read cleanly and still distort a cut: an identifier, and a value a
#   quarter of the column sits on
# - Read the **suggested policy** — a YAML block you paste into your config
# - Fill in the one decision the tool refuses to make for you, apply the policy, and re-run
# - Compare the two runs to see what the corrections actually recovered

# %% [markdown]
# ## What you'll learn
#
# - How to run `metadata-triage` and what its five finding categories mean
# - Why a factor can be *blocking* (recoverable and lost) or merely a *note* (nothing to be done)
# - How to read a factor's distribution chart, and what it tells you about a bin count
# - Where the tool stops guessing, and why
# - How verification distinguishes "this correction ran" from "this correction worked"
# - Why a withheld suggestion is itself a finding
# - What triage cannot see, so you know what it does not cover

# %% [markdown]
# ## What you'll need
#
# - `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`)
# - `maite-datasets` (provides SeaDrone)
# - Internet connection — SeaDrone's validation split downloads on first run (about 1.2 GB)
#
# ```{note}
# Unlike the other tutorials, this one has no `.yaml` twin. SeaDrone arrives as an in-memory MAITE
# dataset, and an in-memory dataset is configured through `DatasetProtocolConfig`, which is
# deliberately not serializable — there is no path or format string that would reproduce it. The
# workflow itself is ordinary and works the same from YAML against a dataset on disk.
# ```

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Data Preparation: Load SeaDrone
#
# SeaDrone is an aerial dataset: a drone flies over water and annotates swimmers, boats and buoys.
# It ships with per-frame telemetry — altitude, heading, speed, GPS — which is the kind of metadata
# that arrives imperfect.

# %% tags=["remove_output"]
from maite_datasets.object_detection import SeaDrone

# Downloads to ./data/seadrone on first run (~1.2 GB for the validation split).
seadrone = SeaDrone(root="./data", image_set="val", download=True)
print(f"{len(seadrone)} images")

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# Triage needs the dataset and nothing else: no model, no embeddings, no image statistics. It reads
# the metadata walk and the encoding decisions from it, which makes it the cheapest workflow here and
# a good first task in a pipeline.
#
# Take a shuffled sample of 200 frames. SeaDrone is ordered by capture, so the first 200 images come
# from a single flight and miss the variety of the full split.

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

# %%
result = run_task(task, config)
print(f"success={result.success}  health={result.health['status']}")

# %%
print(result.report())

# %% [markdown]
# ### Reading the report
#
# The header says it: **15 factors, 24 findings, 3 blocking**. Fifteen columns became factors, and
# twenty-four things happened that nobody asked for. Read the blocking three first; the rest are
# reported once each and collapse into two short lists.
#
# *Blocking* means the run did less than the configuration asked for, without saying so. Three columns
# were dropped outright:
#
# - **`date_time`** — every row holds a different timestamp, so the column identifies rows instead
#   of grouping them. Nothing is wrong with the values; they lack a vocabulary.
# - **`latitude`** and **`longitude`** — 198 rows hold a number and 2 hold text. A column whose values
#   disagree about what they *are* has no single type, so the walk set it aside rather than guessing.
#
# The ratio bar shows that split at a glance, and the report prints the actual values, because a
# correction has to be written against what is literally in the column:
#
# ```text
#   [blocking] latitude [mixed_types @ unit]
#     ████████████████████  198 numeric, 2 text
#     numeric reads: -1, 47.671928, 47.671942, 47.671971, 47.672015, 47.672055 (+114 more)
#     text reads: 'N'
# ```

# %% [markdown]
# ### Where the tool stops guessing
#
# `latitude` holds the text `'N'`. You can see that is a hemisphere marker in a numeric column, and
# that `'N'` alone carries no latitude.
#
# Flow does not know that and will not pretend to. Its suggestion lists every distinct value and
# leaves the code `null` for you to fill in.

# %%
latitude = next(f for f in result.data.raw.findings if f.factor == "latitude")

print("category :", latitude.category)
print("severity :", latitude.severity)
print("reasons  :", latitude.reasons)
print("suggested:", latitude.suggestion.corrections)
print("runnable :", latitude.suggestion.complete)

# %% [markdown]
# `complete=False` is the part to note. The suggestion is a skeleton, not an answer, and nothing
# downstream applies it until you fill in the codes.
#
# Four things are inferred, and only four: a number wearing decoration (`6,000`, `12 kg`), a
# timestamp, a null-ish sentinel (`""`, `"N/A"`, `"unknown"`), and for anything else an enumeration
# with blank codes. Mapping `'N'` to a number would move every bias statistic from this column, and
# nothing in the report would say so.

# %% [markdown]
# ### Distributions, and the argument for a bin count
#
# Ten factors were cut into bins that nobody declared, so their edges come from this sample and
# are not stable across draws. A suggested bin count on its own is a number with no argument
# attached, so the report draws the shape it came from, and deliberately **not** at that bin count:
#
# ```text
#   frame — declare 5 bins
#     178 ▂▂▂▁▁▁▃▁▂▁▄█▅▁ ▂▂▁                 ▁▁▁▁▁ 1.98e+04
#         ├──────████┃█──────────────────────────┤  p25 3465 · p50 5715 · p75 6240
# ```
#
# The top line is a histogram at a fixed display width, the same for every factor. The bottom is a
# box plot on the same scale: quartile box, median, whiskers to the extremes.
#
# ```{note}
# Charts here are built from quantiles, not from the encoding. Drawing a factor at its own bin
# count would show you the answer you were asked to check: a two-bin cut draws two bars, which says
# nothing about whether two was right, and a value holding a quarter of the column can hide inside
# one of them.
# ```
#
# Read `frame`: mass in the low-middle, a wide empty stretch, a small group at the top. That gap is
# real structure, and it argues for five bins rather than the eight the automatic cut used.
#
# A box reaching an end with no whisker means a quarter of the rows or more sit on that extreme.
# Compare `frame` with `altitude`, whose box is flush left.

# ## Step 3: The suggested policy
#
# Every suggestion is merged into one block, shaped exactly like the `metadata:` section of a config
# file. This is the workflow's actual deliverable.

# %%
print(result.data.raw.suggested_policy_yaml)

# %% [markdown]
# Four kinds of remedy land in one place:
#
# - **`parse_datetime` for `date_time`** reads each timestamp as the day it falls in, giving the
#   column a vocabulary. No `format` is pinned: these are ISO-8601, which DataEval reads unaided.
# - **`remap` for `latitude` and `longitude`** are skeletons, with a trailing marker on each line
#   you must complete.
# - **`remap` for the five telemetry columns that floor at `-1.0`** are skeletons too. Confirming a
#   marker is your call.
# - **`exclude` for `object_id`**, and **`continuous_factor_bins`** for the four factors worth
#   pinning.
#
# Note which factors are absent from `continuous_factor_bins`. Ten were cut from this sample, but a
# cut derived from values containing a marker pins an accident, and an identifier should not be cut
# at all. Six of the ten get no bin count, and the report says why for each. A withheld suggestion
# is a finding.
#
# Markers appear only where a decision is outstanding. Where triage recognized a sentinel by its
# spelling (`""`, `"N/A"`) it answered the rule itself with `.nan` and left the line unmarked.

# %% [markdown]
# ## Step 4: Apply the policy and re-run
#
# Now make the decisions the tool refused to make. Both need someone who knows the data:
#
# - `'N'` in a latitude column is a hemisphere letter where a coordinate belongs. It records no
#   position, so it reads as missing.
# - `-1.0` floors five telemetry columns. Altitude, heading and speed can legitimately read at or
#   below zero, so the tool will not decide this for you. In SeaDrone it is the drone's "telemetry
#   unavailable" value, so it also reads as missing.
#
# ```{important}
# "Missing" is spelled `.nan`, not `null`. A `remap` target of `null` is just a non-numeric value:
# the column would hold 198 numbers and 2 nulls, still have no single type, and still be dropped.
# DataEval reads `.nan` as "no reading taken". It makes the column numeric and puts those rows on
# the reserved missing code, where they are counted rather than invented.
#
# Marked lines are the values you must code. Sentinels triage recognized itself are already
# answered with `.nan` and left unmarked.
# ```
#
# Paste the block, replace the marked nulls, and attach it to the pipeline as a named policy.

# %%
from dataeval_flow.config.schemas import MetadataPolicyConfig

policy = MetadataPolicyConfig.model_validate(
    {
        "name": "seadrone",
        "corrections": [
            {"kind": "parse_datetime", "factor": "date_time", "every": "day"},
            # 'N' and 'E' are hemisphere letters, not coordinates: they record no position,
            # so they read as missing -- `float("nan")`, which is `.nan` in YAML.
            {"kind": "remap", "factor": "latitude", "rules": [{"match": "N", "to": float("nan")}]},
            {"kind": "remap", "factor": "longitude", "rules": [{"match": "E", "to": float("nan")}]},
            # SeaDrone's telemetry writes -1 where the drone recorded nothing.
            *(
                {"kind": "remap", "factor": name, "rules": [{"match": -1.0, "to": float("nan")}]}
                for name in ("altitude", "compass_heading", "gimbal_heading", "gimbal_pitch", "speed")
            ),
        ],
        # An identifier groups nothing -- one value per detection -- so it is not a factor.
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
# Every blocking finding is gone, three columns that were not factors now are, and health reads
# `ok`. The factor count is 17 rather than 18 because you dropped one: `object_id` was never worth
# having.
#
# Findings fall only from 24 to 21, which is the interesting part. **Triage is iterative.** Fixing
# one layer makes the next visible. Three things surfaced that could not be seen before:
#
# ```text
#   floor_mass  latitude   a quarter of the rows or more hold -1.0
#   floor_mass  longitude  a quarter of the rows or more hold -1.0
#   floor_mass  speed      a quarter of the rows or more hold 0.0
# ```
#
# `latitude` and `longitude` carry the same `-1` marker as the telemetry columns. It is visible in
# the first report, in `numeric reads: -1, 47.671928, …`, but could not be reported while the column
# was unreadable: a column that never became a factor has no distribution to describe. Repairing the
# text made its numbers describable.
#
# `speed` is the opposite case, and the reason findings are worded as shapes rather than diagnoses.
# Its `-1` is gone. The remaining mass is at `0.0`, and a boat at rest reads zero. That is a real
# reading, not a marker, but a quarter of the column sitting on it still means any cut describes
# the mass rather than the spread.
#
# And five `degenerate` findings appeared where the sentinels were:
#
# ```text
#   altitude         29% missing
#   compass_heading  32% missing
#   gimbal_heading   32% missing
#   gimbal_pitch     29% missing
#   speed            32% missing
# ```
#
# This is the most useful thing the run tells you, and it was invisible before. Roughly a third of
# SeaDrone's telemetry was never recorded. Until you coded `-1` as missing it sat in the lowest bin,
# counted as a real altitude of −1 metres and averaged in. It is now on the missing code, reported
# rather than silently included. The dataset did not change; what you can see about it did.
#
# Check this before running bias analysis over these factors. A third of the rows forming their own
# group is not a defect to fix, it is a fact to know.
#
# Note what `latitude` became once it could be read: an `unbinned` finding with a suggested cut. A
# column has to be readable before you can ask how to group it.

# ### What verification told you
#
# The `VERIFIED` section of the first report is not a restatement of the suggestions. It is what
# happened when they were applied:
#
# ```text
#   latitude: not applied; 1 values still need codes
#   date_time: became a factor, 8 levels
#   frame: 4 bins, 1 empty
#   object_size: 9 bins, 3 empty
# ```
#
# Three different outcomes. `latitude` was **not applied** because its suggestion was incomplete;
# verification does not report recovery from a placeholder. `date_time` became a factor. `frame` was
# asked for five bins and came back with four, one empty. A correction can be well formed, run
# cleanly, and still not do what you hoped.

# %% [markdown]
# ## What triage does not see
#
# Triage reports what failed to read, plus a few shapes that read cleanly and mean nothing. That is
# narrower than "everything wrong with your metadata". Know the boundary before you trust a clean
# report.
#
# **Evidence is structural, never semantic.** Every rule states a shape: values disagreeing about
# their type, a value that never repeats, a number flooring several columns. None of them know what
# a column means. `latitude` was flagged because `'N'` is text among numbers, not because a
# hemisphere letter in a coordinate is absurd.
#
# **A mass is reported; its meaning is not.** `min == p25` says a quarter of a column sits on its
# lowest value, and no more. SeaDrone's `-1` is a marker, `speed`'s `0.0` is a boat at rest, and the
# report cannot tell them apart. A marker held by fewer than a quarter of the rows is not reported.
#
# **Nothing is ranked by consequence.** All three blocking findings print alike, though losing
# `latitude` differs from losing `object_id`, which you would rather lose. Triage tells you what
# happened to your metadata. Whether it mattered is a question about your analysis.

# ## Results Exploration: Export results
#
# The findings, the suggested policy and the verification all travel in the result envelope, so a
# triage run archives and re-reads like any other workflow.

# %%
json_str = result.export(fmt="json")
print(f"JSON output: {len(json_str)} characters")

# %% [markdown]
# ## Conclusion
#
# You ran `metadata-triage` against a dataset with imperfect telemetry. Three columns had never
# become factors. One was an identifier cut into bins as if it were a measurement. Five carried a
# marker value that read as a number and skewed every cut from them. You saw why each happened, the
# values behind it, and got one config block covering all four.
#
# You then made the judgements the tool refused to make: what `'N'` means, and whether `-1` is a
# reading. That revealed that about a third of SeaDrone's telemetry was never recorded, which was
# true before you started but not visible.
#
# Run triage first, and treat a blocking finding as a claim that your later numbers cover less data
# than you think.

# ## What's next
#
# - [Analyze dataset quality across splits](data_analysis) — now that the factors are actually there
# - [Assess dataset coverage](data_coverage) — metadata gaps, once the metadata is trustworthy
# - [Run a full evaluation pipeline end to end](end_to_end)

# %% [markdown]
# ## Related guides
#
# - [Configure metadata binning](../how_to/configure_metadata_binning) — declaring cuts and vocabularies
# - [Build dataset views](../how_to/build_dataset_views) — the `Shuffle` and `Limit` used above
# - [Read evaluation outputs](../how_to/read_evaluation_outputs) — the result envelope and its exports
# - [Reuse results with cache](../how_to/reuse_results_with_cache)
