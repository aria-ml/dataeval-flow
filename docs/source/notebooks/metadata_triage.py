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
# Find the metadata columns a run silently failed to read, and get a config change for each — using
# the `metadata-triage` workflow on SeaDrone drone telemetry.

# %% [markdown]
# **Who this is for** — T&E engineers and data scientists who have pointed DataEval Flow at a dataset
# and want to know whether its metadata actually arrived, before trusting any bias or coverage number
# computed from it.
#
# **Where this fits** — Triage runs *before* everything else. Every workflow that reads metadata —
# [analysis](data_analysis), [coverage](data_coverage), [splitting](dataset_splitting) — reads whatever
# factors survived preprocessing. A column that mixed numbers with text was set aside; a timestamp that
# named its rows rather than grouping them was dropped; a factor nobody cut was cut from this draw.
# None of that raises an error, and none of it is visible in the results. Triage is where you find out.

# %% [markdown]
# ## What you'll do
#
# - Load SeaDrone, an aerial object-detection dataset whose drone telemetry is genuinely messy
# - Run the `metadata-triage` workflow and read its report
# - See which columns were lost, why, and what each one would take to read
# - Read the **suggested policy** — a YAML block you paste into your config
# - Fill in the one decision the tool refuses to make for you, apply the policy, and re-run
# - Compare the two runs to see what the corrections actually recovered

# %% [markdown]
# ## What you'll learn
#
# - How to run `metadata-triage` and what its five finding categories mean
# - Why a factor can be *blocking* (recoverable and lost) or merely a *note* (nothing to be done)
# - How to read a factor's distribution chart, and what it tells you about a bin count
# - **Where the tool stops guessing** — and why that boundary is the point
# - How verification distinguishes "this correction ran" from "this correction worked"
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
# SeaDrone is an aerial dataset in which a drone flies over water and annotates swimmers, boats and
# buoys. It ships with the drone's telemetry — altitude, heading, speed, GPS — recorded per frame,
# which is exactly the kind of metadata that arrives imperfect.

# %% tags=["remove_output"]
from maite_datasets.object_detection import SeaDrone

# Downloads to ./data/seadrone on first run (~1.2 GB for the validation split).
seadrone = SeaDrone(root="./data", image_set="val", download=True)
print(f"{len(seadrone)} images")

# %% [markdown]
# ## Step 1: Build the workflow configuration
#
# Triage needs the dataset and nothing else — no model, no embeddings, no image statistics. It reads
# the metadata walk and the encoding decisions that came out of it, which makes it the cheapest
# workflow in the suite and the natural first task in a pipeline.
#
# We take a shuffled sample of 200 frames. SeaDrone is ordered by capture, so the first 200 images
# would come from a single flight and would not show the variety the full split has.

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
# The header says it: **15 factors, 18 findings, 3 blocking**. Fifteen columns became factors, and
# eighteen things happened that nobody asked for.
#
# *Blocking* means the run did less than the configuration asked for, without saying so. Three columns
# were dropped outright:
#
# - **`date_time`** — every row holds a different timestamp, so the column names its rows rather than
#   grouping them. There is nothing wrong with the values; what they lack is a vocabulary.
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
# `latitude` holds the text `'N'`. A person reading that immediately suspects a hemisphere marker that
# leaked into a numeric column — and knows that `'N'` on its own carries no latitude at all.
#
# Flow does not know that, and it will not pretend to. Look at what it suggests for this column:
# every distinct value is enumerated, and the code is left `null` for you to fill in.

# %%
latitude = next(f for f in result.data.raw.findings if f.factor == "latitude")

print("category :", latitude.category)
print("severity :", latitude.severity)
print("reasons  :", latitude.reasons)
print("suggested:", latitude.suggestion.corrections)
print("runnable :", latitude.suggestion.complete)

# %% [markdown]
# `complete=False` is the important part. The suggestion is a *skeleton*, not an answer, and nothing
# downstream will apply it until you have filled in the codes.
#
# Four things are inferred automatically, and only four: a number wearing decoration (`6,000`,
# `12 kg`), a timestamp, a null-ish sentinel (`""`, `"N/A"`, `"unknown"`), and — for everything else —
# an enumeration with the codes left blank. Mapping `'N'` to a number would move every bias statistic
# computed from this column, and nothing in the report would say it had.

# %% [markdown]
# ### Distributions, and the argument for a bin count
#
# Ten factors were cut into bins that nobody declared, so their edges come from this sample and are not
# stable across draws. A suggested bin count on its own is a number with no argument attached, so the
# report draws the shape it came from:
#
# ```text
#   [warning] frame [unbinned @ unit]
#     ▃▄█▂   ▂    n=17–86
#              < 2630.75  ███████████▉                       34
#      [2630.75, 5083.5)  ██████████████▎                    41
#      [5083.5, 7536.25)  ██████████████████████████████     86
#        [7536.25, 9989)  █████▉                             17
#        [9989, 12441.8)                                      0  empty
#     [12441.8, 14894.5)                                      0  empty
#     [14894.5, 17347.2)                                      0  empty
#             >= 17347.2  ███████▋                           22
# ```
#
# The gap in the sparkline is three empty bins — the automatic cut spent them on a range this sample
# has nothing in. That is the argument for declaring five rather than eight, and you can disagree with
# it, because the evidence is on the page. A bar is never blank unless its bucket is genuinely empty:
# "no rows here" and "one row here" is precisely the distinction a bin count turns on.

# %% [markdown]
# ## Step 3: The suggested policy
#
# Every suggestion is merged into one block, shaped exactly like the `metadata:` section of a config
# file. This is the workflow's actual deliverable.

# %%
print(result.data.raw.suggested_policy_yaml)

# %% [markdown]
# Three kinds of remedy land in one place:
#
# - **`parse_datetime` for `date_time`** — reading each timestamp as the day it falls in gives the
#   column the vocabulary it was missing. No `format` is pinned, because these are ISO-8601 and
#   DataEval reads that without being told.
# - **`remap` for `latitude` and `longitude`** — the skeletons, with a trailing marker on each
#   line you must complete.
# - **`continuous_factor_bins`** — the cuts to pin, carried forward from what the automatic cut
#   actually found rather than replaced with a round number.
#
# The marker appears only where a decision is outstanding. Where triage recognized a sentinel it
# answered the rule itself, with `.nan`, and left the line unmarked.

# %% [markdown]
# ## Step 4: Apply the policy and re-run
#
# Now make the one decision the tool refused to make. `'N'` in a latitude column is a hemisphere letter
# where a coordinate belongs — it records no position, so it reads as missing.
#
# ```{important}
# "Missing" is spelled `.nan`, not `null`. A `remap` target of `null` is simply a non-numeric value:
# the column would still hold 198 numbers and 2 nulls, still have no single type, and still be
# dropped. `.nan` is the value DataEval reads as *no reading was taken* — it makes the column numeric
# and puts those rows on the reserved missing code, where they are counted rather than invented.
#
# The marked lines are the values you must code. Where triage recognized a sentinel itself, it
# already answered with `.nan` and left that line unmarked.
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
        ],
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
# Every blocking finding is gone, and three columns that were not factors now are. The eight findings
# left are all `warning`: cuts and vocabularies still derived from this sample. Pinning those is a
# second pass — export a descriptor once you are happy with the reading, and reference it from
# `encoding:` — and the health line drops to `ok` because nothing is being silently lost any more.
#
# Note what `latitude` became *after* it could be read: an `unbinned` finding with a suggested cut. A
# column has to be readable before anyone can ask how it should be grouped.

# %% [markdown]
# ### What verification told you
#
# Look at the `VERIFIED` section of the first report again. It is not a restatement of the suggestions —
# it is what happened when they were applied:
#
# ```text
#   latitude: not applied; 1 values still need codes
#   date_time: became a factor, 8 levels
#   frame: 4 bins, 1 empty
#   object_size: 9 bins, 3 empty
# ```
#
# Three different outcomes, and the distinction matters. `latitude` was **not applied**, because its
# suggestion was incomplete — verification will not report recovery from a placeholder. `date_time`
# genuinely became a factor. And `frame` was asked for five bins and came back with four, one of them
# empty, which is honest rather than flattering: a correction can be well formed, run cleanly, and
# still not do what you hoped.

# %% [markdown]
# ## What triage does not see
#
# Triage reports what *failed to read*. That is narrower than "everything wrong with your metadata",
# and the difference is worth knowing.
#
# Look again at `latitude`'s numeric values: `-1, 47.671928, 47.671942, ...`. SeaDrone writes `-1` where
# the drone's telemetry was unavailable. That is a sentinel, exactly like the empty `date_time` — but
# it is already a *number*, so the column read cleanly and triage has nothing to report. It will sit in
# its own bin, thousands of kilometres from the real coordinates, and quietly distort every statistic
# computed over that factor.
#
# A sentinel that shares the column's type is invisible here. Reading the distribution chart for a
# factor with a lonely bucket at one end is how you catch it, and a `remap` with a `range` rule is how
# you fix it.

# %% [markdown]
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
# You ran `metadata-triage` against a dataset with genuinely imperfect telemetry and learned that three
# of its columns never became factors at all. You read why each one failed, saw the values behind the
# failure, and got a config block that addresses them. You made the one judgement the tool refused to
# make, applied the result, and confirmed what it recovered.
#
# The habit worth taking away: run triage first, and treat a blocking finding as a claim that your
# later numbers are computed over less data than you think.

# %% [markdown]
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
