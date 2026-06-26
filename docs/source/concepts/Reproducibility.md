# Reproducibility

A data-evaluation result is only as trustworthy as your ability to produce it
again. In test and evaluation, a metric that cannot be reproduced cannot be
defended: you cannot tell whether a number changed because the data changed or
because the process did, you cannot independently verify a finding during an
assessment, and you cannot meaningfully compare two evaluations run weeks apart.
**Reproducibility** is the property that the same evaluation, applied to the same
data, yields the same result — every time, on any machine, by anyone.

This is harder than it sounds when an evaluation is assembled by hand. A notebook
of loading, transforming, embedding, and formatting code is rarely fully
version-controlled, differs from engineer to engineer, and carries hidden state.
The glue is where reproducibility quietly breaks. DataEval Flow's design exists to
remove that glue: it makes the *description* of a run the only thing that
determines its result, so that re-running is deterministic by construction. The
app architecture below is the *how*; reproducibility is the *why*.

## The run is described, not coded

The defining choice of DataEval Flow is that a pipeline is **data, not code**. A
run is a declarative configuration — YAML or JSON — that names the datasets,
transforms, extractors, and workflows involved and binds them into tasks. Nothing
about the run lives in a script that has to be read to understand what happened:
the configuration *is* the run. Because that configuration is a complete,
version-controllable artifact, the same configuration applied to the same data
produces the same result, and that fact can be checked into version control,
reviewed, and replayed.

The configuration is organized as named building blocks — datasets, selections,
preprocessors, extractors, workflows — that tasks reference *by name*. Selecting a
workflow by a registered name rather than by importing a Python class is what
keeps the configuration the single source of truth for what runs: there is no
out-of-band code path that can change the outcome.

## Nothing runs until it validates

Every section of a configuration is backed by a typed schema, so a run is
validated before any computation begins. Required fields are checked, unknown
fields are rejected, and every by-name reference — a task's source, its extractor,
its preprocessors — is resolved up front. A misnamed source or an invalid
parameter fails early and predictably rather than partway through an expensive
run. Deterministic, validated inputs are a precondition for a deterministic
result.

## What is developed interactively is what runs in production

The same declarative pipeline executes two ways. **Interactive** tools — a
terminal UI and a simple CLI config builder — help you author and inspect a
pipeline, but the artifact they produce is just the declarative configuration.
**Headless** execution then runs that exact configuration to completion and exits,
with no human in the loop. This is the mode used in containers, CI pipelines, and
scheduled jobs. The split is what makes runs repeatable: a pipeline explored
interactively is captured as configuration, and that same configuration is what
runs unattended later.

## Portability: the same run on a laptop and in a container

Relative paths in a configuration — to datasets, to model files — are resolved
against a **data root** (an explicit argument, the `DATAEVAL_DATA` environment
variable, or the current directory) using conventional sub-directories. That
indirection is what lets the *same* configuration run unchanged on a developer's
laptop and inside a container where the data is mounted at a fixed location. The
[container reference](../reference/containers.md) documents the mounts and
precedence rules that anchor a run.

## Caching preserves the result, not just the time

Re-running an evaluation should be cheap, but caching is only safe if it never
changes the answer. DataEval Flow caches the expensive, reusable products of a
pipeline — extracted {term}`embeddings <Embeddings>`, image statistics, computed
metadata, clustering results — under a key derived from **everything that affects
the result**:

- a fingerprint of the **dataset**,
- the **selection** (subset or ordering) applied to it, and
- a hash of the producing **configuration** — the extractor, its parameters, and
  the preprocessing pipeline.

Because the key incorporates all of these, a stored artifact is reused only when
the data, the selection, and the configuration all match; change the preprocessing
or an extractor parameter and the key changes, so a stale result is never silently
returned. The custom preprocessors and extractors described in
[Preprocessing and Feature Extraction](PreprocessingAndExtraction.md) are written
to have deterministic representations precisely so they contribute stably to these
keys. The consequence is the property that matters: **a cached run and a cold run
of the same configuration are equivalent.** The cache changes how long a run
takes, never what it produces.

## Reproducibility and provenance

Reproducibility answers "can I get the same result again?"; [provenance](Provenance.md)
answers "where did this result come from?" They are two halves of a trustworthy
result. Reproducibility is the guarantee; provenance is the record of the exact
inputs that guarantee depends on. DataEval Flow records the fully-resolved
configuration of every run in its result envelope, so the inputs needed to
reproduce any finding travel with the finding itself.

## When it matters

- **Regression detection** across program increments — re-run a saved
  configuration and compare; any change in the result is attributable to the data,
  not the process.
- **Independent verification** during a product or program assessment — a reviewer
  can replay the exact run.
- **Automated, scheduled evaluation** — a headless, deterministic run drops into a
  container and CI as a repeatable batch job rather than an ad-hoc activity.

## Related concept pages

- [Provenance](Provenance.md) — the recorded lineage that lets a result be
  reproduced and audited
- [Preprocessing and Feature Extraction](PreprocessingAndExtraction.md) — the
  deterministic transforms and extractors that are the primary cached artifacts

## See this in practice

### Tutorials

- [Cleaning a dataset](../notebooks/data_cleaning) — a complete config-driven run,
  end to end
- [Parameter sweeps](../notebooks/parameter_sweep) — repeated evaluation where
  config-keyed caching reuse is most visible
