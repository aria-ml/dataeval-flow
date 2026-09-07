# Provenance

A finding is only interpretable if you know what produced it. "The dataset has a
12% out-of-distribution rate" means nothing without its lineage: *which* dataset,
under *which* selection, represented by *which* model, evaluated with *which*
configuration, by *which* version of the tool, and *when*. **Provenance** is that
recorded lineage — the context that travels with a result so it can be
interpreted, audited, compared to other runs, and handed to another tool without
out-of-band knowledge.

In test and evaluation this is not a nicety. An assessment archives results and
revisits them later; a program compares an operational dataset against a baseline
collected months earlier; a downstream stage in a pipeline consumes findings it
did not produce. Each of these needs the result to be **self-describing**. A
findings document without provenance is a number stripped of the context that
makes it meaningful. DataEval Flow's output architecture exists to attach that
context automatically — the architecture is the *how*; provenance is the *why*.

## Every result is an envelope

Every workflow returns its result in the same shape: a typed **result envelope**
that pairs a human-readable report with a machine-readable record of the run. The
envelope carries the workflow's name, whether it succeeded, the typed findings, any
errors, and — central to provenance — a **metadata envelope** describing the run
itself. A person reads the report; another tool consumes the structured document;
both are projections of the same envelope, so the two views never drift apart.

## The metadata envelope is the provenance record

The metadata envelope is the machine-readable record of *what was run*, kept
distinct from *what was found*. It captures the context a consumer needs to
interpret the findings, including:

- a schema **version** and a **timestamp**,
- the **dataset identifier(s)** evaluated, and where relevant the label source,
- the **label space** the labels were read under, recorded as a digest when a dataset
  was conformed to a reference vocabulary. The same digest appears on the alignment
  audit, so a result can be matched to the audit that produced its vocabulary,
- the **model**, **preprocessor**, and **selection** identifiers that defined the
  representation,
- human-readable **source descriptions**,
- the **resolved configuration** that produced the result — the fully-resolved
  description of the run, serving as an audit trail, and
- the **tool name and version** plus the **execution time**.

Recording this context in a fixed, versioned structure is what lets results be
archived, compared across runs, and consumed by other tools without out-of-band
knowledge. The same envelope renders a text report for a person — a summary with an
overall health status, optionally followed by per-finding detail and the resolved
configuration — that is a *view* of the result, not a separate computation, so a
reader and an automated consumer are always looking at the same run.

## Resolved configuration closes the loop

The most consequential field in the metadata envelope is the **resolved
configuration**. Because the exact, fully-resolved description of the run travels
with the result, the inputs that produced any finding can be recovered and re-run.
This is precisely where provenance meets [reproducibility](Reproducibility.md):
provenance records the inputs; reproducibility guarantees that replaying them
reproduces the result. A result you can trace is a result you can reproduce.

## A merged corpus records every operand

A source can merge other sources into one corpus. Every envelope field that
names one input then names all of them: **dataset identifiers** list each
operand's dataset, the **selection identifier** lists every view in the chain,
operand views first and the merged source's own view last, the **source
descriptions** spell the merge out, and the **label source** becomes a list
where the operands disagree about where their labels came from.

The resolved configuration recurses too. A merged source expands into a `merge`
list, one entry per operand, with each operand's dataset config and view config
inlined. Nothing about the run has to be looked up elsewhere, so a merged run is
replayable from the envelope alone.

## The label space is what a label means

A label is an integer, and an integer means nothing without the vocabulary it
indexes. Two results that agree on a class name can still disagree on what that
class contains, and no other field separates them. The envelope therefore
records the vocabulary itself, in **label space**.

There is one record per conformed view: one for each operand of a merged source,
plus one for a merged source's own view where that view relabels again. Each
record carries the source whose view applied the rewrite, the ontology and its
digest, the class remapping, the target vocabulary in index order, and a digest
over the three.

One record per operand, rather than one per result, because a merge applies a
different remapping per operand against one shared target. A single record would
have to union those mappings, and a union hashes to a value no audit ever
produced.

Any workflow can declare an ontology, not only `data-coverage`. That is the
join. A coverage audit computes the digest from its own alignment; any other
workflow computes the same digest from the `Relabel` parameters in its source's
view. Declare the same ontology on both and the downstream result's digest
equals the digest of the audit that justified its vocabulary, so matching the
two is a comparison of one value rather than an argument about intent. An
exported dataset carries the same digest in its provenance sidecar, which
extends the join to the corpus itself.

## Provenance enables interoperability

Provenance is also what lets a DataEval Flow result leave the tool and compose
with the rest of the JATIC suite. The mechanism is {term}`MAITE`, the protocol for
interoperable AI/ML components in the JATIC ecosystem, and it operates at both ends
of a run:

- **At the input**, DataEval Flow consumes MAITE-protocol datasets and models
  natively and registers MAITE entry points, so its adapters and task runner are
  discoverable by other MAITE-aware tools. Non-MAITE sources are brought in through
  built-in adapters (HuggingFace, COCO, YOLO, TorchVision, ImageFolder), but native
  interoperability comes from MAITE-compliant inputs.
- **At the output**, the metadata envelope carries the JATIC-required provenance
  fields in a versioned structure, so a result is a well-formed artifact other
  JATIC tools can ingest. Findings produced in one stage compose with the rest of a
  MAITE-conforming pipeline rather than living in an isolated report.

The net effect is that a result is never a dead end: its provenance makes it both
readable by people and traceable by the tools downstream of it.

## Provenance and reproducibility

[Reproducibility](Reproducibility.md) is the guarantee that the same inputs yield
the same result; provenance is the record of what those inputs were. One without
the other is incomplete: a reproducible run whose inputs are not recorded cannot be
re-found, and a richly documented run that cannot be replayed cannot be verified.
DataEval Flow provides both from the same place — the resolved configuration is
recorded in the envelope (provenance) and is the complete description needed to
re-run (reproducibility).

## When it matters

- **Audits and assessments** — a result carries the exact context a reviewer needs
  to judge and replay it.
- **Comparison over time** — versioned, self-describing envelopes can be archived
  and compared run-to-run.
- **Downstream composition** — other JATIC/MAITE tools ingest a result with its
  provenance intact, rather than reverse-engineering what it measured.

## Related concept pages

- [Reproducibility](Reproducibility.md) — the guarantee that the recorded inputs
  reproduce the result
- [Preprocessing and Feature Extraction](PreprocessingAndExtraction.md) — the
  model, preprocessor, and selection identities recorded in the envelope

## See this in practice

### Tutorials

- [Analyzing a dataset](../notebooks/data_analysis.py) — reading a multi-finding
  report and inspecting its result envelope
- [Cleaning a dataset](../notebooks/data_cleaning.py) — exporting a machine-readable
  result alongside the text report
