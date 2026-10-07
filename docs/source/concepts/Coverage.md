# Dataset Coverage

A dataset can be perfectly clean and still be the wrong dataset. Cleaning asks
whether the samples you collected are *sound*; **coverage** asks whether they are
*complete* — whether the collection actually spans the conditions the model will
meet in operation. The failure mode is silent: a class that was never collected
produces no outlier, no duplicate, and no bad label; nothing in the data signals
its absence.

Coverage is measured along two orthogonal axes.

**Which categories you have** is a *label-space* question. Counting the labels
present in a dataset can only tell you about classes that appear at least once; to
notice a class with zero examples you need an external statement of what the
sanctioned label space *is*. That statement is an {term}`ontology <Ontology>` — a
machine-readable vocabulary of the concepts in the domain and how they relate.
Checked against one, a dataset's labels can be validated, and the gaps between the
sanctioned vocabulary and the observed one become nameable.

**How varied each category is** is an *embedding-space* question. A class can be
plentiful by count and still be degenerate: every example drawn from one sensor,
one lighting condition, one pose. Measured in {term}`embedding <Embedding>` space,
that shows up as clustering, low effective dimensionality, or near-duplication —
signals that the class occupies far less of the representation space than its
sample count suggests.

In DataEval Flow, each axis is a preset, a workflow type whose settings expand to a
chain of steps. The `taxonomy` preset checks the observed labels against a
declared ontology. The `scope` preset lists the classes that fall short of
their expected share and, when an {term}`extractor <Extractor>` is configured, adds
per-class embedding variety and dimensional completeness analysis. Run both as two tasks
on one source to measure both axes. Class balance and the metadata factors, including the
class-factor combinations a factor tied to the class leaves under-represented, are the
`bias` preset's; run it on the same source beside them. As with every other workflow, the orchestration layer's
contribution is making this a declarative, reproducible, provenance-carrying
pipeline step; the underlying measures are DataEval's.

The science is explained authoritatively in DataEval's
[Dataset Bias and Coverage explanation](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html),
which covers the coverage and balance measures themselves, and its
[Ontology explanation](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html),
which defines what an ontology is and the reconciliation, alignment, and validation
operations performed over one.

## When to use it

Run coverage assessment *before* training and *before* fixing a reference set —
early enough that a gap can still be closed by collecting more data, which is the
only real remedy. Re-run it whenever the operational scope changes; coverage is
defined relative to the conditions you expect to face.

Coverage also bounds what downstream monitoring can tell you. A drift or OOD
baseline is fit on a reference dataset, so a blind spot in that reference is a blind
spot in the detector: data drifting into a region the reference never covered cannot
be recognized as familiar or flagged as strange in any principled way.

## Related concept pages

- [Data Quality and Cleaning](DataQualityAndCleaning.md) — the complementary
  question of whether the data you *do* have is sound
- [Preprocessing and Feature Extraction](PreprocessingAndExtraction.md) — the
  embedding space in which variety and dimensional completeness are measured
- [Distribution Shift](DistributionShift.md) — why a gap in the reference set
  limits what drift and OOD detection can detect

## See this in practice

### Tutorials

- [Assess dataset coverage](../notebooks/data_coverage.py) — the `scope`
  and `taxonomy` presets end to end on one source, with and without an extractor

### How-tos

- [Declare an ontology](../how_to/declare_an_ontology.md) — define the sanctioned
  label space `taxonomy` checks against

### Reference

- [Preset Catalog: `scope`](../reference/presets.md#scope) and
  [Preset Catalog: `taxonomy`](../reference/presets.md#taxonomy): each preset's chain, settings and `checks:`
  defaults

### Authoritative reference

- DataEval —
  [Dataset Bias and Coverage](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html)
  (coverage and balance measures)
- DataEval —
  [Ontology](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html)
  (label-space vocabulary, reconciliation, alignment, and validation)
