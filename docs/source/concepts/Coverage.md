# Dataset Coverage

A dataset can be clean and still be the wrong dataset. Cleaning asks whether the samples you collected are sound.
**Coverage** asks whether they are complete, meaning whether the collection spans the conditions the model will meet in
operation. The failure is silent. A class that was never collected produces no outlier, no duplicate and no bad label,
and nothing in the data signals its absence.

Coverage is measured along two orthogonal axes.

The first axis is which categories you have, a label-space question. Counting the labels present in a dataset can only
tell you about classes that appear at least once. To notice a class with zero examples, you need an external statement
of what the sanctioned label space is. That statement is an {term}`ontology <Ontology>`, a machine-readable vocabulary
of the concepts in the domain and how they relate. Checked against one, a dataset's labels can be validated, and the
gaps between the sanctioned vocabulary and the observed one become nameable.

The second axis is how varied each category is, an embedding-space question. A class can be plentiful by count and still
be degenerate: every example drawn from one sensor, one lighting condition, one pose. Measured in
{term}`embedding <Embedding>` space, that shows up as clustering, low effective dimensionality, or near-duplication.
These signals show that the class occupies far less of the representation space than its sample count suggests.

In DataEval Flow, each axis is a preset, a workflow type whose settings expand to a chain of steps. The
[`taxonomy`](../reference/presets.md#taxonomy) preset checks the observed labels against a declared ontology. The
[`scope`](../reference/presets.md#scope) preset lists the classes that fall short of their expected share and, when an
{term}`extractor <Extractor>` is configured, adds per-class embedding variety and dimensional completeness analysis. Run
both as two tasks on one source to measure both axes. The [`bias`](../reference/presets.md#bias) preset covers class
balance and the metadata factors, including the class-factor combinations that a factor tied to the class leaves
under-represented. Run it on the same source beside them. As with every other workflow, the orchestration layer makes
this a declarative, reproducible, provenance-carrying pipeline step. The underlying measures are DataEval's.

DataEval's
[Dataset Bias and Coverage explanation](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html) is the
authoritative account of the science. It covers the coverage and balance measures themselves. Its
[Ontology explanation](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html) defines what an ontology is and
the reconciliation, alignment, and validation operations performed over one.

## When to use it

Run coverage assessment before training and before you fix a reference set. Do it early enough that a gap can still be
closed by collecting more data, which is the only real remedy. Run it again whenever the operational scope changes,
because coverage is defined relative to the conditions you expect to face.

Coverage also bounds what downstream monitoring can tell you. A drift or OOD baseline is fit on a reference dataset, so
a blind spot in that reference is a blind spot in the detector. Data that drifts into a region the reference never
covered cannot be recognized as familiar or flagged as strange in any principled way.

## Related concept pages

- [Data Quality and Cleaning](DataQualityAndCleaning.md): The complementary question of whether the data you do have is
  sound.
- [Preprocessing and Feature Extraction](PreprocessingAndExtraction.md): The embedding space in which variety and
  dimensional completeness are measured.
- [Distribution Shift](DistributionShift.md): Why a gap in the reference set limits what drift and OOD detection can
  detect.

## See this in practice

### Tutorials

- [Assess dataset coverage](../notebooks/data_coverage.py): Run the `scope` and `taxonomy` presets end to end on one
  source, with and without an extractor.

### How-to guides

- [Declare an ontology](../how_to/declare_an_ontology.md): Define the sanctioned label space that `taxonomy` checks
  against.
- [Check class representation against an ontology](../how_to/check_class_representation.md): Run the `representation`
  evaluator to find the classes that fall short of their expected share.
- [Find regions the data does not cover](../how_to/find_uncovered_regions.md): Run the `coverage` evaluator on
  embeddings to find the items in regions the rest of the data does not cover.

### Reference

- [`scope`](../reference/presets.md#scope): The Preset Catalog entry with the preset's chain, settings and `checks:`
  defaults.
- [`taxonomy`](../reference/presets.md#taxonomy): The Preset Catalog entry with the preset's chain, settings and
  `checks:` defaults.

## References

1. DataEval, [Dataset Bias and Coverage](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html). The
   coverage and balance measures.
2. DataEval, [Ontology](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html). The label-space vocabulary,
   and the reconciliation, alignment, and validation operations.
