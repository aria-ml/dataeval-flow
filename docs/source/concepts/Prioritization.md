# Data Prioritization

Labels and human review are expensive, and data is abundant. When you have far more candidate samples than you can
afford to label, annotate, or inspect, the question is which data to handle first. **Data prioritization** ranks an
unlabeled or oversized pool so that the most valuable samples come to the front. These are the samples whose labeling
will teach the model most, or whose inspection is most likely to surface a problem.

Value here is a geometric notion. Working in the {term}`embedding <Embeddings>` space, prioritization can rank samples
by how novel they are or by how hard they are. A novel sample is far from existing labeled data, in a sparse or
under-covered region. A hard sample is near a decision boundary or anomalous relative to a reference. Ranking by novelty
spends a labeling budget on territory the dataset does not yet cover. Ranking by difficulty concentrates review where a
model is most likely to fail.

In DataEval Flow, [`prioritization`](../reference/presets.md#prioritization) is a preset. It ranks each pool source
against the reference, and its `selected` output keeps the top of each ranking (`n:` or `fraction:`). Run as a step of a
custom workflow after a [`quality`](../reference/presets.md#quality) step, it ranks clean data. See
[the preset's chain](../reference/presets.md#prioritization). It supports different prioritization methods and
orderings (hardest-first or easiest-first). The orchestration layer makes this a declarative, reproducible step. The
ranking methods, and the embedding and performance-estimation machinery they rest on, are DataEval's. DataEval's
[Embeddings explanation](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html) covers the representation
space the ranking operates in, and its
[Performance Limits explanation](https://dataeval.readthedocs.io/en/latest/concepts/PerformanceLimits.html) covers how
hard samples relate to achievable performance.

## When to use it

Prioritize when a labeling or review budget is the bottleneck. Typical cases are choosing which of many unlabeled
samples to annotate next, selecting a subset for human inspection, and deciding which abundant data is worth the cost of
curation. It pairs naturally with active-learning loops, where each round labels the highest-ranked samples and
retrains.

## Related concept pages

- [Preprocessing and Feature Extraction](PreprocessingAndExtraction.md): Ranking quality depends on the embedding the
  extractor produces.
- [Distribution Shift](DistributionShift.md): OOD scores are one signal of which samples are most worth reviewing.

## See this in practice

### Tutorials

- [Prioritize unlabeled data for labeling](../notebooks/data_prioritization.py): Rank a pool with the `prioritization`
  workflow.

### How-to guides

- [Rank the data to label next](../how_to/rank_data_to_label.md): Run the `prioritization` evaluator on an unlabeled
  pool against a labeled set.

### Reference

- [`prioritization`](../reference/presets.md#prioritization): The Preset Catalog entry with the preset's chain and
  settings.

## References

1. DataEval, [Embeddings](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html). The representation space
   the ranking operates in.
2. DataEval, [Performance Limits](https://dataeval.readthedocs.io/en/latest/concepts/PerformanceLimits.html). How hard
   samples relate to achievable performance.
