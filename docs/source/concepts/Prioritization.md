# Data Prioritization

Labels and human review are expensive; data is abundant. When you have far more
candidate samples than you can afford to label, annotate, or inspect, the question
is not *whether* to review data but *which data first*. **Data prioritization**
ranks an unlabeled or oversized pool so that the most valuable samples come to the
front — the ones whose labeling will teach the model most, or whose inspection is
most likely to surface a problem.

Value here is a geometric notion. Working in the {term}`embedding <Embeddings>`
space, prioritization can rank samples by how *novel* they are — far from existing
labeled data, in sparse or under-covered regions — or by how *hard* they are —
near decision boundaries or anomalous relative to a reference. Ranking by novelty
spends a labeling budget on territory the dataset does not yet cover; ranking by
difficulty concentrates review where a model is most likely to fail. Either way,
prioritization turns a fixed budget into the largest improvement, rather than
labeling at random.

In DataEval Flow, the `data-prioritization` workflow ranks a pool source against a
reference and returns ordered indices with their scores, supporting different
prioritization methods and orderings (hardest-first or easiest-first). The
orchestration layer makes this a declarative, reproducible step; the ranking
methods — and the embedding and performance-estimation machinery they rest on —
are DataEval's. The relevant science is explained authoritatively in DataEval's
[Embeddings explanation](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html)
(the representation space the ranking operates in) and its
[Performance Limits explanation](https://dataeval.readthedocs.io/en/latest/concepts/PerformanceLimits.html)
(how hard samples relate to achievable performance).

## When to use it

Prioritize when a labeling or review budget is the bottleneck: choosing which of
many unlabeled samples to annotate next, selecting a subset for human inspection,
or deciding which abundant data is worth the cost of curation. It pairs naturally
with active-learning loops, where each round labels the highest-ranked samples and
retrains.

## Related concept pages

- [Preprocessing and Feature Extraction](PreprocessingAndExtraction.md) — ranking
  quality depends on the embedding the extractor produces
- [Distribution Shift](DistributionShift.md) — OOD scores are one signal of which
  samples are most worth reviewing

## See this in practice

### Tutorials

- [Prioritizing data](../notebooks/data_prioritization) — ranking a pool with the
  `data-prioritization` workflow

### Authoritative reference

- DataEval —
  [Embeddings](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html)
  and
  [Performance Limits](https://dataeval.readthedocs.io/en/latest/concepts/PerformanceLimits.html)
