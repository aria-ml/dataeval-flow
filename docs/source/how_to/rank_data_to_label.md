# Rank the data to label next

Given an unlabeled pool and what is already labeled, which unlabeled items are the best fit for the labeling budget? Run
the [`prioritization`](../reference/evaluators.md#prioritization) evaluator on the pool to rank it.
[Data Prioritization](../concepts/Prioritization.md) explains how the ranking works. An evaluator reports DataEval's
answer to its question and renders no health status. The [Evaluator Catalog](../reference/evaluators.md) lists every
parameter it takes. You need a pipeline that defines `datasets:`, the sources `unlabeled` and `labeled`, and the
extractor `bovw_ext` from [Run a single evaluator](run_a_single_evaluator.md).

## Define the evaluator and task

```yaml
evaluators:
  - name: next_to_label
    type: prioritization
    order: hard_first

tasks:
  - name: rank_unlabeled
    evaluator: next_to_label
    sources: [unlabeled, labeled]
    extractor: bovw_ext
```

Keep two points in mind:

- Sources are `[data, reference]`. The first is the source being ranked, and the second, optional, is the labeled set it
  is ranked against.
- `data` is the ranked item indices, and `extras.scores` their scores, in the same order.

## Read the output

`rank_unlabeled`'s output (trimmed; from a small run):

```json
{
  "shape": "array",
  "data": [7, 9, 21, 16, 32],
  "extras": {
    "scores": [0.5686631202697754, 0.4072543978691101, 0.40523257851600647, 0.4052129089832306, 0.40487241744995117]
  }
}
```

`order: hard_first` puts the item the unlabeled set finds hardest relative to the labeled set first. That item has the
highest score, and item 7 is the top labeling candidate here.

## Run it from Python

```python
from dataeval_flow import run
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.evaluators.scope import PrioritizationConfig

bovw_ext = BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32)
result = run(
    PrioritizationConfig(order="hard_first"),
    {"data": unlabeled_dataset, "reference": labeled_dataset},
    extractor=bovw_ext,
)
```

## Next steps

- [Prioritize unlabeled data for labeling](../notebooks/data_prioritization.py) — Rank a real pool and choose what to
  annotate first.
- [Data Prioritization](../concepts/Prioritization.md) — Learn how a ranking turns a fixed budget into the largest
  improvement.
- [Read the result envelope](read_the_result_envelope.md) — Read the envelope that evaluators and workflows share.
