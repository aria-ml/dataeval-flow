# Look for shortcuts in your metadata

Which metadata factors predict the class label, on their own or through each other? Run the
[`balance`](../reference/evaluators.md#balance) and [`parity`](../reference/evaluators.md#parity) evaluators on a source
to find out. DataEval's [Dataset Bias explanation](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html)
covers the idea behind them. An evaluator reports DataEval's answer to its question and renders no health status. The
[Evaluator Catalog](../reference/evaluators.md) lists every parameter each evaluator takes. You need a pipeline that
defines `datasets:` and a source named `train`.

## Define the evaluators and tasks

```yaml
metadata:
  - name: standard
    exclude: [id]

evaluators:
  - name: balance
    type: balance
    metadata: standard
  - name: parity
    type: parity
    metadata: standard

tasks:
  - name: shortcuts
    evaluator: balance
    sources: train
  - name: label_parity
    evaluator: parity
    sources: train
```

Keep three points in mind:

- `metadata:` names a policy, so both evaluators read their factors under the one encoding.
- `balance`'s `factor_source`, left unset, follows the policy's `factor_source` where it sets one.
- Neither task names an extractor, since the bias evaluators read metadata only and refuse one.

## Read the balance output

`shortcuts`' output (trimmed; from a small run):

```json
{
  "shape": "mapping",
  "data": {
    "balance": {
      "shape": "table",
      "columns": ["factor_name", "mi_value"],
      "rows": [
        {"factor_name": "class_label", "mi_value": 1.0},
        {"factor_name": "angle", "mi_value": 0.0},
        {"factor_name": "site", "mi_value": 1.0}
      ]
    }
  }
}
```

`site` carries as much mutual information with the class as the class carries with itself, and `angle` carries none.
`site` is the shortcut the toy data plants.

## Read the parity output

`label_parity`'s output (trimmed; from a small run):

```json
{
  "shape": "mapping",
  "data": {
    "factors": {
      "shape": "table",
      "columns": ["factor_name", "status", "score", "p_value_adjusted", "is_associated", "has_insufficient_data"],
      "rows": [
        {"factor_name": "angle", "status": "inconclusive", "score": 0.0, "p_value_adjusted": 0.9999971962630967,
         "is_associated": false, "has_insufficient_data": true},
        {"factor_name": "site", "status": "associated", "score": 0.9999999999999999,
         "p_value_adjusted": 1.0682943130489845e-24, "is_associated": true, "has_insufficient_data": false}
      ]
    }
  }
}
```

`site`'s association with the class is significant at any reasonable threshold. `angle`'s is not, and its cell counts
are too small to trust either way. Its `inconclusive` status says so and does not call it independent. A second table,
`cells`, has one row per `site` or `angle` level and class, with a status such as `underrepresented` for a combination
the data holds less of than the rest predicts.

## Run it from Python

```python
from dataeval_flow import run
from dataeval_flow.config import MetadataPolicyConfig
from dataeval_flow.evaluators.bias import BalanceConfig, ParityConfig

policy = MetadataPolicyConfig(name="standard", exclude=["id"])
balance = run(BalanceConfig(metadata="standard"), dataset, definitions=[policy])
parity = run(ParityConfig(metadata="standard"), dataset, definitions=[policy])
```

## Next steps

- [Configure metadata binning](configure_metadata_binning.md) — Choose how a `metadata:` policy encodes the factors the
  bias evaluators read.
- [Run a single evaluator](run_a_single_evaluator.md) — Learn how an evaluator, its task and its result fit together.
- [Read the result envelope](read_the_result_envelope.md) — Read the envelope that evaluators and workflows share.
