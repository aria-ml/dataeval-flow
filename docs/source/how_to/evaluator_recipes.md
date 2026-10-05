# Evaluator Recipes

Each recipe below answers one question with one evaluator. They assume the pipeline already defines `datasets:`
and the sources these recipes name — `train`, `test`, `validation`, `operational`, `labeled`, `unlabeled` — and,
where a recipe embeds, the extractor `bovw_ext` from [Run a single evaluator](run_a_single_evaluator.md). The
[Evaluator Catalog](../reference/evaluators.md) lists every parameter each evaluator takes. An evaluator reports
DataEval's answer to its question; it renders no health verdict.

## Look for shortcuts in your metadata

Which metadata factors predict the class label, on their own or through each other?

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

Notice:

- `metadata:` names a policy, so both evaluators read their factors under the one encoding.
- `balance`'s `factor_source`, left unset, follows the policy's `factor_source` where it sets one.
- Neither task names an extractor — the bias evaluators read metadata only, and refuse one.

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

`site` carries as much mutual information with the class as the class carries with itself; `angle` carries none —
exactly the shortcut the toy data plants.

`label_parity`'s output (trimmed; from a small run):

```json
{
  "shape": "mapping",
  "data": {
    "factors": {
      "shape": "table",
      "columns": ["factor_name", "score", "p_value", "is_significant", "has_insufficient_data"],
      "rows": [
        {"factor_name": "angle", "score": 0.0, "p_value": 0.9999971962630967, "is_significant": false,
         "has_insufficient_data": true},
        {"factor_name": "site", "score": 0.9999999999999999, "p_value": 5.341471565244922e-25,
         "is_significant": true, "has_insufficient_data": false}
      ]
    }
  }
}
```

`site`'s association with the class is significant at any reasonable threshold; `angle`'s is not, and its cell
counts are too small to trust either way.

From Python:

```python
from dataeval_flow import run
from dataeval_flow.config import MetadataPolicyConfig
from dataeval_flow.evaluators.bias import BalanceConfig, ParityConfig

policy = MetadataPolicyConfig(name="standard", exclude=["id"])
balance = run(BalanceConfig(metadata="standard"), dataset, definitions=[policy])
parity = run(ParityConfig(metadata="standard"), dataset, definitions=[policy])
```

## Check class representation against an ontology

Which classes fall short of their expected share of the source?

```yaml
ontologies:
  - name: vehicles
    source: config/label_ontology.jsonld
    concepts:
      - id: http://example.org/cv#FreightCar
        label: Freight Car
        synonyms: [freight_car, freight car]
        parents: [http://example.org/cv#LandVehicle]

evaluators:
  - name: representation
    type: representation
  - name: representation_vehicles
    type: representation
    ontology: vehicles
    expected: {truck: 0.2}

tasks:
  - name: representation
    evaluator: representation
    sources: train
  - name: representation_vehicles
    evaluator: representation_vehicles
    sources: train
```

Notice:

- `representation` names no `ontology:`, so a flat one is synthesized from the dataset's `index2label` — it can
  only ever name classes the dataset already declares.
- `representation_vehicles` names `ontology: vehicles`, an entry under `ontologies:` shared the same way as
  [Declare an ontology](declare_an_ontology.md) sets one up; `expected: {truck: 0.2}` floors `truck` at 20% instead
  of a uniform share.
- Without labels, the run fails, naming the source.
- Expected floors are checked in `extras.violations`; classes without one keep the uniform target.

`representation`'s output (trimmed; from a small run — `representation_vehicles` needs a `truck` class the toy
data does not have, so it is not run here):

```json
{
  "shape": "table",
  "columns": ["concept", "label", "parent", "action", "count", "target", "deficit"],
  "rows": [],
  "extras": {
    "leaf_coverage": 1.0,
    "total_deficit": 0,
    "violations": {"shape": "table", "columns": ["concept", "label", "floor", "actual", "shortfall"], "rows": []},
    "dark_branches": {"shape": "table", "columns": ["concept", "label", "leaves"], "rows": []}
  }
}
```

The worklist is empty because the toy's two classes are evenly split and meet their uniform target; a class
short of its floor would show up as a row here, and in `extras.violations` if the floor came from `expected`.

From Python:

```python
from dataeval_flow import run
from dataeval_flow.evaluators.scope import RepresentationConfig

result = run(RepresentationConfig(), dataset)  # a flat ontology synthesized from index2label
```

## Find regions the data does not cover

Which items sit in embedding-space regions the rest of the data does not cover?

```yaml
evaluators:
  - name: coverage
    type: coverage
    num_observations: 20

tasks:
  - name: coverage_check
    evaluator: coverage
    sources: train
    extractor: bovw_ext
```

Notice:

- The task needs an extractor: coverage measures in embedding space.
- `num_observations` must be fewer than the source's items.
- Without one label per item — an unlabeled or detection dataset — it logs a warning and reports one class, `0`,
  covering every item.
- `extras` holds `uncovered_indices`, `coverage_radius` and `critical_value_radii`.

`coverage_check`'s output (trimmed; from a small run):

```json
{
  "shape": "table",
  "columns": ["class", "count", "uncovered", "uncovered_fraction", "dispersion", "isotropy",
              "near_duplicate_fraction", "assessable"],
  "rows": [
    {"class": "a", "count": 20, "uncovered": 0, "uncovered_fraction": 0.0, "dispersion": 0.9783207970130985,
     "isotropy": null, "near_duplicate_fraction": 0.0, "assessable": true},
    {"class": "b", "count": 20, "uncovered": 1, "uncovered_fraction": 0.05, "dispersion": 1.0216792029869015,
     "isotropy": null, "near_duplicate_fraction": 0.0, "assessable": true}
  ],
  "extras": {
    "uncovered_indices": [7],
    "coverage_radius": 13.830693244934082,
    "critical_value_radii": [11.495150566101074, 11.507107734680176, 11.457376480102539, 11.39704704284668]
  }
}
```

Class `b` has one item, index 7, that the rest of the data does not cover; `critical_value_radii` holds one radius
per item, the value its own neighborhood needed to reach `num_observations`.

From Python:

```python
from dataeval_flow import run
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.evaluators.scope import CoverageConfig

bovw_ext = BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32)
result = run(CoverageConfig(num_observations=20), dataset, extractor=bovw_ext)
```

## Rank the data to label next

Given an unlabeled pool and what is already labeled, which unlabeled items are the best fit for the labeling budget?

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

Notice:

- Sources are `[data, reference]`: the first is the source being ranked, and the second, optional, is the labeled
  set it is ranked against.
- `data` is the ranked item indices, and `extras.scores` their scores, in the same order.

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

`order: hard_first` puts the item the unlabeled set finds hardest relative to the labeled set — the one with the
highest score — first; item 7 is the top labeling candidate here.

From Python:

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

## Detect drift

Has the operational data drifted from the reference it was validated against?

```yaml
evaluators:
  - name: mmd
    type: drift-mmd
  - name: mmd_chunked
    type: drift-mmd
    chunking:
      chunk_size: 100
      threshold: [zscore, 3.0]
  - name: wasserstein
    type: drift-wasserstein

tasks:
  - name: drift
    evaluator: mmd
    sources: [train, operational]
    extractor: bovw_ext
  - name: drift_by_chunk
    evaluator: mmd_chunked
    sources: [train, operational]
    extractor: bovw_ext
  - name: drift_against_validation
    evaluator: wasserstein
    sources: [train, validation, operational]
    extractor: bovw_ext
```

Notice:

- Sources run the reference first, then the data to test.
- `chunking:` tests each chunk of the data against the spread of the reference's chunks; DataEval needs at least
  three reference chunks, so `chunk_size` must be small enough to give it that many.
- `drift-wasserstein` takes a validation source between the two.

`drift`'s output (trimmed; from a small run):

```json
{
  "shape": "mapping",
  "data": {
    "drifted": true,
    "threshold": 0.05,
    "distance": 0.41380244493484497,
    "metric_name": "mmd2",
    "details": {"p_val": 0.0, "distance_threshold": 0.01657271385192871},
    "feature_names": []
  }
}
```

`drift_by_chunk`'s output (trimmed; from a small run — the toy sources hold 40 items each, too few for
`chunk_size: 100`, so this excerpt used `chunk_count: 4` instead):

```json
{
  "shape": "mapping",
  "data": {
    "drifted": true,
    "threshold": 0.030687406659126282,
    "distance": 0.4389404356479645,
    "metric_name": "mmd2",
    "details": {
      "shape": "table",
      "columns": ["key", "index", "start_index", "end_index", "value", "upper_threshold", "lower_threshold",
                  "drifted"],
      "rows": [
        {"key": "[0:9]", "index": 0, "start_index": 0, "end_index": 9, "value": 0.46045851707458496,
         "upper_threshold": 0.030687406659126282, "lower_threshold": -0.0239151269197464, "drifted": true},
        {"key": "[10:19]", "index": 1, "start_index": 10, "end_index": 19, "value": 0.43915778398513794,
         "upper_threshold": 0.030687406659126282, "lower_threshold": -0.0239151269197464, "drifted": true}
      ]
    },
    "feature_names": []
  }
}
```

`drift_against_validation`'s output (trimmed; from a small run):

```json
{
  "shape": "mapping",
  "data": {
    "drifted": true,
    "threshold": 1.4,
    "distance": 5.285390853881836,
    "metric_name": "wasserstein_ratio",
    "details": {
      "ratio": 5.285390853881836,
      "feature_drift": [true, true, true, true, true, true, true, false]
    }
  }
}
```

All three declare drift on this toy's brightened operational set; `drift_by_chunk`'s `details` breaks that call
down by chunk instead of reporting it whole, and `drift_against_validation`'s `feature_drift` has one entry per
embedding dimension.

From Python:

```python
from dataeval_flow import run
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.evaluators.shift import ChunkedDriftConfig, DriftMMDConfig, DriftWassersteinConfig

bovw_ext = BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32)
sources = {"reference": reference_dataset, "test": operational_dataset}

drift = run(DriftMMDConfig(), sources, extractor=bovw_ext)
drift_by_chunk = run(
    DriftMMDConfig(chunking=ChunkedDriftConfig(chunk_size=100, threshold=("zscore", 3.0))),
    sources,
    extractor=bovw_ext,
)

validation_sources = {"reference": reference_dataset, "validation": validation_dataset, "test": operational_dataset}
drift_against_validation = run(DriftWassersteinConfig(), validation_sources, extractor=bovw_ext)
```

## Flag out-of-distribution items

Which items of the operational data sit outside the distribution the reference established?

```yaml
evaluators:
  - name: ood
    type: ood-kneighbors
    distance_metric: euclidean

tasks:
  - name: ood_check
    evaluator: ood
    sources: [train, operational]
    extractor: bovw_ext
```

Notice:

- `is_ood` and `instance_score` hold one value per item of the data to test.
- The console report cuts them to ten plus a count, while `-v` and `result.txt` show them all.
- Cosine distance, the default, barely moves under a brightness change, where euclidean does — hence
  `distance_metric: euclidean` here.

`ood_check`'s output (trimmed; from a small run):

```json
{
  "shape": "mapping",
  "data": {
    "is_ood": [true, true, true, true, true],
    "instance_score": [3128.990966796875, 3156.11474609375, 3145.77978515625, 3122.916748046875, 3138.1005859375],
    "feature_score": null
  }
}
```

Every item of this toy's brightened operational set reads as out of distribution under euclidean distance —
brightening every pixel moves the whole embedding, which euclidean distance is sensitive to.

From Python:

```python
from dataeval_flow import run
from dataeval_flow.config.extractors import BoVWExtractorConfig
from dataeval_flow.evaluators.shift import OODKNeighborsConfig

bovw_ext = BoVWExtractorConfig(name="bovw_ext", vocab_size=512, batch_size=32)
sources = {"reference": reference_dataset, "test": operational_dataset}
result = run(OODKNeighborsConfig(distance_metric="euclidean"), sources, extractor=bovw_ext)
```

## See also

- [Evaluator Catalog](../reference/evaluators.md) — every evaluator and parameter
- [Run a single evaluator](run_a_single_evaluator.md) — the mechanics: defining an evaluator, adding a task, and
  reading the result
- [Declare an ontology](declare_an_ontology.md) — defining a label space `representation` can check against
- [Configure metadata binning](configure_metadata_binning.md) — how a `metadata:` policy encodes the factors bias
  evaluators read
- [Read evaluation outputs](read_evaluation_outputs.md) — the result envelope both evaluators and workflows share
