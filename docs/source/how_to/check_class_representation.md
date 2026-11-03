# Check class representation against an ontology

Which classes fall short of their expected share of the source? Run the
[`representation`](../reference/evaluators.md#representation) evaluator on a source to find out. DataEval's
[Ontology explanation](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html) covers the idea behind it. An
evaluator reports DataEval's answer to its question and renders no health status. The
[Evaluator Catalog](../reference/evaluators.md) lists every parameter it takes. You need a pipeline that defines
`datasets:` and a source named `train`. The second evaluator below also needs an ontology.

## Define the evaluators and tasks

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

Keep four points in mind:

- `representation` names no `ontology:`, so a flat one is synthesized from the dataset's `index2label`. It can only ever
  name classes the dataset already declares.
- `representation_vehicles` names `ontology: vehicles`, an entry under `ontologies:` shared the same way as
  [Declare an ontology](declare_an_ontology.md) sets one up. `expected: {truck: 0.2}` floors `truck` at 20% instead of a
  uniform share.
- Without labels, the run fails and names the source.
- Expected floors are checked in `extras.violations`. Classes without a floor keep the uniform target.

## Read the output

`representation`'s output (trimmed; from a small run). `representation_vehicles` needs a `truck` class the toy data does
not have, so it is not run here:

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

The worklist is empty because the toy's two classes are evenly split and meet their uniform target. A class short of its
floor would show up as a row here, and in `extras.violations` if the floor came from `expected`.

## Run it from Python

```python
from dataeval_flow import run
from dataeval_flow.evaluators.scope import RepresentationConfig

result = run(RepresentationConfig(), dataset)  # a flat ontology synthesized from index2label
```

## Next steps

- [Declare an ontology](declare_an_ontology.md) — Define the label space that `representation` checks against.
- [Run a single evaluator](run_a_single_evaluator.md) — Learn how an evaluator, its task and its result fit together.
- [Read the result envelope](read_the_result_envelope.md) — Read the envelope that evaluators and workflows share.
