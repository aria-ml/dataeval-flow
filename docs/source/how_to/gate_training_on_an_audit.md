# Gate training on an audit

An `audit` run says whether a dataset is ready to train on, and records a content digest of each split it judged. A
training job can check both before it starts: that the verdict allows training, and that the data it is about to read
is the data that was audited. This guide covers running the audit, reading its verdict, and refusing data whose digest
doesn't match.

## Used in these tutorials

- {doc}`Audit a set of splits before training <../notebooks/audit>`

## Run the audit

Name the splits in order: the first source is train, and every later source is an evaluation split. The example
assumes the pipeline defines `datasets:`, the sources `train`, `validation` and `test`, and the extractor `bovw_ext`, as
[Evaluator recipes](evaluator_recipes.md) does.

```yaml
workflows:
  - name: release-audit
    type: audit
    outliers: {flags: [dimension, pixel, visual], outlier_threshold: adaptive}
    accepted:
      class-imbalance: "Rare class by design; weighted loss in training."

tasks:
  - {name: audit-splits, workflow: release-audit, sources: [train, validation, test], extractor: bovw_ext}
```

Run it, writing the results under `./output`:

```bash
dataeval-flow --config pipeline.yaml --output ./output
```

`output/results/result.json` holds the task's result under its name, `audit-splits`. The
[Preset Catalog](../reference/presets.md#audit) lists every setting, and what `blocking:` and `accepted:` change.

## Read the verdict

`verdict.level` is `not-ready`, `ready-with-caveats` or `ready`. The [Preset Catalog](../reference/presets.md#audit)
gives the rule for each level, and the verdict's other fields. From Python the level is `result.verdict.level`, and in
the JSON it is `verdict.level`.

Gate on it with `--require`, or `result: require:` in the config; `DATAEVAL_REQUIRE` sets the requirement too, and the
flag overrides both. Its value names the worst verdict that passes, and the command exits 4 when a task's verdict is
worse:

```bash
dataeval-flow --config pipeline.yaml --output ./output --require ready-with-accepted-risks
```

A pipeline that reads the JSON can apply the same gate with `jq -e`, which exits 1 when the expression is false:

| `--require` | Refuses | The same gate with `jq -e` |
| --- | --- | --- |
| `ready-with-caveats` | Not ready | `.["audit-splits"].verdict.level \| IN("ready", "ready-with-caveats")` |
| `ready-with-accepted-risks` | Not ready, and Ready with caveats unless its only caveats are accepted risks | `.["audit-splits"].verdict \| .level != "not-ready" and .warnings == [] and .not_assessed == []` |
| `ready` | Anything but Ready | `.["audit-splits"].verdict.level == "ready"` |

```bash
jq -e '.["audit-splits"].verdict | .level != "not-ready" and .warnings == [] and .not_assessed == []' output/results/result.json
```

`ready-with-caveats` refuses only a blocking check that warned with no acceptance covering it.
`ready-with-accepted-risks` lets through an audit whose only caveats are accepted risks; any other warning, or a check
not assessed, stops training until a person fixes the data, accepts the warning under `accepted:`, or gives the check
what it needs. `ready` refuses those too, and an accepted warning as well: an acceptance whose check warns keeps the
verdict at `ready-with-caveats`, so once a warning is accepted, this gate refuses until the data stops warning.

A task whose workflow gives no verdict isn't judged. A task that failed has no verdict, and falls short; each `jq`
expression refuses it too. A run in which no task gives a verdict is refused, and exits 1, before any task starts.

Exit codes take this order: 1, for a failed task or export, unless `result: fail_on: never`; then 4; then 3, which
`fail_on: warning` gives for any warning, an accepted one included, since health counts the warnings the data has and
the verdict records the decision made about them. So an audit whose only caveats are accepted risks passes
`--require ready-with-accepted-risks` and still exits 3 under `fail_on: warning`: keep the default `fail_on: failure`
when you gate with `--require`. Without `--require`, the exit code doesn't read the verdict, and a `not-ready` audit
exits 0 under `fail_on: failure`.

## Refuse data that was not audited

Each split's `content-digest` step records a SHA-256 digest over every item's image and labels, and the class names.
In the training job, recompute it with `dataeval_flow.dataset_digest()` and refuse a mismatch. The digest matches only
the dataset as Flow loads it, before any training transform: load it with `load_dataset`, passing the `datasets:`
entry's format and options. Data in another shape gives another digest: resized or normalized images, `(image, int)`
tuples, images in height-width-channel order, or a dataset with no `index2label`.

```python
import json
from pathlib import Path

from dataeval_flow import dataset_digest, load_dataset

audit = json.loads(Path("output/results/result.json").read_text()).get("audit-splits", {})
verdict = audit.get("verdict")
if verdict is None:
    raise SystemExit(f"audit failed: {audit.get('errors')}")
if verdict["level"] == "not-ready" or verdict["warnings"] or verdict["not_assessed"]:
    raise SystemExit(f"not cleared to train: {verdict['level']}")

recorded = audit["steps"]["content-digest-train"]["output"]["data"]
train = load_dataset(Path("data/train"), dataset_format="coco")  # the `datasets:` entry train's source reads
if dataset_digest(train).content != recorded["content"]:
    raise SystemExit("train is not the data that was audited")
```

The snippet applies the middle gate above. The evaluation splits' digests are under `content-digest-evals`, by source
name: `audit["steps"]["content-digest-evals"]["elements"]["test"]["output"]["data"]`. From Python, the same values are
`result.steps["content-digest-train"].output.data()` and
`result.steps["content-digest-evals"].elements["test"].output.data()`.

Notice:

- The digest ignores item order, so an unseeded `Shuffle` still matches. Adding, removing or editing any item, or
  renaming a class, changes it.
- Metadata has a digest of its own, so a loader that yields images and labels with no metadata still matches the
  content digest. Gate on the content digest across machines: the metadata digest covers metadata as stored, paths
  included, so metadata holding an absolute file path digests differently on each machine.
- A split read through views, or made by a chain step, digests the items the views or the step kept, and the record
  says when it was. Flow has no public way to rebuild such a split outside a run, so for a gate, audit whole splits.
  Or [export](export_a_dataset.md) the split, audit the export as a source, and train on the export. `export` writes
  object-detection Datasets only.
- The digest hashes decoded pixels and targets as Flow reads them. Compare digests made with the same image libraries,
  which the result records in `metadata.library_versions`.

## See also

- [Preset Catalog](../reference/presets.md#audit): the audit's chain, settings, checks and verdict rule
- [Read evaluation outputs](read_evaluation_outputs.md): the result envelope, and the steps the digests are in
- [Provenance](../concepts/Provenance.md): what a result records about the data it read
