# Gate training on an audit

An `audit` run says whether a dataset is ready to train on, and records a content digest of each split it judged. A
training job can check both before it starts: that the verdict allows training, and that the data it is about to read
is the data that was audited. This guide covers running the audit, reading its verdict, and refusing data whose digest
doesn't match.

## Used in these tutorials

- {doc}`Audit a set of splits before training <../notebooks/audit>`

## Run the audit

Name the splits in order: the first source is train, and every later source is an evaluation split.

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

Run it with `--output ./output`, and `output/results/result.json` holds the task's result under its name,
`audit-splits`. The [Preset Catalog](../reference/presets.md#audit) lists every setting, and what `blocking:` and
`accepted:` change.

## Read the verdict

`verdict.level` is one of three values:

| Level | Means |
| --- | --- |
| `not-ready` | A blocking check warned, and no acceptance covers it. |
| `ready-with-caveats` | Another check warned, an accepted check warned, or a check was not assessed. The report lists each. |
| `ready` | No check warned, and every check was assessed. |

From Python it is `result.verdict.level`, and in the JSON it is `verdict.level`. Refuse `not-ready`. A stricter gate
also refuses `ready-with-caveats`, so a person reads the caveats before training starts.

Gate on the verdict, not on the result's health. `result: fail_on: warning` fails the run on any warning, an accepted
one included, since health counts the warnings the data has and the verdict records the decision made about them.

## Refuse data that was not audited

Each split's `content-digest` step records a SHA-256 digest over every item's image and labels, and the class names.
In the training job, recompute it with `dataeval_flow.dataset_digest()` on the data as the job will train on it, and
refuse a mismatch:

```python
import json
from pathlib import Path

from dataeval_flow import dataset_digest, load_dataset

audit = json.loads(Path("output/results/result.json").read_text())["audit-splits"]
if audit["verdict"]["level"] == "not-ready":
    raise SystemExit(f"Not ready: {[item['brief'] for item in audit['verdict']['blocking']]}")

recorded = audit["steps"]["content-digest-train"]["output"]["data"]
train = load_dataset(Path("data/train"), dataset_format="coco")  # the files the audit's train source read
if dataset_digest(train).content != recorded["content"]:
    raise SystemExit("train is not the data that was audited")
```

The evaluation splits' digests are under `content-digest-evals`, by source name:
`audit["steps"]["content-digest-evals"]["elements"]["test"]["output"]["data"]`. From Python, the same values are
`result.steps["content-digest-train"].output.data()` and
`result.steps["content-digest-evals"].elements["test"].output.data()`.

Notice:

- The digest ignores item order, so an unseeded `Shuffle` still matches. Adding, removing or editing any item, or
  renaming a class, changes it.
- Metadata has a digest of its own, so a loader that yields images and labels with no metadata still matches the
  content digest. Gate on the content digest across machines: the metadata digest covers metadata as stored, paths
  included, so metadata holding an absolute file path digests differently on each machine.
- A split read through a view, or made by a chain step, digests the items the view kept. The record says when it was.
  The job matches only if it reads the same items.
- The digest hashes decoded pixels and targets as Flow reads them. Compare digests made with the same image libraries,
  which the result records in `metadata.library_versions`.

## See also

- [Preset Catalog](../reference/presets.md#audit): the audit's chain, settings, checks and verdict rule
- [Read evaluation outputs](read_evaluation_outputs.md): the result envelope, and the steps the digests are in
- [Provenance](../concepts/Provenance.md): what a result records about the data it read
