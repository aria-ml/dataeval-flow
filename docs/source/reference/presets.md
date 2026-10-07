# Preset Catalog

A preset is a workflow type whose settings expand to a chain of steps. It runs as a task, or as a step of a custom
workflow. Each section below gives the question the preset answers, its chain, its settings, its `checks:` defaults and
an example. See [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md#workflow-types-as-presets) for how a
preset runs. The chain tables show the chain that the settings above each table build; other settings add, drop or
retune steps. A block's row lists every field the block takes; it may take fewer than its step does, and refuses any
other. Each example assumes the pipeline defines `datasets:`, the sources `train`, `test`, `validation`, `operational`,
`labeled` and `unlabeled`, and the extractor `bovw_ext`, as [Evaluator recipes](../how_to/evaluator_recipes.md) does.

## Settings every preset shares

Most presets take `ontology`, and some take `stats` or `metadata`, as each settings table shows. Each means the same in
every preset that takes it:

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space the preset's labels are read under: a name under the top-level `ontologies:` key, a path to a serialized RDF artifact resolved against the data root, or a nested mapping of concept to children, read as an inline hierarchy. It is recorded in the result envelope's `label_space`, so a run conformed by a `taxonomy` entry's stanza carries that entry's digest and can be matched back to it. Declare it wherever a source's view applies a `Relabel`. `taxonomy` requires it and judges labels against it, and `audit` judges each split's labels against it with `label-conformance`. `scope` refuses it as the config loads, so a scope run on a conformed source records no label space of its own; judge its labels with a `taxonomy` entry on the same source. |
| `stats` | a stats policy name, or `null` | `null` | The name of a policy under the top-level `stats:` key, which the preset's image statistics are measured under. Declare one to measure named band groups or the image background; leave it unset to measure the whole image. `audit` and `quality` pass it to their `outliers` and `duplicates` steps, and outlier detection reads the policy's `outliers_from` views. `shift` passes it to `factor-predictors` and `factor-deviation`, which read the statistics beside the metadata factors. |
| `metadata` | a metadata policy name, or `null` | `null` | The name of a policy under the top-level `metadata:` key, which the preset's metadata factors are read under. A policy is defined once and shared, so entries meant to be compared read their factors under one encoding. Leave it unset for DataEval's defaults. The preset passes it to every step of its chain that reads metadata. |

## `audit`

Audits one or more splits before training: a verdict, a record of what was audited, and findings under five questions.

- **Answers:** [Is the data clean?](index.md#is-the-data-clean),
  [Are the labels sound?](index.md#are-the-labels-sound),
  [Does the data cover what the model must handle?](index.md#does-the-data-cover-what-the-model-must-handle),
  [Could the model learn a shortcut?](index.md#could-the-model-learn-a-shortcut) and
  [Are the splits fit to evaluate on?](index.md#are-the-splits-fit-to-evaluate-on)
- **Reads:** `train`, then `evals`: the first source is train, and each later source is an evaluation split. `evals`
  may be empty.
- **Makes:** no Dataset; its verdict and findings are its result.

The verdict is one of three levels, worst first:

| Verdict | `level` | When |
| --- | --- | --- |
| Not ready | `not-ready` | A check that `blocking` names warned, and no `accepted` key covers that run. |
| Ready with caveats | `ready-with-caveats` | Any other check warned and isn't accepted, an accepted check warned, or a check was not assessed. |
| Ready | `ready` | No check warned, and every check was assessed. |

A blocking check that could not run is a caveat, not a block. An acceptance keyed by check type covers that check on
every split; one keyed by a check step covers all that step's runs, and `step[split]` one run. Copy a step key from the
verdict's `warnings[].step`. Only a step that runs once per evaluation split takes `[split]`: `class-imbalance-evals`,
`image-outliers-evals`, `image-duplicates-evals`, `factor-issues-evals`, `label-conformance-evals`, `eval-coverage`,
`embedding-divergence` and `class-stratification`. Each acceptance holds on this run and later ones; the accepted finding keeps
its severity and its evidence, and health still counts it.

A `blocking` entry that names no check the chain runs, or an `accepted` key that names neither a check nor a check step
it runs, is refused as the config loads: `label-conformance` without `ontology`, `uncovered-items` unless
`coverage: {method: naive}`, and `factor-coverage-gaps` with `factor-gaps: false`. So is a `[split]` on a step that
runs once, such as `leakage[test]`. A `[split]` naming no evaluation split of the task refuses the run before any step
runs: the command exits 1 and writes no result.json.

The result's `verdict`, `result.verdict` in Python and `verdict` in the JSON, holds:

- `level`: `not-ready`, `ready-with-caveats` or `ready`;
- `blocking` and `warnings`: each unaccepted warning, of a blocking check and of any other, as
  `{check, step, title, brief}`;
- `accepted`: each acceptance, as `{check, reason, state}`, where `check` is the key as written, such as
  `image-outliers` or `image-outliers-evals[test]`, and `state` is `warned`, `did-not-warn` or `not-assessed`;
- `not_assessed`: each check, or element of one, that judged nothing, as `{check, step, reason}`.

A task that fails has no verdict: `result.verdict` is `None`, and the JSON has no `verdict`. Run as a step of a custom
workflow, on splits a chain made, it gives the task the same verdict, record and questions. The verdict names its steps
`audit/...`; write `accepted:` keys without the prefix. A workflow runs one such step, and it may not be `optional:`.
See [Check a set of splits](../how_to/write_a_custom_workflow.md#11-check-a-set-of-splits).

**Chain**, from `outliers: {flags: [pixel], outlier_threshold: zscore}`, `ontology: {animal: {cat: null}}`,
`factor-leakage: {factors: [site]}` and `coverage: {method: naive}`, with an extractor for `ood-kneighbors`,
`divergence`, `coverage` and `completeness`. A step reading `evals` runs once per evaluation split, one reading `train`
and `evals` runs once per evaluation split with train, and one reading "each pair" runs once per pair of evaluation
splits. `crops` crops detection data and passes other Datasets through:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `label-health-train` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `train` |
| `label-health-evals` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `evals` |
| `class-imbalance-train` | check | [`class-imbalance`](checks.md#class-imbalance) | `input`: `label-health-train` |
| `class-imbalance-evals` | check | [`class-imbalance`](checks.md#class-imbalance) | `input`: `label-health-evals` |
| `outliers-train` | evaluator | [`outliers`](evaluators.md#outliers) | `input`: `train` |
| `outliers-evals` | evaluator | [`outliers`](evaluators.md#outliers) | `input`: `evals` |
| `image-outliers-train` | check | [`image-outliers`](checks.md#image-outliers) | `input`: `outliers-train` |
| `image-outliers-evals` | check | [`image-outliers`](checks.md#image-outliers) | `input`: `outliers-evals` |
| `duplicates-train` | evaluator | [`duplicates`](evaluators.md#duplicates) | `input`: `train` |
| `duplicates-evals` | evaluator | [`duplicates`](evaluators.md#duplicates) | `input`: `evals` |
| `image-duplicates-train` | check | [`image-duplicates`](checks.md#image-duplicates) | `input`: `duplicates-train` |
| `image-duplicates-evals` | check | [`image-duplicates`](checks.md#image-duplicates) | `input`: `duplicates-evals` |
| `factor-triage-train` | evaluator | [`factor-triage`](evaluators.md#factor-triage) | `input`: `train` |
| `factor-triage-evals` | evaluator | [`factor-triage`](evaluators.md#factor-triage) | `input`: `evals` |
| `factor-issues-train` | check | [`factor-issues`](checks.md#factor-issues) | `input`: `factor-triage-train` |
| `factor-issues-evals` | check | [`factor-issues`](checks.md#factor-issues) | `input`: `factor-triage-evals` |
| `content-digest-train` | evaluator | [`content-digest`](evaluators.md#content-digest) | `input`: `train` |
| `content-digest-evals` | evaluator | [`content-digest`](evaluators.md#content-digest) | `input`: `evals` |
| `label-reconciliation-train` | evaluator | [`label-reconciliation`](evaluators.md#label-reconciliation) | `input`: `train` |
| `label-reconciliation-evals` | evaluator | [`label-reconciliation`](evaluators.md#label-reconciliation) | `input`: `evals` |
| `label-conformance-train` | check | [`label-conformance`](checks.md#label-conformance) | `input`: `label-reconciliation-train` |
| `label-conformance-evals` | check | [`label-conformance`](checks.md#label-conformance) | `input`: `label-reconciliation-evals` |
| `ood-kneighbors` | evaluator | [`ood-kneighbors`](evaluators.md#ood-kneighbors) | `input`: `train`, `evals` |
| `eval-coverage` | check | [`eval-coverage`](checks.md#eval-coverage) | `input`: `ood-kneighbors` |
| `divergence` | evaluator | [`divergence`](evaluators.md#divergence) | `input`: `train`, `evals` |
| `embedding-divergence` | check | [`embedding-divergence`](checks.md#embedding-divergence) | `input`: `divergence` |
| `duplicates-cross` | evaluator | [`duplicates`](evaluators.md#duplicates) | `input`: `train`, `evals` |
| `duplicates-pairs` | evaluator | [`duplicates`](evaluators.md#duplicates) | `input`: `evals`, each pair |
| `factor-leakage-cross` | evaluator | [`factor-leakage`](evaluators.md#factor-leakage) | `input`: `train`, `evals` |
| `factor-leakage-pairs` | evaluator | [`factor-leakage`](evaluators.md#factor-leakage) | `input`: `evals`, each pair |
| `leakage` | check | [`leakage`](checks.md#leakage) | `duplicates`: `duplicates-cross`, `duplicates-pairs`; `factors`: `factor-leakage-cross`, `factor-leakage-pairs` |
| `class-sufficiency` | check | [`class-sufficiency`](checks.md#class-sufficiency) | `input`: `label-health-train`; `evals`: `label-health-evals` |
| `untrained-classes` | check | [`untrained-classes`](checks.md#untrained-classes) | `input`: `label-health-train`; `evals`: `label-health-evals` |
| `class-stratification` | check | [`class-stratification`](checks.md#class-stratification) | `input`: `label-health-train`; `parts`: `label-health-evals` |
| `crops` | transform | [`wrap`](transforms.md#wrap) | `input`: `train` |
| `coverage` | evaluator | [`coverage`](evaluators.md#coverage) | `input`: `crops` |
| `class-coverage` | check | [`class-coverage`](checks.md#class-coverage) | `input`: `coverage` |
| `uncovered-items` | check | [`uncovered-items`](checks.md#uncovered-items) | `input`: `coverage` |
| `completeness` | evaluator | [`completeness`](evaluators.md#completeness) | `input`: `crops` |
| `dimensional-completeness` | check | [`dimensional-completeness`](checks.md#dimensional-completeness) | `input`: `completeness` |
| `factor-summary` | evaluator | [`factor-summary`](evaluators.md#factor-summary) | `input`: `train` |
| `balance` | evaluator | [`balance`](evaluators.md#balance) | `input`: `train` |
| `diversity` | evaluator | [`diversity`](evaluators.md#diversity) | `input`: `train` |
| `shortcut-risk` | check | [`shortcut-risk`](checks.md#shortcut-risk) | `input`: `balance` |
| `factor-gaps` | combine | [`factor-gaps`](combines.md#factor-gaps) | `input`: `train`; `balance`: `balance` |
| `factor-coverage-gaps` | check | [`factor-coverage-gaps`](checks.md#factor-coverage-gaps) | `input`: `factor-gaps` |

**Settings** ({py:class}`~dataeval_flow.workflows.audit.AuditConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `stats` | a stats policy name, or `null` | `null` | The stats policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares). Every split's factors are encoded like train's, or like the source the policy's `reference_split` names |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; set, it adds `label-reconciliation` and `label-conformance` on each split; see [Settings every preset shares](#settings-every-preset-shares) |
| `outliers` | a block | required | [`outliers`](evaluators.md#outliers)'s `flags` and `outlier_threshold`, both required, and `cluster_threshold`, `cluster_algorithm` and `n_clusters`; the preset sets `per_target: false` |
| `coverage` | a block | unset: DataEval's defaults, `adaptive`, 20, 0.01, 20 and 0.5 | [`coverage`](evaluators.md#coverage)'s `method`, `num_observations`, `percent`, `min_class_samples`, `isotropy_min_samples` and `near_duplicate_factor`; the step runs on train when the task names an extractor |
| `wrap` | a block | `params: {padding: 0.0, min_size: 1}` | [`wrap`](transforms.md#wrap)'s `params`, used on detection data only; the preset fixes the wrapper |
| `factor-gaps` | a block, or `false` | `mi_threshold: 0.1`, `min_representation: 5` | [`factor-gaps`](combines.md#factor-gaps)'s `mi_threshold` and `min_representation`; `false` leaves out the gap analysis and its check |
| `factor-leakage` | a block, or `null` | `null` | [`factor-leakage`](evaluators.md#factor-leakage)'s `factors`, at least one: the group factors, such as a scene or site, whose values must not sit in two splits; unset leaves group leakage out. A factor a split's metadata lacks fails the task, which then has no verdict |
| `diversity` | a block | `method: simpson` | [`diversity`](evaluators.md#diversity)'s `method` |
| `divergence` | a block | `method: mst` | [`divergence`](evaluators.md#divergence)'s `method`; the step runs when the task names an extractor |
| `ood-kneighbors` | a block | unset: DataEval's defaults, with `threshold_perc` 95 | [`ood-kneighbors`](evaluators.md#ood-kneighbors)'s `k`, `distance_metric` and `threshold_perc`; the step is fitted on train and runs on each evaluation split when the task names an extractor |
| `blocking` | a list of check types | `[leakage, untrained-classes]` | The check types whose unaccepted warning makes the verdict "Not ready"; each must name a check the chain runs |
| `accepted` | a mapping of a check type, check step or `step[split]` to a reason | `{}` | Why each warning is accepted, by check type (`image-outliers`), by check step for all its runs (`image-outliers-evals`), or by `step[split]` for one run of a step that runs once per evaluation split (`image-outliers-evals[test]`), as the verdict's `warnings[].step` names it: an accepted warning can't make the verdict not ready, but it still leaves it ready with caveats; each key must name a check or check step the chain runs, and a reason may not be blank |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.audit.AuditChecks`):

| Check | Default settings |
| --- | --- |
| [`image-outliers`](checks.md#image-outliers) | `warning: 3.0` |
| [`image-duplicates`](checks.md#image-duplicates) | `exact: 0.0`, `near: 5.0` |
| [`factor-issues`](checks.md#factor-issues) | `max_examples: 20` |
| [`class-imbalance`](checks.md#class-imbalance) | `warning: 5.0`, `info: null`, `empty: true` |
| [`class-sufficiency`](checks.md#class-sufficiency) | `train: 20`, `eval: 30` |
| [`untrained-classes`](checks.md#untrained-classes) | `declared: false` |
| [`label-conformance`](checks.md#label-conformance) | `warning: 0`; run only where `ontology` is set |
| [`class-coverage`](checks.md#class-coverage) | `dispersion: 0.5`, `isotropy: 0.5`, `near_duplicates: 0.1` |
| [`uncovered-items`](checks.md#uncovered-items) | `warning: 10.0`; run only under `coverage: {method: naive}` |
| [`dimensional-completeness`](checks.md#dimensional-completeness) | `warning: 0.5`, `info: 0.8` |
| [`factor-coverage-gaps`](checks.md#factor-coverage-gaps) | `warning: 2`; run unless `factor-gaps: false` |
| [`shortcut-risk`](checks.md#shortcut-risk) | `warning: 0.1` |
| [`leakage`](checks.md#leakage) | `exact: 0`, `near: 0`, `groups: 0` |
| [`eval-coverage`](checks.md#eval-coverage) | `warning: 9.0`, `info: 1.0`, points past the split's baseline |
| [`class-stratification`](checks.md#class-stratification) | `info: 2.0`, `warning: 10.0` |
| [`embedding-divergence`](checks.md#embedding-divergence) | `warning: 0.5`, and `info` 0.4 times `warning` |

A check over each split applies its settings to every split. The settings of a check the chain does not run, such as
`checks.uncovered-items` under adaptive coverage, are unused. `coverage`, `completeness`, `divergence` and
`ood-kneighbors` are optional: with no extractor they are skipped with "requires an extractor", and their checks are
not assessed. `balance`, `diversity` and `factor-gaps` are optional too, since DataEval refuses them on metadata with
no factors. Any other step that fails fails the task, which then has no verdict. With one source, `evals` is empty,
and every check of *Are the splits fit to evaluate on?* is not assessed, with "no evaluation split given", so a
one-split audit is at best Ready with caveats. A check over each split then judges train alone, and its check over
`evals`, which judged nothing, is left out of the verdict and the questions. `class-sufficiency` and
`untrained-classes` still judge train.

The report gives the verdict, then a record of what was audited: a column per split, with its items, labels, classes,
metadata factors and the digests of its content and metadata, then the run and the criteria: the settings of each
check the chain ran, the blocking checks and the acceptances. The findings follow under the five questions, then next
steps: what to do about each check that warned, and about each check left unassessed. The console's short report
leaves out the evidence and the next steps; `-v` prints them, and `result.txt` and `result.html` hold them unless
`result: detail: summary`. {doc}`Audit a set of splits before training <../notebooks/audit>` walks through a run, and
[Gate training on an audit](../how_to/gate_training_on_an_audit.md) reads the verdict in a training job. See
[Dataset Splitting](../concepts/DatasetSplitting.md) for why leakage and unrepresentative splits make a test score
untrustworthy.

```yaml
workflows:
  - name: release-audit
    type: audit
    outliers: {flags: [dimension, pixel, visual], outlier_threshold: modzscore}
    factor-leakage: {factors: [site]}
    accepted:
      class-imbalance: "Rare class by design; weighted loss in training."
    checks:
      class-sufficiency: {eval: 50}

tasks:
  - {name: audit-splits, workflow: release-audit, sources: [train, validation, test], extractor: bovw_ext}
```

## `quality`

Outlier and duplicate detection for image datasets, and the dataset without them.

- **Answers:** [Is the data clean?](index.md#is-the-data-clean)
- **Reads:** `data`, the task's one source.
- **Makes:** `clean`, the Dataset without its flagged outliers and duplicates.

**Chain**, from `outliers: {flags: [pixel], outlier_threshold: zscore}`:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `outliers` | evaluator | [`outliers`](evaluators.md#outliers) | `input`: `data` |
| `label-health` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `data` |
| `outliers-by-class` | combine | [`outliers-by-class`](combines.md#outliers-by-class) | `input`: `data`; `outliers`: `outliers` |
| `duplicates` | evaluator | [`duplicates`](evaluators.md#duplicates) | `input`: `data` |
| `image-outliers` | check | [`image-outliers`](checks.md#image-outliers) | `input`: `outliers` |
| `target-outliers` | check | [`target-outliers`](checks.md#target-outliers) | `input`: `outliers`; `labels`: `label-health` |
| `class-outliers` | check | [`class-outliers`](checks.md#class-outliers) | `input`: `outliers-by-class` |
| `image-duplicates` | check | [`image-duplicates`](checks.md#image-duplicates) | `input`: `duplicates` |
| `clean` | transform | [`remove`](transforms.md#remove) | `input`: `data`; `plans`: `duplicates`, `outliers` |

**Settings** ({py:class}`~dataeval_flow.workflows.quality.QualityConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `stats` | a stats policy name, or `null` | `null` | The stats policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `outliers` | a block | required | [`outliers`](evaluators.md#outliers)'s `flags` and `outlier_threshold`, both required, and `cluster_threshold`, `cluster_algorithm` and `n_clusters`; the preset sets `per_target: true` |
| `duplicates` | a block | `merge_near_duplicates: true`, and the step's own defaults otherwise | [`duplicates`](evaluators.md#duplicates)'s `flags`, `merge_near_duplicates`, `cluster_sensitivity`, `cluster_algorithm` and `n_clusters` |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.quality.QualityChecks`):

| Check | Default settings |
| --- | --- |
| [`image-outliers`](checks.md#image-outliers) | `warning: 3.0` |
| [`target-outliers`](checks.md#target-outliers) | `warning: 3.0` |
| [`class-outliers`](checks.md#class-outliers) | `warning: 3.0` |
| [`image-duplicates`](checks.md#image-duplicates) | `exact: 0.0`, `near: 5.0` |

The other settings go to the evaluators: the `outliers` block to `outliers`, the `duplicates` block to `duplicates`,
`metadata` to `label-health`, and `stats` to both `outliers` and `duplicates`. Each block holds the step settings its
row lists, spelled as the step spells them. Each `checks:` entry holds the settings of the check it names. `clean`
removes each image and box with at least one outlier flag, and each exact or near duplicate but the first of its group.

Its report gives each finding a section, with the evaluators it judged below it: the flagged images and boxes under
Image Outliers, and the duplicate groups under Image Duplicates. The class counts sit under Target Outliers where any box
was flagged, else in a Label Health section of their own; class balance is [`bias`](#bias)'s. A finding
that read a step shown
already names the finding it is under, as Class Outliers names Image Outliers for the outliers `outliers-by-class`
counted. `clean`'s section follows, saying how many images it kept and what each plan named. On MILCO's reference
campaigns, as {doc}`View a report as HTML <../notebooks/view_html_reports>` runs it: "Kept 162 of 261 images. Removed 99
images and 32 detections: 90 images named by `duplicates`, 11 images and 32 detections by `outliers`." Two images were
named by both plans. A Steps table lists every step, what it read, and why it made nothing where it did not.

Run as a step of a custom workflow, `<step>.clean` reads the cleaned Dataset. See
[Workflow types as presets](../concepts/WorkflowsAsChains.md#workflow-types-as-presets).

```yaml
workflows:
  - name: cleaning
    type: quality
    outliers: {flags: [pixel, visual], outlier_threshold: zscore}
    checks:
      image-outliers: {warning: 5.0}

tasks:
  - {name: clean-train, workflow: cleaning, sources: [train]}
```

## `taxonomy`

Judges a Dataset's labels against a declared ontology: leaf coverage, conformance, alignment and structure.

- **Answers:** [Do the labels match an ontology?](index.md#do-the-labels-match-an-ontology)
- **Reads:** `data`, the task's one source.
- **Makes:** no Dataset; its findings are its result.

**Chain**, from `ontology: {animal: {cat: null}}`:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `representation` | evaluator | [`representation`](evaluators.md#representation) | `input`: `data` |
| `leaf-coverage` | check | [`leaf-coverage`](checks.md#leaf-coverage) | `input`: `representation` |
| `label-reconciliation` | evaluator | [`label-reconciliation`](evaluators.md#label-reconciliation) | `input`: `data` |
| `label-conformance` | check | [`label-conformance`](checks.md#label-conformance) | `input`: `label-reconciliation` |
| `label-alignment` | evaluator | [`label-alignment`](evaluators.md#label-alignment) | `input`: `data` |
| `label-mergeability` | check | [`label-mergeability`](checks.md#label-mergeability) | `input`: `label-alignment` |
| `ontology-validation` | evaluator | [`ontology-validation`](evaluators.md#ontology-validation) | `input`: `data` |
| `ontology-structure` | check | [`ontology-structure`](checks.md#ontology-structure) | `input`: `ontology-validation` |

**Settings** ({py:class}`~dataeval_flow.workflows.taxonomy.TaxonomyConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `ontology` | an ontology name, a path, or a nested mapping | required | The ontology to judge labels against: a name under the top-level `ontologies:` key, a path to a serialized RDF artifact resolved against the data root, or a nested mapping of concept to children. |
| `representation` | a block | the step's own defaults | [`representation`](evaluators.md#representation)'s `expected`: each class's minimum share; the preset sets its `ontology` |
| `ontology-validation` | a block | the step's own defaults | [`ontology-validation`](evaluators.md#ontology-validation)'s `label_pattern`; the preset sets its `ontology` |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.taxonomy.TaxonomyChecks`):

| Check | Default settings |
| --- | --- |
| [`leaf-coverage`](checks.md#leaf-coverage) | `coverage: 0.9`, `empty_branches: 0` |
| [`label-conformance`](checks.md#label-conformance) | `warning: 0` |

The `label-mergeability` and `ontology-structure` checks take no `checks:` entry. The ontology is the sanctioned label space,
so the preset can name a class that was never collected, which counting the dataset's own labels cannot. It is the only
preset that judges labels against an ontology: `scope` refuses `ontology:`, so run a `taxonomy` entry on the
same source beside it. The ontology is declared inline, loaded from an RDF file, or named from the shared `ontologies:`
block; `ontology:` has no default. See [Declare an ontology](../how_to/declare_an_ontology.md).

```yaml
workflows:
  - name: vocab-check
    type: taxonomy
    ontology:
      animal:
        mammal: [cat, dog]
        bird: [owl]

tasks:
  - {name: vocab, workflow: vocab-check, sources: [train]}
```

## `scope`

Judges how a Dataset's embeddings cover their space, and what to acquire per class; detections are cropped first.

- **Answers:** [Does the data cover what the model must handle?](index.md#does-the-data-cover-what-the-model-must-handle)
- **Reads:** `data`, the task's one source.
- **Makes:** no Dataset; its findings are its result.

**Chain**, from `coverage: {method: naive}`, with an extractor for `coverage` and `completeness`; `crops` crops
detection data and passes other Datasets through:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `crops` | transform | [`wrap`](transforms.md#wrap) | `input`: `data` |
| `coverage` | evaluator | [`coverage`](evaluators.md#coverage) | `input`: `crops` |
| `class-coverage` | check | [`class-coverage`](checks.md#class-coverage) | `input`: `coverage` |
| `uncovered-items` | check | [`uncovered-items`](checks.md#uncovered-items) | `input`: `coverage` |
| `completeness` | evaluator | [`completeness`](evaluators.md#completeness) | `input`: `crops` |
| `dimensional-completeness` | check | [`dimensional-completeness`](checks.md#dimensional-completeness) | `input`: `completeness` |
| `representation` | evaluator | [`representation`](evaluators.md#representation) | `input`: `data` |
| `class-shortfall` | check | [`class-shortfall`](checks.md#class-shortfall) | `input`: `representation` |

**Settings** ({py:class}`~dataeval_flow.workflows.scope.ScopeConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `ontology` | `null`; any other value is refused | `null` | Refused when set, as the config loads: judge the labels with a `taxonomy` entry on the same source; see [Settings every preset shares](#settings-every-preset-shares) |
| `representation` | a block | the step's own defaults | [`representation`](evaluators.md#representation)'s `expected`: each class's minimum share |
| `coverage` | a block | unset: DataEval's defaults, `adaptive`, 20, 0.01, 20 and 0.5 | [`coverage`](evaluators.md#coverage)'s `method`, `num_observations`, `percent`, `min_class_samples`, `isotropy_min_samples` and `near_duplicate_factor`; the step runs when the task names an extractor |
| `wrap` | a block | `params: {padding: 0.0, min_size: 1}` | [`wrap`](transforms.md#wrap)'s `params`, used on detection data only; the preset fixes the wrapper |
| `completeness` | `true` or `false` | `true` | Whether the completeness steps run, when the task names an extractor |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.scope.ScopeChecks`):

| Check | Default settings |
| --- | --- |
| [`class-coverage`](checks.md#class-coverage) | `dispersion: 0.5`, `isotropy: 0.5`, `near_duplicates: 0.1` |
| [`uncovered-items`](checks.md#uncovered-items) | `warning: 10.0` |
| [`dimensional-completeness`](checks.md#dimensional-completeness) | `warning: 0.5`, `info: 0.8` |

Coverage asks whether a collection spans the conditions the model will meet, where cleaning asks whether its samples are
sound. The preset lists the classes that fall short of their expected share. When the task names an
{term}`extractor <Extractor>`, it adds per-class embedding variety and dimensional completeness. A class can be
plentiful by count and still occupy little of the representation space, which the embedding steps show. `naive` coverage
is judged by an `uncovered-items` step. The chain is the same without an extractor, but the `coverage` and
`completeness` steps are then skipped with "requires an extractor". Class balance and the metadata factors, with the
class-factor combinations a factor tied to the class leaves under-represented, are [`bias`](#bias)'s; run it
on the same source. Run both before training and before fixing a reference set, while a gap can still be closed by
collecting more data. See [Dataset Coverage](../concepts/Coverage.md).

```yaml
workflows:
  - name: coverage
    type: scope
    coverage: {method: naive}
    checks:
      uncovered-items: {warning: 5.0}

tasks:
  - {name: coverage-train, workflow: coverage, sources: [train], extractor: bovw_ext}
```

## `bias`

Judges a Dataset's class balance and how its metadata factors relate to the class: shortcuts, association and
under-represented combinations.

- **Answers:** [Could the model learn a shortcut?](index.md#could-the-model-learn-a-shortcut) and
  [Are the labels sound?](index.md#are-the-labels-sound)
- **Reads:** `data`, the task's one source.
- **Makes:** no Dataset; its findings are its result.

**Chain**, from the defaults:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `label-health` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `data` |
| `class-imbalance` | check | [`class-imbalance`](checks.md#class-imbalance) | `input`: `label-health` |
| `factor-summary` | evaluator | [`factor-summary`](evaluators.md#factor-summary) | `input`: `data` |
| `balance` | evaluator | [`balance`](evaluators.md#balance) | `input`: `data` |
| `diversity` | evaluator | [`diversity`](evaluators.md#diversity) | `input`: `data` |
| `shortcut-risk` | check | [`shortcut-risk`](checks.md#shortcut-risk) | `input`: `balance` |
| `parity` | evaluator | [`parity`](evaluators.md#parity) | `input`: `data` |
| `factor-parity` | check | [`factor-parity`](checks.md#factor-parity) | `input`: `parity` |
| `factor-gaps` | combine | [`factor-gaps`](combines.md#factor-gaps) | `input`: `data`; `balance`: `balance` |
| `factor-coverage-gaps` | check | [`factor-coverage-gaps`](checks.md#factor-coverage-gaps) | `input`: `factor-gaps` |

**Settings** ({py:class}`~dataeval_flow.workflows.bias.BiasConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `diversity` | a block | `method: simpson` | [`diversity`](evaluators.md#diversity)'s `method` |
| `factor-gaps` | a block, or `false` | `mi_threshold: 0.1`, `min_representation: 5` | [`factor-gaps`](combines.md#factor-gaps)'s `mi_threshold` and `min_representation`; `false` leaves out the gap analysis and its check |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.bias.BiasChecks`):

| Check | Default settings |
| --- | --- |
| [`class-imbalance`](checks.md#class-imbalance) | `warning: 5.0`, `info: null` |
| [`shortcut-risk`](checks.md#shortcut-risk) | `warning: 0.1` |
| [`factor-parity`](checks.md#factor-parity) | `warning: 0.3`, `p_value: 0.05` |
| [`factor-coverage-gaps`](checks.md#factor-coverage-gaps) | `warning: 2` |

Bias asks whether something other than the task tells the classes apart. The preset judges how far the largest class
outnumbers the smallest, then, for each metadata factor, how much it tells about the class (`shortcut-risk`, by mutual
information) and whether that association is significant (`factor-parity`, by Cramér's V and a chi-square test). Among
the factors tied to the class, `factor-coverage-gaps` lists the class-factor-value combinations held too rarely, which
collecting more data can close. Diversity and the factor summary are report sections. It reads labels and metadata
only, so it needs no extractor; a source with no metadata factors still judges its class balance, and its factor checks
are not assessed. Run it beside [`scope`](#scope) on the same source: each judges what the other
leaves out. See DataEval's [Dataset Bias
explanation](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html).

```yaml
workflows:
  - name: bias
    type: bias
    checks:
      shortcut-risk: {warning: 0.2}

tasks:
  - {name: bias-train, workflow: bias, sources: [train]}
```

## `splits`

Splits a Dataset into train, val and test, or k folds, and judges each part's stratification; with `folds` of 2 or
more, `train` and `val` are lists keyed by fold.

- **Answers:** [Are the splits fit to evaluate on?](index.md#are-the-splits-fit-to-evaluate-on)
- **Reads:** `data`, the task's one source.
- **Makes:** `train`, `val` and `test`.

**Chain**, from `rebalance: interclass`:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `label-health` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `data` |
| `split` | transform | [`split`](transforms.md#split) | `input`: `data` |
| `rebalanced` | transform | [`view`](transforms.md#view) | `input`: `split.train` |
| `label-health-train` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `split.train` |
| `label-health-val` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `split.val` |
| `label-health-test` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `split.test` |
| `label-health-rebalanced` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `rebalanced` |
| `class-stratification` | check | [`class-stratification`](checks.md#class-stratification) | `input`: `label-health`; `parts`: `label-health-train`, `label-health-val`, `label-health-test`; `shown`: `label-health-rebalanced` |

**Settings** ({py:class}`~dataeval_flow.workflows.splits.SplitsConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `folds` | a count | `1` | `1` splits once into train, val and test; 2 or more make that many train and val folds and one test. |
| `test_frac` | a fraction | `0.2` | The share held out as `test`; `0` holds out none. |
| `val_frac` | a fraction, or `null` | `null` | The share held out as `val` with `folds: 1`; unset is 0.1. With `folds` 2 or more, each fold's val is its 1/k, and setting this is refused. |
| `stratify` | `true` or `false` | `true` | Whether each part keeps the whole's class proportions. |
| `split_on` | a list of metadata factors, or `null` | `null` | Metadata factors whose values never straddle parts, such as a scene or site. Classification data only: DataEval ignores it on detection data, with a warning in the log. |
| `rebalance` | `global`, `interclass`, or `null` | `null` | DataEval's `ClassBalance` method applied to each train; unset rebalances nothing. |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.splits.SplitsChecks`):

| Check | Default settings |
| --- | --- |
| [`class-stratification`](checks.md#class-stratification) | `info: 2.0`, `warning: 10.0` |

The chain reads the whole set's labels, splits it (`folds: 1`) or cuts it into k folds (`folds` of 2 or more) with a
shared test part, optionally rebalances each train, and judges each part's labels and its stratification against the
whole. Its findings are Class Stratification for each fold. The whole set's class balance and metadata factors are
[`bias`](#bias)'s; run it on the source before splitting it. The result is a `ChainResult`, and each part's
indices into the source are in `result.steps["split"].details["indices"]`. Run as a step of a custom workflow, the
entry hands on three Datasets: `<step>.train` (the rebalanced train, where the entry sets `rebalance:`), `<step>.val`
and `<step>.test`. The preset does not judge coverage, leakage or shift: run [`audit`](#audit) as a step after it, as
[Check a set of splits](../how_to/write_a_custom_workflow.md#11-check-a-set-of-splits) does, or give the parts to an
`audit` task as sources, train first. See [Dataset Splitting](../concepts/DatasetSplitting.md).

```yaml
workflows:
  - name: splitting
    type: splits
    test_frac: 0.2
    rebalance: interclass

tasks:
  - {name: split-train, workflow: splitting, sources: [train]}
```

## `shift`

Tests each incoming source against a reference for drift and for out-of-distribution images,
with the metadata behind them.

- **Answers:** [Has new data drifted?](index.md#has-new-data-drifted), [Which items are out of distribution?](index.md#which-items-are-out-of-distribution)
- **Reads:** `reference`, then `tests`: the first source is the reference, and each later source is tested against it.
- **Makes:** no Dataset; its findings are its result.

**Chain**, from the detectors `mmd` (`drift-mmd`, chunked), `knn` (`ood-kneighbors`) and `dc`
(`ood-domain-classifier`), with `classwise: {mmd: class}` and the task's extractor or each detector's own:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `mmd` | evaluator | [`drift-mmd`](evaluators.md#drift-mmd) | `input`: `reference`, `tests` |
| `mmd-check` | check | [`drift`](checks.md#drift) | `input`: `mmd` |
| `knn` | evaluator | [`ood-kneighbors`](evaluators.md#ood-kneighbors) | `input`: `reference`, `tests` |
| `knn-check` | check | [`ood`](checks.md#ood) | `input`: `knn` |
| `dc` | evaluator | [`ood-domain-classifier`](evaluators.md#ood-domain-classifier) | `input`: `reference`, `tests` |
| `dc-check` | check | [`ood`](checks.md#ood) | `input`: `dc` |
| `mmd-by-class` | evaluator | [`drift-mmd`](evaluators.md#drift-mmd) | `input`: `reference`, `tests` |
| `mmd-by-class-check` | check | [`drift`](checks.md#drift) | `input`: `mmd-by-class` |
| `ood-union` | combine | [`ood-union`](combines.md#ood-union) | `input`: `knn`, `dc` |
| `ood-agreement` | check | [`ood-agreement`](checks.md#ood-agreement) | `input`: `ood-union` |
| `factor-predictors` | combine | [`factor-predictors`](combines.md#factor-predictors) | `ood`: `ood-union`; `reference`: `reference`; `input`: `tests` |
| `factor-deviation` | combine | [`factor-deviation`](combines.md#factor-deviation) | `ood`: `ood-union`; `reference`: `reference`; `input`: `tests` |

Each detector adds its evaluator and its check, in list order: `drift` for a drift detector, `ood` for an OOD one. The
`classwise` steps follow, and then the OOD steps. `ood-union`, `factor-predictors` and `factor-deviation` run only where
the list holds an OOD detector, and `ood-agreement` only with two or more, so a list of drift detectors alone adds none
of them.

**Settings** ({py:class}`~dataeval_flow.workflows.shift.ShiftConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `stats` | a stats policy name, or `null` | `null` | The stats policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `detectors` | a list of drift and OOD evaluator entries | `[drift-univariate, ood-kneighbors]` | Drift (`drift-univariate`, `drift-mmd`, `drift-kneighbors`, `drift-domain-classifier`) and OOD (`ood-kneighbors`, `ood-domain-classifier`) evaluator entries, each testing every test source against the reference. An entry's `name` names its step. An entry may name its own `extractor:`. |
| `classwise` | a mapping of drift detector name to `by:` | `{}` | Drift detectors to also run per key, unchunked, each with its `by:`: `{drift-mmd: class}`, `{uncertainty: predicted}` (see [Drift in a model's uncertainty](../how_to/monitor_drift.md#6-drift-in-a-models-uncertainty)), or with settings; `min_items` is 2 unless written. An OOD detector here is refused. |
| `factor-predictors` | `false`, or `null` | `null` | `false` leaves out the `factor-predictors` step; it takes no settings. |
| `factor-deviation` | a block, or `false` | `max_items: 50` | [`factor-deviation`](combines.md#factor-deviation)'s `max_items`; `false` leaves it out. |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.shift.ShiftChecks`):

| Check | Default settings |
| --- | --- |
| [`drift`](checks.md#drift) | `warn_on_drift: true`, `chunk_percent: 10.0`, `consecutive_chunks: 2` |
| [`ood`](checks.md#ood) | `warning: 10.0`, `info: 1.0` |
| [`ood-agreement`](checks.md#ood-agreement) | `warning: 10.0`, `info: 1.0` |

With no `detectors:`, it runs `drift-univariate` (a KS test per embedding dimension, with Bonferroni correction) and
`ood-kneighbors`, each with DataEval's settings. `drift-univariate` can miss correlated shifts across embedding
dimensions: add `drift-mmd` or `drift-domain-classifier` to catch those.

Drift asks whether a whole batch moved, and out-of-distribution detection asks of each image whether it is anomalous
relative to the reference. A task with a reference and two test sources runs every detector once for each test source.
The entries of `detectors:` are evaluator entries, so each takes the fields its evaluator takes in the
[Evaluator Catalog](evaluators.md), and its name defaults to its type. A `drift` check judges each drift detector, and
an `ood` check each OOD detector's flagged share. `ood-union` joins the OOD detectors' flags, `ood-agreement` judges how
far they agree, and `factor-predictors` and `factor-deviation` read the flagged images' metadata. Use it to monitor
operational data for drift, and during ingestion or operation to flag individual inputs outside the training
distribution. The report groups each source's findings under the source's name, and the result is a `ChainResult` with
one element per test source for each step. See [Distribution Shift](../concepts/DistributionShift.md) and
[Monitor drift with steps](../how_to/monitor_drift.md).

```yaml
workflows:
  - name: shift
    type: shift
    detectors:
      - {type: drift-univariate, method: ks, p_val: 0.01}
      - {name: mmd-chunked, type: drift-mmd, chunking: {chunk_count: 10}}
      - {name: knn, type: ood-kneighbors, distance_metric: euclidean}
    checks:
      drift: {chunk_percent: 20.0}
      ood: {warning: 5.0}

tasks:
  - {name: cameras, workflow: shift, sources: [train, test, operational], extractor: bovw_ext}
```

## `prioritization`

Ranks each pool against a reference for labeling, and keeps the top.

- **Answers:** [Which items should be labeled next?](index.md#which-items-should-be-labeled-next)
- **Reads:** `reference`, then `pools`: the first source is the reference and the rest are the pools. This is the
  reverse of the [`prioritization`](evaluators.md#prioritization) evaluator's order, the data to rank and then the
  reference, and the chain hands them to it in that order.
- **Makes:** `selected`, each pool's top-ranked items.

**Chain**, from its defaults, with an extractor for `prioritization`:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `prioritization` | evaluator | [`prioritization`](evaluators.md#prioritization) | `input`: `pools`, `reference` |
| `selected` | transform | [`select`](transforms.md#select) | `input`: `pools`; `ranking`: `prioritization` |

**Settings** ({py:class}`~dataeval_flow.workflows.prioritization.PrioritizationWorkflowConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `prioritization` | a block | `order: hard_first` (DataEval's default is `easy_first`); `method`, `n_init`, `policy` and `num_bins` unset, so DataEval's `knn`, `auto`, `difficulty` and 50 apply | [`prioritization`](evaluators.md#prioritization)'s `method`, `k`, `c`, `n_init`, `max_cluster_size`, `order`, `policy` and `num_bins`: how each pool is ranked |
| `select` | a block | `n: null`, `fraction: null` | [`select`](transforms.md#select)'s `n` and `fraction`: how much of each pool's ranking `selected` keeps |

The preset has no `checks:`. `prioritization` ranks each pool against the reference, `hard_first` putting novel or
challenging items first. `selected` keeps the top of each pool's ranking: `select.n` items, or `select.fraction` of
them. With neither, `selected` keeps every item (`fraction: 1.0`). The chain has no checks, so it makes no findings.
See [Data Prioritization](../concepts/Prioritization.md).

```yaml
workflows:
  - name: prioritization
    type: prioritization
    prioritization: {method: knn, k: 5}
    select: {n: 200}

tasks:
  - {name: next-labels, workflow: prioritization, sources: [labeled, unlabeled], extractor: bovw_ext}
```

To rank clean data, run the preset as a step of a custom workflow, after a [`quality`](#quality) step on
the reference and one on the pools. A step over the pools runs once per pool, so `pool-clean.clean` is a list keyed by
pool, and `rank.selected` is too:

```yaml
workflows:
  - name: cleaning
    type: quality
    outliers: {flags: [pixel, visual], outlier_threshold: zscore}

  - name: prioritization
    type: prioritization
    prioritization: {method: knn, k: 5}
    select: {n: 200}

  - name: clean_then_rank
    inputs: [reference, {name: pools, list: true}]
    steps:
      - {name: reference-clean, workflow: cleaning, input: reference}
      - {name: pool-clean, workflow: cleaning, input: pools}
      - {name: rank, workflow: prioritization, input: [reference-clean.clean, pool-clean.clean]}

tasks:
  - {name: next-labels, workflow: clean_then_rank, sources: [labeled, unlabeled], extractor: bovw_ext}
```

`quality` removes exact and near duplicates. To keep near duplicates, clean with an `outliers` step, a
`duplicates` step and a `remove` step whose duplicates plan sets `dup_types: [exact]`; see
[`remove`](transforms.md#remove).

## `triage`

Reports unreadable and unpinned metadata factors, with suggested corrections.

Checks what can be read from metadata and annotations alone, without pixels or embeddings, so it is cheap and runs first. Label and box checks are planned.

- **Answers:** [Is the metadata readable?](index.md#is-the-metadata-readable)
- **Reads:** `data`, the task's one source.
- **Makes:** no Dataset; its findings are its result.

**Chain**, from its defaults:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `factor-triage` | evaluator | [`factor-triage`](evaluators.md#factor-triage) | `input`: `data` |
| `factor-issues` | check | [`factor-issues`](checks.md#factor-issues) | `input`: `factor-triage` |

**Settings** ({py:class}`~dataeval_flow.workflows.triage.TriageConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `checks` | a block | the defaults below | The `factor-issues` check's settings, keyed by check type. |
| `verify` | `true` or `false` | `true` | Re-read the metadata under the complete suggestions and report what they recover. Costs no second dataset walk: `repair` returns a copy sharing the store. |
| `default_bins` | a count | `10` | Bin count a suggestion falls back to where the run left no fit to read. Where there is one, the populated bins of the derived cut are carried forward instead, which pins the cut the run used rather than substituting a different one. |
| `min_missing_fraction` | a fraction | `0.2` | Share of rows recording no value above which a factor is called degenerate. |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.triage.TriageChecks`):

| Check | Default settings |
| --- | --- |
| [`factor-issues`](checks.md#factor-issues) | `max_examples: 20` |

The settings expand to two steps on the task's one source, `data`. `metadata:`, `verify`, `default_bins` and
`min_missing_fraction` are `factor-triage`'s settings, and `max_examples` is `factor-issues`', set under
`checks.factor-issues`. Its findings are `factor-issues`': one per kind of issue, then the suggested policy and
what verification recovered. The chain makes no Dataset, so it declares no output. Its result's `metadata_binning`
records the encoding `factor-triage` read, which `dataeval-flow encoding` writes out.

```yaml
workflows:
  - name: triage
    type: triage
    verify: true
    checks:
      factor-issues: {max_examples: 10}

tasks:
  - {name: triage-train, workflow: triage, sources: [train]}
```
