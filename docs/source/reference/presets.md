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
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space the preset's labels are read under: a name under the top-level `ontologies:` key, a path to a serialized RDF artifact resolved against the data root, or a nested mapping of concept to children, read as an inline hierarchy. It is recorded in the result envelope's `label_space`, so a run conformed by a `label-space` entry's stanza carries that entry's digest and can be matched back to it. Declare it wherever a source's view applies a `Relabel`. `label-space` requires it and judges labels against it, and `audit` judges each split's labels against it with `label-conformance`. `data-coverage` refuses it as the config loads, so a data-coverage run on a conformed source records no label space of its own; judge its labels with a `label-space` entry on the same source. |
| `stats` | a stats policy name, or `null` | `null` | The name of a policy under the top-level `stats:` key, which the preset's image statistics are measured under. Declare one to measure named band groups or the image background; leave it unset to measure the whole image. `audit`, `data-cleaning` and `data-prioritization` pass it to their `outliers` and `duplicates` steps, and outlier detection reads the policy's `outliers_from` views. `ood-detection` passes it to `factor-predictors` and `factor-deviation`, which read the statistics beside the metadata factors. |
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
`image-outliers-evals`, `image-duplicates-evals`, `metadata-issues-evals`, `label-conformance-evals`, `eval-coverage`,
`distribution-shift` and `stratification`. Each acceptance holds on this run and later ones; the accepted finding keeps
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
workflow, audit gives no verdict, record or questions, so run it as a task.

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
| `metadata-issues-train` | check | [`metadata-issues`](checks.md#metadata-issues) | `input`: `factor-triage-train` |
| `metadata-issues-evals` | check | [`metadata-issues`](checks.md#metadata-issues) | `input`: `factor-triage-evals` |
| `content-digest-train` | evaluator | [`content-digest`](evaluators.md#content-digest) | `input`: `train` |
| `content-digest-evals` | evaluator | [`content-digest`](evaluators.md#content-digest) | `input`: `evals` |
| `label-reconciliation-train` | evaluator | [`label-reconciliation`](evaluators.md#label-reconciliation) | `input`: `train` |
| `label-reconciliation-evals` | evaluator | [`label-reconciliation`](evaluators.md#label-reconciliation) | `input`: `evals` |
| `label-conformance-train` | check | [`label-conformance`](checks.md#label-conformance) | `input`: `label-reconciliation-train` |
| `label-conformance-evals` | check | [`label-conformance`](checks.md#label-conformance) | `input`: `label-reconciliation-evals` |
| `ood-kneighbors` | evaluator | [`ood-kneighbors`](evaluators.md#ood-kneighbors) | `input`: `train`, `evals` |
| `eval-coverage` | check | [`eval-coverage`](checks.md#eval-coverage) | `input`: `ood-kneighbors` |
| `divergence` | evaluator | [`divergence`](evaluators.md#divergence) | `input`: `train`, `evals` |
| `distribution-shift` | check | [`distribution-shift`](checks.md#distribution-shift) | `input`: `divergence` |
| `duplicates-cross` | evaluator | [`duplicates`](evaluators.md#duplicates) | `input`: `train`, `evals` |
| `duplicates-pairs` | evaluator | [`duplicates`](evaluators.md#duplicates) | `input`: `evals`, each pair |
| `factor-leakage-cross` | evaluator | [`factor-leakage`](evaluators.md#factor-leakage) | `input`: `train`, `evals` |
| `factor-leakage-pairs` | evaluator | [`factor-leakage`](evaluators.md#factor-leakage) | `input`: `evals`, each pair |
| `leakage` | check | [`leakage`](checks.md#leakage) | `duplicates`: `duplicates-cross`, `duplicates-pairs`; `factors`: `factor-leakage-cross`, `factor-leakage-pairs` |
| `class-sufficiency` | check | [`class-sufficiency`](checks.md#class-sufficiency) | `input`: `label-health-train`; `evals`: `label-health-evals` |
| `untrained-classes` | check | [`untrained-classes`](checks.md#untrained-classes) | `input`: `label-health-train`; `evals`: `label-health-evals` |
| `stratification` | check | [`stratification`](checks.md#stratification) | `input`: `label-health-train`; `parts`: `label-health-evals` |
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
| `coverage` | a block | `method: adaptive`, `num_observations: 50` where the step's own default is DataEval's 20, and the step's other defaults | [`coverage`](evaluators.md#coverage)'s `method`, `num_observations`, `percent`, `min_class_samples`, `isotropy_min_samples` and `near_duplicate_factor`; the step runs on train when the task names an extractor |
| `wrap` | a block | `params: {padding: 0.0, min_size: 1}` | [`wrap`](transforms.md#wrap)'s `params`, used on detection data only; the preset fixes the wrapper |
| `factor-gaps` | a block, or `false` | `mi_threshold: 0.1`, `min_representation: 5` | [`factor-gaps`](combines.md#factor-gaps)'s `mi_threshold` and `min_representation`; `false` leaves out the gap analysis and its check |
| `factor-leakage` | a block, or `null` | `null` | [`factor-leakage`](evaluators.md#factor-leakage)'s `factors`, at least one: the group factors, such as a scene or site, whose values must not sit in two splits; unset leaves group leakage out. A factor a split's metadata lacks fails the task, which then has no verdict |
| `diversity` | a block | `method: simpson` | [`diversity`](evaluators.md#diversity)'s `method` |
| `divergence` | a block | `method: mst` | [`divergence`](evaluators.md#divergence)'s `method`; the step runs when the task names an extractor |
| `ood-kneighbors` | a block | `threshold_perc: 99.0`, and the step's own defaults otherwise | [`ood-kneighbors`](evaluators.md#ood-kneighbors)'s `k`, `distance_metric` and `threshold_perc`, where the step's own default is DataEval's 95; the step is fitted on train and runs on each evaluation split when the task names an extractor |
| `blocking` | a list of check types | `[leakage, untrained-classes]` | The check types whose unaccepted warning makes the verdict "Not ready"; each must name a check the chain runs |
| `accepted` | a mapping of a check type, check step or `step[split]` to a reason | `{}` | Why each warning is accepted, by check type (`image-outliers`), by check step for all its runs (`image-outliers-evals`), or by `step[split]` for one run of a step that runs once per evaluation split (`image-outliers-evals[test]`), as the verdict's `warnings[].step` names it: an accepted warning can't make the verdict not ready, but it still leaves it ready with caveats; each key must name a check or check step the chain runs, and a reason may not be blank |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.audit.AuditChecks`):

| Check | Default settings |
| --- | --- |
| [`image-outliers`](checks.md#image-outliers) | `warning: 3.0` |
| [`image-duplicates`](checks.md#image-duplicates) | `exact: 0.0`, `near: 5.0` |
| [`metadata-issues`](checks.md#metadata-issues) | `max_examples: 20` |
| [`class-imbalance`](checks.md#class-imbalance) | `warning: 5.0`, `info: null`, `empty: false` |
| [`class-sufficiency`](checks.md#class-sufficiency) | `train: 20`, `eval: 30` |
| [`untrained-classes`](checks.md#untrained-classes) | `declared: false` |
| [`label-conformance`](checks.md#label-conformance) | `warning: 0`; run only where `ontology` is set |
| [`class-coverage`](checks.md#class-coverage) | `dispersion: 0.5`, `isotropy: 0.5`, `near_duplicates: 0.1` |
| [`uncovered-items`](checks.md#uncovered-items) | `warning: 10.0`; run only under `coverage: {method: naive}` |
| [`dimensional-completeness`](checks.md#dimensional-completeness) | `warning: 0.5`, `info: 0.8` |
| [`factor-coverage-gaps`](checks.md#factor-coverage-gaps) | `warning: 2`; run unless `factor-gaps: false` |
| [`shortcut-risk`](checks.md#shortcut-risk) | `warning: 0.1` |
| [`leakage`](checks.md#leakage) | `exact: 0`, `near: 0`, `groups: 0` |
| [`eval-coverage`](checks.md#eval-coverage) | `warning: 10.0`, `info: 2.0` |
| [`stratification`](checks.md#stratification) | `info: 2.0`, `warning: 10.0` |
| [`distribution-shift`](checks.md#distribution-shift) | `warning: 0.5`, and `info` 0.4 times `warning` |

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

## `data-cleaning`

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
| `classwise-outliers` | check | [`classwise-outliers`](checks.md#classwise-outliers) | `input`: `outliers-by-class` |
| `image-duplicates` | check | [`image-duplicates`](checks.md#image-duplicates) | `input`: `duplicates` |
| `class-imbalance` | check | [`class-imbalance`](checks.md#class-imbalance) | `input`: `label-health` |
| `clean` | transform | [`remove`](transforms.md#remove) | `input`: `data`; `plans`: `duplicates`, `outliers` |

**Settings** ({py:class}`~dataeval_flow.workflows.data_cleaning.DataCleaningConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `stats` | a stats policy name, or `null` | `null` | The stats policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `outliers` | a block | required | [`outliers`](evaluators.md#outliers)'s `flags` and `outlier_threshold`, both required, and `cluster_threshold`, `cluster_algorithm` and `n_clusters`; the preset sets `per_target: true` |
| `duplicates` | a block | `merge_near_duplicates: true`, and the step's own defaults otherwise | [`duplicates`](evaluators.md#duplicates)'s `flags`, `merge_near_duplicates`, `cluster_sensitivity`, `cluster_algorithm` and `n_clusters` |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.data_cleaning.DataCleaningChecks`):

| Check | Default settings |
| --- | --- |
| [`image-outliers`](checks.md#image-outliers) | `warning: 3.0` |
| [`target-outliers`](checks.md#target-outliers) | `warning: 3.0` |
| [`classwise-outliers`](checks.md#classwise-outliers) | `warning: 3.0` |
| [`image-duplicates`](checks.md#image-duplicates) | `exact: 0.0`, `near: 5.0` |
| [`class-imbalance`](checks.md#class-imbalance) | `warning: 5.0` |

The other settings go to the evaluators: the `outliers` block to `outliers`, the `duplicates` block to `duplicates`,
`metadata` to `label-health`, and `stats` to both `outliers` and `duplicates`. Each block holds the step settings its
row lists, spelled as the step spells them. Each `checks:` entry holds the settings of the check it names. `clean`
removes each image and box with at least one outlier flag, and each exact or near duplicate but the first of its group.

Its report gives each finding a section, with the evaluators it judged below it: the flagged images and boxes under
Image Outliers, and the duplicate groups under Image Duplicates. The class counts sit under the first finding that read
`label-health`: Target Outliers where any box was flagged, else Class Imbalance. A finding that read a step shown
already names the finding it is under, as Classwise Outliers names Image Outliers for the outliers `outliers-by-class`
counted. `clean`'s section follows, saying how many images it kept and what each plan named. On MILCO's reference
campaigns, as {doc}`View a report as HTML <../notebooks/view_html_reports>` runs it: "Kept 162 of 261 images. Removed 99
images and 32 detections: 90 images named by `duplicates`, 11 images and 32 detections by `outliers`." Two images were
named by both plans. A Steps table lists every step, what it read, and why it made nothing where it did not.

Run as a step of a custom workflow, `<step>.clean` reads the cleaned Dataset. See
[Workflow types as presets](../concepts/WorkflowsAsChains.md#workflow-types-as-presets).

```yaml
workflows:
  - name: cleaning
    type: data-cleaning
    outliers: {flags: [pixel, visual], outlier_threshold: zscore}
    checks:
      image-outliers: {warning: 5.0}

tasks:
  - {name: clean-train, workflow: cleaning, sources: [train]}
```

## `label-space`

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
| `mergeability` | check | [`mergeability`](checks.md#mergeability) | `input`: `label-alignment` |
| `ontology-validation` | evaluator | [`ontology-validation`](evaluators.md#ontology-validation) | `input`: `data` |
| `ontology-structure` | check | [`ontology-structure`](checks.md#ontology-structure) | `input`: `ontology-validation` |

**Settings** ({py:class}`~dataeval_flow.workflows.label_space.LabelSpaceConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `ontology` | an ontology name, a path, or a nested mapping | required | The ontology to judge labels against: a name under the top-level `ontologies:` key, a path to a serialized RDF artifact resolved against the data root, or a nested mapping of concept to children. |
| `representation` | a block | the step's own defaults | [`representation`](evaluators.md#representation)'s `expected`: each class's minimum share; the preset sets its `ontology` |
| `ontology-validation` | a block | the step's own defaults | [`ontology-validation`](evaluators.md#ontology-validation)'s `label_pattern`; the preset sets its `ontology` |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.label_space.LabelSpaceChecks`):

| Check | Default settings |
| --- | --- |
| [`leaf-coverage`](checks.md#leaf-coverage) | `coverage: 0.9`, `empty_branches: 0` |
| [`label-conformance`](checks.md#label-conformance) | `warning: 0` |

The `mergeability` and `ontology-structure` checks take no `checks:` entry. The ontology is the sanctioned label space,
so the preset can name a class that was never collected, which counting the dataset's own labels cannot. It is the only
preset that judges labels against an ontology: `data-coverage` refuses `ontology:`, so run a `label-space` entry on the
same source beside it. The ontology is declared inline, loaded from an RDF file, or named from the shared `ontologies:`
block; `ontology:` has no default. See [Declare an ontology](../how_to/declare_an_ontology.md).

```yaml
workflows:
  - name: vocab-check
    type: label-space
    ontology:
      animal:
        mammal: [cat, dog]
        bird: [owl]

tasks:
  - {name: vocab, workflow: vocab-check, sources: [train]}
```

## `data-coverage`

Judges how a Dataset's embeddings cover their space, its class balance and metadata gaps, and what to acquire per class;
detections are cropped first.

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
| `label-health` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `data` |
| `class-imbalance` | check | [`class-imbalance`](checks.md#class-imbalance) | `input`: `label-health` |
| `factor-summary` | evaluator | [`factor-summary`](evaluators.md#factor-summary) | `input`: `data` |
| `balance` | evaluator | [`balance`](evaluators.md#balance) | `input`: `data` |
| `diversity` | evaluator | [`diversity`](evaluators.md#diversity) | `input`: `data` |
| `factor-gaps` | combine | [`factor-gaps`](combines.md#factor-gaps) | `input`: `data`; `balance`: `balance` |
| `factor-coverage-gaps` | check | [`factor-coverage-gaps`](checks.md#factor-coverage-gaps) | `input`: `factor-gaps` |
| `representation` | evaluator | [`representation`](evaluators.md#representation) | `input`: `data` |
| `class-shortfall` | check | [`class-shortfall`](checks.md#class-shortfall) | `input`: `representation` |

**Settings** ({py:class}`~dataeval_flow.workflows.data_coverage.DataCoverageConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | `null`; any other value is refused | `null` | Refused when set, as the config loads: judge the labels with a `label-space` entry on the same source; see [Settings every preset shares](#settings-every-preset-shares) |
| `representation` | a block | the step's own defaults | [`representation`](evaluators.md#representation)'s `expected`: each class's minimum share |
| `coverage` | a block | `method: adaptive`, `num_observations: 50` where the step's own default is DataEval's 20, and the step's other defaults | [`coverage`](evaluators.md#coverage)'s `method`, `num_observations`, `percent`, `min_class_samples`, `isotropy_min_samples` and `near_duplicate_factor`; the step runs when the task names an extractor |
| `wrap` | a block | `params: {padding: 0.0, min_size: 1}` | [`wrap`](transforms.md#wrap)'s `params`, used on detection data only; the preset fixes the wrapper |
| `completeness` | `true` or `false` | `true` | Whether the completeness steps run, when the task names an extractor |
| `diversity` | a block | `method: simpson` | [`diversity`](evaluators.md#diversity)'s `method` |
| `factor-gaps` | a block, or `false` | `mi_threshold: 0.1`, `min_representation: 5` | [`factor-gaps`](combines.md#factor-gaps)'s `mi_threshold` and `min_representation`; `false` leaves out the gap analysis and its check |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.data_coverage.DataCoverageChecks`):

| Check | Default settings |
| --- | --- |
| [`class-imbalance`](checks.md#class-imbalance) | `warning: 5.0`, `info: 2.0` |
| [`factor-coverage-gaps`](checks.md#factor-coverage-gaps) | `warning: 2` |
| [`class-coverage`](checks.md#class-coverage) | `dispersion: 0.5`, `isotropy: 0.5`, `near_duplicates: 0.1` |
| [`uncovered-items`](checks.md#uncovered-items) | `warning: 10.0` |
| [`dimensional-completeness`](checks.md#dimensional-completeness) | `warning: 0.5`, `info: 0.8` |

Coverage asks whether a collection spans the conditions the model will meet, where cleaning asks whether its samples are
sound. The preset evaluates class balance and metadata factor gaps. When the task names an
{term}`extractor <Extractor>`, it adds per-class embedding variety and dimensional completeness. A class can be
plentiful by count and still occupy little of the representation space, which the embedding steps show. `naive` coverage
is judged by an `uncovered-items` step. The chain is the same without an extractor, but the `coverage` and
`completeness` steps are then skipped with "requires an extractor". Run it before training and before fixing a reference
set, while a gap can still be closed by collecting more data. See [Dataset Coverage](../concepts/Coverage.md).

```yaml
workflows:
  - name: coverage
    type: data-coverage
    coverage: {method: naive}
    checks:
      uncovered-items: {warning: 5.0}

tasks:
  - {name: coverage-train, workflow: coverage, sources: [train], extractor: bovw_ext}
```

## `data-splitting`

Splits a Dataset into train, val and test, or k folds, and judges its balance, stratification and coverage; with `folds`
of 2 or more, `train` and `val` are lists keyed by fold.

- **Answers:** [Are the splits fit to evaluate on?](index.md#are-the-splits-fit-to-evaluate-on)
- **Reads:** `data`, the task's one source.
- **Makes:** `train`, `val` and `test`.

**Chain**, from `rebalance: interclass` and `coverage: {method: naive}`, with an extractor for the `coverage` steps:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `label-health` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `data` |
| `class-imbalance` | check | [`class-imbalance`](checks.md#class-imbalance) | `input`: `label-health` |
| `balance` | evaluator | [`balance`](evaluators.md#balance) | `input`: `data` |
| `diversity` | evaluator | [`diversity`](evaluators.md#diversity) | `input`: `data` |
| `coverage` | evaluator | [`coverage`](evaluators.md#coverage) | `input`: `data` |
| `uncovered-items` | check | [`uncovered-items`](checks.md#uncovered-items) | `input`: `coverage` |
| `split` | transform | [`split`](transforms.md#split) | `input`: `data` |
| `rebalanced` | transform | [`view`](transforms.md#view) | `input`: `split.train` |
| `label-health-train` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `split.train` |
| `label-health-val` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `split.val` |
| `label-health-test` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `split.test` |
| `label-health-rebalanced` | evaluator | [`label-health`](evaluators.md#label-health) | `input`: `rebalanced` |
| `stratification` | check | [`stratification`](checks.md#stratification) | `input`: `label-health`; `parts`: `label-health-train`, `label-health-val`, `label-health-test`; `shown`: `label-health-rebalanced` |
| `coverage-train` | evaluator | [`coverage`](evaluators.md#coverage) | `input`: `rebalanced` |
| `uncovered-items-train` | check | [`uncovered-items`](checks.md#uncovered-items) | `input`: `coverage-train` |
| `coverage-val` | evaluator | [`coverage`](evaluators.md#coverage) | `input`: `split.val` |
| `uncovered-items-val` | check | [`uncovered-items`](checks.md#uncovered-items) | `input`: `coverage-val` |
| `coverage-test` | evaluator | [`coverage`](evaluators.md#coverage) | `input`: `split.test` |
| `uncovered-items-test` | check | [`uncovered-items`](checks.md#uncovered-items) | `input`: `coverage-test` |

**Settings** ({py:class}`~dataeval_flow.workflows.data_splitting.DataSplittingConfig`):

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
| `coverage` | a block | `method: adaptive`, `num_observations: 50`, `percent: 0.01` | [`coverage`](evaluators.md#coverage)'s `method`, `num_observations` and `percent`; the steps run on the whole set and each part when the task names an extractor |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.data_splitting.DataSplittingChecks`):

| Check | Default settings |
| --- | --- |
| [`class-imbalance`](checks.md#class-imbalance) | `warning: 10.0` |
| [`stratification`](checks.md#stratification) | `info: 2.0`, `warning: 10.0` |
| [`uncovered-items`](checks.md#uncovered-items) | `warning: 5.0` |

The chain judges the whole set's labels, balance and diversity, splits it (`folds: 1`) or cuts it into k folds
(`folds` of 2 or more) with a shared test part, optionally rebalances each train, and judges each part's labels, its
stratification against the whole, and, when the task names an extractor, its coverage. Its findings are Class
Imbalance for the whole set, Stratification for each fold, and Uncovered Items under `naive` coverage; balance and
diversity are report sections. The result is a `ChainResult`, and each part's indices into the source are in
`result.steps["split"].details["indices"]`. Run as a step of a custom workflow, the entry hands on three Datasets:
`<step>.train` (the rebalanced train, where the entry sets `rebalance:`), `<step>.val` and `<step>.test`. The preset
does not judge leakage, shift or evaluation coverage. [`audit`](#audit) does, given the parts as sources, train first,
and [Check a set of splits](../how_to/write_a_custom_workflow.md#11-check-a-set-of-splits) chains the same steps
after a split. See [Dataset Splitting](../concepts/DatasetSplitting.md).

```yaml
workflows:
  - name: splitting
    type: data-splitting
    test_frac: 0.2
    rebalance: interclass

tasks:
  - {name: split-train, workflow: splitting, sources: [train]}
```

## `drift-monitoring`

Tests each incoming source for drift from a reference, whole, by chunk and by class.

- **Answers:** [Has new data drifted?](index.md#has-new-data-drifted)
- **Reads:** `reference`, then `tests`: the first source is the reference, and each later source is tested against it.
- **Makes:** no Dataset; its findings are its result.

**Chain**, from `detectors: [{name: mmd, type: drift-mmd, chunking: {chunk_count: 5}}]` and
`classwise: {mmd: class}`, with the task's extractor or the detector's own:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `mmd` | evaluator | [`drift-mmd`](evaluators.md#drift-mmd) | `input`: `reference`, `tests` |
| `mmd-check` | check | [`drift`](checks.md#drift) | `input`: `mmd` |
| `mmd-by-class` | evaluator | [`drift-mmd`](evaluators.md#drift-mmd) | `input`: `reference`, `tests` |
| `mmd-by-class-check` | check | [`drift`](checks.md#drift) | `input`: `mmd-by-class` |

**Settings** ({py:class}`~dataeval_flow.workflows.drift_monitoring.DriftMonitoringConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `detectors` | a list of drift evaluator entries | required | Drift evaluator entries (`drift-univariate`, `drift-mmd`, `drift-kneighbors`, `drift-domain-classifier`), each tested on every test source against the reference. An entry's `name` names its step. An entry may name its own `extractor:`. |
| `classwise` | a mapping of detector name to `by:` | `{}` | Detectors to also run per key, unchunked, each with its `by:`: `{drift-mmd: class}`, `{uncertainty: predicted}` (see [Drift in a model's uncertainty](../how_to/monitor_drift.md#6-drift-in-a-models-uncertainty)), or with settings; `min_items` is 2 unless written. |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.drift_monitoring.DriftMonitoringChecks`):

| Check | Default settings |
| --- | --- |
| [`drift`](checks.md#drift) | `warn_on_drift: true`, `chunk_percent: 10.0`, `consecutive_chunks: 2` |

A task with a reference and two test sources runs every detector once for each test source. The entries of
`detectors:` are drift evaluator entries, so each takes the fields its evaluator takes in the
[Evaluator Catalog](evaluators.md). An entry's `name` defaults to its type, and it names the detector's step. The
`drift` check is the step `<detector>-check`. The report groups each source's findings under the source's name, and
the result is a `ChainResult` with one element per test source for each step. See
[Monitor drift with steps](../how_to/monitor_drift.md).

```yaml
workflows:
  - name: drift
    type: drift-monitoring
    detectors:
      - {type: drift-univariate, method: ks, p_val: 0.01}
      - {name: mmd-chunked, type: drift-mmd, chunking: {chunk_count: 10}}
    checks:
      drift: {chunk_percent: 20.0}

tasks:
  - {name: cameras, workflow: drift, sources: [train, test, operational], extractor: bovw_ext}
```

## `ood-detection`

Flags each test source's images unlike the reference, by each detector and by their agreement, with the metadata behind
them.

- **Answers:** [Which items are out of distribution?](index.md#which-items-are-out-of-distribution)
- **Reads:** `reference`, then `tests`: the first source is the reference, and each later source is scored against it.
- **Makes:** no Dataset; its findings are its result.

**Chain**, from the detectors `knn` (`ood-kneighbors`, with `distance_metric: euclidean`) and `dc`
(`ood-domain-classifier`), with the task's extractor or each detector's own:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `knn` | evaluator | [`ood-kneighbors`](evaluators.md#ood-kneighbors) | `input`: `reference`, `tests` |
| `knn-check` | check | [`ood`](checks.md#ood) | `input`: `knn` |
| `dc` | evaluator | [`ood-domain-classifier`](evaluators.md#ood-domain-classifier) | `input`: `reference`, `tests` |
| `dc-check` | check | [`ood`](checks.md#ood) | `input`: `dc` |
| `ood-union` | combine | [`ood-union`](combines.md#ood-union) | `input`: `knn`, `dc` |
| `ood-agreement` | check | [`ood-agreement`](checks.md#ood-agreement) | `input`: `ood-union` |
| `factor-predictors` | combine | [`factor-predictors`](combines.md#factor-predictors) | `ood`: `ood-union`; `reference`: `reference`; `input`: `tests` |
| `factor-deviation` | combine | [`factor-deviation`](combines.md#factor-deviation) | `ood`: `ood-union`; `reference`: `reference`; `input`: `tests` |

`ood-agreement` runs only with two or more detectors, so one detector gives no `ood-agreement` step.

**Settings** ({py:class}`~dataeval_flow.workflows.ood_detection.OODDetectionConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `stats` | a stats policy name, or `null` | `null` | The stats policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `detectors` | a list of OOD evaluator entries | required | OOD evaluator entries (`ood-kneighbors`, `ood-domain-classifier`), each scoring every test source against the reference. An entry's `name` names its step. An entry may name its own `extractor:`. |
| `factor-predictors` | `false`, or `null` | `null` | `false` leaves out the `factor-predictors` step; it takes no settings. |
| `factor-deviation` | a block, or `false` | `max_items: 50` | [`factor-deviation`](combines.md#factor-deviation)'s `max_items`; `false` leaves it out. |
| `checks` | a block | the defaults below | When findings warn, keyed by check type |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.ood_detection.OODDetectionChecks`):

| Check | Default settings |
| --- | --- |
| [`ood`](checks.md#ood) | `warning: 10.0`, `info: 1.0` |
| [`ood-agreement`](checks.md#ood-agreement) | `warning: 10.0`, `info: 1.0` |

Out-of-distribution detection asks of each image whether it is anomalous relative to the reference, where drift asks
whether a whole batch moved. Each detector scores every test source against the reference, and an `ood` check
judges each detector's flagged share. The `ood-union` combine joins the detectors' flags, and `ood-agreement` judges how
far they agree when there are two or more detectors. `factor-predictors` and `factor-deviation` read the flagged
images' metadata. Use it during data ingestion, to flag anomalous samples before they reach a model, and in
operation, to flag individual inputs outside the training distribution. See [Distribution Shift](../concepts/DistributionShift.md).

```yaml
workflows:
  - name: ood
    type: ood-detection
    detectors:
      - {name: knn, type: ood-kneighbors, distance_metric: euclidean}
      - {name: dc, type: ood-domain-classifier}
    checks:
      ood: {warning: 5.0}

tasks:
  - {name: ood-operational, workflow: ood, sources: [train, operational], extractor: bovw_ext}
```

## `data-prioritization`

Ranks each pool against a reference for labeling, after optional cleaning, and keeps the top.

- **Answers:** [Which items should be labeled next?](index.md#which-items-should-be-labeled-next)
- **Reads:** `reference`, then `pools`: the first source is the reference and the rest are the pools. This is the
  reverse of the [`prioritization`](evaluators.md#prioritization) evaluator's order, the data to rank and then the
  reference, and the chain hands them to it in that order.
- **Makes:** `selected`, each pool's top-ranked items.

**Chain**, from `cleaning: {outliers: {flags: [pixel], outlier_threshold: zscore}}`, with an extractor for
`prioritization`:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `outliers-reference` | evaluator | [`outliers`](evaluators.md#outliers) | `input`: `reference` |
| `duplicates-reference` | evaluator | [`duplicates`](evaluators.md#duplicates) | `input`: `reference` |
| `reference-clean` | transform | [`remove`](transforms.md#remove) | `input`: `reference`; `plans`: `duplicates-reference`, `outliers-reference` |
| `outliers-pool` | evaluator | [`outliers`](evaluators.md#outliers) | `input`: `pools` |
| `duplicates-pool` | evaluator | [`duplicates`](evaluators.md#duplicates) | `input`: `pools` |
| `pool-clean` | transform | [`remove`](transforms.md#remove) | `input`: `pools`; `plans`: `duplicates-pool`, `outliers-pool` |
| `prioritization` | evaluator | [`prioritization`](evaluators.md#prioritization) | `input`: `pool-clean`, `reference-clean` |
| `selected` | transform | [`select`](transforms.md#select) | `input`: `pool-clean`; `ranking`: `prioritization` |

**Settings** ({py:class}`~dataeval_flow.workflows.data_prioritization.DataPrioritizationConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `stats` | a stats policy name, or `null` | `null` | The stats policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `method` | `knn`, `kmeans_distance`, `kmeans_complexity`, `hdbscan_distance` or `hdbscan_complexity` | `knn` | The ranking method. |
| `k` | a count, or `null` | `null` | The neighbors the `knn` method counts; unset uses the square root of the number of samples. |
| `c` | a count, or `null` | `null` | The clusters the `kmeans` and `hdbscan` methods make; unset uses the square root of the number of samples. |
| `n_init` | a count, or `auto` | `auto` | The K-means initializations, for the `kmeans` methods only. |
| `max_cluster_size` | a count, or `null` | `null` | The largest cluster, for the `hdbscan` methods only; unset sets no limit. |
| `order` | `easy_first` or `hard_first` | `hard_first` | The sort direction: `easy_first` puts prototypical items first, and `hard_first` puts novel or challenging items first. |
| `policy` | `difficulty`, `stratified` or `class_balanced` | `difficulty` | The selection policy: `difficulty` keeps the ranking's order, `stratified` selects across bins of it, and `class_balanced` balances the classes. |
| `num_bins` | a count | `50` | The bins of the `stratified` policy. |
| `cleaning` | a block, or `null` | `null` | Cleaning before ranking: `outliers`, required, and `duplicates`, which take the fields of [`data-cleaning`](#data-cleaning)'s blocks of the same names, with the same defaults; and `dup_types`, the duplicate kinds removed, `[exact, near]` unless set. Unset ranks the data as it is |
| `select` | a block | `n: null`, `fraction: null` | [`select`](transforms.md#select)'s `n` and `fraction`: how much of each pool's ranking `selected` keeps |

The preset has no `checks:`. With `cleaning:` set, the reference and each pool are cleaned by their own `outliers`
and `duplicates` steps and a `remove` step, and `method`, `k`, `c`, `n_init`, `max_cluster_size`, `order`, `policy` and
`num_bins` are the settings of `prioritization`. `selected` keeps the top of each pool's ranking: `select.n` items, or
`select.fraction` of them. Without `cleaning:`, only `prioritization` and `selected` run, reading `pools` and
`reference`. `cleaning.dup_types: [exact]` makes both plans' `dup_types` `[exact]`. With neither `select.n` nor
`select.fraction`, `selected` keeps every item (`fraction: 1.0`). The chain has no checks, so it makes no findings. See
[Data Prioritization](../concepts/Prioritization.md).

```yaml
workflows:
  - name: prioritization
    type: data-prioritization
    method: knn
    k: 5
    cleaning: {outliers: {flags: [pixel, visual], outlier_threshold: zscore}}
    select: {n: 200}

tasks:
  - {name: next-labels, workflow: prioritization, sources: [labeled, unlabeled], extractor: bovw_ext}
```

## `metadata-triage`

Reports unreadable and unpinned metadata factors, with suggested corrections.

- **Answers:** [Is the metadata readable?](index.md#is-the-metadata-readable)
- **Reads:** `data`, the task's one source.
- **Makes:** no Dataset; its findings are its result.

**Chain**, from its defaults:

| Step | Kind | Type | Reads |
| --- | --- | --- | --- |
| `factor-triage` | evaluator | [`factor-triage`](evaluators.md#factor-triage) | `input`: `data` |
| `metadata-issues` | check | [`metadata-issues`](checks.md#metadata-issues) | `input`: `factor-triage` |

**Settings** ({py:class}`~dataeval_flow.workflows.metadata_triage.MetadataTriageConfig`):

| Setting | Takes | Default | Description |
| --- | --- | --- | --- |
| `metadata` | a metadata policy name, or `null` | `null` | The metadata policy; see [Settings every preset shares](#settings-every-preset-shares) |
| `ontology` | an ontology name, a path, or a nested mapping, or `null` | `null` | The label space; see [Settings every preset shares](#settings-every-preset-shares) |
| `checks` | a block | the defaults below | The `metadata-issues` check's settings, keyed by check type. |
| `verify` | `true` or `false` | `true` | Re-read the metadata under the complete suggestions and report what they recover. Costs no second dataset walk: `repair` returns a copy sharing the store. |
| `default_bins` | a count | `10` | Bin count a suggestion falls back to where the run left no fit to read. Where there is one, the populated bins of the derived cut are carried forward instead, which pins the cut the run used rather than substituting a different one. |
| `min_missing_fraction` | a fraction | `0.2` | Share of rows recording no value above which a factor is called degenerate. |

**Checks**, under `checks:` ({py:class}`~dataeval_flow.workflows.metadata_triage.MetadataTriageChecks`):

| Check | Default settings |
| --- | --- |
| [`metadata-issues`](checks.md#metadata-issues) | `max_examples: 20` |

The settings expand to two steps on the task's one source, `data`. `metadata:`, `verify`, `default_bins` and
`min_missing_fraction` are `factor-triage`'s settings, and `max_examples` is `metadata-issues`', set under
`checks.metadata-issues`. Its findings are `metadata-issues`': one per kind of issue, then the suggested policy and
what verification recovered. The chain makes no Dataset, so it declares no output. Its result's `metadata_binning`
records the encoding `factor-triage` read, which `dataeval-flow encoding` writes out.

```yaml
workflows:
  - name: triage
    type: metadata-triage
    verify: true
    checks:
      metadata-issues: {max_examples: 10}

tasks:
  - {name: triage-train, workflow: triage, sources: [train]}
```
