# Find the Right Step

The steps are grouped by the question a user brings. Under each question, the page names the presets that answer it,
then the evaluators, combines, checks and transforms on that question that their chains run, so a step can appear under
several questions. A last bullet names the steps on the question that no listed preset runs, which you chain in a
[workflow of your own](../how_to/write_a_custom_workflow.md). Each name links its catalog entry. With no preset, chain
the steps in a workflow of your own.

To ask whether a set of splits is ready to train on, run [`audit`](presets.md#audit). One task answers
[Is the data clean?](#is-the-data-clean), [Are the labels sound?](#are-the-labels-sound),
[Does the data cover what the model must handle?](#does-the-data-cover-what-the-model-must-handle),
[Could the model learn a shortcut?](#could-the-model-learn-a-shortcut) and
[Are the splits fit to evaluate on?](#are-the-splits-fit-to-evaluate-on), and gives a verdict over them.

## Is the data clean?

Finds outlier and duplicate images and boxes, and removes them. `audit` judges each split's outliers, duplicates and
metadata issues.

- **Preset:** [`audit`](presets.md#audit) and [`quality`](presets.md#quality)
- **Evaluators:** [`outliers`](evaluators.md#outliers), [`duplicates`](evaluators.md#duplicates),
  [`label-health`](evaluators.md#label-health), [`factor-triage`](evaluators.md#factor-triage)
- **Combines:** [`outliers-by-class`](combines.md#outliers-by-class)
- **Checks:** [`image-outliers`](checks.md#image-outliers), [`target-outliers`](checks.md#target-outliers),
  [`class-outliers`](checks.md#class-outliers), [`image-duplicates`](checks.md#image-duplicates),
  [`factor-issues`](checks.md#factor-issues)
- **Transforms:** [`remove`](transforms.md#remove)

## Are the labels sound?

Measures how a Dataset's labels spread over its classes, and warns on classes that are too few, too uneven or missing
from train, and, where `ontology:` is set, on class names that resolve to no ontology concept, or to several.
`bias` judges one Dataset's class balance.

- **Preset:** [`audit`](presets.md#audit) and [`bias`](presets.md#bias)
- **Evaluators:** [`label-health`](evaluators.md#label-health),
  [`label-reconciliation`](evaluators.md#label-reconciliation)
- **Checks:** [`class-imbalance`](checks.md#class-imbalance), [`class-sufficiency`](checks.md#class-sufficiency),
  [`untrained-classes`](checks.md#untrained-classes), [`label-conformance`](checks.md#label-conformance)

## Do the labels match an ontology?

Judges a Dataset's class names against a declared ontology. In a workflow of your own, `conform` relabels the
Dataset onto the ontology.

- **Preset:** [`taxonomy`](presets.md#taxonomy); [`audit`](presets.md#audit) runs `label-reconciliation` and
  `label-conformance` on each split where `ontology:` is set
- **Evaluators:** [`representation`](evaluators.md#representation),
  [`label-reconciliation`](evaluators.md#label-reconciliation), [`label-alignment`](evaluators.md#label-alignment),
  [`ontology-validation`](evaluators.md#ontology-validation)
- **Checks:** [`leaf-coverage`](checks.md#leaf-coverage), [`label-conformance`](checks.md#label-conformance),
  [`label-mergeability`](checks.md#label-mergeability), [`ontology-structure`](checks.md#ontology-structure)
- **In a workflow of your own:** [`conform`](transforms.md#conform); see the
  [workflow of your own](../how_to/write_a_custom_workflow.md)

## Does the data cover what the model must handle?

Judges how a Dataset's embeddings cover their space and which classes fall short of their expected share; detections are
cropped first. Among the metadata factors tied to the class, `bias` and `audit` list the class-factor-value
combinations held too rarely.

- **Preset:** [`audit`](presets.md#audit), on train, and [`scope`](presets.md#scope);
  [`bias`](presets.md#bias) runs `factor-gaps` and `factor-coverage-gaps`
- **Evaluators:** [`coverage`](evaluators.md#coverage), [`completeness`](evaluators.md#completeness),
  [`representation`](evaluators.md#representation), [`balance`](evaluators.md#balance),
  [`factor-summary`](evaluators.md#factor-summary), [`diversity`](evaluators.md#diversity)
- **Combines:** [`factor-gaps`](combines.md#factor-gaps)
- **Checks:** [`class-coverage`](checks.md#class-coverage), [`uncovered-items`](checks.md#uncovered-items),
  [`dimensional-completeness`](checks.md#dimensional-completeness),
  [`factor-coverage-gaps`](checks.md#factor-coverage-gaps), [`class-shortfall`](checks.md#class-shortfall)
- **Transforms:** [`wrap`](transforms.md#wrap)

## Could the model learn a shortcut?

Measures how metadata factors relate to the class labels and how evenly their values spread, summarizes each factor, and
warns when a factor tells much about the class or is significantly associated with it. `audit` judges
`shortcut-risk` alone; `parity` and `factor-parity` are `bias`'s.

- **Preset:** [`bias`](presets.md#bias) and [`audit`](presets.md#audit), on train
- **Evaluators:** [`balance`](evaluators.md#balance), [`parity`](evaluators.md#parity),
  [`diversity`](evaluators.md#diversity), [`factor-summary`](evaluators.md#factor-summary)
- **Checks:** [`shortcut-risk`](checks.md#shortcut-risk), [`factor-parity`](checks.md#factor-parity)

## Are the splits fit to evaluate on?

`splits` splits a Dataset, or cuts it into k folds, and judges each part's class shares against the whole's.
`audit` judges splits already made, as a task over sources or as a step after `splits`: the items and group
values two splits share, how far apart the splits sit, how much of each evaluation split lies beyond what train covers,
each split's class shares against train's, whether each class train holds has enough labels in every split, and whether
an evaluation split holds a class train lacks.

- **Preset:** [`audit`](presets.md#audit) and [`splits`](presets.md#splits)
- **Evaluators:** [`label-health`](evaluators.md#label-health), [`coverage`](evaluators.md#coverage),
  [`duplicates`](evaluators.md#duplicates), [`factor-leakage`](evaluators.md#factor-leakage),
  [`divergence`](evaluators.md#divergence), [`ood-kneighbors`](evaluators.md#ood-kneighbors)
- **Checks:** [`uncovered-items`](checks.md#uncovered-items),
  [`class-stratification`](checks.md#class-stratification), [`leakage`](checks.md#leakage),
  [`embedding-divergence`](checks.md#embedding-divergence), [`eval-coverage`](checks.md#eval-coverage),
  [`class-sufficiency`](checks.md#class-sufficiency), [`untrained-classes`](checks.md#untrained-classes)
- **Transforms:** [`split`](transforms.md#split), [`kfold`](transforms.md#kfold), [`view`](transforms.md#view)

## Has new data drifted?

Tests whether incoming data has drifted from a reference, whole, chunk by chunk and by class. `drift-wasserstein`
needs a validation set as a third source, which the preset does not take, so it runs in a workflow of your own.

- **Preset:** [`drift-monitoring`](presets.md#drift-monitoring)
- **Evaluators:** [`drift-univariate`](evaluators.md#drift-univariate), [`drift-mmd`](evaluators.md#drift-mmd),
  [`drift-kneighbors`](evaluators.md#drift-kneighbors),
  [`drift-domain-classifier`](evaluators.md#drift-domain-classifier)
- **Checks:** [`drift`](checks.md#drift)
- **In a workflow of your own:** [`drift-wasserstein`](evaluators.md#drift-wasserstein); see
  [Drift in a model's uncertainty](../how_to/monitor_drift.md#6-drift-in-a-models-uncertainty)

## Which items are out of distribution?

Flags the test images unlike the reference, by each detector and by their agreement, and names the metadata factors that
go with them.

- **Preset:** [`ood-detection`](presets.md#ood-detection)
- **Evaluators:** [`ood-kneighbors`](evaluators.md#ood-kneighbors),
  [`ood-domain-classifier`](evaluators.md#ood-domain-classifier)
- **Combines:** [`ood-union`](combines.md#ood-union), [`factor-predictors`](combines.md#factor-predictors),
  [`factor-deviation`](combines.md#factor-deviation)
- **Checks:** [`ood`](checks.md#ood), [`ood-agreement`](checks.md#ood-agreement)

## Which items should be labeled next?

Ranks each pool's items against a reference for labeling, after optional cleaning of outliers and duplicates, and
keeps the top.

- **Preset:** [`prioritization`](presets.md#prioritization)
- **Evaluators:** [`prioritization`](evaluators.md#prioritization), [`outliers`](evaluators.md#outliers),
  [`duplicates`](evaluators.md#duplicates)
- **Transforms:** [`select`](transforms.md#select), [`remove`](transforms.md#remove)

## Is the metadata readable?

Reports the metadata factors a run could not read as configured, and a policy that repairs them.

- **Preset:** [`triage`](presets.md#triage); [`audit`](presets.md#audit) runs `factor-triage` and
  `factor-issues` on each split
- **Evaluators:** [`factor-triage`](evaluators.md#factor-triage)
- **Checks:** [`factor-issues`](checks.md#factor-issues)

## What exactly was evaluated?

Records SHA-256 digests of every item's image and labels, and of its metadata.

- **Preset:** [`audit`](presets.md#audit), which records each split's content and metadata digests in its record of
  what was audited
- **Evaluators:** [`content-digest`](evaluators.md#content-digest)

## Preparing data

Shapes, combines and writes out Datasets, for the steps that read them.

- **Preset:** none yet; chain its steps in a [workflow of your own](../how_to/write_a_custom_workflow.md)
- **Transforms:** [`view`](transforms.md#view), [`wrap`](transforms.md#wrap), [`merge`](transforms.md#merge),
  [`collect`](transforms.md#collect), [`conform`](transforms.md#conform), [`remove`](transforms.md#remove),
  [`export`](transforms.md#export)
