# Find the Right Step

The steps are grouped by the question a user brings. Under each question, the page names the preset that answers it,
then the evaluators, combines, checks and transforms its chain runs, so a step can appear under several questions. A
last bullet names the steps on the question that the preset does not run, which you chain in a
[workflow of your own](../how_to/write_a_custom_workflow.md). Each name links its catalog entry. With no preset, chain
the steps in a workflow of your own.

## Is the data clean?

Finds outlier and duplicate images and boxes, judges class imbalance, and removes the outliers and duplicates.

- **Preset:** [`data-cleaning`](presets.md#data-cleaning)
- **Evaluators:** [`outliers`](evaluators.md#outliers), [`duplicates`](evaluators.md#duplicates),
  [`label-health`](evaluators.md#label-health)
- **Combines:** [`outliers-by-class`](combines.md#outliers-by-class)
- **Checks:** [`image-outliers`](checks.md#image-outliers), [`target-outliers`](checks.md#target-outliers),
  [`classwise-outliers`](checks.md#classwise-outliers), [`image-duplicates`](checks.md#image-duplicates),
  [`class-imbalance`](checks.md#class-imbalance)
- **Transforms:** [`remove`](transforms.md#remove)

## Are the labels sound?

Measures how a Dataset's labels spread over its classes, and warns on classes that are too few, too uneven or missing
from train.

- **Preset:** none yet; chain its steps in a [workflow of your own](../how_to/write_a_custom_workflow.md)
- **Evaluators:** [`label-health`](evaluators.md#label-health)
- **Checks:** [`class-imbalance`](checks.md#class-imbalance), [`class-sufficiency`](checks.md#class-sufficiency),
  [`untrained-classes`](checks.md#untrained-classes)

## Do the labels match an ontology?

Judges a Dataset's class names against a declared ontology. In a workflow of your own, `conform` relabels the
Dataset onto the ontology.

- **Preset:** [`label-space`](presets.md#label-space)
- **Evaluators:** [`representation`](evaluators.md#representation),
  [`label-reconciliation`](evaluators.md#label-reconciliation), [`label-alignment`](evaluators.md#label-alignment),
  [`ontology-validation`](evaluators.md#ontology-validation)
- **Checks:** [`leaf-coverage`](checks.md#leaf-coverage), [`label-conformance`](checks.md#label-conformance),
  [`mergeability`](checks.md#mergeability), [`ontology-structure`](checks.md#ontology-structure)
- **In a workflow of your own:** [`conform`](transforms.md#conform); see the [workflow of your own](../how_to/write_a_custom_workflow.md)

## Does the data cover what the model must handle?

Judges how a Dataset's embeddings, classes and metadata factors cover their space, its class balance, and which
classes fall short, and summarizes its metadata factors; detections are cropped first.

- **Preset:** [`data-coverage`](presets.md#data-coverage)
- **Evaluators:** [`coverage`](evaluators.md#coverage), [`completeness`](evaluators.md#completeness),
  [`label-health`](evaluators.md#label-health), [`factor-summary`](evaluators.md#factor-summary),
  [`balance`](evaluators.md#balance), [`diversity`](evaluators.md#diversity),
  [`representation`](evaluators.md#representation)
- **Combines:** [`factor-gaps`](combines.md#factor-gaps)
- **Checks:** [`class-coverage`](checks.md#class-coverage), [`uncovered-items`](checks.md#uncovered-items),
  [`dimensional-completeness`](checks.md#dimensional-completeness), [`class-imbalance`](checks.md#class-imbalance),
  [`factor-coverage-gaps`](checks.md#factor-coverage-gaps), [`class-shortfall`](checks.md#class-shortfall)
- **Transforms:** [`wrap`](transforms.md#wrap)

## Could the model learn a shortcut?

Measures how metadata factors relate to the class labels and how evenly their values spread, summarizes each factor, and
warns when a factor tells much about the class.

- **Preset:** none yet; chain its steps in a [workflow of your own](../how_to/write_a_custom_workflow.md)
- **Evaluators:** [`balance`](evaluators.md#balance), [`parity`](evaluators.md#parity),
  [`diversity`](evaluators.md#diversity), [`factor-summary`](evaluators.md#factor-summary)
- **Checks:** [`shortcut-risk`](checks.md#shortcut-risk)

## Are the splits fit to evaluate on?

Splits a Dataset, or cuts it into k folds, and judges the whole set's class balance, each part's class shares against
the whole's and, under `naive` coverage when the task names an extractor, the items the whole set and each part leave
uncovered. It does not judge leakage. A workflow of your own adds the items and group values two splits share, how far
apart the splits sit, how much of each evaluation split lies beyond what train covers, whether each class train holds
has enough labels in every split, and whether an evaluation split holds a class train lacks.

- **Preset:** [`data-splitting`](presets.md#data-splitting)
- **Evaluators:** [`label-health`](evaluators.md#label-health), [`balance`](evaluators.md#balance),
  [`diversity`](evaluators.md#diversity), [`coverage`](evaluators.md#coverage)
- **Checks:** [`class-imbalance`](checks.md#class-imbalance), [`uncovered-items`](checks.md#uncovered-items),
  [`stratification`](checks.md#stratification)
- **Transforms:** [`split`](transforms.md#split), [`kfold`](transforms.md#kfold), [`view`](transforms.md#view)
- **In a workflow of your own:** [`duplicates`](evaluators.md#duplicates),
  [`factor-leakage`](evaluators.md#factor-leakage), [`divergence`](evaluators.md#divergence),
  [`ood-kneighbors`](evaluators.md#ood-kneighbors), [`leakage`](checks.md#leakage),
  [`distribution-shift`](checks.md#distribution-shift), [`eval-coverage`](checks.md#eval-coverage),
  [`class-sufficiency`](checks.md#class-sufficiency), [`untrained-classes`](checks.md#untrained-classes); see
  [Audit a set of splits](../how_to/write_a_custom_workflow.md#11-audit-a-set-of-splits)

## Has new data drifted?

Tests whether incoming data has drifted from a reference, whole, chunk by chunk and by class. `drift-wasserstein`
needs a validation set as a third source, which the preset does not take, so it runs in a workflow of your own.

- **Preset:** [`drift-monitoring`](presets.md#drift-monitoring)
- **Evaluators:** [`drift-univariate`](evaluators.md#drift-univariate), [`drift-mmd`](evaluators.md#drift-mmd),
  [`drift-kneighbors`](evaluators.md#drift-kneighbors), [`drift-domain-classifier`](evaluators.md#drift-domain-classifier)
- **Checks:** [`drift`](checks.md#drift)
- **In a workflow of your own:** [`drift-wasserstein`](evaluators.md#drift-wasserstein); see the [workflow of your own](../how_to/write_a_custom_workflow.md)

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

- **Preset:** [`data-prioritization`](presets.md#data-prioritization)
- **Evaluators:** [`prioritization`](evaluators.md#prioritization), [`outliers`](evaluators.md#outliers),
  [`duplicates`](evaluators.md#duplicates)
- **Transforms:** [`select`](transforms.md#select), [`remove`](transforms.md#remove)

## Is the metadata readable?

Reports the metadata factors a run could not read as configured, and a policy that repairs them.

- **Preset:** [`metadata-triage`](presets.md#metadata-triage)
- **Evaluators:** [`factor-triage`](evaluators.md#factor-triage)
- **Checks:** [`metadata-issues`](checks.md#metadata-issues)

## What exactly was evaluated?

Records SHA-256 digests of every item's image and labels, and of its metadata.

- **Preset:** none yet; chain its steps in a [workflow of your own](../how_to/write_a_custom_workflow.md)
- **Evaluators:** [`content-digest`](evaluators.md#content-digest)

## Preparing data

Shapes, combines and writes out Datasets, for the steps that read them.

- **Preset:** none yet; chain its steps in a [workflow of your own](../how_to/write_a_custom_workflow.md)
- **Transforms:** [`view`](transforms.md#view), [`wrap`](transforms.md#wrap), [`merge`](transforms.md#merge),
  [`conform`](transforms.md#conform), [`remove`](transforms.md#remove), [`export`](transforms.md#export)
