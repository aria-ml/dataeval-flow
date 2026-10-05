# Find the Right Step

The steps are grouped by the question a user brings. Under each question, the page names the preset that answers it,
then the evaluators, combines and checks it uses, then any transforms. Each name links its catalog entry. With no
preset, chain the steps in a [custom workflow](../how_to/write_a_custom_workflow.md).

## Is the data clean?

Finds outlier and duplicate images and boxes, and removes them.

- **Preset:** [`data-cleaning`](presets.md#data-cleaning)
- **Evaluators:** [`outliers`](evaluators.md#outliers), [`duplicates`](evaluators.md#duplicates)
- **Combines:** [`outliers-by-class`](combines.md#outliers-by-class)
- **Checks:** [`image-outliers`](checks.md#image-outliers), [`target-outliers`](checks.md#target-outliers),
  [`classwise-outliers`](checks.md#classwise-outliers), [`image-duplicates`](checks.md#image-duplicates)
- **Transforms:** [`remove`](transforms.md#remove)

## Are the labels sound?

Measures how a Dataset's labels spread over its classes, and warns on classes that are too few, too uneven or missing
from train.

- **Preset:** none yet
- **Evaluators:** [`label-health`](evaluators.md#label-health)
- **Checks:** [`class-imbalance`](checks.md#class-imbalance), [`class-sufficiency`](checks.md#class-sufficiency),
  [`untrained-classes`](checks.md#untrained-classes)

## Do the labels match an ontology?

Judges a Dataset's class names against a declared ontology, and conforms the Dataset to it.

- **Preset:** [`label-space`](presets.md#label-space)
- **Evaluators:** [`representation`](evaluators.md#representation),
  [`label-reconciliation`](evaluators.md#label-reconciliation), [`label-alignment`](evaluators.md#label-alignment),
  [`ontology-validation`](evaluators.md#ontology-validation)
- **Checks:** [`leaf-coverage`](checks.md#leaf-coverage), [`label-conformance`](checks.md#label-conformance),
  [`mergeability`](checks.md#mergeability), [`ontology-structure`](checks.md#ontology-structure)
- **Transforms:** [`conform`](transforms.md#conform)

## Does the data cover what the model must handle?

Judges how a Dataset's embeddings, classes and metadata factors cover their space, and which classes fall short.

- **Preset:** [`data-coverage`](presets.md#data-coverage)
- **Evaluators:** [`coverage`](evaluators.md#coverage), [`completeness`](evaluators.md#completeness),
  [`representation`](evaluators.md#representation)
- **Combines:** [`factor-gaps`](combines.md#factor-gaps)
- **Checks:** [`class-coverage`](checks.md#class-coverage), [`uncovered-items`](checks.md#uncovered-items),
  [`dimensional-completeness`](checks.md#dimensional-completeness),
  [`factor-coverage-gaps`](checks.md#factor-coverage-gaps), [`class-shortfall`](checks.md#class-shortfall)

## Could the model learn a shortcut?

Measures how metadata factors relate to the class labels and how evenly their values spread, summarizes each factor, and
warns when a factor tells much about the class.

- **Preset:** none yet
- **Evaluators:** [`balance`](evaluators.md#balance), [`parity`](evaluators.md#parity),
  [`diversity`](evaluators.md#diversity), [`factor-summary`](evaluators.md#factor-summary)
- **Checks:** [`shortcut-risk`](checks.md#shortcut-risk)

## Are the splits fit to evaluate on?

Splits a Dataset, and judges whether the splits keep class shares, hold no leaked items or group values, sit close in
distribution and lie within what train covers.

- **Preset:** [`data-splitting`](presets.md#data-splitting)
- **Evaluators:** [`factor-leakage`](evaluators.md#factor-leakage), [`divergence`](evaluators.md#divergence),
  [`duplicates`](evaluators.md#duplicates), [`ood-kneighbors`](evaluators.md#ood-kneighbors)
- **Checks:** [`stratification`](checks.md#stratification), [`leakage`](checks.md#leakage),
  [`distribution-shift`](checks.md#distribution-shift), [`eval-coverage`](checks.md#eval-coverage)
- **Transforms:** [`split`](transforms.md#split), [`kfold`](transforms.md#kfold)

## Has new data drifted?

Tests whether incoming data has drifted from a reference, whole or chunk by chunk.

- **Preset:** [`drift-monitoring`](presets.md#drift-monitoring)
- **Evaluators:** [`drift-univariate`](evaluators.md#drift-univariate), [`drift-mmd`](evaluators.md#drift-mmd),
  [`drift-kneighbors`](evaluators.md#drift-kneighbors), [`drift-wasserstein`](evaluators.md#drift-wasserstein),
  [`drift-domain-classifier`](evaluators.md#drift-domain-classifier)
- **Checks:** [`drift`](checks.md#drift)

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

Ranks items by difficulty, optionally against a reference, and keeps the top.

- **Preset:** [`data-prioritization`](presets.md#data-prioritization)
- **Evaluators:** [`prioritization`](evaluators.md#prioritization)
- **Transforms:** [`select`](transforms.md#select)

## Is the metadata readable?

Reports the metadata factors a run could not read as configured, and a policy that repairs them.

- **Preset:** [`metadata-triage`](presets.md#metadata-triage)
- **Evaluators:** [`factor-triage`](evaluators.md#factor-triage)
- **Checks:** [`metadata-issues`](checks.md#metadata-issues)

## What exactly was evaluated?

Records SHA-256 digests of every item's image and labels, and of its metadata.

- **Preset:** none yet
- **Evaluators:** [`content-digest`](evaluators.md#content-digest)

## Preparing data

Shapes, combines and writes out Datasets, for the steps that read them.

- **Preset:** none yet
- **Transforms:** [`view`](transforms.md#view), [`wrap`](transforms.md#wrap), [`merge`](transforms.md#merge),
  [`conform`](transforms.md#conform), [`remove`](transforms.md#remove), [`export`](transforms.md#export)
