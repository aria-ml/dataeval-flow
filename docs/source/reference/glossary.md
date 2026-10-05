# Glossary

Key vocabulary for using DataEval Flow. Orchestration terms (workflow, pipeline,
source, extractor, …) are defined here in full; for the underlying evaluator
science, each entry links to the
[DataEval explanations](https://dataeval.readthedocs.io/en/latest/) and the
DataEval Flow [Explanation pages](../concepts/index.md).

```{glossary}
Accepted Risk
    A check type named under an `audit` entry's `accepted:`, with the reason its warning is accepted, on every split.
    Its warning can't make the {term}`verdict<Verdict>` `not-ready`, but a warning it covers leaves the verdict
    `ready-with-caveats`, keeps its severity and evidence, and still counts toward health. See
    [Preset Catalog](presets.md#audit).

Bag-of-Visual-Words (BoVW)
    A model-free {term}`feature extractor<Extractor>` that builds an
    {term}`embedding<Embedding>` from a histogram of quantized local image
    features (SIFT descriptors). Useful when no trained model is available.

Binning
    Cutting a continuous {term}`factor<Factor>` into intervals so evaluators read
    interval codes rather than measured values. Bias, balance, diversity, and
    parity all operate on binned codes, so where the cuts fall changes the
    numbers they report. A categorical factor is *digitized*: mapped to
    ordinals, one code per distinct value. See
    [Configure metadata binning](../how_to/configure_metadata_binning.md).

Blocking Check
    A check type named in an `audit` entry's `blocking:` list, `leakage` and `untrained-classes` unless the entry says
    otherwise. Its warning makes the {term}`verdict<Verdict>` `not-ready` unless an
    {term}`acceptance<Accepted Risk>` covers it. A blocking check that could not run is a caveat, not a block. See
    [Preset Catalog](presets.md#audit).

Brief
    A {term}`finding's<Finding>` one-line summary: the value on its line in the report, such as `3 items`, `ok` or
    `not assessed`. A finding's `title` names it, and its `brief` says what was found. See
    [Check Catalog](checks.md).

Caching
    Reuse of intermediate computation (loaded datasets, embeddings, evaluator
    results) keyed on the configuration and inputs that produced them, so that
    re-running an unchanged pipeline does not recompute it. DataEval Flow
    supports in-memory caching and a disk-backed cache at the `/cache` mount.
    Artifacts live under a version directory (currently `v1`) so incompatible
    formats can coexist.

Chain
    The `inputs:` and `steps:` of a custom {term}`workflow<Workflow>`: {term}`steps<Step>` run in the order they are
    written, each reading the inputs and the steps above it. A {term}`preset<Preset>` expands to a chain too. See
    [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md).

Check
    A {term}`step<Step>` that judges what an evaluator or {term}`combine<Combine>` reported against
    {term}`thresholds<Threshold>` and makes {term}`findings<Finding>`. Evaluators judge nothing; a check does. See the
    [Check Catalog](checks.md).

Classwise Drift
    {term}`Drift` measured separately for each class, showing which classes a
    distribution shift affects most.

Combine
    A {term}`step<Step>` that makes one {term}`Output` from other Outputs, and from the Datasets they were computed on,
    for a {term}`check<Check>` to read, such as `outliers-by-class` or `ood-union`. See the
    [Combine Catalog](combines.md).

Content Digest
    A SHA-256 digest over every item's image and labels, and the class names, whatever the items' order, which
    `audit`'s `content-digest` step records for each split. A training job recomputes it with
    `dataeval_flow.dataset_digest()` and refuses data whose digest differs. The metadata digest, recorded beside it,
    covers each item's metadata, so a loader that yields no metadata still matches the content digest. See
    [Preset Catalog](presets.md#audit) and [Gate training on an audit](../how_to/gate_training_on_an_audit.md).

Coverage
    How completely a dataset spans the conditions a model will meet in operation,
    measured along two axes: which categories are present (labels checked against
    an {term}`ontology<Ontology>`) and how varied each one is
    ({term}`embeddings<Embedding>` checked for clustering, low dimensionality, and
    duplication). See [Dataset Coverage](../concepts/Coverage.md) and the
    [DataEval Dataset Bias and Coverage explanation](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html).

Data Cleaning
    The process of identifying and flagging quality issues in a dataset, such as
    {term}`outliers<Outlier>`, {term}`duplicates<Duplicates>` and label anomalies.
    The `data-cleaning` preset flags outliers and duplicates, judges class
    imbalance and lists the images with no labels. Its `clean` step hands on the
    dataset without the outliers and duplicates, which an `export` step of a custom
    workflow can write to disk. See the
    [DataEval Data Integrity explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html).

DataEval
    The core evaluation library that provides the statistical analysis, outlier
    detection, drift, OOD, and data-quality algorithms. DataEval Flow
    orchestrates these evaluators behind a declarative configuration.

Dataset
    One loaded dataset that {term}`steps<Step>` read and make: a {term}`source's<Source>` data once loaded, possibly
    through a {term}`view<View>`, or what a {term}`transform<Transform>` made from one, such as a split's `train`. A
    source is the configured input a task binds; the Dataset is the data a step reads. See
    [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md).

Determination
    What an {term}`evaluator<Evaluator>` reports: a flag, a group or a p-value
    that DataEval's threshold produced. A determination says what was found;
    the judgment of whether it is a problem belongs to a {term}`check<Check>`.

Domain Classifier
    A drift/OOD method that trains a classifier to distinguish reference data
    from incoming data; the better it succeeds, the larger the distributional
    difference.

Drift
    A change over time in the statistical properties of data relative to the
    training {term}`reference dataset<Reference Dataset>`, which degrades model
    performance. DataEval Flow's `drift-monitoring` workflow detects population-level drift;
    see the
    [DataEval Distribution Shift explanation](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html).

Duplicates
    Exact or near-identical samples in a dataset. Exact matches are found via a
    byte hash; near matches via a perception-based hash or
    {term}`embedding<Embedding>` distance.

Element
    One {term}`Dataset`, or one {term}`Output`, of a list. A list comes from a list input, keyed by source name, or
    from `kfold`'s `train` and `val`. A step that reads one Dataset, handed a list, runs once per element, and a step
    with `pairs: true` runs once per pair of a list's elements. Each element has a {term}`key<Key>`. See
    [Addresses and lists](../concepts/WorkflowsAsChains.md#addresses-and-lists).

Embedding
Embeddings
    A compact, fixed-length vector representation of an image produced by a
    {term}`feature extractor<Extractor>`. Geometric distance between embeddings
    is a proxy for semantic similarity, and most embedding-space evaluators
    (drift, OOD, prioritization) operate on them. See the
    [DataEval Embeddings explanation](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html).

Evaluation Split
    A `val` or `test` split whose labels or coverage are judged against `train`'s: a split `data-splitting` makes, or
    a source after the first that `audit` names. `class-sufficiency` and `untrained-classes` read them. It is not a
    {term}`test source<Test Source>`, which is tested against a {term}`reference<Reference Dataset>`, the data a
    detector fits on. See
    [Are the splits fit to evaluate on?](index.md#are-the-splits-fit-to-evaluate-on).

Evaluator
    One DataEval evaluator run by DataEval Flow, configured under `evaluators:`
    and run by a {term}`task<Task>` that names `evaluator:`, or by an evaluator
    {term}`step<Step>` in a {term}`chain<Chain>`. It reports its
    {term}`determinations<Determination>` with no health status. See
    [Workflows and Evaluators](../concepts/WorkflowsAndEvaluators.md).

Extractor
    Also *feature extractor*. A component that turns images into
    {term}`embeddings<Embedding>`. DataEval Flow ships ONNX, PyTorch,
    {term}`BoVW<Bag-of-Visual-Words (BoVW)>`, Flatten, and Uncertainty
    extractors; an extractor may reference a {term}`preprocessor<Preprocessor>`.

Factor
    One field of {term}`metadata<Metadata>` that evaluators analyze (a sensor name, a
    timestamp, an elevation, a class label). Factors are what bias, balance, and
    coverage analyses correlate against each other and against the class labels.
    Each carries a type (categorical, discrete, or continuous) and a
    {term}`level<Metadata Level>`, and reaches evaluators through
    {term}`binning<Binning>`.

Finding
    A {term}`check's<Check>` judgment: a title, a {term}`severity<Severity>`, a {term}`brief<Brief>`, a description and
    the evidence it judged. See [Check Catalog](checks.md).

Key
    An {term}`element's<Element>` name in its list: a source name, a fold number from `0` to `k-1`, or a pair key
    such as `val_vs_test`, which a step with `pairs: true` gives each pair it runs on. It appears in an address, as
    `0` in `kfold.train[0]`, and in a finding's `step`, as `train` in `class-imbalance[train]`. See
    [Addresses and lists](../concepts/WorkflowsAsChains.md#addresses-and-lists).

MAITE
    Modular AI Trustworthy Engineering — the JATIC protocol for interoperable
    AI/ML datasets, models, and components. MAITE-compliant inputs give DataEval
    Flow native interoperability with the rest of the JATIC suite.

Matrix
    A task's `matrix:`: it runs the task's entry once per combination of the
    values it lists, and compares the runs in one result. See
    [Sweep settings with a matrix](../how_to/run_a_matrix.md).

Maximum Mean Discrepancy (MMD)
    A multivariate drift statistic that measures the distance between the mean
    {term}`embeddings<Embedding>` of a reference and an incoming sample in a
    kernel feature space.

Metadata
    Everything a dataset carries for each item beyond its image and label. A {term}`factor<Factor>` is one field of it
    that an evaluator reads: a metadata field is a factor once an evaluator analyzes it. `metadata-triage` reports which
    fields Flow could read.

Metadata Level
    The entity a {term}`factor<Factor>` describes: `sequence`, `unit`, `track`, or
    `instance`. For a still-image task `unit` is the image and `instance` is the
    target. A factor is summarized and binned at its own level, so a per-image
    factor is not weighted by how many detections each image happens to carry.
    Introduced in DataEval v1.1, replacing the fixed image/target split. Distinct
    from the `level` reported on duplicate groups, which is `item` or `target`.

Not Assessed
    A {term}`check<Check>` that had nothing to judge. Where an input failed or was skipped, or the check could not
    assess it, the check makes one `info` {term}`finding<Finding>` briefed `not assessed`, and its description says
    why. See [How thresholds work](checks.md#how-thresholds-work).

ONNX
    Open Neural Network Exchange — an open model format. DataEval Flow can use an
    ONNX model as a {term}`feature extractor<Extractor>` for embedding
    extraction.

Ontology
    A machine-readable statement of the sanctioned label space — the concepts in a
    domain and how they relate. Declaring one lets the `label-space` workflow
    validate a dataset's labels and name classes missing entirely. See the
    [DataEval Ontology explanation](https://dataeval.readthedocs.io/en/latest/concepts/Ontology.html).

Out-of-Distribution (OOD)
    A sample that differs significantly from the training distribution. {term}`Drift<Drift>`
    is a population-level signal; OOD detection scores individual samples. See the
    [DataEval Distribution Shift explanation](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html).

Outlier
    A sample that deviates significantly from the rest of a dataset, detected
    via statistical methods (adaptive, modified z-score, z-score, or IQR) over image
    statistics or {term}`embeddings<Embedding>`.

Output
    An evaluator's or {term}`combine's<Combine>` result object: what DataEval determined, which a
    {term}`check<Check>` reads. An Output judges nothing; the check's {term}`findings<Finding>` do. See the
    [Evaluator Catalog](evaluators.md).

Pipeline
    The full sequence executed in a single DataEval Flow run: one or more
    {term}`sources<Source>` flow through {term}`preprocessing<Preprocessor>` and
    {term}`extraction<Extractor>` into one or more
    {term}`workflow<Workflow>` evaluators, producing reports and
    {term}`result envelopes<Result Envelope>`.

Port
    A {term}`step's<Step>` named input or output. `dataeval-flow steps NAME` prints one step's ports; an address
    names a step's output port, as `split.train`. See
    [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md).

Preprocessor
    A named, ordered image-transform pipeline (built on torchvision transforms)
    applied to samples before {term}`extraction<Extractor>`.

Preset
    A {term}`workflow<Workflow>` type whose settings expand to a chain of steps:
    evaluators, the combines that join their Outputs, the checks that judge what
    they found, and the transforms that make Datasets. Every built-in workflow
    type, such as `data-cleaning`, is one. See the [Preset Catalog](presets.md) and
    [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md).

Prioritization
    Ranking abundant or unlabeled samples by how informative they are for
    labeling or review, typically using {term}`embedding<Embedding>` structure.

Provenance
    The lineage a result carries: which dataset, under which {term}`view<View>`,
    represented by which model, evaluated with which resolved configuration, by
    which version of the tool, and when. Recorded in the
    {term}`result envelope<Result Envelope>`, provenance is what makes a finding
    auditable. See [Provenance](../concepts/Provenance.md).

Reference Dataset
    The baseline (typically training) dataset against which {term}`drift<Drift>`
    and {term}`OOD<Out-of-Distribution (OOD)>` detectors are calibrated: what a
    detector fits on. The sources tested against it are
    {term}`test sources<Test Source>`; a val or test split judged against train is
    an {term}`evaluation split<Evaluation Split>`. Detection quality is bounded by
    how representative the reference is.

Reproducibility
    The property that the same evaluation over the same data yields the same
    result. DataEval Flow pursues it through declarative configuration, config
    validation, and config-keyed {term}`caching<Caching>`. See
    [Reproducibility](../concepts/Reproducibility.md).

Result Envelope
    The machine-readable output object emitted by a workflow alongside the
    human-readable report, carrying results and metadata in a structured form
    that satisfies JATIC interoperability requirements.

Severity
    A {term}`finding's<Finding>` judgment, one of three. `ok`: its check judged it, and it is within every bound.
    `info`: it falls in an `info` band or meets a criterion its check informs on, its check had nothing to judge (it is
    briefed `not assessed`), or no bound was set to judge it by. `warning`: it passes a {term}`threshold<Threshold>`
    or meets another criterion its check warns on. Warnings count toward the result's health and `--fail-on-warning`.
    Each check's entry in the [Check Catalog](checks.md) says which applies when.

Source
    A configured dataset input — a {term}`MAITE`-compatible dataset or one
    consumed through an adapter (HuggingFace, COCO, YOLO, TorchVision,
    ImageFolder).

Step
    One unit of a {term}`chain<Chain>`: an evaluator, a {term}`transform<Transform>`, a {term}`combine<Combine>` or a
    {term}`check<Check>`, each with named {term}`ports<Port>`. Find one with
    [Find the Right Step](index.md). See [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md).

Stratified Split
    A dataset split that preserves class proportions across the resulting
    subsets, produced by the `split` or `kfold` transform with `stratify: true`.

Subject
    A `drift` or `ood` check's `subject:` setting: the title its {term}`finding<Finding>` takes, so a finding names
    the detector it judged. Unset, the title is the evaluator's title, followed by its entry's name where that differs
    from its type, as in `Drift (MMD) · mmd`. See [Naming Conventions](naming.md).

Task
    A single configured unit of work within a {term}`pipeline<Pipeline>` —
    binding a {term}`source<Source>` (or sources) to exactly one
    {term}`workflow<Workflow>` or {term}`evaluator<Evaluator>`.

Test Source
    A source a detector tests against the {term}`reference<Reference Dataset>`, as `drift-monitoring` and
    `ood-detection` take after their reference. It is not an {term}`evaluation split<Evaluation Split>`, which is judged
    against `train` and not against a reference. See [Has new data drifted?](index.md#has-new-data-drifted).

Threshold
    A {term}`check's<Check>` bound on a measured value, in the unit its check names: a percentage, a ratio, a count,
    a fraction, a score, percentage points or mutual information. A {term}`finding<Finding>` warns only past the
    bound, and `null` switches it off. A {term}`preset<Preset>`'s defaults are under `checks:`. See
    [How thresholds work](checks.md#how-thresholds-work).

Transform
    A {term}`step<Step>` that makes Datasets from a Dataset, or writes one to disk: `split`, `remove`, `export` and the
    others. See the [Transform Catalog](transforms.md).

Verdict
    The level an `audit` gives the data: `not-ready`, `ready-with-caveats` or `ready`, from its checks' warnings, its
    {term}`blocking checks<Blocking Check>` and its {term}`accepted risks<Accepted Risk>`, and the checks
    {term}`not assessed<Not Assessed>`. Health still counts every warning. See [Preset Catalog](presets.md#audit).

View
    A named, ordered pipeline of dataset operations (`Limit`, `ClassFilter`,
    `Shuffle`, …) applied to a dataset before evaluation, referenced by name from
    a {term}`source<Source>`. The config-layer counterpart of `dataeval.data.View`.

Workflow
    A {term}`chain<Chain>` of {term}`steps<Step>`: either a custom workflow, which
    you write, or a built-in workflow type, which expands to a chain. Every
    built-in type is a {term}`preset<Preset>`. Each has its own
    configuration schema, defaults, and {term}`caching<Caching>` contract.

Workflow Configuration
    A YAML or JSON file specifying the {term}`sources<Source>`,
    {term}`tasks<Task>`, {term}`extractors<Extractor>`,
    {term}`preprocessors<Preprocessor>`, and workflow parameters for a run.
    Files at the data root are auto-discovered and merged.

```
