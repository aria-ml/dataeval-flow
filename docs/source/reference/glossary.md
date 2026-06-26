# Glossary

Key vocabulary for using DataEval Flow. Orchestration terms (workflow, pipeline,
source, extractor, …) are defined here in full; for the underlying evaluator
science, each entry links to the authoritative
[DataEval explanations](https://dataeval.readthedocs.io/en/latest/) and the
DataEval Flow [Explanation pages](../concepts/index.md).

```{glossary}
Bag-of-Visual-Words (BoVW)
    A model-free {term}`feature extractor<Extractor>` that builds an
    {term}`embedding<Embedding>` from a histogram of quantized local image
    features (SIFT descriptors). Useful when no trained model is available.

Caching
    Reuse of intermediate computation (loaded datasets, embeddings, evaluator
    results) keyed on the configuration and inputs that produced them, so that
    re-running an unchanged pipeline does not recompute it. DataEval Flow
    supports in-memory caching and a disk-backed cache at the `/cache` mount.

Classwise Drift
    {term}`Drift` measured separately for each class, revealing which classes a
    distribution shift most affects rather than only an aggregate signal.

Data Cleaning
    The process of identifying and flagging quality issues in a dataset —
    {term}`outliers<Outlier>`, {term}`duplicates<Duplicates>`, and label
    anomalies. See the
    [DataEval Data Integrity explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html).

DataEval
    The core evaluation library that provides the statistical analysis, outlier
    detection, drift, OOD, and data-quality algorithms. DataEval Flow
    orchestrates these evaluators behind a declarative configuration.

Domain Classifier
    A drift/OOD method that trains a classifier to distinguish reference data
    from incoming data; the better it succeeds, the larger the distributional
    difference.

Drift
    A change over time in the statistical properties of data relative to the
    training {term}`reference dataset<Reference Dataset>`, which degrades model
    performance. DataEval Flow's drift workflow detects population-level drift;
    see the
    [DataEval Distribution Shift explanation](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html).

Duplicates
    Exact or near-identical samples in a dataset. Exact matches are found via a
    byte hash; near matches via a perception-based hash or
    {term}`embedding<Embedding>` distance.

Embedding
Embeddings
    A compact, fixed-length vector representation of an image produced by a
    {term}`feature extractor<Extractor>`. Geometric distance between embeddings
    is a proxy for semantic similarity, and most embedding-space evaluators
    (drift, OOD, prioritization) operate on them. See the
    [DataEval Embeddings explanation](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html).

Extractor
    Also *feature extractor*. A component that turns images into
    {term}`embeddings<Embedding>`. DataEval Flow ships ONNX, PyTorch,
    {term}`BoVW<Bag-of-Visual-Words (BoVW)>`, Flatten, and Uncertainty
    extractors; an extractor may reference a {term}`preprocessor<Preprocessor>`.

MAITE
    Modular AI Trustworthy Engineering — the JATIC protocol for interoperable
    AI/ML datasets, models, and components. MAITE-compliant inputs give DataEval
    Flow native interoperability with the rest of the JATIC suite.

Maximum Mean Discrepancy (MMD)
    A multivariate drift statistic that measures the distance between the mean
    {term}`embeddings<Embedding>` of a reference and an incoming sample in a
    kernel feature space.

ONNX
    Open Neural Network Exchange — an open model format. DataEval Flow can use an
    ONNX model as a {term}`feature extractor<Extractor>` for embedding
    extraction.

Out-of-Distribution (OOD)
    A sample that differs significantly from the training distribution. Where
    {term}`drift<Drift>` is a population-level signal, OOD detection scores
    individual samples. See the
    [DataEval Distribution Shift explanation](https://dataeval.readthedocs.io/en/latest/concepts/DistributionShift.html).

Outlier
    A sample that deviates significantly from the rest of a dataset, detected
    via statistical methods (modified z-score, z-score, or IQR) over image
    statistics or {term}`embeddings<Embedding>`.

Parameter Sweep
    A workflow that runs another workflow repeatedly across a grid of parameter
    values and compares the per-configuration results.

Pipeline
    The full sequence executed in a single DataEval Flow run: one or more
    {term}`sources<Source>` flow through {term}`preprocessing<Preprocessor>` and
    {term}`extraction<Extractor>` into one or more
    {term}`workflow<Workflow>` evaluators, producing reports and
    {term}`result envelopes<Result Envelope>`.

Preprocessor
    A named, ordered image-transform pipeline (built on torchvision transforms)
    applied to samples before {term}`extraction<Extractor>`.

Prioritization
    Ranking abundant or unlabeled samples by how informative they are for
    labeling or review, typically using {term}`embedding<Embedding>` structure.

Reference Dataset
    The baseline (typically training) dataset against which {term}`drift<Drift>`
    and {term}`OOD<Out-of-Distribution (OOD)>` detectors are calibrated.
    Detection quality is bounded by how representative the reference is.

Result Envelope
    The machine-readable output object emitted by a workflow alongside the
    human-readable report, carrying results and metadata in a structured form
    that satisfies JATIC interoperability requirements.

Source
    A configured dataset input — a {term}`MAITE`-compatible dataset or one
    consumed through an adapter (HuggingFace, COCO, YOLO, TorchVision,
    ImageFolder).

Stratified Split
    A dataset split that preserves class proportions across the resulting
    subsets (as opposed to a purely random split), produced by the dataset
    splitting workflow.

Task
    A single configured unit of work within a {term}`pipeline<Pipeline>` —
    binding a {term}`source<Source>` (or sources) to a {term}`workflow<Workflow>`
    and its parameters.

Workflow
    A built-in evaluator type DataEval Flow can run: Data Cleaning, Data
    Analysis, Dataset Splitting, Drift Detection, OOD Detection, Prioritization,
    or Parameter Sweep. Each has its own configuration schema, defaults, and
    {term}`caching<Caching>` contract.

Workflow Configuration
    A YAML or JSON file specifying the {term}`sources<Source>`,
    {term}`tasks<Task>`, {term}`extractors<Extractor>`,
    {term}`preprocessors<Preprocessor>`, and workflow parameters for a run.
    Files at the data root are auto-discovered and merged.
```
