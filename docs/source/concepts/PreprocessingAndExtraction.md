# Preprocessing and Feature Extraction

Almost every DataEval evaluator works not on raw pixels but on
{term}`embeddings <Embeddings>` — compact numerical vectors that capture what
matters about an image while suppressing irrelevant variation. Getting from a
folder of images to those vectors involves two steps that DataEval Flow makes
explicit and configurable: **preprocessing** the images into a consistent form,
and **extracting** features from them. Both are declared in the configuration, so
the representation an evaluation runs on is part of the reproducible record.

## Why the representation matters

Raw pixels are a poor space for comparison. Lighting changes, small spatial
shifts, compression artifacts, and noise can make identical scenes look
completely different to a pixel-wise distance, while genuinely important
differences can produce only small pixel changes. An embedding transforms images
into a space where task-relevant visual and semantic structure is emphasized and
irrelevant variation is suppressed — so that geometric distance becomes a
meaningful proxy for "how similar are these images." DataEval's
[Embeddings explanation](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html)
covers what embeddings are, what they capture, and why extractor choice is
consequential; this page covers how DataEval Flow produces them.

## Preprocessing pipelines

A preprocessor is an ordered list of image transforms applied before feature
extraction. The pipeline is built on torchvision's transform library, so the full
range of standard transforms — resize, normalize, crop, and so on — is available
by name, specified as ordered steps with their parameters in the configuration.

DataEval Flow also provides custom steps for needs that arise specifically in
evaluation. **ToRGB**, for example, coerces an image to three channels —
repeating a grayscale channel or dropping an alpha channel — so that a mixed
dataset presents a consistent channel layout to a model that expects RGB input.
Custom steps are written to have a deterministic representation so they
participate cleanly in cache keying (see [Reproducibility](Reproducibility.md)).

The ordering is the pipeline: transforms apply in the sequence declared, and the
declared sequence is part of the reproducible configuration.

## Extractor types

An extractor turns preprocessed images into feature vectors. DataEval Flow
exposes several types, selected by the extractor's model field, because no single
extractor is right for every situation:

- **ONNX** — runs an {term}`ONNX`-format model. This is the portable, framework-
  agnostic option: an exported model file produces embeddings without requiring
  the original training framework, which suits containerized and reproducible
  deployments.
- **PyTorch** — runs a PyTorch model and extracts features from a named layer.
  When you have a task-trained model, embeddings drawn directly from it best
  reflect what matters for your task.
- **BoVW** — a Bag-of-Visual-Words extractor built on classical local features.
  It needs no neural network and gives a training-free, lightweight feature
  representation, useful as a baseline or when a learned model is unavailable.
- **Flatten** — passes images through as flattened vectors with no learned model.
  This is the explicit "raw features" option; it is fast but inherits the
  weaknesses of pixel space described above, and is appropriate mainly for
  low-dimensional or already-feature-like inputs.
- **Uncertainty** — derives features from a classifier's prediction confidence
  rather than from image content. This is the representation that model-aware
  drift detection uses, because it is sensitive specifically to shift that moves
  the model into uncertain territory.

## Why extractor choice matters

The extractor defines the space every downstream evaluator measures in. A drift
detector, an OOD scorer, a divergence estimate, or a prioritization ranking is
only as informative as the embedding it operates on: if the extractor does not
represent the dimension along which data actually varies, the evaluator cannot
see it. Choosing an extractor whose training objective aligns with the task — and,
where possible, drawing embeddings from a model trained on the target task — is
therefore one of the most consequential decisions in a pipeline. DataEval's
[Embeddings explanation](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html)
discusses how different model families (classification, detection,
self-supervised, contrastive) shape what embeddings capture and which metrics
they suit.

## Related concept pages

- [Reproducibility](Reproducibility.md) — the declarative configuration these
  `preprocessors` and `extractors` belong to, and how computed embeddings are
  cached for deterministic re-runs
- [Provenance](Provenance.md) — the model, preprocessor, and selection identities
  recorded with every result
- [Distribution Shift](DistributionShift.md) and
  [Data Prioritization](Prioritization.md) — evaluators that operate in the
  embedding space

## See this in practice

### Tutorials

- [Extracting embeddings with ONNX](../notebooks/onnx_embeddings) — configuring an
  ONNX extractor and a preprocessing pipeline
- [Analyzing a dataset](../notebooks/data_analysis) — embeddings feeding a
  multi-finding analysis

### Authoritative reference

- DataEval —
  [Embeddings](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html)
  (what embeddings are and why extractor choice matters)
