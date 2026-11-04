# Monitor a model's uncertainty

Embeddings say whether the images changed. A deployed model's uncertainty says whether the model still knows them, and
it needs no labels: when the model is less sure on incoming data than on its validation data, the data has moved away
from what it learned. The `uncertainty` extractor runs an ONNX classifier or detector and gives each prediction's
normalized entropy, one column, which a drift detector tests. This guide tests it with
[`drift-mmd`](../reference/evaluators.md#drift-mmd), [`drift-univariate`](../reference/evaluators.md#drift-univariate)
and [`drift-wasserstein`](../reference/evaluators.md#drift-wasserstein), and flags single images with
[`ood-kneighbors`](../reference/evaluators.md#ood-kneighbors). [Distribution Shift](../concepts/DistributionShift.md)
explains what drift and OOD detection ask. You need an ONNX classifier or detector with its DataEval model metadata, a
reference source and one or more test sources.

## Test the uncertainty for drift

```yaml
extractors:
  - name: yolo-uncertainty
    model: uncertainty
    model_path: yolo.onnx
    metadata_path: yolo.json   # DataEval's model metadata: task, input size, class count
    preds_type: sigmoid
    confidence: 0.25
    batch_size: 16

workflows:
  - name: drift
    type: shift
    detectors:
      - {type: drift-mmd}
      - name: uncertainty
        type: drift-univariate
        extractor: yolo-uncertainty
        chunking: {chunk_count: 5}
    classwise:
      uncertainty: predicted
```

Check these settings and behaviors:

- `preds_type` says what the model emits: `logits`, `probs` that sum to 1, or per-class `sigmoid` scores, as YOLO-family
  detectors emit. Check the export: `logits` given probabilities fails silently, squashing every entropy.
- `image_height` and `image_width` are set together, and needed when the model's metadata leaves its input size open.
- A classifier gives one row per image. A detector gives one row per box whose top score is at least `confidence`. The
  model returns a fixed number of boxes per image, padding included, so a detector needs `confidence`; `0` keeps every
  box, and caches every one.
- Drift and OOD detection read it. A detector's rows are detections, not images, so load refuses the extractor on any
  other evaluator. A detector entry's own `extractor:` lets one task mix it with the task's embeddings.
- Chunks hold whole images. `chunk_size` and `chunk_count` count images, and a chunk whose images hold no detections is
  listed as not assessed. The section says how many detections each source held, in how many images.
- `by: predicted` keys rows by the class the model predicts, which unlabeled data allows. A detection counts toward
  every class whose sigmoid score is at least `threshold` (default 0.99) of its top one, and `min_items` counts
  detections. Classes are named by the reference's `index2label` where its ids are the model's outputs, `0` to `n − 1`,
  which holds when the reference is the model's validation data; otherwise they are named by index.
- Detections in one image are correlated, so p-values read optimistic, as they do for crops. A preprocessor that returns
  floats must return them in [0, 1]: DataEval scales only integer images.

## Test the uncertainty against a validation baseline

DataEval's own examples test uncertainty with `drift-wasserstein`, against a validation baseline. `shift` takes no
validation source, so run it as a step of a custom workflow:

```yaml
evaluators:
  - {name: wasserstein, type: drift-wasserstein}

workflows:
  - name: uncertainty_wasserstein
    inputs: [reference, validation, {name: tests, list: true}]
    steps:
      - name: w
        evaluator: wasserstein
        input: [reference, validation, tests]
        extractor: yolo-uncertainty
```

## Flag out-of-distribution images

The same extractor feeds OOD detectors, which flag individual images where drift tests a whole batch.

```yaml
workflows:
  - name: ood
    type: shift
    detectors:
      - {type: ood-kneighbors, k: 10}          # the task's embeddings
      - name: uncertain
        type: ood-kneighbors
        extractor: yolo-uncertainty
        distance_metric: euclidean             # required: an uncertainty row is one number
```

- Images are judged, and detections reported beside them. On a detector's rows, an image is out of distribution when any
  of its detections is, and scores as its most out-of-distribution detection. The finding's brief adds the detections
  flagged: "9/300 images OOD (3.0%; 12/4,812 detections at confidence ≥ 0.25)".
- An image with no detection at the confidence is not assessed. The detector's section lists it, and the percent counts
  assessed images only. A source with no detection at all fails the detector's step.
- `ood-kneighbors` needs `distance_metric: euclidean` here: cosine distance, DataEval's default, cannot rank one number,
  and load refuses it.
- A detector on embeddings and one on uncertainty agree per image, so the `ood-union` step combines them, and its
  section pictures each flagged image once.

## Next steps

- [Monitor incoming data for drift](../notebooks/drift_monitoring.py) — Run the `shift` preset on real imagery with
  chunked detectors.
- [Detect out-of-distribution images](../notebooks/ood_detection.py) — Run the `shift` preset with OOD detectors on
  embeddings.
- [Distribution Shift](../concepts/DistributionShift.md) — Learn what drift and out-of-distribution detection ask.
