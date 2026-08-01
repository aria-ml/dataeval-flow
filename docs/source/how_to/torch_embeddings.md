# Use a PyTorch model for embeddings

You have a trained PyTorch model — often the very model under test — and want the evaluators to measure in *its*
representation space rather than a generic one. This guide covers configuring the `torch` extractor, choosing which
layer to read, and the device and container implications.

## Used in these tutorials

- {doc}`Prioritize unlabeled data for labeling <../notebooks/data_prioritization>` — trains a small CNN, saves it, and
  reads its `embed` layer through a `torch` extractor

## Minimal configuration

```yaml
extractors:
  - name: torch_ext
    model: torch
    model_path: models/resnet50.pt   # relative to the data root
    batch_size: 64
```

`model_path` is required and must be relative — it resolves against the data root, and a path not found directly is
also looked up under the conventional `models/` subfolder. Absolute paths in a config are rejected so that a config
stays portable between a laptop and a container.

## Choose the layer to read

By default the extractor takes the model's final output. That is rarely what you want for an embedding: the last layer
of a classifier is a vector of class scores, which throws away exactly the representational detail that drift, OOD, and
prioritization depend on. Use `layer_name` to install a forward hook on an intermediate layer instead:

```yaml
extractors:
  - name: torch_ext
    model: torch
    model_path: models/resnet50.pt
    layer_name: layer4       # penultimate feature map, not the logits
    use_output: true         # capture the layer's output (false = its input)
```

`layer_name` must match the attribute path of a module in the loaded model. If you are unsure what is available:

```python
import torch

model = torch.load("models/resnet50.pt", weights_only=False)
for name, _ in model.named_modules():
    print(name)
```

`use_output: false` captures what was fed *into* the named layer instead of what came out. That is the easier way to
grab the input to a classifier head when the head itself is what you want to bypass.

## Select a device

```yaml
    device: cuda:0     # or cpu; omit to let DataEval Flow choose
```

Setting `device: cuda:0` requires a CUDA image variant (`cu118` / `cu128`) run with `--gpus all`. On the `cpu` image
the model runs on CPU regardless of this field. A GPU is never required — it only speeds up extraction, and matters
most on embedding-heavy workflows over large datasets.

## Add preprocessing

A trained model expects the input pipeline it was trained with. Define a named preprocessor and reference it:

```yaml
preprocessors:
  - name: resnet_preprocess
    steps:
      - step: Resize
        params: { size: [256, 256], antialias: true }
      - step: CenterCrop
        params: { size: [224, 224] }
      - step: ToDtype
        params: { dtype: float32, scale: true }
      - step: Normalize
        params:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]

extractors:
  - name: torch_ext
    model: torch
    model_path: models/resnet50.pt
    layer_name: layer4
    preprocessor: resnet_preprocess
```

A `preprocessor` reference must name a preprocessor defined in the same configuration. Getting normalization wrong is
the most common cause of embeddings that look plausible but produce nonsense distances — see
[Preprocessing and Feature Extraction](../concepts/PreprocessingAndExtraction.md).

## PyTorch, ONNX, or BoVW?

| Extractor | Needs | Use when |
| --- | --- | --- |
| `torch` | a `.pt` model | you want the representation of the exact model under test, or need an intermediate layer |
| `onnx` | a `.onnx` model | you want a portable, framework-independent artifact, or a published pretrained model |
| `bovw` | nothing | no trained model is available and you need a model-free baseline |
| `flatten` | nothing | the images are small and raw pixels are an acceptable representation |

Prefer `onnx` when the model is a fixed, shared artifact — it pins the graph and the input contract. Prefer `torch`
when the model is yours, still moving, or when you need a layer that the exported ONNX graph does not expose. See
{doc}`../notebooks/onnx_embeddings` for the ONNX path.

## Related material

- [Preprocessing and Feature Extraction](../concepts/PreprocessingAndExtraction.md) — why the representation a
  workflow measures in matters
- [DataEval Embeddings explanation](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html) — the
  authoritative treatment of embeddings
- {doc}`API Reference <../reference/autoapi/dataeval_flow/index>` — every field on `TorchExtractorConfig`
