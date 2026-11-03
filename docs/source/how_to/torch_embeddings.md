# Use a PyTorch model for embeddings

To make the evaluators measure in the representation space of your own trained PyTorch model, often the model
under test, configure the `torch` extractor, choose the layer to read and add the preprocessing the model expects.
[Preprocessing and Feature Extraction](../concepts/PreprocessingAndExtraction.md) explains why the representation a
workflow measures in matters, and DataEval's
[Embeddings explanation](https://dataeval.readthedocs.io/en/latest/concepts/Embeddings.html) is the authoritative
treatment of embeddings. You need a `.pt` model file under your data root. The
{doc}`API Reference <../reference/autoapi/dataeval_flow/index>` lists every field on `TorchExtractorConfig`.

## Set the model path

```yaml
extractors:
  - name: torch_ext
    model: torch
    model_path: models/resnet50.pt   # relative to the data root
    batch_size: 64
```

`model_path` is required and must be relative. It resolves against the data root, and a path not found directly is also
looked up under the conventional `models/` subfolder. A config rejects absolute paths so that it stays portable between
a laptop and a container.

## Choose the layer to read

By default the extractor takes the model's final output. The last layer of a classifier is a vector of class scores,
which throws away the representational detail that drift, OOD, and prioritization depend on. Use `layer_name` to
install a forward hook on an intermediate layer:

```yaml
extractors:
  - name: torch_ext
    model: torch
    model_path: models/resnet50.pt
    layer_name: layer4       # penultimate feature map, not the logits
    use_output: true         # capture the layer's output (false = its input)
```

`layer_name` must match the attribute path of a module in the loaded model. If you are unsure what is available, list
the modules:

```python
import torch

model = torch.load("models/resnet50.pt", weights_only=False)
for name, _ in model.named_modules():
    print(name)
```

`use_output: false` captures what was fed into the named layer. That is the easier way to grab the input to a
classifier head when you want to bypass the head itself.

## Select a device

The device is chosen once for the whole run, not per extractor, and by the machine rather than the config. DataEval
Flow computes on CUDA when PyTorch sees a GPU and on the CPU otherwise. Each result's `metadata.device` records the
device its task ran on.

To choose, hide or pick GPUs, set `CUDA_VISIBLE_DEVICES`: empty runs on the CPU, and `1` on the second GPU. From
Python, `set_device` chooses for every task run after it:

```python
from dataeval_flow import set_device

set_device("cuda:1")  # or "cpu"; None returns to the default
```

A GPU needs a CUDA image variant (`cu126` or `cu130`) run with `--gpus all`, and the `cpu` image computes on the CPU.
See [Run workflows in containers](containerized_workflows.md). A GPU is not required. It only speeds up extraction, and
it matters most on embedding-heavy workflows over large datasets.

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
the most common cause of embeddings that look plausible but produce nonsense distances.

## Choose an extractor

| Extractor | Needs | Use when |
| --- | --- | --- |
| `torch` | a `.pt` model | you want the representation of the exact model under test, or need an intermediate layer |
| `onnx` | a `.onnx` model | you want a portable, framework-independent artifact, or a published pretrained model |
| `bovw` | nothing | no trained model is available and you need a model-free baseline |
| `flatten` | nothing | the images are small and raw pixels are an acceptable representation |

Prefer `onnx` when the model is a fixed, shared artifact, because it pins the graph and the input contract. Prefer
`torch` when the model is yours, still moving, or when you need a layer that the exported ONNX graph does not expose.
[Use an ONNX model for embeddings](../notebooks/onnx_embeddings.py) shows the ONNX path.

## Next steps

- [Prioritize unlabeled data for labeling](../notebooks/data_prioritization.py) — Train a small CNN, save it, and read
  its `embed` layer through a `torch` extractor.
- [Preprocessing and Feature Extraction](../concepts/PreprocessingAndExtraction.md) — Learn why the representation a
  workflow measures in matters.
- [Use an ONNX model for embeddings](../notebooks/onnx_embeddings.py) — Configure a pretrained ONNX model in place of
  a PyTorch model.
