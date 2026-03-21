# Use an ONNX model for embeddings

This guide shows how to configure an ONNX extractor with preprocessing
transforms.  ONNX extractors use pretrained models (e.g. ResNet50) for
higher-fidelity embeddings than lightweight methods like BoVW.

## YAML

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
  - name: resnet50_ext
    model: onnx
    model_path: "./models/resnet50-v2-7.onnx"
    output_name: resnetv24_dense0_fwd
    preprocessor: resnet_preprocess
    batch_size: 32
```

## Python

```python
from dataeval_flow.config import OnnxExtractorConfig, PreprocessorConfig
from dataeval_flow.preprocessing import PreprocessingStep

onnx_extractor = OnnxExtractorConfig(
    name="resnet50_ext",
    model="onnx",
    model_path="./models/resnet50-v2-7.onnx",
    output_name="resnetv24_dense0_fwd",
    batch_size=32,
    preprocessor=PreprocessorConfig(
        name="resnet_preprocess",
        steps=[
            PreprocessingStep(step="Resize", params={"size": [256, 256], "antialias": True}),
            PreprocessingStep(step="CenterCrop", params={"size": [224, 224]}),
            PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True}),
            PreprocessingStep(
                step="Normalize",
                params={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ),
        ],
    ),
)
```

## Key fields

| Field | Description |
| --- | --- |
| `model_path` | Path to the `.onnx` file |
| `output_name` | Name of the output node to extract embeddings from (use [Netron](https://netron.app) to inspect) |
| `preprocessor` | Named preprocessor with torchvision v2 transform steps |
| `batch_size` | Images per inference batch |

## When to use ONNX vs BoVW

| Feature | ONNX | BoVW |
| --- | --- | --- |
| Model file | Required (~100 MB) | None |
| Preprocessing | Required | None |
| Embedding quality | High (pretrained features) | Good (learned visual words) |
| Setup complexity | Higher | Minimal |

Use ONNX when cluster-based detection benefits from richer feature
representations.  Use BoVW (see the [data cleaning tutorial](../notebooks/data_cleaning))
for a simpler setup with no external model dependencies.
