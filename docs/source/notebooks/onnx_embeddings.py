# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dataeval-flow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Use an ONNX model for embeddings
#
# To generate embeddings from a pretrained model such as ResNet-50, configure an ONNX extractor with preprocessing
# transforms. Pretrained models provide higher-fidelity embeddings than lightweight methods like BoVW.
# {doc}`Preprocessing and Feature Extraction <../concepts/PreprocessingAndExtraction>` explains why the representation a
# workflow measures in matters. You need the `onnx` extra and the `.onnx` model file.
#
# :::{important}
# ONNX support is an optional extra and is not installed by default. Without it, your configuration validates, but task
# execution fails on import:
#
# ```bash
# pip install "dataeval-flow[onnx]"          # CPU
# pip install "dataeval-flow[onnx-cu126]"    # CUDA 12.6
# pip install "dataeval-flow[onnx-cu130]"    # CUDA 13.0
# ```
#
# Select the CUDA variant that matches your PyTorch installation.
# :::

# %% [markdown]
# ## Configure the extractor in YAML
#
# ```yaml
# preprocessors:
#   - name: resnet_preprocess
#     steps:
#       - step: Resize
#         params: { size: [256, 256], antialias: true }
#       - step: CenterCrop
#         params: { size: [224, 224] }
#       - step: ToDtype
#         params: { dtype: float32, scale: true }
#       - step: Normalize
#         params:
#           mean: [0.485, 0.456, 0.406]
#           std: [0.229, 0.224, 0.225]
#
# extractors:
#   - name: resnet50_ext
#     model: onnx
#     model_path: "./models/resnet50-v2-7.onnx"
#     output_name: resnetv24_dense0_fwd
#     preprocessor: resnet_preprocess
#     batch_size: 32
# ```

# %% [markdown]
# ## Configure the extractor in Python

# %% tags=["remove_output"]
from dataeval_flow.config import PreprocessingStep, PreprocessorConfig
from dataeval_flow.config.extractors import OnnxExtractorConfig

onnx_extractor = OnnxExtractorConfig(
    name="resnet50_ext",
    model_path="./models/resnet50-v2-7.onnx",
    output_name="resnetv24_dense0_fwd",
    batch_size=32,
    preprocessor="resnet_preprocess",
)

resnet_preprocess = PreprocessorConfig(
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
)

# %% [markdown]
# ## Set the key fields
#
# | Field | Description |
# | --- | --- |
# | `model_path` | Path to the `.onnx` file |
# | `output_name` | Name of the output node to extract embeddings from (use [Netron](https://netron.app) to inspect) |
# | `preprocessor` | Name of a preprocessor defined in the `preprocessors` section |
# | `batch_size` | Images per inference batch |

# %% [markdown]
# ## Choose between ONNX and BoVW
#
# | Feature | ONNX | BoVW |
# | --- | --- | --- |
# | Model file | Required (~100 MB) | None |
# | Preprocessing | Required | None |
# | Embedding quality | High (pretrained features) | Good (learned visual words) |
# | Setup complexity | Higher | Minimal |
#
# Use ONNX when your tasks require rich feature representations from pretrained models. Use BoVW (see
# {doc}`Clean a dataset <data_cleaning>`) when you need a lightweight setup without external model files.

# %% [markdown]
# ## Next steps
#
# - {doc}`data_coverage` — Measure how well a dataset covers the embedding space of an extractor.
# - {doc}`../how_to/torch_embeddings` — Read an intermediate layer of your own PyTorch model instead.
# - {doc}`../concepts/PreprocessingAndExtraction` — Learn why the representation a workflow measures in matters.
