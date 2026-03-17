"""Extractor configuration schemas — one class per model type."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ExtractorConfigBase(BaseModel):
    """Common fields shared by all extractor model types."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str = Field(description="Identifier for the extractor")
    preprocessor: str | None = Field(default=None, description="Reference to a preprocessor name (optional)")
    batch_size: int | None = Field(default=None, description="Batch size for embedding extraction")


class OnnxExtractorConfig(_ExtractorConfigBase):
    """Extractor config for ONNX models.

    YAML example::

        extractors:
          - name: resnet_extractor
            model: onnx
            model_path: "./resnet50.onnx"
            output_name: "flatten0"
            preprocessor: resnet_preprocess
            batch_size: 64
    """

    model: Literal["onnx"] = "onnx"
    model_path: str = Field(description="Path to ONNX model file.")
    output_name: str | None = Field(default=None, description="Output layer name.")
    flatten: bool = Field(default=True, description="Flatten output to (N, D) shape.")


class BoVWExtractorConfig(_ExtractorConfigBase):
    """Extractor config for Bag-of-Visual-Words.

    YAML example::

        extractors:
          - name: bovw_extractor
            model: bovw
            vocab_size: 1024
            batch_size: 32
    """

    model: Literal["bovw"] = "bovw"
    vocab_size: int = Field(default=2048, ge=256, le=4096, description="Visual word count.")


class FlattenExtractorConfig(_ExtractorConfigBase):
    """Extractor config for simple flattening (no model).

    YAML example::

        extractors:
          - name: flat_extractor
            model: flatten
    """

    model: Literal["flatten"] = "flatten"


class TorchExtractorConfig(_ExtractorConfigBase):
    """Extractor config for PyTorch models.

    YAML example::

        extractors:
          - name: torch_extractor
            model: torch
            model_path: "./resnet.pt"
            layer_name: layer4
            device: cpu
    """

    model: Literal["torch"] = "torch"
    model_path: str = Field(description="Path to PyTorch model file.")
    layer_name: str | None = Field(default=None, description="Layer for forward hook extraction.")
    use_output: bool = Field(default=True, description="Capture layer output (True) or input (False).")
    device: str | None = Field(default=None, description="Device (e.g., 'cpu', 'cuda:0').")


class UncertaintyExtractorConfig(_ExtractorConfigBase):
    """Extractor config for uncertainty estimation models.

    YAML example::

        extractors:
          - name: unc_extractor
            model: uncertainty
            model_path: "./classifier.pt"
            preds_type: logits
    """

    model: Literal["uncertainty"] = "uncertainty"
    model_path: str = Field(description="Path to model file.")
    preds_type: Literal["probs", "logits"] | None = Field(default=None, description="Model output format.")
    device: str | None = Field(default=None, description="Device (e.g., 'cpu', 'cuda:0').")
