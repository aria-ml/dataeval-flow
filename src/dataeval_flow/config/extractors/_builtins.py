"""The built-in extractors, and the config each one reads."""

__all__ = [
    "BoVWExtractorConfig",
    "FlattenExtractorConfig",
    "OnnxExtractorConfig",
    "TorchExtractorConfig",
    "UncertaintyExtractorConfig",
]

from collections.abc import Callable
from typing import Any, ClassVar, Literal

from dataeval.extractors import BoVWExtractor, FlattenExtractor, OnnxExtractor, TorchExtractor
from dataeval.protocols import FeatureExtractor
from pydantic import Field, field_validator, model_validator

from dataeval_flow.config._paths import validate_config_path
from dataeval_flow.config.extractors._base import Extractor, ExtractorConfig

# --- Configs ---


class OnnxExtractorConfig(ExtractorConfig):
    """Extractor config for ONNX models.

    YAML example::

        extractors:
          - name: resnet_extractor
            model: onnx
            model_path: "./resnet50.onnx"
            output_name: "flatten0"
            preprocessor: resnet_preprocess
            batch_size: 64
            image_height: 224
            image_width: 224
    """

    model: str = Field(default="onnx", description="The extractor this entry configures: `onnx`.")
    model_path: str = Field(description="Path to ONNX model file (relative to data root).")
    output_name: str | None = Field(default=None, description="Output layer name.")
    flatten: bool = Field(default=True, description="Flatten output to (N, D) shape.")
    # User-imposed model input configuration [IR-3.1-S-4]. When both height and
    # width are set, input images are resized to (height, width), preferred over
    # the model's native input size. Batch size is the inherited `batch_size`.
    image_height: int | None = Field(
        default=None, gt=0, description="Override model input image height; resizes inputs (IR-3.1-S-4)."
    )
    image_width: int | None = Field(
        default=None, gt=0, description="Override model input image width; resizes inputs (IR-3.1-S-4)."
    )

    @field_validator("model_path")
    @classmethod
    def _model_path_must_be_relative(cls, v: str) -> str:
        return validate_config_path(v)

    @model_validator(mode="after")
    def _image_size_requires_both_dims(self) -> "OnnxExtractorConfig":
        if (self.image_height is None) != (self.image_width is None):
            raise ValueError("image_height and image_width must be set together to resize model inputs.")
        return self


class BoVWExtractorConfig(ExtractorConfig):
    """Extractor config for Bag-of-Visual-Words.

    YAML example::

        extractors:
          - name: bovw_extractor
            model: bovw
            vocab_size: 1024
            batch_size: 32
    """

    model: str = Field(default="bovw", description="The extractor this entry configures: `bovw`.")
    vocab_size: int = Field(default=2048, ge=256, le=4096, description="Visual word count.")


class FlattenExtractorConfig(ExtractorConfig):
    """Extractor config for simple flattening (no model).

    YAML example::

        extractors:
          - name: flat_extractor
            model: flatten
    """

    model: str = Field(default="flatten", description="The extractor this entry configures: `flatten`.")


class TorchExtractorConfig(ExtractorConfig):
    """Extractor config for PyTorch models.

    YAML example::

        extractors:
          - name: torch_extractor
            model: torch
            model_path: "./resnet.pt"
            layer_name: layer4
            device: cpu
    """

    model: str = Field(default="torch", description="The extractor this entry configures: `torch`.")
    model_path: str = Field(description="Path to PyTorch model file (relative to data root).")
    layer_name: str | None = Field(default=None, description="Layer for forward hook extraction.")
    use_output: bool = Field(default=True, description="Capture layer output (True) or input (False).")
    device: str | None = Field(default=None, description="Device (e.g., 'cpu', 'cuda:0').")

    @field_validator("model_path")
    @classmethod
    def _model_path_must_be_relative(cls, v: str) -> str:
        return validate_config_path(v)


class UncertaintyExtractorConfig(ExtractorConfig):
    """Extractor config for uncertainty estimation models.

    YAML example::

        extractors:
          - name: unc_extractor
            model: uncertainty
            model_path: "./classifier.pt"
            preds_type: logits
    """

    model: str = Field(default="uncertainty", description="The extractor this entry configures: `uncertainty`.")
    model_path: str = Field(description="Path to model file (relative to data root).")
    preds_type: Literal["probs", "logits"] | None = Field(default=None, description="Model output format.")
    device: str | None = Field(default=None, description="Device (e.g., 'cpu', 'cuda:0').")

    @field_validator("model_path")
    @classmethod
    def _model_path_must_be_relative(cls, v: str) -> str:
        return validate_config_path(v)


# --- Extractors ---


def _make_resize_transform(height: int, width: int) -> Callable[[Any], Any]:
    """Build a CHW-image resize transform to ``(height, width)`` for IR-3.1-S-4.

    The returned callable bilinearly resizes a single CHW image so that a
    user-imposed model input size overrides the model's native input size.
    """

    def _resize(image: Any) -> Any:
        import numpy as np

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "Resizing ONNX model inputs (image_height/image_width) requires torch; "
                "install dataeval-flow[cpu] or a cuda extra."
            ) from exc

        tensor = torch.as_tensor(np.asarray(image)).float()
        if tensor.ndim != 3:
            raise ValueError(f"ONNX input resize expects CHW images; got shape {tuple(tensor.shape)}")
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
        return resized.squeeze(0).numpy()

    return _resize


class _OnnxExtractor(Extractor[OnnxExtractorConfig]):
    """Embeds with an ONNX model: DataEval's ``OnnxExtractor``."""

    name: ClassVar[str] = "onnx"
    description: ClassVar[str] = "Embeddings from an ONNX model's output layer."

    def build(self, config: OnnxExtractorConfig, transforms: Callable[[Any], Any] | None) -> FeatureExtractor:
        """Build DataEval's ``OnnxExtractor``, resizing inputs last when the config sets a size."""
        # IR-3.1-S-4: honor user-imposed model input size. The pinned dataeval
        # OnnxExtractor takes its input size from the model, so we resize via the
        # transform pipeline instead: append a resize (applied last, after any
        # user preprocessing) when both height and width are configured. The
        # config validator guarantees they are set together.
        onnx_transforms: Any = transforms
        if config.image_height is not None and config.image_width is not None:
            # Append the resize last, keeping any user preprocessing ahead of it.
            resize = _make_resize_transform(config.image_height, config.image_width)
            onnx_transforms = [t for t in (transforms,) if t is not None] + [resize]
        return OnnxExtractor(
            config.model_path,
            transforms=onnx_transforms,
            output_name=config.output_name,
            flatten=config.flatten,
        )


class _BoVWExtractor(Extractor[BoVWExtractorConfig]):
    """Embeds as bag-of-visual-words histograms: DataEval's ``BoVWExtractor``."""

    name: ClassVar[str] = "bovw"
    description: ClassVar[str] = "Bag-of-visual-words histograms over a vocabulary fitted to the data."
    stateful: ClassVar[bool] = True

    def build(
        self,
        config: BoVWExtractorConfig,
        transforms: Callable[[Any], Any] | None,  # noqa: ARG002
    ) -> FeatureExtractor:
        """Build DataEval's ``BoVWExtractor``. It takes no preprocessing, so `transforms` is not applied."""
        return BoVWExtractor(vocab_size=config.vocab_size)


class _FlattenExtractor(Extractor[FlattenExtractorConfig]):
    """Embeds each image as its flattened pixels: DataEval's ``FlattenExtractor``."""

    name: ClassVar[str] = "flatten"
    description: ClassVar[str] = "Each image's pixels, flattened. No model."

    def build(
        self,
        config: FlattenExtractorConfig,  # noqa: ARG002
        transforms: Callable[[Any], Any] | None,  # noqa: ARG002
    ) -> FeatureExtractor:
        """Build DataEval's ``FlattenExtractor``. It takes no preprocessing, so `transforms` is not applied."""
        return FlattenExtractor()


class _TorchExtractor(Extractor[TorchExtractorConfig]):
    """Embeds with a PyTorch model loaded from disk: DataEval's ``TorchExtractor``."""

    name: ClassVar[str] = "torch"
    description: ClassVar[str] = "Embeddings from a PyTorch model's layer."

    def build(self, config: TorchExtractorConfig, transforms: Callable[[Any], Any] | None) -> FeatureExtractor:
        """Build DataEval's ``TorchExtractor``, loading the model from disk.

        The `transforms` from ``build_preprocessing`` is a numpy→numpy wrapper.
        ``TorchExtractor`` handles tensor conversion internally and expects raw
        torchvision transforms, so the wrapped ``v2.Compose`` is unwrapped when possible.
        """
        import torch
        from torchvision.transforms import v2

        # Unwrap the numpy wrapper produced by build_preprocessing to get the
        # raw v2.Compose that TorchExtractor expects (it handles tensor
        # conversion internally via torch.as_tensor).
        torch_transforms: Any = None
        if transforms is not None:
            inner = getattr(transforms, "__wrapped__", None)
            torch_transforms = inner if isinstance(inner, v2.Compose) else transforms

        model = torch.load(config.model_path, map_location="cpu", weights_only=False)
        return TorchExtractor(
            model,
            transforms=torch_transforms,
            device=config.device,
            layer_name=config.layer_name,
            use_output=config.use_output,
        )


class _UncertaintyExtractor(Extractor[UncertaintyExtractorConfig]):
    """Reserved for model-uncertainty embeddings; building one is not yet supported."""

    name: ClassVar[str] = "uncertainty"
    description: ClassVar[str] = "Model prediction uncertainty (not yet implemented)."

    def build(
        self,
        config: UncertaintyExtractorConfig,
        transforms: Callable[[Any], Any] | None,  # noqa: ARG002
    ) -> FeatureExtractor:
        """Refuse: no DataEval extractor backs this model type yet.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(
            f"Extractor type '{config.model}' is not yet implemented. Currently supported: onnx, bovw, flatten, torch."
        )
