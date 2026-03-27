"""Embeddings convenience builder wrapping DataEval."""

__all__ = [
    "build_embeddings",
    "build_extractor",
]

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dataeval import Embeddings
from dataeval.extractors import BoVWExtractor, FlattenExtractor, OnnxExtractor, TorchExtractor
from dataeval.protocols import AnnotatedDataset

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dataeval_flow.config.schemas import ExtractorConfig


def build_embeddings(
    dataset: AnnotatedDataset[Any],
    extractor_config: "ExtractorConfig",
    transforms: Callable | None = None,
    batch_size: int | None = None,
) -> Embeddings:
    """Build Embeddings from dataset and extractor config.

    Creates the appropriate extractor based on the config's model type and
    wraps it in a DataEval Embeddings instance.

    Parameters
    ----------
    dataset : MaiteDataset
        Input dataset.
    extractor_config : ExtractorConfig
        Extractor configuration with model type and params.
    transforms : Callable | None
        Preprocessing transforms to apply before encoding.
        Only used by extractor types that accept it (onnx, torch, uncertainty).

    Returns
    -------
    Embeddings
        DataEval Embeddings instance (implements FeatureExtractor).
    """

    extractor = build_extractor(extractor_config, transforms)
    return Embeddings(dataset, extractor=extractor, batch_size=batch_size)


def build_extractor(extractor_config: "ExtractorConfig", transforms: Callable | None = None) -> Callable:
    """Build a standalone extractor (not wrapped in Embeddings).

    Used for workflows that need to apply the extractor separately from embedding extraction
    (e.g. to extract metadata features for evaluation).

    Parameters
    ----------
    extractor_config : ExtractorConfig
        Extractor configuration with model type and params.
    transforms : Callable | None
        Preprocessing transforms to apply before encoding.
        Only used by extractor types that accept it (onnx, torch, uncertainty).

    Returns
    -------
    Callable
        A callable extractor function that takes a dataset and returns extracted features.
    """
    from dataeval_flow.config.schemas._extractor import (
        BoVWExtractorConfig,
        FlattenExtractorConfig,
        OnnxExtractorConfig,
        TorchExtractorConfig,
    )

    logger.debug("Building %s extractor", extractor_config.model)

    if isinstance(extractor_config, OnnxExtractorConfig):
        extractor = OnnxExtractor(
            extractor_config.model_path,
            transforms=transforms,
            output_name=extractor_config.output_name,
            flatten=extractor_config.flatten,
        )
    elif isinstance(extractor_config, BoVWExtractorConfig):
        extractor = BoVWExtractor(vocab_size=extractor_config.vocab_size)
    elif isinstance(extractor_config, FlattenExtractorConfig):
        extractor = FlattenExtractor()
    elif isinstance(extractor_config, TorchExtractorConfig):
        extractor = _build_torch_extractor(extractor_config, transforms)
    else:
        raise ValueError(
            f"Extractor type '{extractor_config.model}' is not yet implemented. "
            f"Currently supported: onnx, bovw, flatten, torch."
        )
    return extractor


def _build_torch_extractor(
    config: Any,
    transforms: Callable | None = None,
) -> TorchExtractor:
    """Build a TorchExtractor from config, loading the model from disk.

    The *transforms* from ``build_preprocessing`` is a numpy→numpy wrapper.
    ``TorchExtractor`` handles tensor conversion internally and expects raw
    torchvision transforms, so we unwrap the ``v2.Compose`` when possible.
    """
    import torch
    from torchvision.transforms import v2

    # Unwrap the numpy wrapper produced by build_preprocessing to get the
    # raw v2.Compose that TorchExtractor expects (it handles tensor
    # conversion internally via torch.as_tensor).
    torch_transforms: v2.Compose | Callable | None = None
    if transforms is not None:
        inner = getattr(transforms, "__wrapped__", None)
        if isinstance(inner, v2.Compose):
            torch_transforms = inner
        elif isinstance(transforms, v2.Compose):
            torch_transforms = transforms
        else:
            torch_transforms = transforms

    model = torch.load(config.model_path, map_location="cpu", weights_only=False)
    return TorchExtractor(
        model,
        transforms=torch_transforms,
        device=config.device,
        layer_name=config.layer_name,
        use_output=config.use_output,
    )
