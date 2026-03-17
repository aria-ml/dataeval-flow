"""Embeddings convenience builder wrapping DataEval."""

__all__ = [
    "build_embeddings",
    "build_extractor",
]

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from dataeval import Embeddings
from dataeval.extractors import BoVWExtractor, FlattenExtractor, OnnxExtractor
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
    else:
        raise ValueError(
            f"Extractor type '{extractor_config.model}' is not yet implemented. "
            f"Currently supported: onnx, bovw, flatten."
        )
    return extractor
