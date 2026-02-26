"""Embeddings convenience builder wrapping DataEval."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from dataeval import Embeddings
from dataeval.extractors import BoVWExtractor, FlattenExtractor, OnnxExtractor

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dataeval_app.config.models import ExtractorConfig
    from dataeval_app.dataset import MaiteDataset

__all__ = [
    "build_embeddings",
    "build_extractor",
]


def build_embeddings(
    dataset: "MaiteDataset",
    extractor_config: "ExtractorConfig",
    transforms: Callable | None = None,
    batch_size: int | None = None,
) -> Embeddings:
    """Build Embeddings from dataset and extractor config.

    Creates the appropriate extractor based on the config type and wraps it
    in a DataEval Embeddings instance.

    Parameters
    ----------
    dataset : MaiteDataset
        Input dataset.
    extractor_config : ExtractorConfig
        Extractor configuration (discriminated union).
    transforms : Callable | None
        Preprocessing transforms to apply before encoding.
        Only used by extractor types that accept it (onnx, torch, uncertainty).

    Returns
    -------
    Embeddings
        DataEval Embeddings instance (implements FeatureExtractor).
    """

    # MaiteDataset conforms to DataEval's dataset protocol at runtime (duck typing);
    # pyright can't verify cross-library structural conformance.
    extractor = build_extractor(extractor_config, transforms)
    return Embeddings(dataset, extractor=extractor, batch_size=batch_size)  # type: ignore[arg-type]


def build_extractor(extractor_config: "ExtractorConfig", transforms: Callable | None = None) -> Callable:
    """Build a standalone extractor (not wrapped in Embeddings).

    Used for workflows that need to apply the extractor separately from embedding extraction
    (e.g. to extract metadata features for evaluation).

    Parameters
    ----------
    extractor_config : ExtractorConfig
        Extractor configuration (discriminated union).
    transforms : Callable | None
        Preprocessing transforms to apply before encoding.
        Only used by extractor types that accept it (onnx, torch, uncertainty).

    Returns
    -------
    Callable
        A callable extractor function that takes a dataset and returns extracted features.
    """

    from dataeval_app.config.models import (
        BoVWExtractorConfig,
        FlattenExtractorConfig,
        OnnxExtractorConfig,
    )

    logger.debug("Building %s extractor", extractor_config.type)

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
            f"Extractor type '{extractor_config.type}' is not yet implemented. "
            f"Currently supported: onnx, bovw, flatten."
        )
    return extractor
