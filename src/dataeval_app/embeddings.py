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
        bovw = BoVWExtractor(vocab_size=extractor_config.vocab_size)
        bovw.fit(dataset)  # type: ignore[arg-type]  # MaiteDataset conforms at runtime
        extractor = bovw
    elif isinstance(extractor_config, FlattenExtractorConfig):
        extractor = FlattenExtractor()
    else:
        raise ValueError(
            f"Extractor type '{extractor_config.type}' is not yet implemented. "
            f"Currently supported: onnx, bovw, flatten."
        )

    # MaiteDataset conforms to DataEval's dataset protocol at runtime (duck typing);
    # pyright can't verify cross-library structural conformance.
    return Embeddings(dataset, extractor=extractor, batch_size=batch_size)  # type: ignore[arg-type]
