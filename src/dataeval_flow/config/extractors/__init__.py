"""Extractors: the framework for writing one, and the built-in extractors.

An extractor turns images into embeddings. Each ``extractors:`` entry names one by its ``model:`` and is validated
with that extractor's config; the extractor then builds the DataEval ``FeatureExtractor`` the entry describes.
"""

from dataeval_flow.config.extractors._base import Extractor, ExtractorConfig
from dataeval_flow.config.extractors._builtins import (
    BoVWExtractorConfig,
    FlattenExtractorConfig,
    OnnxExtractorConfig,
    TorchExtractorConfig,
    UncertaintyExtractorConfig,
)
from dataeval_flow.config.extractors._registry import get_extractor, list_extractors

__all__ = [
    "BoVWExtractorConfig",
    "Extractor",
    "ExtractorConfig",
    "FlattenExtractorConfig",
    "OnnxExtractorConfig",
    "TorchExtractorConfig",
    "UncertaintyExtractorConfig",
    "get_extractor",
    "list_extractors",
]
