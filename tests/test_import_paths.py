"""Internal modules are private, and each config type is imported from one place."""

import importlib

import pytest

import dataeval_flow
import dataeval_flow.config as config
import dataeval_flow.config.extractors as extractors

MADE_PRIVATE = [
    "binning",
    "cache",
    "dataset",
    "embeddings",
    "export",
    "label_space",
    "metadata",
    "policy",
    "preprocessing",
    "result",
    "runner",
    "sources",
    "stats",
    "triage",
    "view",
]

CONFIG_TYPES = [
    "AggregatorConfig",
    "DatasetConfig",
    "ExportConfig",
    "LoggingConfig",
    "MetadataPolicyConfig",
    "OntologyConceptConfig",
    "OntologyConfig",
    "ParseDateTimeCorrectionConfig",
    "ParseValueCorrectionConfig",
    "PreprocessingStep",
    "ReductionOptionsConfig",
    "RemapCorrectionConfig",
    "RemapRuleConfig",
    "RescaleCorrectionConfig",
    "StatsMeasureConfig",
    "StatsPolicyConfig",
]


@pytest.mark.parametrize("name", MADE_PRIVATE)
def test_internal_modules_are_private(name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"dataeval_flow.{name}")
    assert importlib.import_module(f"dataeval_flow._{name}")


@pytest.mark.parametrize("kind", ["extractors", "transforms"])
def test_input_side_kinds_live_under_config(kind: str) -> None:
    """The top level holds the doers; a kind that only feeds them lives under ``config``."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"dataeval_flow.{kind}")
    assert importlib.import_module(f"dataeval_flow.config.{kind}")


def test_the_preprocessors_module_is_gone() -> None:
    """Its `ToRGB` is a registered transform now, imported from `dataeval_flow.config.transforms`."""
    for name in ("dataeval_flow.preprocessors", "dataeval_flow._preprocessors"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_config_schemas_is_private() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dataeval_flow.config.schemas")


@pytest.mark.parametrize("name", CONFIG_TYPES)
def test_config_types_come_from_config(name: str) -> None:
    assert name in config.__all__
    assert getattr(config, name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "BoVWExtractorConfig",
        "FlattenExtractorConfig",
        "OnnxExtractorConfig",
        "TorchExtractorConfig",
        "UncertaintyExtractorConfig",
    ],
)
def test_extractor_configs_come_from_extractors(name: str) -> None:
    assert name in extractors.__all__
    assert not hasattr(config, name)


def test_deprecated_selection_aliases_are_gone() -> None:
    for name in ("SelectionConfig", "SelectionStep"):
        assert not hasattr(dataeval_flow, name)
        assert not hasattr(config, name)


def test_the_pipeline_config_and_its_loader_come_from_the_top_level() -> None:
    """One import path per name: `dataeval_flow` exports them, and `dataeval_flow.config` does not."""
    for name in ("PipelineConfig", "load_config"):
        assert name in dataeval_flow.__all__
        assert name not in config.__all__
        assert not hasattr(config, name)


def test_the_folder_loader_and_the_schema_export_are_gone() -> None:
    """`load_config` reads a folder; `PipelineConfig.model_json_schema()` is the schema."""
    for name in ("load_config_folder", "export_params_schema"):
        assert not hasattr(dataeval_flow, name)
        assert not hasattr(config, name)
