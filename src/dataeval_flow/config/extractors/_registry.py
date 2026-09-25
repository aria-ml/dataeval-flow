"""The extractor registry: the built-in table and the `dataeval_flow.extractors` entry points."""

from typing import Any

from dataeval_flow._kind import type_id_problem
from dataeval_flow._registry import Registry
from dataeval_flow.config.extractors._base import Extractor

__all__ = ["EXTRACTORS", "get_extractor", "list_extractors"]

_BUILTINS = {
    "bovw": "dataeval_flow.config.extractors._builtins:_BoVWExtractor",
    "flatten": "dataeval_flow.config.extractors._builtins:_FlattenExtractor",
    "onnx": "dataeval_flow.config.extractors._builtins:_OnnxExtractor",
    "torch": "dataeval_flow.config.extractors._builtins:_TorchExtractor",
    "uncertainty": "dataeval_flow.config.extractors._builtins:_UncertaintyExtractor",
}


def _config_model_matches(name: str, cls: type[Extractor[Any]]) -> str | None:
    """A registered extractor's config must configure the `model` it is registered under."""
    return type_id_problem(name, cls.config_type, "model")


EXTRACTORS: Registry[Extractor[Any]] = Registry(
    kind="extractor",
    group="dataeval_flow.extractors",
    base=lambda: Extractor,
    builtins=_BUILTINS,
    check=_config_model_matches,
)


def get_extractor(name: str) -> type[Extractor[Any]]:
    """The extractor registered as `name`, built-in or plugin.

    Parameters
    ----------
    name : str
        The extractor's name: its YAML ``model:`` value, e.g. ``"onnx"``.

    Returns
    -------
    type[Extractor]
        The extractor class, whose ``name``, ``description`` and ``config_type`` describe it. Flow builds its
        instances.

    Raises
    ------
    ValueError
        When nothing is registered under `name`, or the plugin registered under it failed to load.
    """
    return EXTRACTORS.get(name)


def list_extractors() -> list[type[Extractor[Any]]]:
    """Every installed extractor, built-in or plugin, sorted by name.

    A plugin that failed to load is left out; :func:`get_extractor` raises its error.

    Returns
    -------
    list[type[Extractor]]
        The extractor classes.
    """
    return EXTRACTORS.list()
