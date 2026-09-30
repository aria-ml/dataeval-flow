"""Named config entries handed over beside a config (``run()``'s and ``save()``'s ``definitions``): the pools they
belong to, and each as a config file writes it."""

__all__ = ["definition_pools", "entry_as_written"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel

if TYPE_CHECKING:
    from dataeval_flow.config._schemas._metadata import MetadataPolicyConfig
    from dataeval_flow.config._schemas._ontology import OntologyConfig
    from dataeval_flow.config._schemas._preprocessor import PreprocessorConfig
    from dataeval_flow.config._schemas._stats import StatsPolicyConfig
    from dataeval_flow.config._schemas._view import ViewConfig
    from dataeval_flow.config.extractors._base import ExtractorConfig
    from dataeval_flow.evaluators._base import EvaluatorConfig
    from dataeval_flow.workflows._base import WorkflowConfig

    # A named entry a config refers to: a policy, an ontology, a preprocessor, an evaluator or workflow entry, a view,
    # or an extractor.
    Definition: TypeAlias = (
        MetadataPolicyConfig
        | StatsPolicyConfig
        | OntologyConfig
        | PreprocessorConfig
        | EvaluatorConfig[Any]
        | WorkflowConfig[Any]
        | ViewConfig
        | ExtractorConfig
    )

# The fields an entry is found and dispatched by: every entry's `name`, and the `type` (evaluators, workflows) or
# `model` (extractors) its pool reads to pick the entry's config class. They are written even at their defaults.
_KEYS = ("name", "type", "model")


def definition_pools(definitions: Sequence[object], *, caller: str) -> dict[str, list[Any]]:
    """Sort `definitions` into the ``PipelineConfig`` pools they belong to, keyed by field name, in the order given.

    Raises
    ------
    TypeError
        When a definition is none of the types a pool holds; the message names `caller`, such as ``run()``.
    """
    from dataeval_flow.config._schemas._metadata import MetadataPolicyConfig
    from dataeval_flow.config._schemas._ontology import OntologyConfig
    from dataeval_flow.config._schemas._preprocessor import PreprocessorConfig
    from dataeval_flow.config._schemas._stats import StatsPolicyConfig
    from dataeval_flow.config._schemas._view import ViewConfig
    from dataeval_flow.config.extractors._base import ExtractorConfig
    from dataeval_flow.evaluators._base import EvaluatorConfig
    from dataeval_flow.workflows._base import WorkflowConfig

    fields: dict[type, str] = {
        MetadataPolicyConfig: "metadata",
        StatsPolicyConfig: "stats",
        OntologyConfig: "ontologies",
        PreprocessorConfig: "preprocessors",
        EvaluatorConfig: "evaluators",
        WorkflowConfig: "workflows",
        ViewConfig: "views",
        ExtractorConfig: "extractors",
    }
    pools: dict[str, list[Any]] = {}
    for definition in definitions:
        field = next((name for cls, name in fields.items() if isinstance(definition, cls)), None)
        if field is None:
            raise TypeError(
                f"{caller} cannot use a {type(definition).__name__} as a definition; it takes MetadataPolicyConfig, "
                "StatsPolicyConfig, OntologyConfig, PreprocessorConfig, EvaluatorConfig, WorkflowConfig, ViewConfig "
                "and ExtractorConfig."
            )
        pools.setdefault(field, []).append(definition)
    return pools


def entry_as_written(entry: BaseModel) -> dict[str, Any]:
    """`entry` as a config file writes it: its name and dispatch key, then each setting that differs from its default.

    A setting at its default is left out, since loading gives it back. A setting given as ``None`` where the default is
    not stays in, since ``None`` means something of its own there, such as a threshold that judges nothing.
    """
    kept = {
        name
        for name, field in type(entry).model_fields.items()
        if name in _KEYS or field.is_required() or getattr(entry, name) != field.get_default(call_default_factory=True)
    }
    return entry.model_dump(mode="json", include=kept)
