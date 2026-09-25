"""The pipeline's JSON schema, built from the registry so it includes installed plugins."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Union

from pydantic import BaseModel, Discriminator, Field, Tag, create_model

__all__ = ["registry_twin"]


def _union(configs: Sequence[type[BaseModel]], key: str = "type") -> Any:
    """The configs as one union, told apart by the type id each states as `key`'s default."""

    def type_of(value: Any) -> str | None:
        return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)

    members = tuple(Annotated[config, Tag(config.model_fields[key].default)] for config in configs)
    return Annotated[Union[members], Discriminator(type_of)]  # noqa: UP007 - built from a runtime tuple


def registry_twin(*, plugins: bool = True) -> type[BaseModel]:
    """``PipelineConfig`` with its pluggable pools typed as unions of the registered config classes.

    Without `plugins`, the unions hold the built-ins alone: the checked-in schema must not depend on what happens
    to be installed where it was generated.
    """
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config.extractors._registry import EXTRACTORS
    from dataeval_flow.evaluators._registry import EVALUATORS
    from dataeval_flow.workflows._registry import WORKFLOWS

    workflows = _union([cls.config_type for cls in WORKFLOWS.list(plugins=plugins)])
    evaluators = _union([cls.config_type for cls in EVALUATORS.list(plugins=plugins)])
    extractors = _union([cls.config_type for cls in EXTRACTORS.list(plugins=plugins)], key="model")
    fields = PipelineConfig.model_fields
    return create_model(
        "PipelineConfig",
        __base__=PipelineConfig,
        __doc__=PipelineConfig.__doc__,
        extractors=(Sequence[extractors] | None, Field(default=None, description=fields["extractors"].description)),
        workflows=(Sequence[workflows] | None, Field(default=None, description=fields["workflows"].description)),
        evaluators=(Sequence[evaluators] | None, Field(default=None, description=fields["evaluators"].description)),
    )
