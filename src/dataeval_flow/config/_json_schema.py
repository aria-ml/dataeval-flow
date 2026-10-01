"""The pipeline's JSON schema, built from the registry so it includes installed plugins."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, create_model

__all__ = ["registry_twin"]

_CUSTOM = "<steps>"  # the tag of a custom workflow; `<` keeps it apart from every type id


def _union(configs: Sequence[type[BaseModel]], key: str = "type") -> Any:
    """The configs as one union, told apart by the type id each states as `key`'s default."""

    def type_of(value: Any) -> str | None:
        return value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)

    members = tuple(Annotated[config, Tag(config.model_fields[key].default)] for config in configs)
    return Annotated[Union[members], Discriminator(type_of)]  # noqa: UP007 - built from a runtime tuple


def _workflow_union(configs: Sequence[type[BaseModel]], custom: type[BaseModel]) -> Any:
    """The workflow types, plus custom workflows (entries with `steps:` and no `type:`)."""

    def tag_of(value: Any) -> Any:
        if isinstance(value, Mapping):
            return _CUSTOM if "steps" in value and "type" not in value else value.get("type")
        return getattr(value, "type", _CUSTOM)

    members = (
        *(Annotated[config, Tag(config.model_fields["type"].default)] for config in configs),
        Annotated[custom, Tag(_CUSTOM)],
    )
    return Annotated[Union[members], Discriminator(tag_of)]  # noqa: UP007


def _step_union(plugins: bool) -> Any:
    """One schema branch per kind of step, and one per registered transform, combine and check with its settings."""
    from dataeval_flow.steps._by import ByConfig
    from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS
    from dataeval_flow.steps._workflow import StepEntry

    common = {
        "name": (str, StepEntry.model_fields["name"]),
        "extractor": (str | None, StepEntry.model_fields["extractor"]),
        "optional": (bool, StepEntry.model_fields["optional"]),
    }
    forbid = ConfigDict(extra="forbid")

    def required(key: str) -> Any:
        """`StepEntry`'s field `key`, without its default: a step of this kind must write it."""
        return Field(description=StepEntry.model_fields[key].description)

    evaluator = create_model(
        "EvaluatorStep",
        __config__=forbid,
        evaluator=(str, required("evaluator")),
        input=(str | list[str], required("input")),
        by=(ByConfig | None, StepEntry.model_fields["by"]),
        **common,
    )
    workflow = create_model(
        "WorkflowStep",
        __config__=forbid,
        workflow=(str, required("workflow")),
        input=(str | list[str], required("input")),
        **common,
    )
    # An inline step embeds nothing, so its step takes no `extractor:`; naming one fails the load.
    plain = {key: field for key, field in common.items() if key != "extractor"}
    check_by = {"by": (Literal["class", "predicted"] | None, StepEntry.model_fields["by"])}
    inline = [
        create_model(
            f"{kind.title()}Step_{cls.name}",
            __base__=cls.config_type,
            **{kind: (Literal[cls.name], Field(description=f"`{cls.name}`: {cls.description}"))},  # type: ignore[valid-type]
            **plain,
            **(check_by if kind == "check" else {}),
        )
        for kind, registry in (("transform", TRANSFORMS), ("combine", COMBINES), ("check", CHECKS))
        for cls in registry.list(plugins=plugins)
    ]
    return Union[(evaluator, workflow, *inline)]


def registry_twin(*, plugins: bool = True) -> type[BaseModel]:
    """``PipelineConfig`` with its pluggable pools typed as unions of the registered config classes.

    Without `plugins`, the unions hold the built-ins alone: the checked-in schema must not depend on what happens
    to be installed where it was generated.
    """
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config.extractors._registry import EXTRACTORS
    from dataeval_flow.evaluators._registry import EVALUATORS
    from dataeval_flow.steps._workflow import CustomWorkflowConfig
    from dataeval_flow.workflows._registry import WORKFLOWS

    custom = create_model(
        "CustomWorkflowConfig",
        __base__=CustomWorkflowConfig,
        __doc__=CustomWorkflowConfig.__doc__,
        steps=(
            list[_step_union(plugins)],
            Field(min_length=1, description=CustomWorkflowConfig.model_fields["steps"].description),
        ),
    )
    workflows = _workflow_union([cls.config_type for cls in WORKFLOWS.list(plugins=plugins)], custom)
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
