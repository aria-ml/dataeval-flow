"""The step catalog: every registered step described as data, so a palette needs to import none of them (spec §8)."""

__all__ = ["PortEntry", "StepCatalog", "StepCatalogEntry", "list_steps"]

import sys
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow.steps._port import DATASET_KINDS, DataType, Port
from dataeval_flow.steps._step import StepKind


class PortEntry(BaseModel):
    """One port of a step, as the catalog describes it."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    port: str = Field(description="The config field that feeds an input, or an output's name.")
    type: DataType = Field(description="What flows along it: `dataset`, `output`, `export` or `findings`.")
    kinds: list[str] | None = Field(
        default=None,
        description=(
            'On a Dataset port, the Dataset kinds it takes or makes, or `["any"]`; `null` on a port that carries '
            "no Dataset."
        ),
    )
    is_list: bool = Field(default=False, alias="list", description="Whether it takes or gives a whole keyed list.")
    count: str | None = Field(default=None, description="How many addresses a Dataset port takes, such as `1+`.")
    derives: list[str] = Field(default_factory=list, description="What Flow derives from an evaluator's Dataset port.")
    classes: list[str] = Field(
        default_factory=list, description="The output classes an output port takes or gives, by import path."
    )


class StepCatalogEntry(BaseModel):
    """One step type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    kind: StepKind = Field(description="`evaluator`, `transform`, `combine`, `check` or `workflow`.")
    type: str = Field(description="The name a step uses under its kind key.")
    title: str = Field(description="Its friendly name, such as `K-Fold`; its `type` where it declares none.")
    description: str = Field(description="One line on what it does.")
    origin: str = Field(description="The distribution that registered it: `dataeval-flow` for a built-in.")
    inputs: list[PortEntry] = Field(description="Its input ports.")
    outputs: list[PortEntry] = Field(description="Its output ports.")
    judges: list[str] = Field(
        default_factory=list, description="On a check, the evaluator and combine types whose Outputs it judges."
    )
    judged_by: list[str] = Field(
        default_factory=list, description="On an evaluator or combine, the check types that judge its Output."
    )
    config_schema: dict[str, Any] = Field(description="The JSON Schema of its settings.")


class StepCatalog(BaseModel):
    """Every step Flow can chain, and what produced this description."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    format: Literal[1] = Field(default=1, description="The document's format version, raised when its shape breaks.")
    flow_version: str = Field(description="The dataeval-flow version that produced it.")
    dataeval_version: str = Field(description="The DataEval version installed beside it.")
    data_types: list[str] = Field(description="What can flow along a port.")
    dataset_kinds: list[str] = Field(description="The Dataset kinds a port may name.")
    steps: list[StepCatalogEntry] = Field(description="Every step, by kind, then name.")


def _class_path(cls: type) -> str:
    """Where `cls` imports from: the shortest package enclosing its module that exports it, else that module."""
    parts = cls.__module__.split(".")
    for end in range(1, len(parts)):
        package = sys.modules.get(".".join(parts[:end]))  # imported already: a module's packages import before it
        if package is not None and getattr(package, cls.__qualname__, None) is cls:
            return f"{package.__name__}.{cls.__qualname__}"
    return f"{cls.__module__}.{cls.__qualname__}"


def _kinds(port: Port) -> list[str] | None:
    """The kinds a Dataset port names, ``["any"]`` when it names none; ``None`` for a port carrying no Dataset."""
    if port.type != DataType.DATASET:
        return None
    return sorted(port.kinds) if port.kinds is not None else ["any"]


def _port(port: Port) -> PortEntry:
    return PortEntry(
        port=port.name,
        type=port.type,
        kinds=_kinds(port),
        list=port.is_list,  # by its alias, the name a type checker gives the constructor's parameter
        count=str(port.count) if port.count is not None else None,
        derives=sorted(str(kind) for kind in port.derives),
        classes=[_class_path(cls) for cls in port.classes],
    )


def _feeds(producer: Any, check: Any) -> bool:
    """Whether an Output `producer` gives is of a class one of `check`'s ports takes."""
    made = [cls for port in producer.output_ports() if port.type == DataType.OUTPUT for cls in port.classes]
    taken = [cls for port in check.input_ports() if port.type == DataType.OUTPUT for cls in port.classes]
    return any(issubclass(out, into) for out in made for into in taken)


def list_steps(*, plugins: bool = True) -> StepCatalog:
    """Every registered step as data: kind, name, description, origin, ports and settings schema.

    Custom workflows are config, not step types, so they do not appear.

    Parameters
    ----------
    plugins : bool
        Whether to include steps installed by other packages. Without them, only dataeval-flow's own.

    Returns
    -------
    StepCatalog
        The catalog, ordered by kind, then name.
    """
    import dataeval

    from dataeval_flow import __version__
    from dataeval_flow.evaluators._registry import EVALUATORS
    from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS
    from dataeval_flow.workflows._registry import WORKFLOWS

    found = [
        (registry, cls)
        for registry in (EVALUATORS, TRANSFORMS, COMBINES, CHECKS, WORKFLOWS)  # in kind order
        for cls in registry.list(plugins=plugins)
    ]
    producers = [cls for _, cls in found if cls.kind in ("evaluator", "combine")]
    checks = [cls for _, cls in found if cls.kind == "check"]
    judges = {check.name: sorted({p.name for p in producers if _feeds(p, check)}) for check in checks}
    judged_by = {(p.kind, p.name): sorted({c.name for c in checks if _feeds(p, c)}) for p in producers}
    entries = [
        StepCatalogEntry(
            kind=cls.kind,
            type=cls.name,
            title=cls.title,
            description=cls.description,
            origin=registry.origin(cls.name),
            inputs=[_port(port) for port in cls.input_ports()],
            outputs=[_port(port) for port in cls.output_ports()],
            config_schema=cls.config_type.model_json_schema(),
            judges=judges.get(cls.name, []) if cls.kind == "check" else [],
            judged_by=judged_by.get((cls.kind, cls.name), []),
        )
        for registry, cls in found
    ]
    return StepCatalog(
        flow_version=__version__,
        dataeval_version=dataeval.__version__,
        data_types=[str(member) for member in DataType],
        dataset_kinds=list(DATASET_KINDS),
        steps=entries,
    )
