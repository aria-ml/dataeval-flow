"""The workflow framework: the bases a workflow type and its config subclass."""

__all__ = ["Finding", "Workflow", "WorkflowConfig", "render_label_source"]

import typing
from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from dataeval_flow._blocks import Block
from dataeval_flow._kind import KindConfig, bind_implementation, bind_result_type, is_abstract, type_arguments
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Step, StepKind

if TYPE_CHECKING:
    from dataeval_flow.steps._result import ChainResult


# --- Config ---

R = TypeVar("R")


class WorkflowConfig(KindConfig, Generic[R]):
    """The settings of one workflow entry, and the result class its workflow returns.

    Each workflow has one config class. A pipeline's ``workflows:`` entry is validated with the config class its
    ``type`` names, and :func:`~dataeval_flow.run` takes an instance directly. Every entry has a ``name``, which
    tasks reference it by and which defaults to its ``type``, and the ``ontology`` field below.

    Subclassing
    -----------
    Parameterize ``WorkflowConfig`` with :class:`~dataeval_flow.steps.ChainResult`, the result every workflow type
    returns. That binds ``result_type``, the type :func:`~dataeval_flow.run` returns for it. A workflow whose config
    is not parameterized with a result class raises ``TypeError`` when the workflow class is defined. Then define:

    - ``type``: a ``str`` field whose default is the workflow's type id. A validator refuses any other value,
      and the JSON schema states it as a ``const``.
    - ``inputs``: a ``ClassVar[InputSpec]`` naming what the workflow reads and how many sources a task gives it.
      A task that does not meet it is refused when the pipeline config loads, and again before a run.
    - The workflow's settings, as pydantic fields, each with ``Field(description=...)``. The JSON schema, the TUI
      and the interactive CLI show the descriptions.

    Override :meth:`wanted_kinds` when a setting switches on one of ``inputs.optional``: a task needs an extractor
    only when a kind it wants is made with one. Override :meth:`check_inputs` when a setting limits the source
    count beyond ``inputs.sources``. Mix in :class:`~dataeval_flow.config.MetadataConfigMixin` or
    :class:`~dataeval_flow.config.StatsConfigMixin` to read a source's metadata or statistics under a policy the
    pipeline names.

    A config has no entry point of its own: Flow finds it through its workflow's ``config_type`` (see
    :class:`Workflow`).

    Examples
    --------
    The config of :class:`Workflow`'s example:

    >>> from typing import ClassVar
    >>> from pydantic import Field
    >>> from dataeval_flow import InputKind, InputSpec, SourceCount
    >>> from dataeval_flow.steps import ChainResult
    >>> from dataeval_flow.workflows import WorkflowConfig
    >>> class DedupeConfig(WorkflowConfig[ChainResult]):
    ...     type: str = "example.dedupe"
    ...     inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE)
    ...     exact: float = Field(default=0.0, ge=0, description="Share of exact duplicates that warns.")
    >>> DedupeConfig.result_type is ChainResult, DedupeConfig(exact=0.01).name
    (True, 'example.dedupe')

    A pipeline entry for it:

    .. code-block:: yaml

        workflows:
          - name: dedupe
            type: example.dedupe
            exact: 0.01
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    ontology: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Label space this workflow's labels are read under. Name an entry under the "
            "top-level `ontologies:` key, or give a path to a serialized RDF artifact "
            "resolved against the data root; a nested mapping of concept to children is "
            "read as an inline hierarchy. Recorded in the result envelope's `label_space`, "
            "so a run conformed by a `taxonomy` entry's stanza carries that entry's "
            "digest and can be matched back to it. Declare it wherever a source's view "
            "applies a `Relabel`. `taxonomy` judges labels against it; `scope` "
            "refuses it, so a scope run on a conformed source records no label "
            "space of its own."
        ),
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``result_type`` from the result class this config was parameterized with."""
        super().__pydantic_init_subclass__(**kwargs)
        bind_result_type(cls)


def render_label_source(label_source: "str | Sequence[str] | None") -> str:
    """Render a label provenance for a report line.

    A merged dataset carries one entry per operand, so join them rather than printing the
    list: the reader wants the provenances, not their repr.
    """
    if label_source is None:
        return ""
    if isinstance(label_source, str):
        return label_source
    return ", ".join(label_source)


# --- Findings ---


class Finding(BaseModel):
    """What a check step found: a titled piece of evidence, and whether it breached a health threshold.

    ``severity`` is the verdict: a ``"warning"`` finding counts toward the result's health and
    ``--fail-on-warning``. ``brief`` is the value on the finding's summary line, ``description`` the lede
    under its heading, and ``blocks`` the evidence: report blocks that the text report draws and
    ``results.json`` holds, one object per block with its ``type`` tag. A finding accepts no other field.
    ``step`` names the check step that made it; Flow fills it in, so a check leaves it unset.

    Examples
    --------
    >>> from dataeval_flow.steps import Finding
    >>> finding = Finding(title="train items", severity="warning", brief="3 items", description="Fewer than 10.")
    >>> finding.brief
    '3 items'
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    severity: Literal["ok", "info", "warning"] = Field(
        default="info",
        description=(
            "`warning` when the finding breached its health threshold, `ok` when it passed one, `info` when "
            "nothing judged it. Warnings count toward the result's health and `--fail-on-warning`."
        ),
    )
    title: str = Field(description="A short label: the finding's summary line and its detail section's heading.")
    brief: str | None = Field(
        default=None, description="A short value shown on the summary line and beside the detail section's heading."
    )
    description: str | None = Field(
        default=None, description="The lede: a sentence or two of plain prose under the detail section's heading."
    )
    blocks: list[Block] = Field(default_factory=list, description="The evidence, as report blocks, in order.")
    step: str | None = Field(
        default=None,
        description=(
            "The check step that made it, with the element's key where the check ran once per element of a list, "
            "such as `class-imbalance[train]`. Flow fills it in; left out until then."
        ),
    )

    @model_serializer(mode="wrap")
    def _without_defaults(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202 - an annotation would flatten the serialization schema
        """Leave `step` out while it holds its default."""
        data = handler(self)
        if self.step is None:
            data.pop("step", None)
        return data


# --- Workflow ---


def _require_the_configs_result(cls: type) -> None:
    """Refuse a result type argument that is neither the result class the config names nor a subclass of it.

    :func:`~dataeval_flow.run` is typed by the config's result class, so a workflow producing another class would
    make that type a lie. Checked only where both arguments are classes: an abstract base may pass type variables on.
    """
    arguments = type_arguments(cls, Workflow)
    if len(arguments) != 2:
        return
    config, result = arguments
    result = typing.get_origin(result) or result
    expected = getattr(config, "result_type", None)
    if not isinstance(result, type) or not isinstance(expected, type) or issubclass(result, expected):
        return
    raise TypeError(
        f"{cls.__name__} returns {result.__name__}, but its config {config.__name__} names {expected.__name__}: "
        f"parameterize it as `Workflow[{config.__name__}, {expected.__name__}]`, or with a subclass of "
        f"{expected.__name__}."
    )


ConfigT = TypeVar("ConfigT", bound="WorkflowConfig[Any]")
ResultT = TypeVar("ResultT", bound="ChainResult")


class Workflow(Step, ABC, Generic[ConfigT, ResultT]):
    """A workflow type: settings that expand to a chain of steps, whose checks judge findings against health
    thresholds.

    Every workflow type is a preset: Flow runs the chain its :class:`~dataeval_flow.workflows.Preset` mixin builds,
    as it runs a custom workflow's, and returns a :class:`~dataeval_flow.steps.ChainResult`. Flow finds a workflow
    type by its ``name``: a pipeline entry's ``type:``, :func:`get_workflow` and ``dataeval-flow workflows`` all
    use it.

    Subclassing
    -----------
    Mix in ``Preset`` ahead of ``Workflow``, and parameterize ``Workflow`` with the config class and ``ChainResult``,
    ``class DedupeWorkflow(Preset, Workflow[DedupeConfig, ChainResult])``, which binds ``config_type`` when the class
    is defined. The arguments must be given to ``Workflow`` itself: an abstract base of your own may take them for
    its subclasses, but a generic one (``class Shared(Workflow[C, R])``, subclassed as
    ``Shared[DedupeConfig, ChainResult]``) is refused. Then define:

    - ``name: ClassVar[str]``: the type id. It must equal the config's ``type`` default and the entry-point name.
    - ``description: ClassVar[str]``: one line, which ``dataeval-flow workflows`` prints.
    - What ``Preset`` asks for: ``slots``, ``outputs`` where a custom workflow may read a Dataset it makes, and
      :meth:`~dataeval_flow.workflows.Preset.chain`. The steps may be built-in or registered by any plugin.

    A concrete workflow that is not a preset, has no ``name`` or ``description``, is not parameterized, whose config
    is not parameterized with a result class, or whose result argument is neither that class nor a subclass of it,
    raises ``TypeError`` when the class is defined. Register the class under the ``dataeval_flow.workflows``
    entry-point group, named by ``name``. Flow loads it on the first registry lookup and leaves it out, logging why,
    when it fails to import, is not a ``Workflow``, its entry-point name, ``name`` and the config's ``type`` default
    disagree, its config declares no ``inputs``, or another workflow has its name.

    Examples
    --------
    A workflow type that finds duplicates, judges how many are exact, and removes them, with the config defined in
    :class:`WorkflowConfig`'s example:

    >>> from typing import ClassVar
    >>> from dataeval_flow.evaluators.quality import DuplicatesConfig
    >>> from dataeval_flow.steps import ChainResult, InputSlot
    >>> from dataeval_flow.steps._port import DataType, Port
    >>> from dataeval_flow.workflows import Preset, PresetChain, Workflow
    >>> class DedupeWorkflow(Preset, Workflow[DedupeConfig, ChainResult]):
    ...     name: ClassVar[str] = "example.dedupe"
    ...     description: ClassVar[str] = "Finds, judges and removes duplicates."
    ...     slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)
    ...     outputs: ClassVar[tuple[Port, ...]] = (Port("kept", DataType.DATASET),)
    ...
    ...     @classmethod
    ...     def chain(cls, config: DedupeConfig) -> PresetChain:
    ...         return PresetChain(
    ...             steps=[
    ...                 {"name": "duplicates", "evaluator": "duplicates", "input": "data"},
    ...                 {"name": "image-duplicates", "check": "image-duplicates", "input": "duplicates",
    ...                  "exact": config.exact},
    ...                 {"name": "kept", "transform": "remove", "input": "data", "plans": {"duplicates": {}}},
    ...             ],
    ...             evaluators=[DuplicatesConfig(name="duplicates")],
    ...         )

    Register it in the plugin's ``pyproject.toml``:

    .. code-block:: toml

        [project.entry-points."dataeval_flow.workflows"]
        "example.dedupe" = "my_package:DedupeWorkflow"

    Once the plugin is installed, it runs like a built-in:

    >>> from dataeval_flow import run
    >>> result = run(DedupeConfig(exact=0.01), dataset)  # doctest: +SKIP
    >>> result.steps["kept"].output  # the Dataset without its duplicates  # doctest: +SKIP
    """

    name: ClassVar[str]
    description: ClassVar[str]
    kind: ClassVar[StepKind] = "workflow"
    config_type: ClassVar["type[WorkflowConfig[Any]]"]  # type: ignore[reportIncompatibleVariableOverride]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``config_type`` from the type arguments, and require identity and the preset mixin on a concrete
        workflow."""
        from dataeval_flow.workflows._preset import Preset

        super().__init_subclass__(**kwargs)
        if not is_abstract(cls) and not issubclass(cls, Preset):
            raise TypeError(
                f"{cls.__name__} is a workflow type, and every workflow type is a preset: mix in `Preset` ahead of "
                f"`Workflow`, `class {cls.__name__}(Preset, Workflow[...])`, and declare `slots` and `chain`."
            )
        bind_implementation(cls, Workflow)
        _require_the_configs_result(cls)

    @classmethod
    def input_ports(cls) -> tuple[Port, ...]:
        """One Dataset port, fed one address per source the workflow reads."""
        spec = cls.config_type.inputs
        return (Port("input", DataType.DATASET, kinds=spec.dataset_kinds, count=spec.sources, derives=spec.kinds),)
