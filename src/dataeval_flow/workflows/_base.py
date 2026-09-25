"""The workflow framework: the bases a workflow, its config and its output subclass."""

__all__ = [
    "Finding",
    "Workflow",
    "WorkflowConfig",
    "WorkflowOutput",
    "WorkflowRawOutput",
    "WorkflowReport",
    "effective_value_range",
    "raw_field",
    "render_label_source",
]

import typing
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from dataeval_flow._kind import KindConfig, bind_implementation, bind_result_type, type_arguments

if TYPE_CHECKING:
    from dataeval_flow.workflows._context import WorkflowContext
    from dataeval_flow.workflows._result import WorkflowResult


# --- Config ---

R = TypeVar("R")


class WorkflowConfig(KindConfig, Generic[R]):
    """The settings of one workflow entry, and the result class its workflow returns.

    Each workflow has one config class. A pipeline's ``workflows:`` entry is validated with the config class its
    ``type`` names, and :func:`~dataeval_flow.run` takes an instance directly. Every entry has a ``name``, which
    tasks reference it by and which defaults to its ``type``, and the ``mode`` and ``ontology`` fields below.

    Subclassing
    -----------
    Parameterize ``WorkflowConfig`` with the workflow's result class. That binds ``result_type``: the class of
    every result a run of this config returns, a failed run's included, and the type :func:`~dataeval_flow.run`
    returns for it. A workflow whose config is not parameterized with a result class raises ``TypeError`` when
    the workflow class is defined. Then define:

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
    The config of :class:`Workflow`'s example, with a stand-in for the ``CountResult`` defined there:

    >>> from typing import ClassVar
    >>> from pydantic import Field
    >>> from dataeval_flow import InputSpec, ResultMetadata, SourceCount
    >>> from dataeval_flow.workflows import (
    ...     WorkflowConfig, WorkflowOutput, WorkflowRawOutput, WorkflowReport, WorkflowResult,
    ... )
    >>> class CountResult(WorkflowResult[ResultMetadata, WorkflowOutput[WorkflowRawOutput, WorkflowReport]]):
    ...     pass
    >>> class CountConfig(WorkflowConfig[CountResult]):
    ...     type: str = "example.count"
    ...     inputs: ClassVar[InputSpec] = InputSpec(required=frozenset(), sources=SourceCount.ONE_OR_MORE)
    ...     minimum: int = Field(default=0, ge=0, description="Fewest items a source may hold before it warns.")
    >>> CountConfig.result_type is CountResult, CountConfig(minimum=5).name
    (True, 'example.count')

    A pipeline entry for it:

    .. code-block:: yaml

        workflows:
          - name: count
            type: example.count
            minimum: 100
    """

    mode: Literal["advisory", "preparatory"] = Field(
        default="advisory",
        description="advisory: report only, preparatory: modify dataset",
    )
    ontology: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Label space this workflow's labels are read under. Name an entry under the "
            "top-level `ontologies:` key, or give a path to a serialized RDF artifact "
            "resolved against the data root; a nested mapping of concept to children is "
            "read as an inline hierarchy. Recorded in the result envelope's `label_space`, "
            "so a run conformed by a `data-coverage` audit's stanza carries that audit's "
            "digest and can be matched back to it. Declare it wherever a source's view "
            "applies a `Relabel`."
        ),
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``result_type`` from the result class this config was parameterized with."""
        super().__pydantic_init_subclass__(**kwargs)
        bind_result_type(cls)


class _LegacyValueRangeMixin(BaseModel):
    """The deprecated workflow field ``value_range``, kept on the configs that carried it."""

    value_range: tuple[float, float] | None = Field(
        default=None,
        deprecated=(
            "Superseded by `value_range` on the dataset, which every workflow reading that "
            "dataset shares. Removed in the next minor version."
        ),
        description=(
            "Interval the image data occupies, as (low, high). Integer encodings state "
            "their own range; float data does not, and the statistics that need one "
            "answer NaN without it — the whole visual family, pixel histogram and "
            "entropy, and dimension depth. Leave unset for integer imagery."
        ),
    )


def raw_field(params: Any, name: str, default: Any = None) -> Any:
    """A params field's stored value, without tripping pydantic's deprecation descriptor.

    ``Field(deprecated=...)`` warns on every attribute read, the framework's own included,
    so a plain ``getattr`` here would scold a user who did exactly the right thing —
    declared the value on the dataset or the policy and never touched the deprecated
    param. Deprecation belongs to what the user wrote; the instance dict is what they
    wrote, and the warning is raised where the value is actually honoured.
    """
    stored = getattr(params, "__dict__", None)
    if stored is not None and name in stored:
        return stored[name]
    return getattr(params, name, default)


def effective_value_range(
    dataset_context: Any,
    params: Any,
) -> tuple[float, float] | None:
    """The range to measure statistics against: the dataset's, else the deprecated param.

    The dataset's declaration wins because it is the one every consumer sees — the
    injection pass inside ``build_metadata`` reads it off the policy the orchestrator
    stamped, and a workflow reading its own param instead would land in a different stats
    cache scope and buy a second full pass over the data.

    Raises
    ------
    ValueError
        When both are set and disagree.  There is no sound precedence between two people
        stating different facts about the same imagery.
    """
    from_dataset = getattr(dataset_context, "value_range", None)
    from_params = raw_field(params, "value_range")
    if from_dataset is not None and from_params is not None and tuple(from_dataset) != tuple(from_params):
        raise ValueError(
            f"The dataset declares value_range={tuple(from_dataset)} and this workflow's "
            f"params declare {tuple(from_params)}. The workflow param is deprecated — "
            "remove it and keep the dataset's, which every workflow reading that dataset "
            "shares.",
        )
    return from_dataset if from_dataset is not None else from_params


def render_label_source(label_source: "str | Sequence[str] | None") -> str:
    """Render a label provenance for a report line.

    A merged corpus carries one entry per operand, so join them rather than printing the
    list: the reader wants the provenances, not their repr.
    """
    if label_source is None:
        return ""
    if isinstance(label_source, str):
        return label_source
    return ", ".join(label_source)


# --- Output bases ---


class Finding(BaseModel):
    """One item of a workflow's report: a titled piece of evidence, and whether it breached a health threshold.

    ``severity`` is the verdict: a ``"warning"`` finding counts toward the result's health and
    ``--fail-on-warning``. ``report_type`` says how ``data`` is laid out, so the text report can render it.
    """

    report_type: Literal["table", "key_value", "image", "text", "pivot_table", "chunk_table", "classwise_table"] = (
        Field(
            description=(
                "How `data` is laid out, which decides how the text report renders it: `key_value` (a mapping "
                "rendered as name and value lines), `table`, `pivot_table`, `chunk_table`, `classwise_table`, "
                "`text` or `image`."
            )
        )
    )
    severity: Literal["ok", "info", "warning"] = Field(
        default="info",
        description=(
            "`warning` when the finding breached its health threshold, `ok` when it passed one, `info` when "
            "nothing judged it. Warnings count toward the result's health and `--fail-on-warning`."
        ),
    )
    title: str = Field(description="A short label: the finding's summary line and its detail section's heading.")
    data: dict[str, Any] | list[dict[str, Any]] | str = Field(
        description="The evidence, laid out as `report_type` says. A mapping's `brief` shows on the summary line."
    )
    description: str | None = Field(default=None, description="A sentence shown under the detail section's heading.")


class WorkflowRawOutput(BaseModel):
    """A workflow's machine-readable output: what the run measured, before it was judged.

    Subclassing
    -----------
    Subclass it once per workflow and add what the workflow measured as pydantic fields, each with a description.
    Flow reads none of the fields; ``to_dict()`` and ``export()`` write them under ``raw``, so each must
    serialize in pydantic's JSON mode. Name the subclass as the raw type argument of the workflow's
    :class:`WorkflowOutput`.

    Examples
    --------
    >>> from pydantic import Field
    >>> from dataeval_flow.workflows import WorkflowRawOutput
    >>> class CountRaw(WorkflowRawOutput):
    ...     counts: dict[str, int] = Field(default_factory=dict, description="Items in each source.")
    >>> CountRaw(dataset_size=12, counts={"train": 12}).counts
    {'train': 12}
    """

    dataset_size: int = Field(description="Number of items in dataset")


class WorkflowReport(BaseModel):
    """A workflow's human-readable report: a one-line summary, and findings judged against health thresholds.

    Flow renders the report as text and rolls its findings up into the result's health: a finding whose severity
    is ``"warning"`` counts toward :attr:`WorkflowResult.warning_count` and ``--fail-on-warning``.

    Subclassing
    -----------
    Use it as it is when a summary and findings say everything. Otherwise subclass it once per workflow, add
    report fields as pydantic fields with descriptions, and name the subclass as the report type argument of the
    workflow's :class:`WorkflowOutput`. Flow reads only ``summary`` and ``findings``; ``to_dict()`` and
    ``export()`` write every field under ``report``.

    Examples
    --------
    >>> from pydantic import Field
    >>> from dataeval_flow.workflows import Finding, WorkflowReport
    >>> class CountReport(WorkflowReport):
    ...     smallest: str | None = Field(default=None, description="The source holding the fewest items.")
    >>> finding = Finding(report_type="key_value", severity="warning", title="train items", data={"items": 3})
    >>> CountReport(summary="Item counts", findings=[finding], smallest="train").smallest
    'train'
    """

    summary: str = Field(description="One line saying what the run found: the text report's banner.")
    findings: list[Finding] = Field(
        default_factory=list, description="What the run found, each judged against its health threshold."
    )


RawT = TypeVar("RawT", bound=WorkflowRawOutput)
ReportT = TypeVar("ReportT", bound=WorkflowReport)


class WorkflowOutput(BaseModel, Generic[RawT, ReportT]):
    """Everything a workflow run produced: its raw output, and the report drawn from it.

    A successful :class:`WorkflowResult`'s ``output``. ``to_dict()`` and ``export()`` write both fields, under
    ``raw`` and ``report``.

    Subclassing
    -----------
    Subclass it once per workflow, parameterized by the workflow's raw output and report classes, and name the
    subclass as the output type argument of the workflow's :class:`WorkflowResult`. The parameters type the
    result's ``output.raw`` and ``output.report``; the subclass needs no body.

    Examples
    --------
    >>> from pydantic import Field
    >>> from dataeval_flow.workflows import WorkflowOutput, WorkflowRawOutput, WorkflowReport
    >>> class CountRaw(WorkflowRawOutput):
    ...     counts: dict[str, int] = Field(default_factory=dict, description="Items in each source.")
    >>> class CountOutput(WorkflowOutput[CountRaw, WorkflowReport]):
    ...     pass
    >>> output = CountOutput(raw=CountRaw(dataset_size=3, counts={"train": 3}), report=WorkflowReport(summary="3"))
    >>> output.raw.counts
    {'train': 3}
    """

    raw: RawT = Field(description="What the run measured.")
    report: ReportT = Field(description="The summary and findings drawn from `raw`.")


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
ResultT = TypeVar("ResultT", bound="WorkflowResult[Any, Any]")


class Workflow(ABC, Generic[ConfigT, ResultT]):
    """One analysis over a task's sources that ends in a verdict: findings judged against health thresholds.

    A workflow reads what its :class:`WorkflowContext` offers for each source (the dataset, its statistics,
    embeddings, clusters, metadata and labels) and returns a :class:`WorkflowResult` whose findings say what is
    healthy and what is not. Flow finds a workflow by its ``name``: a pipeline entry's ``type:``,
    :func:`get_workflow` and ``dataeval-flow workflows`` all use it.

    Subclassing
    -----------
    Parameterize ``Workflow`` with the workflow's config and result classes,
    ``class CountWorkflow(Workflow[CountConfig, CountResult])``, which binds ``config_type`` when the class is
    defined. The arguments must be given to ``Workflow`` itself: an abstract base of your own may take them for its
    subclasses, but a generic one (``class Shared(Workflow[C, R])``, subclassed as
    ``Shared[CountConfig, CountResult]``) is refused. Then define:

    - ``name: ClassVar[str]``: the type id. It must equal the config's ``type`` default and the entry-point name.
    - ``description: ClassVar[str]``: one line, which ``dataeval-flow workflows`` prints.
    - :meth:`run`: the analysis.

    A concrete workflow without ``name`` or ``description``, not parameterized, whose config is not parameterized
    with a result class, or whose result argument is neither that class nor a subclass of it, raises ``TypeError``
    when the class is defined. Register the class under the
    ``dataeval_flow.workflows`` entry-point group, named by ``name``. Flow loads it on the first registry lookup
    and leaves it out, logging why, when it fails to import, is not a ``Workflow``, its entry-point name, ``name``
    and the config's ``type`` default disagree, its config declares no ``inputs``, or another workflow has its
    name.

    For each task that runs the workflow, Flow builds an instance with no arguments and calls :meth:`run` once. It
    guarantees that:

    - ``config`` is an instance of ``config_type``, validated when it was built or loaded;
    - the task meets ``config.inputs``: it names as many sources as the workflow takes, and an extractor where
      one is needed;
    - an exception raised in :meth:`run`, or a return value that is not an instance of the config's result class,
      becomes a failed result of that class that records the error;
    - once :meth:`run` returns, Flow fills in the result's envelope: the datasets and views read, the extractor,
      the timing, the resolved configuration and any diagnostics DataEval raised.

    Examples
    --------
    A workflow that counts each source's items, with the config and result classes it needs:

    >>> from typing import ClassVar
    >>> from pydantic import Field
    >>> from dataeval_flow import InputSpec, ResultMetadata, SourceCount
    >>> from dataeval_flow.workflows import (
    ...     Finding, Workflow, WorkflowConfig, WorkflowContext, WorkflowOutput, WorkflowRawOutput, WorkflowReport,
    ...     WorkflowResult,
    ... )
    >>> class CountRaw(WorkflowRawOutput):
    ...     counts: dict[str, int] = Field(default_factory=dict, description="Items in each source.")
    >>> class CountOutput(WorkflowOutput[CountRaw, WorkflowReport]):
    ...     pass
    >>> class CountResult(WorkflowResult[ResultMetadata, CountOutput]):
    ...     pass
    >>> class CountConfig(WorkflowConfig[CountResult]):
    ...     type: str = "example.count"
    ...     inputs: ClassVar[InputSpec] = InputSpec(required=frozenset(), sources=SourceCount.ONE_OR_MORE)
    ...     minimum: int = Field(default=0, ge=0, description="Fewest items a source may hold before it warns.")
    >>> class CountWorkflow(Workflow[CountConfig, CountResult]):
    ...     name: ClassVar[str] = "example.count"
    ...     description: ClassVar[str] = "Counts the items in each source."
    ...
    ...     def run(self, config: CountConfig, context: WorkflowContext) -> CountResult:
    ...         counts = {source: len(context.dataset(source)) for source in context.sources}
    ...         findings = [
    ...             Finding(
    ...                 report_type="key_value",
    ...                 severity="warning" if n < config.minimum else "ok",
    ...                 title=f"{source} items",
    ...                 data={"items": n},
    ...             )
    ...             for source, n in counts.items()
    ...         ]
    ...         raw = CountRaw(dataset_size=sum(counts.values()), counts=counts)
    ...         output = CountOutput(raw=raw, report=WorkflowReport(summary="Item counts", findings=findings))
    ...         return CountResult(type=self.name, success=True, output=output, metadata=ResultMetadata())

    Register it in the plugin's ``pyproject.toml``:

    .. code-block:: toml

        [project.entry-points."dataeval_flow.workflows"]
        "example.count" = "my_package:CountWorkflow"

    Once the plugin is installed, it runs like a built-in:

    >>> from dataeval_flow import run
    >>> result = run(CountConfig(minimum=100), dataset)  # doctest: +SKIP
    >>> result.warning_count  # doctest: +SKIP
    1
    """

    name: ClassVar[str]
    description: ClassVar[str]
    config_type: ClassVar["type[WorkflowConfig[Any]]"]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``config_type`` from the type arguments, and require identity on a concrete workflow."""
        super().__init_subclass__(**kwargs)
        bind_implementation(cls, Workflow)
        _require_the_configs_result(cls)

    @abstractmethod
    def run(self, config: ConfigT, context: "WorkflowContext") -> ResultT:
        """Run the workflow on the task's sources and return its result.

        Parameters
        ----------
        config : ConfigT
            This entry's settings, an instance of ``config_type``.
        context : WorkflowContext
            The task's sources, in the order the task names them, with cached access to what each yields.

        Returns
        -------
        ResultT
            A successful result of the config's result class, built with ``type=self.name``.

        Raises
        ------
        Exception
            Anything, to fail the run: Flow records the error on a failed result of the config's result class.
        """
