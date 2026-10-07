"""A toy workflow type, `test.count`: the stand-in for tests about machinery every workflow type shares.

It declares what quality declares (statistics and metadata, clusters when asked, one source), so a test about
the orchestrator or the envelope keeps the inputs it was written against while running a workflow of its own. Its
chain is a toy combine that counts the source's items and a toy check that warns below `minimum`. Serve it through
the `plugins` fixture with :func:`register_count` before building any pipeline that names it, or build a result of
it by hand with :func:`count_result`.

A second, `test.union`, holds a discriminated-union list for the TUI's forms, served with :func:`register_union`.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import (
    ChainMetadata,
    ChainOutput,
    ChainResult,
    Check,
    CheckConfig,
    CheckContext,
    Combine,
    CombineConfig,
    CombineContext,
    DataType,
    Finding,
    InputSlot,
    Port,
    StepResult,
)
from dataeval_flow.workflows import Preset, PresetChain, Workflow, WorkflowConfig


class ItemCount(BaseModel):
    """How many items a Dataset holds: `test.items`'s output."""

    items: int


class ToyItemsConfig(CombineConfig):
    input: str


class ToyItems(Combine[ToyItemsConfig]):
    """Counts a Dataset's items."""

    name: ClassVar[str] = "test.items"
    description: ClassVar[str] = "Counts a Dataset's items."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(ItemCount,)),)

    def run(self, config: ToyItemsConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        return {"output": ItemCount(items=len(inputs["input"].value))}


class ToyAtLeastConfig(CheckConfig):
    input: str
    minimum: int = 0


class ToyAtLeast(Check[ToyAtLeastConfig]):
    """One finding: the count, a warning below `minimum`."""

    name: ClassVar[str] = "test.at-least"
    description: ClassVar[str] = "Warns below a count of items."
    title: ClassVar[str] = "Items"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(ItemCount,)),)

    def run(self, config: ToyAtLeastConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
        n = inputs["input"].value.items
        return [Finding(severity="warning" if n < config.minimum else "info", title=self.title, brief=f"{n} items")]


class ToyCountConfig(WorkflowConfig[ChainResult], MetadataConfigMixin, StatsConfigMixin):
    """The settings of one `test.count` entry."""

    type: str = Field(default="test.count", description="The workflow type this entry configures: `test.count`.")
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.STATS, InputKind.METADATA}),
        optional=frozenset({InputKind.CLUSTERS}),
        sources=SourceCount.ONE,
    )
    minimum: int = Field(default=0, ge=0, description="Fewest items the source may hold before it warns.")
    clusters: bool = Field(default=False, description="Whether the run asks for clusters, which need an extractor.")

    def wanted_kinds(self) -> frozenset[InputKind]:
        """Statistics and metadata, and clusters when asked."""
        return self.inputs.required | (frozenset({InputKind.CLUSTERS}) if self.clusters else frozenset())


class ToyCountWorkflow(Preset, Workflow[ToyCountConfig, ChainResult]):
    """Counts the one source's items, and warns below `minimum`."""

    name: ClassVar[str] = "test.count"
    description: ClassVar[str] = "Counts the source's items."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: ToyCountConfig) -> PresetChain:
        """The count, then the check on it."""
        return PresetChain(
            steps=[
                {"name": "items", "combine": "test.items", "input": "data"},
                {"name": "at-least", "check": "test.at-least", "input": "items", "minimum": config.minimum},
            ]
        )


def register_count(plugins: dict[str, list[tuple[str, str]]]) -> None:
    """Serve `test.count`, with its combine and check, through the `plugins` fixture, before the first registry
    lookup."""
    plugins.setdefault("dataeval_flow.workflows", []).append(("test.count", "tests.workflow_toys:ToyCountWorkflow"))
    plugins.setdefault("dataeval_flow.combines", []).append(("test.items", "tests.workflow_toys:ToyItems"))
    plugins.setdefault("dataeval_flow.checks", []).append(("test.at-least", "tests.workflow_toys:ToyAtLeast"))


def count_result(*findings: Finding, metadata: ChainMetadata | None = None) -> ChainResult:
    """A successful `test.count` result whose check step made `findings`, for a test of how any workflow's result reads
    without running one."""
    steps = {
        "at-least": StepResult(
            name="at-least", kind="check", type="test.at-least", inputs=["items"], status="ok", output=list(findings)
        )
    }
    return ChainResult(
        type="test.count", success=True, metadata=metadata or ChainMetadata(), output=ChainOutput(steps), steps=steps
    )


class _Near(BaseModel):
    """A `near` variant of `test.union`'s detectors."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["near"] = Field(default="near", description="Selects this variant: `near`.")
    k: int = Field(default=5, gt=0, description="Neighbors.")
    # A select and a float, so the TUI's variant-widget tests still read each kind of field.
    distance_metric: Literal["cosine", "euclidean"] = Field(default="cosine", description="Distance.")
    threshold_perc: float = Field(default=95.0, description="Percent treated as normal.")


class _Far(BaseModel):
    """A `far` variant of `test.union`'s detectors."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["far"] = Field(default="far", description="Selects this variant: `far`.")
    n_folds: int = Field(default=3, ge=2, description="Folds.")


class _Limits(BaseModel):
    """`test.union`'s nested thresholds, which the TUI edits field by field."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float = Field(default=10.0, description="Warns from this percent.")
    info: float = Field(default=1.0, description="Is info from this percent.")


class ToyUnionConfig(WorkflowConfig[ChainResult]):
    """The settings of one `test.union` entry: a list of discriminated variants, the shape the TUI's union-list form
    edits."""

    type: str = Field(default="test.union", description="The workflow type this entry configures: `test.union`.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE)
    detectors: Sequence[Annotated[_Near | _Far, Field(discriminator="method")]] = Field(
        min_length=1, description="The variants, at least one."
    )
    # A plain list, which the TUI edits as JSON.
    exclude: list[str] = Field(default_factory=list, description="Names to leave out.")
    health_thresholds: _Limits = Field(default_factory=_Limits, description="When it warns.")


class ToyUnionWorkflow(Preset, Workflow[ToyUnionConfig, ChainResult]):
    """Stands in for a workflow whose config holds a discriminated-union list; the TUI's tests never run it."""

    name: ClassVar[str] = "test.union"
    description: ClassVar[str] = "A discriminated-union list, for the TUI's forms."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: ToyUnionConfig) -> PresetChain:  # noqa: ARG003 - one chain whatever the settings
        """Duplicates, which read the statistics its config declares."""
        return PresetChain(
            steps=[{"name": "duplicates", "evaluator": "duplicates", "input": "data"}],
            evaluators=[DuplicatesConfig(name="duplicates")],
        )


def register_union(plugins: dict[str, list[tuple[str, str]]]) -> None:
    """Serve `test.union` through the `plugins` fixture, before the first registry lookup."""
    plugins.setdefault("dataeval_flow.workflows", []).append(("test.union", "tests.workflow_toys:ToyUnionWorkflow"))
