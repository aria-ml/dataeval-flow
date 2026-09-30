"""A toy workflow type, `test.count`: the stand-in for tests about machinery every workflow type shares.

It reads what data-cleaning read (statistics and metadata, clusters when asked, one source), so a test about the
orchestrator or the envelope keeps the inputs it was written against while running a workflow of its own. Serve it
through the `plugins` fixture with :func:`register_count` before building any pipeline that names it.
"""

from typing import ClassVar

from pydantic import Field

from dataeval_flow import ResultMetadata
from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.workflows import (
    Finding,
    Workflow,
    WorkflowConfig,
    WorkflowContext,
    WorkflowOutput,
    WorkflowRawOutput,
    WorkflowReport,
    WorkflowResult,
)


class ToyCountRaw(WorkflowRawOutput):
    """How many items the source holds, in `dataset_size`."""


class ToyCountOutput(WorkflowOutput[ToyCountRaw, WorkflowReport]):
    """The count, and one finding on it."""


class ToyCountMetadata(ResultMetadata):
    """The envelope of a `test.count` result: its own class, so a failed result's metadata can be told apart."""


class ToyCountResult(WorkflowResult[ToyCountMetadata, ToyCountOutput]):
    """What a `test.count` run returns."""


class ToyCountConfig(WorkflowConfig[ToyCountResult], MetadataConfigMixin, StatsConfigMixin):
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


class ToyCountWorkflow(Workflow[ToyCountConfig, ToyCountResult]):
    """Counts the one source's items, and warns below `minimum`."""

    name: ClassVar[str] = "test.count"
    description: ClassVar[str] = "Counts the source's items."

    def run(self, config: ToyCountConfig, context: WorkflowContext) -> ToyCountResult:
        """One finding: the count, a warning below `minimum`."""
        (source,) = context.sources
        n = len(context.dataset(source))
        finding = Finding(severity="warning" if n < config.minimum else "info", title="Items", brief=f"{n} items")
        report = WorkflowReport(summary="Items counted.", findings=[finding])
        output = ToyCountOutput(raw=ToyCountRaw(dataset_size=n), report=report)
        return ToyCountResult(type=self.name, success=True, output=output, metadata=ToyCountMetadata())


def register_count(plugins: dict[str, list[tuple[str, str]]]) -> None:
    """Serve `test.count` through the `plugins` fixture, before the first registry lookup."""
    plugins.setdefault("dataeval_flow.workflows", []).append(("test.count", "tests.workflow_toys:ToyCountWorkflow"))
