"""Task configuration schema."""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, model_validator

AutoBinMethod = Literal["uniform_width", "uniform_count", "clusters"]


class TaskConfig(BaseModel):
    """Task/workflow configuration schema.

    Tasks reference sources (dataset+selection bundles) and an optional
    extractor (model+preprocessor+batch_size bundle) by name.
    """

    name: str
    workflow: str  # reference to WorkflowConfig.name (e.g. "clean_zscore_stats")
    enabled: bool = Field(default=True, description="Whether this task is included when running the pipeline.")
    sources: str | Sequence[str]  # reference to SourceConfig.name
    extractor: str | None = None  # reference to ExtractorConfig.name


class MultiSourceTaskConfig(TaskConfig):
    """TaskConfig subclass that validates multiple sources/datasets for drift/ood tasks.

    Validates that at least two sources are specified (reference + test).
    The ``workflow`` field references a workflow instance whose type must
    be either ``drift-monitoring`` or ``ood-detection`` — enforced at runtime
    by the orchestrator.
    """

    @model_validator(mode="after")
    def _require_multiple_sources(self) -> "MultiSourceTaskConfig":
        srcs = self.sources if isinstance(self.sources, list) else [self.sources]
        if len(srcs) < 2:
            raise ValueError(f"{self.workflow} requires at least 2 sources (reference + test), got {len(srcs)}: {srcs}")
        return self


class DataAnalysisTaskConfig(TaskConfig):
    """Task config for ``data-analysis`` workflows.

    A typed subclass of :class:`TaskConfig` that enables typed overloads
    on :func:`~dataeval_flow.workflow.orchestrator.run_tasks`, returning a
    result with full access to analysis-specific metadata.
    """


class DataCleaningTaskConfig(TaskConfig):
    """Task config for ``data-cleaning`` workflows.

    A typed subclass of :class:`TaskConfig` that enables typed overloads
    on :func:`~dataeval_flow.workflow.orchestrator.run_tasks`, returning a
    :class:`~dataeval_flow.workflows.cleaning.outputs.DataCleaningResult`
    with full access to cleaning-specific metadata (``mode``,
    ``clean_indices``, ``flagged_indices``, ``removed_count``).
    """


class DataSplittingTaskConfig(TaskConfig):
    """Task config for ``data-splitting`` workflows."""


class DriftMonitoringTaskConfig(MultiSourceTaskConfig):
    """Task config that validates drift-monitoring constraints.

    Validates that at least two sources are specified (reference + test).
    The ``workflow`` field references a workflow instance whose type must
    be ``drift-monitoring`` — enforced at runtime by the orchestrator.
    """


class OODDetectionTaskConfig(MultiSourceTaskConfig):
    """Task config that validates OOD detection constraints.

    Validates that at least two datasets are specified (reference + test).
    The ``workflow`` field references a workflow instance whose type must
    be ``ood-detection`` — enforced at runtime by the orchestrator.
    """


class DataPrioritizationTaskConfig(MultiSourceTaskConfig):
    """Task config for ``data-prioritization`` workflows.

    Requires at least two sources: a reference (labeled) dataset and one
    or more additional datasets to prioritize for labeling.
    """
