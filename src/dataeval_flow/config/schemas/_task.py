"""Task configuration schema."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    GetJsonSchemaHandler,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

AutoBinMethod = Literal["uniform_width", "uniform_count", "clusters"]
FactorSource = Literal["coded", "values", "auto"]

TaskKind = Literal["workflow", "evaluator"]


class TaskConfig(BaseModel):
    """Task/workflow configuration schema.

    Tasks reference sources (dataset+selection bundles) and an optional
    extractor (model+preprocessor+batch_size bundle) by name.

    A config file names what a task runs under ``workflow:`` or ``evaluator:``, so the
    file says which kind of task it is. Loaded, either lands in ``workflow``, and ``kind``
    records which key the file used. Running a task reads ``kind`` only to look the name
    up in ``workflows:`` or ``evaluators:``; saved, the task is written back under its key.
    """

    name: str
    workflow: str = Field(
        description=(
            "What this task runs, by name: an entry of `workflows:` or of `evaluators:`, as `kind` says. "
            "A config file names an evaluator under `evaluator:` instead."
        )
    )
    kind: TaskKind = Field(
        default="workflow",
        description=(
            "Whether `workflow` names a workflow or an evaluator, read from the key the config file used. "
            "A result carries the same value."
        ),
    )
    enabled: bool = Field(default=True, description="Whether this task is included when running the pipeline.")
    sources: str | Sequence[str]  # reference to SourceConfig.name
    extractor: str | None = None  # reference to ExtractorConfig.name

    @property
    def source_names(self) -> list[str]:
        """The sources this task reads, as a list whether one or several were named."""
        return [self.sources] if isinstance(self.sources, str) else list(self.sources)

    @model_validator(mode="before")
    @classmethod
    def _read_target_key(cls, data: Any) -> Any:
        """Read the file's ``workflow:`` or ``evaluator:`` into ``workflow`` and ``kind``.

        An empty key (``null`` in YAML, or a TUI form's unset field) counts as absent.
        """
        if not isinstance(data, Mapping):
            return data
        name = data.get("name", "<unnamed>")
        workflow, evaluator = data.get("workflow"), data.get("evaluator")
        if workflow and evaluator:
            raise ValueError(f"Task '{name}' names both a workflow and an evaluator; a task runs exactly one.")
        if not workflow and not evaluator:
            raise ValueError(
                f"Task '{name}' names neither a workflow nor an evaluator; set `workflow:` or `evaluator:`."
            )
        fields = {key: value for key, value in data.items() if key != "evaluator"}
        if evaluator:
            if fields.get("kind", "evaluator") != "evaluator":
                raise ValueError(f"Task '{name}' names an evaluator, so its kind cannot be '{fields['kind']}'.")
            fields.update(workflow=evaluator, kind="evaluator")
        return fields

    @model_serializer(mode="wrap")
    def _write_target_key(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Write the task as a config file names it: under ``workflow:`` or ``evaluator:``, with no ``kind``."""
        data = handler(self)
        data.pop("kind", None)
        if self.kind == "evaluator" and "workflow" in data:
            data["evaluator"] = data.pop("workflow")
        return data

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Describe the file's shape: exactly one of ``workflow:`` and ``evaluator:``, and no ``kind``."""
        schema = handler.resolve_ref_schema(handler(core_schema))
        properties = schema["properties"]
        properties.pop("kind", None)
        properties["workflow"]["description"] = "The workflow this task runs, by its name in `workflows:`."
        properties["evaluator"] = {
            "type": "string",
            "title": "Evaluator",
            "description": "The evaluator this task runs, by its name in `evaluators:`. Set this or `workflow`.",
        }
        schema["required"] = [key for key in schema["required"] if key != "workflow"]
        schema["oneOf"] = [{"required": ["workflow"]}, {"required": ["evaluator"]}]
        return schema


class EvaluatorTaskConfig(TaskConfig):
    """A task that runs one evaluator: a :class:`TaskConfig` whose ``kind`` is always ``"evaluator"``.

    Like the typed workflow tasks, it exists so :func:`~dataeval_flow.workflow.run_task` can
    return an :class:`~dataeval_flow.evaluator.result.EvaluatorResult`. ``workflow`` holds the
    evaluator's name, as on any evaluator task; a config file names it under ``evaluator:``.
    Its evaluator's input spec decides how many sources it takes and whether it needs an
    extractor, and the config refuses a task that breaks either rule when it loads.
    """

    @model_validator(mode="before")
    @classmethod
    def _always_an_evaluator(cls, data: Any) -> Any:
        """Pin ``kind`` to ``"evaluator"``, refusing a task built as anything else."""
        if not isinstance(data, Mapping):
            return data
        if data.get("kind", "evaluator") != "evaluator":
            raise ValueError(f"An EvaluatorTaskConfig runs an evaluator, so its kind cannot be '{data['kind']}'.")
        return {**data, "kind": "evaluator"}


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


class DataCoverageTaskConfig(TaskConfig):
    """Task config for ``data-coverage`` workflows."""


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


class ParameterSweepTaskConfig(TaskConfig):
    """Task config for ``parameter-sweep`` workflows."""


class MetadataTriageTaskConfig(TaskConfig):
    """Task config for ``metadata-triage`` workflows."""
