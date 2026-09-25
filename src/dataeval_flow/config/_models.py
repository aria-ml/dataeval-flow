"""Pipeline and workflow composition models — SourceConfig, PipelineConfig."""

__all__ = [
    "LoggingConfig",
    "PipelineConfig",
    "SourceConfig",
]

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny, model_validator

from dataeval_flow._kind import input_problem
from dataeval_flow.config._schemas import (
    DatasetConfig,
    DatasetProtocolConfig,
    ExportConfig,
    MetadataPolicyConfig,
    OntologyConfig,
    PreprocessorConfig,
    StatsPolicyConfig,
    TaskConfig,
    ViewConfig,
)
from dataeval_flow.config.extractors._base import ExtractorConfig
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.workflows._base import WorkflowConfig

if TYPE_CHECKING:
    _WorkflowBase = WorkflowConfig[Any]
    _EvaluatorBase = EvaluatorConfig[Any]
else:
    # Bare at runtime: an instance of any parameterization is an instance of the bare class, whereas
    # `WorkflowConfig[Any]` is a class of its own that would rebuild each entry as the base, dropping its fields.
    _WorkflowBase = WorkflowConfig
    _EvaluatorBase = EvaluatorConfig

# ---------------------------------------------------------------------------
# Source — dataset + optional view
# ---------------------------------------------------------------------------


class SourceConfig(BaseModel):
    """Named source definition — bundles a dataset with an optional view.

    Name a `dataset` to read one dataset, or `merge` to concatenate other sources
    into one corpus.

    YAML example::

        sources:
          - name: cifar_train_subset
            dataset: cifar10_train
            view: first_5k
          - name: merged
            merge: [m3fd_conformed, drone_conformed]

    The legacy ``selection`` key is accepted as a deprecated alias for ``view``.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)

    name: str = Field(description="Identifier for the source")
    dataset: str | None = Field(
        default=None,
        description="Reference to a dataset name. Name either this or `merge`, not both.",
    )
    merge: Sequence[str] | None = Field(
        default=None,
        description=(
            "Sources to concatenate into one corpus, in the order given. Name either this "
            "or `dataset`, not both. Give every operand a view whose `Relabel` passes the "
            "identical `target`, or their integer labels denote different classes and the "
            "merge is refused. Each datum's id becomes '<position>:<id>', so an item keeps "
            "its source's identity."
        ),
    )
    view: str | None = Field(
        default=None,
        validation_alias=AliasChoices("view", "selection"),
        description="Reference to a view name (optional)",
    )

    @model_validator(mode="after")
    def _exactly_one_input(self) -> "SourceConfig":
        """Refuse a source that names both a dataset and a merge, or neither."""
        if self.merge is not None and len(self.merge) < 2:
            raise ValueError(f"Source '{self.name}' merges at least two sources; this one names {len(self.merge)}.")
        if self.dataset is not None and self.merge is not None:
            raise ValueError(
                f"Source '{self.name}' names both `dataset` and `merge`. Name one: `dataset` "
                "reads one dataset, `merge` concatenates other sources."
            )
        if self.dataset is None and self.merge is None:
            raise ValueError(f"Source '{self.name}' names neither `dataset` nor `merge`. Name one.")
        return self


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    """The log levels of a pipeline's ``logging:`` key.

    Only a pipeline run by the ``dataeval-flow`` command applies them: :func:`~dataeval_flow.run_tasks` and
    :func:`~dataeval_flow.run` leave logging to their caller.

    YAML example::

        logging:
          app_level: INFO
          lib_level: ERROR
    """

    app_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="DEBUG", description="Level of dataeval-flow's own loggers."
    )
    lib_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="WARNING",
        description=(
            "Level of the root logger, so of the libraries dataeval-flow calls. DataEval's diagnostics still show at "
            "WARNING when this is set higher."
        ),
    )


# ---------------------------------------------------------------------------
# Pipeline (top-level)
# ---------------------------------------------------------------------------


def _dispatch(entry: Any, *, kind: str, resolve: Callable[[str], type[Any]], key: str = "type") -> Any:
    """Validate a mapping entry with the config class its `key` names; leave an instance as it is."""
    if not isinstance(entry, Mapping):
        return entry
    type_id = entry.get(key)
    if not isinstance(type_id, str):
        raise ValueError(f"Each `{kind}s:` entry needs a `{key}:`.")
    return resolve(type_id).config_type.model_validate(entry)


def _workflow_entry(entry: Any) -> Any:
    from dataeval_flow.workflows._registry import get_workflow

    return _dispatch(entry, kind="workflow", resolve=get_workflow)


def _evaluator_entry(entry: Any) -> Any:
    from dataeval_flow.evaluators._registry import get_evaluator

    return _dispatch(entry, kind="evaluator", resolve=get_evaluator)


def _extractor_entry(entry: Any) -> Any:
    from dataeval_flow.config.extractors._registry import get_extractor

    return _dispatch(entry, kind="extractor", resolve=get_extractor, key="model")


# One `workflows:` / `evaluators:` / `extractors:` entry, validated with the config class its registered type
# (an extractor's `model`) names. Dispatched per entry, so an error's location carries the entry's index;
# serialized as its own class, so a dump keeps a subclass's fields.
_WorkflowEntry = Annotated[SerializeAsAny[_WorkflowBase], BeforeValidator(_workflow_entry)]
_EvaluatorEntry = Annotated[SerializeAsAny[_EvaluatorBase], BeforeValidator(_evaluator_entry)]
_ExtractorEntry = Annotated[SerializeAsAny[ExtractorConfig], BeforeValidator(_extractor_entry)]


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration.

    All sections use a define-once, reference-by-name pattern.
    Sources compose datasets with optional views; extractors
    compose model type/params with optional preprocessors.
    Tasks reference workflows, sources, and extractors by name.

    The legacy ``selections`` key is accepted as a deprecated alias for
    ``views`` (with a :class:`DeprecationWarning`).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)

    # Logging
    logging: LoggingConfig | None = Field(
        default=None,
        description="Log levels for dataeval-flow and the libraries it calls, applied when the CLI runs the pipeline",
    )

    # Reproducibility [CR-7-S-1]
    seed: int | None = Field(
        default=None,
        description=(
            "Seed for every stochastic component of the run (clustering, random splits, "
            "shuffled views, sampling). Applied through DataEval's seed configuration before "
            "each task, so a task's result does not depend on what ran before it. "
            "None (the default) leaves randomness unseeded."
        ),
    )
    deterministic: bool = Field(
        default=False,
        description=(
            "Force PyTorch to use deterministic algorithms. Only meaningful alongside `seed`. "
            "Improves run-to-run reproducibility on GPU at some cost to performance."
        ),
    )

    # Named resource pools
    datasets: Sequence[DatasetConfig | DatasetProtocolConfig] | None = Field(
        default=None, description="Named dataset definitions (format + path), referenced by sources"
    )
    preprocessors: Sequence[PreprocessorConfig] | None = Field(
        default=None, description="Named preprocessor definitions (transform steps), referenced by extractors"
    )
    views: Sequence[ViewConfig] | None = Field(
        default=None,
        validation_alias=AliasChoices("views", "selections"),
        description="Named view pipeline definitions (dataset operations), referenced by sources",
    )

    metadata: Sequence[MetadataPolicyConfig] | None = Field(
        default=None,
        description=(
            "Named metadata policy definitions (encoding, vocabularies, exclusions), "
            "referenced by workflows. Defined once and shared so that workflows meant to "
            "be compared read their factors under one encoding."
        ),
    )

    ontologies: Sequence[OntologyConfig] | None = Field(
        default=None,
        description=(
            "Named label-space definitions, referenced by workflows. Defined once and shared "
            "so that workflows meant to be compared read the same vocabulary."
        ),
    )

    stats: Sequence[StatsPolicyConfig] | None = Field(
        default=None,
        description=(
            "Named stats policy definitions (which statistics over which views), referenced "
            "by workflows. Defined once and shared so that workflows meant to be compared "
            "measure the same things."
        ),
    )

    # Composition layers
    sources: Sequence[SourceConfig] | None = Field(
        default=None,
        description="Named source definitions (dataset + optional view)",
    )
    extractors: Sequence[_ExtractorEntry] | None = Field(
        default=None,
        description="Named extractor definitions (model type + params + optional preprocessor + batch_size)",
    )
    exports: Sequence[ExportConfig] | None = Field(
        default=None,
        description=(
            "Named datasets to write out, referenced by source. Declared here rather than "
            "on a task, so a corpus is written whether or not a task reads it."
        ),
    )

    # Execution
    workflows: Sequence[_WorkflowEntry] | None = Field(
        default=None,
        description="Named workflow configurations (type + params), referenced by tasks",
    )
    evaluators: Sequence[_EvaluatorEntry] | None = Field(
        default=None,
        description=(
            "Named evaluator configurations (type + params), referenced by tasks. An evaluator runs one DataEval "
            "evaluator and reports its determinations, with no health status."
        ),
    )
    tasks: Sequence[TaskConfig] | None = Field(
        default=None,
        description="What to run: each task runs a workflow or evaluator on named sources, with an optional extractor",
    )

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """The schema a config file follows, with one branch per installed workflow, evaluator and extractor."""
        if cls is PipelineConfig:
            from dataeval_flow.config._json_schema import registry_twin

            return registry_twin().model_json_schema(*args, **kwargs)
        return super().model_json_schema(*args, **kwargs)

    @model_validator(mode="before")
    @classmethod
    def _warn_legacy_selection_keys(cls, data: Any) -> Any:
        """Emit deprecation warnings for the legacy ``selections``/``selection`` keys."""
        if isinstance(data, Mapping):
            if "selections" in data and "views" not in data:
                warnings.warn(
                    "The 'selections' key is deprecated; use 'views' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            for source in data.get("sources") or []:
                if isinstance(source, Mapping) and "selection" in source and "view" not in source:
                    warnings.warn(
                        "The 'selection' key in a source is deprecated; use 'view' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
        return data

    @model_validator(mode="after")
    def _check_task_inputs(self) -> "PipelineConfig":
        """Refuse a task its workflow or evaluator cannot run, before any data is read."""
        workflows = {workflow.name: workflow for workflow in self.workflows or ()}
        evaluators = {evaluator.name: evaluator for evaluator in self.evaluators or ()}
        for task in self.tasks or ():
            kind = task.kind
            pool = evaluators if kind == "evaluator" else workflows
            target = pool.get(task.workflow)
            if target is None:
                raise ValueError(
                    f"Task '{task.name}' names {kind} '{task.workflow}', which `{kind}s:` does not define. "
                    f"Defined: {sorted(pool)}"
                )
            problem = input_problem(
                target, source_count=len(task.source_names), has_extractor=task.extractor is not None
            )
            if problem is not None:
                raise ValueError(f"Task '{task.name}' runs {kind} '{target.name}' ({target.type}), which {problem}")
        return self

    @model_validator(mode="after")
    def _check_unique_names(self) -> "PipelineConfig":
        """Raise if any named pool contains duplicate names.

        The pools are discovered rather than listed.  A reference resolves by first match,
        so a pool omitted from a hand-maintained list silently keeps the first definition
        and drops the second — most likely the one the user just edited — with the run
        reporting no problem.
        """
        for section_name in type(self).model_fields:
            items = getattr(self, section_name, None)
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                continue
            seen: set[str] = set()
            for item in items:
                name = getattr(item, "name", None)
                if name is None:
                    break
                if name in seen:
                    raise ValueError(f"Duplicate name '{name}' in {section_name}")
                seen.add(name)
        return self
