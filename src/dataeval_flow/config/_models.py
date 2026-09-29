"""Pipeline and workflow composition models — SourceConfig, PipelineConfig."""

__all__ = [
    "LoggingConfig",
    "PipelineConfig",
    "SourceConfig",
]

import difflib
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from dataeval_flow._blocks._table import DEFAULT_PREVIEW, DEFAULT_ROWS
from dataeval_flow._blocks._text import DEFAULT_WIDTH, MIN_WIDTH
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
from dataeval_flow.steps._workflow import CustomWorkflowConfig
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

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, extra="forbid")

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

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

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
# Results
# ---------------------------------------------------------------------------


class ResultConfig(BaseModel):
    """What each task's result carries, and the files the ``dataeval-flow`` command writes it to, under ``result:``.

    ``name``, ``formats``, ``detail``, ``per_task`` and ``width`` shape what the command writes to ``--output`` and
    prints; :func:`~dataeval_flow.run_tasks` and :func:`~dataeval_flow.run` return results and write nothing.

    YAML example::

        result:
          name: audit
          formats: [text, html]
          detail: summary
          per_task: true
          max_images: 100
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str = Field(
        default="result",
        pattern=r"^\w[\w.-]*$",
        description=(
            "The results' file name, without its extension: `result.json`, `result.txt`, `result.html`. Letters, "
            "digits, `_`, `.` and `-`, starting with a letter, digit or `_`."
        ),
    )
    formats: list[Literal["json", "text", "html", "junit", "markdown"]] = Field(
        default_factory=lambda: ["json", "text", "html"],
        min_length=1,
        description=(
            "Which files to write: the JSON record, the text report, the HTML page, a JUnit report (`.xml`) for a CI's "
            "test view, and a Markdown summary (`.md`) for a job summary or a merge-request comment."
        ),
    )
    detail: Literal["full", "summary"] = Field(
        default="full",
        description=(
            "How much of each report the text and HTML files hold: every finding in `full`, the summary table in "
            "`summary`. The JSON always holds everything."
        ),
    )
    per_task: bool = Field(
        default=False,
        description="Write each task's results to files of its own, `<name>-<task>.<ext>`, not one for the run.",
    )
    fail_on: Literal["never", "failure", "warning"] = Field(
        default="failure",
        description=(
            "What makes the command's exit code non-zero: a failed task (`failure`: 1), also a finding past its health "
            "threshold (`warning`: 3), or nothing (`never`). `--fail-on-warning` and `DATAEVAL_FAIL_ON_WARNING` "
            "override it."
        ),
    )
    width: int = Field(
        default=DEFAULT_WIDTH,
        ge=MIN_WIDTH,
        description=(
            "Characters per line of the text report, on the console and in its file. `--report-width` and "
            "`DATAEVAL_REPORT_WIDTH` override it."
        ),
    )
    max_images: int = Field(
        default=200,
        ge=-1,
        description=(
            "Most thumbnails a task's result embeds, shared evenly between the report's findings that name items, "
            "and each finding's share between its tables; a share more than its items need goes to the rest. "
            "0 embeds none, and -1 every item the report names."
        ),
    )
    max_rows: int = Field(
        default=DEFAULT_ROWS,
        ge=-1,
        description=(
            "Most rows a table of items lists, such as flagged images or duplicate groups; a paragraph after it "
            "names the rest, which the JSON's raw output holds. -1 lists every row."
        ),
    )
    preview_rows: int = Field(
        default=DEFAULT_PREVIEW,
        ge=-1,
        description=(
            "How many rows of a table of items the text report and the TUI show, before a line counting the rest; "
            "the HTML page shows every row. -1 shows every row."
        ),
    )

    @field_validator("name")
    @classmethod
    def _not_the_descriptor(cls, value: str) -> str:
        """``encoding.json`` beside the results is the run's encoding descriptor, which would overwrite them."""
        if value == "encoding":
            raise ValueError("name 'encoding' is taken by the run's encoding descriptor, encoding.json")
        return value

    @field_validator("formats")
    @classmethod
    def _each_once(cls, value: list[Any]) -> list[Any]:
        """A format named twice, as merging a config folder's files can, is written once."""
        return list(dict.fromkeys(value))

    @field_validator("max_rows")
    @classmethod
    def _lists_some_rows(cls, value: int) -> int:
        """A table listing no row would say only that it left every one out."""
        if value == 0:
            raise ValueError("max_rows must be at least 1, or -1 for every row")
        return value


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

    if isinstance(entry, Mapping) and "steps" in entry:
        if "type" in entry:
            raise ValueError(
                f"Workflow '{entry.get('name', '<unnamed>')}' names both `type` and `steps`. A `workflows:` entry is "
                "a workflow type (`type:`) or a chain of steps (`steps:`), not both."
            )
        return CustomWorkflowConfig.model_validate(entry)
    if isinstance(entry, Mapping) and not isinstance(entry.get("type"), str):
        raise ValueError("Each `workflows:` entry needs a `type:`, or `steps:` for a chain of steps.")
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
_WorkflowEntry = Annotated[SerializeAsAny["_WorkflowBase | CustomWorkflowConfig"], BeforeValidator(_workflow_entry)]
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

    A key that is none of these sections is refused, naming the section it most resembles: a misspelled section
    would otherwise be dropped, and the run go ahead without it.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, extra="forbid")

    # Logging
    logging: LoggingConfig | None = Field(
        default=None,
        description="Log levels for dataeval-flow and the libraries it calls, applied when the CLI runs the pipeline",
    )

    # What each result carries
    result: ResultConfig = Field(
        default_factory=ResultConfig,
        description=(
            "What each task's result carries, and the files and exit code the `dataeval-flow` command makes of it"
        ),
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
    def _refuse_unknown_keys(cls, data: Any) -> Any:
        """Refuse a key that is no section, naming the section it most resembles."""
        if isinstance(data, Mapping):
            problem = unknown_keys_problem(data)
            if problem is not None:
                raise ValueError(problem)
        return data

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
            if isinstance(target, CustomWorkflowConfig):
                problem = target.binding_problem(len(task.source_names))
                if problem is not None:
                    raise ValueError(f"Task '{task.name}' runs workflow '{target.name}', which {problem}")
                continue
            problem = input_problem(
                target, source_count=len(task.source_names), has_extractor=task.extractor is not None
            )
            if problem is not None:
                raise ValueError(f"Task '{task.name}' runs {kind} '{target.name}' ({target.type}), which {problem}")
        return self

    @model_validator(mode="after")
    def _check_custom_workflows(self) -> "PipelineConfig":
        """Refuse a custom workflow whose steps do not connect, and a task it cannot run, before any data is read."""
        from dataeval_flow._chain._graph import build_graph, task_problems

        for workflow in self.workflows or ():
            if isinstance(workflow, CustomWorkflowConfig):
                build_graph(workflow, self)
        problems = task_problems(self)
        if problems:
            raise ValueError(" ".join(problems))
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

    @model_validator(mode="after")
    def _check_per_task_names(self) -> "PipelineConfig":
        """Refuse a task name no file can carry, where ``result: per_task`` names each task's files after it."""
        if self.result.per_task:
            for task in self.tasks or ():
                if "/" in task.name or "\\" in task.name:
                    raise ValueError(
                        f"Task '{task.name}' can't name its result files, as `result: per_task` would: "
                        "a task name there can't hold '/' or '\\'"
                    )
        return self


def top_level_keys() -> frozenset[str]:
    """Every key a pipeline config's top level may hold: its sections, and their legacy aliases."""
    keys = set(PipelineConfig.model_fields)
    for field in PipelineConfig.model_fields.values():
        alias = field.validation_alias
        if isinstance(alias, AliasChoices):
            keys.update(choice for choice in alias.choices if isinstance(choice, str))
        elif isinstance(alias, str):
            keys.add(alias)
    return frozenset(keys)


def unknown_keys_problem(keys: Iterable[Any]) -> str | None:
    """Name each of `keys` that is no top-level section, with the section it most resembles; ``None`` if none is.

    A guess is offered only among the current section names, never a legacy alias.
    """
    known = top_level_keys()
    unknown = [key for key in keys if key not in known]
    if not unknown:
        return None
    sections = sorted(PipelineConfig.model_fields)
    named = []
    for key in unknown:
        guess = difflib.get_close_matches(str(key), sections, n=1)
        named.append(f"'{key}'" + (f" (did you mean '{guess[0]}'?)" if guess else ""))
    noun = "key" if len(unknown) == 1 else "keys"
    return f"Unknown top-level {noun} {', '.join(named)}. The sections are: {', '.join(sections)}."
