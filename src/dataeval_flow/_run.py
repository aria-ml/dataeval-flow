"""``run``: one workflow, evaluator or custom workflow on datasets already in memory."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

__all__ = ["run"]

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset, FeatureExtractor

    from dataeval_flow.config._definitions import Definition as _Definition
    from dataeval_flow.config.extractors._base import ExtractorConfig
    from dataeval_flow.evaluators._base import EvaluatorConfig
    from dataeval_flow.steps._result import ChainResult
    from dataeval_flow.steps._workflow import CustomWorkflowConfig
    from dataeval_flow.workflows._base import WorkflowConfig

R = TypeVar("R")


@overload
def run(
    config: "CustomWorkflowConfig",
    data: "AnnotatedDataset[Any] | Mapping[str, AnnotatedDataset[Any]]",
    *,
    extractor: "ExtractorConfig | FeatureExtractor | None" = None,
    definitions: "Sequence[_Definition]" = (),
    cache_dir: Path | None = None,
    report_images: bool = True,
    output_dir: Path | None = None,
) -> "ChainResult": ...


@overload
def run(
    config: "WorkflowConfig[R] | EvaluatorConfig[R]",
    data: "AnnotatedDataset[Any] | Mapping[str, AnnotatedDataset[Any]]",
    *,
    extractor: "ExtractorConfig | FeatureExtractor | None" = None,
    definitions: "Sequence[_Definition]" = (),
    cache_dir: Path | None = None,
    report_images: bool = True,
    output_dir: Path | None = None,
) -> R: ...


def run(
    config: "WorkflowConfig[R] | EvaluatorConfig[R] | CustomWorkflowConfig",
    data: "AnnotatedDataset[Any] | Mapping[str, AnnotatedDataset[Any]]",
    *,
    extractor: "ExtractorConfig | FeatureExtractor | None" = None,
    definitions: "Sequence[_Definition]" = (),
    cache_dir: Path | None = None,
    report_images: bool = True,
    output_dir: Path | None = None,
) -> "R | ChainResult":
    """Run one workflow, evaluator or custom workflow on datasets already in memory.

    The data is checked against what `config` consumes before anything runs. The run then goes through
    :func:`~dataeval_flow.run_task` as a one-task pipeline, so it returns what that pipeline's task would.

    The compute device is the one chosen with :func:`~dataeval_flow.set_device`, else CUDA where PyTorch sees a GPU,
    otherwise the CPU. It is applied before each task, so it overrides one set with ``dataeval.config.set_device``.

    Parameters
    ----------
    config : WorkflowConfig, EvaluatorConfig or CustomWorkflowConfig
        What to run. A workflow's or an evaluator's type parameter decides the result type returned; a custom
        workflow returns a :class:`~dataeval_flow.steps.ChainResult`.
    data : AnnotatedDataset or Mapping[str, AnnotatedDataset]
        One dataset, read as the source ``"dataset"``, or source names mapped to datasets in the order the
        workflow reads them — reference first for multi-source workflows, and in input order for a custom
        workflow.
    extractor : ExtractorConfig or FeatureExtractor, optional
        An extractor config, whose embeddings are disk-cached like any pipeline's, or any object satisfying
        DataEval's ``FeatureExtractor`` protocol, whose embeddings and clusters are never cached: it has no stable
        cache key. An object without its own ``batch_size`` runs at DataEval's global batch size.
    definitions : Sequence of named config entries
        The entries `config` and `extractor` refer to by name: ``MetadataPolicyConfig``, ``StatsPolicyConfig``,
        ``OntologyConfig`` and ``PreprocessorConfig``; and, for a custom workflow, the ``EvaluatorConfig``,
        ``WorkflowConfig``, ``ViewConfig`` and ``ExtractorConfig`` entries its steps name.
    cache_dir : Path, optional
        Directory for the disk cache. ``None`` keeps the cache in memory.
    report_images : bool
        Whether the result keeps thumbnails of the items its report names, for its HTML report. ``False``
        reads no item and keeps none.
    output_dir : Path, optional
        Where export steps write, under ``<output_dir>/datasets/``. ``None`` writes nothing, and export steps are
        skipped with a reason.

    Returns
    -------
    R or ChainResult
        The config's result class, e.g. ``DuplicatesResult`` for a ``DuplicatesConfig``. A run that raised
        returns a failed result of that class. A custom workflow returns a
        :class:`~dataeval_flow.steps.ChainResult`, holding every step's outcome whether or not one failed.

    Raises
    ------
    ValueError
        When `data` is an empty mapping; when a policy or preprocessor that `config` or `extractor` names is not
        among `definitions`; or when no built-in and no installed entry point registers `config`'s type, as with a
        plugin class defined in a notebook: Flow finds a workflow or evaluator by its type id alone.
    TypeError
        When a definition is none of the types `definitions` takes.
    pydantic.ValidationError
        When the data does not meet the config's inputs (source count, extractor), before anything runs.

    Examples
    --------
    >>> from dataeval_flow import run
    >>> from dataeval_flow.evaluators.quality import DuplicatesConfig
    >>> result = run(DuplicatesConfig(), dataset)  # doctest: +SKIP
    >>> result.output.aggregate_by_image()  # doctest: +SKIP

    Several sources, in the order the workflow reads them::

        from dataeval_flow.config.extractors import FlattenExtractorConfig
        from dataeval_flow.evaluators.shift import DriftMMDConfig
        from dataeval_flow.workflows.shift import ShiftConfig

        drift = run(
            ShiftConfig(detectors=[DriftMMDConfig()]),
            {"reference": train, "test": incoming},
            extractor=FlattenExtractorConfig(batch_size=64),
        )
    """
    from dataeval_flow._orchestrator import _run_single_task
    from dataeval_flow.config._definitions import definition_pools
    from dataeval_flow.config._models import PipelineConfig, ResultConfig, SourceConfig
    from dataeval_flow.config._schemas._dataset import DatasetProtocolConfig
    from dataeval_flow.config._schemas._task import TaskConfig
    from dataeval_flow.config.extractors._base import ExtractorConfig, _InstanceExtractorConfig
    from dataeval_flow.evaluators._base import EvaluatorConfig

    datasets: dict[str, AnnotatedDataset[Any]] = dict(data) if isinstance(data, Mapping) else {"dataset": data}
    if not datasets:
        raise ValueError("run() needs at least one dataset.")
    pools = definition_pools(definitions, caller="run()")
    extractors: list[ExtractorConfig] = []
    if extractor is not None:
        extractors.append(
            extractor
            if isinstance(extractor, ExtractorConfig)
            else _InstanceExtractorConfig(name="extractor", extractor=extractor)
        )
    # `config` joins its pool beside the definitions: a custom workflow's steps name entries of both.
    if isinstance(config, EvaluatorConfig):
        kind, workflows, evaluators = "evaluator", pools.get("workflows"), [config, *pools.get("evaluators", [])]
    else:
        kind, workflows, evaluators = "workflow", [config, *pools.get("workflows", [])], pools.get("evaluators")
    task = TaskConfig(
        name=config.name,
        workflow=config.name,
        kind=kind,
        sources=list(datasets),
        extractor=extractors[0].name if extractors else None,
    )
    pipeline = PipelineConfig(
        datasets=[
            DatasetProtocolConfig(name=name, format="maite", dataset=dataset) for name, dataset in datasets.items()
        ],
        sources=[SourceConfig(name=name, dataset=name) for name in datasets],
        extractors=[*extractors, *pools.get("extractors", [])] or None,
        workflows=workflows,
        evaluators=evaluators,
        tasks=[task],
        metadata=pools.get("metadata"),
        stats=pools.get("stats"),
        ontologies=pools.get("ontologies"),
        preprocessors=pools.get("preprocessors"),
        views=pools.get("views"),
        result=ResultConfig() if report_images else ResultConfig(max_images=0),
    )
    return cast(
        "R | ChainResult",
        _run_single_task(task, pipeline, cache_dir=cache_dir, output_dir=output_dir),
    )
