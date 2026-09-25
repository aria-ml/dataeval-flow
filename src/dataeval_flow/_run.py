"""``run``: one workflow or evaluator on datasets already in memory."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

__all__ = ["run"]

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset, FeatureExtractor

    from dataeval_flow.config._schemas._metadata import MetadataPolicyConfig
    from dataeval_flow.config._schemas._ontology import OntologyConfig
    from dataeval_flow.config._schemas._preprocessor import PreprocessorConfig
    from dataeval_flow.config._schemas._stats import StatsPolicyConfig
    from dataeval_flow.config.extractors._base import ExtractorConfig
    from dataeval_flow.evaluators._base import EvaluatorConfig
    from dataeval_flow.workflows._base import WorkflowConfig

R = TypeVar("R")


def run(
    config: "WorkflowConfig[R] | EvaluatorConfig[R]",
    data: "AnnotatedDataset[Any] | Mapping[str, AnnotatedDataset[Any]]",
    *,
    extractor: "ExtractorConfig | FeatureExtractor | None" = None,
    definitions: "Sequence[MetadataPolicyConfig | StatsPolicyConfig | OntologyConfig | PreprocessorConfig]" = (),
    cache_dir: Path | None = None,
) -> R:
    """Run one workflow or evaluator on datasets already in memory.

    The data is checked against what `config` consumes before anything runs. The run then goes through
    :func:`~dataeval_flow.run_task` as a one-task pipeline, so it returns what that pipeline's task would.

    Parameters
    ----------
    config : WorkflowConfig or EvaluatorConfig
        What to run. Its type parameter decides the result type returned.
    data : AnnotatedDataset or Mapping[str, AnnotatedDataset]
        One dataset, read as the source ``"dataset"``, or source names mapped to datasets in the order the
        workflow reads them — reference first for multi-source workflows.
    extractor : ExtractorConfig or FeatureExtractor, optional
        An extractor config, whose embeddings are disk-cached like any pipeline's, or any object satisfying
        DataEval's ``FeatureExtractor`` protocol, whose embeddings and clusters are never cached: it has no stable
        cache key. An object without its own ``batch_size`` runs at DataEval's global batch size.
    definitions : Sequence of MetadataPolicyConfig, StatsPolicyConfig, OntologyConfig or PreprocessorConfig
        The named policies, ontologies and preprocessors `config` and `extractor` refer to by name.
    cache_dir : Path, optional
        Directory for the disk cache. ``None`` keeps the cache in memory.

    Returns
    -------
    R
        The config's result class, e.g. ``DuplicatesResult`` for a ``DuplicatesConfig``. A run that raised
        returns a failed result of that class.

    Raises
    ------
    ValueError
        When `data` is an empty mapping; when a policy or preprocessor that `config` or `extractor` names is not
        among `definitions`; or when no built-in and no installed entry point registers `config`'s type, as with a
        plugin class defined in a notebook: Flow finds a workflow or evaluator by its type id alone.
    TypeError
        When a definition is none of the four types `definitions` takes.
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
        from dataeval_flow.workflows.drift_monitoring import DriftDetectorMMD, DriftMonitoringConfig

        drift = run(
            DriftMonitoringConfig(detectors=[DriftDetectorMMD()]),
            {"reference": train, "test": incoming},
            extractor=FlattenExtractorConfig(batch_size=64),
        )
    """
    from dataeval_flow._orchestrator import run_task
    from dataeval_flow.config._models import PipelineConfig, SourceConfig
    from dataeval_flow.config._schemas._dataset import DatasetProtocolConfig
    from dataeval_flow.config._schemas._task import TaskConfig
    from dataeval_flow.config.extractors._base import ExtractorConfig, _InstanceExtractorConfig
    from dataeval_flow.evaluators._base import EvaluatorConfig

    datasets: dict[str, AnnotatedDataset[Any]] = dict(data) if isinstance(data, Mapping) else {"dataset": data}
    if not datasets:
        raise ValueError("run() needs at least one dataset.")
    pools = _pools(definitions)
    extractors: list[ExtractorConfig] = []
    if extractor is not None:
        extractors.append(
            extractor
            if isinstance(extractor, ExtractorConfig)
            else _InstanceExtractorConfig(name="extractor", extractor=extractor)
        )
    if isinstance(config, EvaluatorConfig):
        kind, workflows, evaluators = "evaluator", None, [config]
    else:
        kind, workflows, evaluators = "workflow", [config], None
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
        extractors=extractors or None,
        workflows=workflows,
        evaluators=evaluators,
        tasks=[task],
        metadata=pools.get("metadata"),
        stats=pools.get("stats"),
        ontologies=pools.get("ontologies"),
        preprocessors=pools.get("preprocessors"),
    )
    return cast("R", run_task(task, pipeline, cache_dir=cache_dir))


def _pools(definitions: Sequence[object]) -> dict[str, list[Any]]:
    """Sort `definitions` into the ``PipelineConfig`` pools they belong to, keyed by field name."""
    from dataeval_flow.config._schemas._metadata import MetadataPolicyConfig
    from dataeval_flow.config._schemas._ontology import OntologyConfig
    from dataeval_flow.config._schemas._preprocessor import PreprocessorConfig
    from dataeval_flow.config._schemas._stats import StatsPolicyConfig

    fields: dict[type, str] = {
        MetadataPolicyConfig: "metadata",
        StatsPolicyConfig: "stats",
        OntologyConfig: "ontologies",
        PreprocessorConfig: "preprocessors",
    }
    pools: dict[str, list[Any]] = {}
    for definition in definitions:
        field = next((name for cls, name in fields.items() if isinstance(definition, cls)), None)
        if field is None:
            raise TypeError(
                f"run() cannot use a {type(definition).__name__} as a definition; it takes MetadataPolicyConfig, "
                "StatsPolicyConfig, OntologyConfig and PreprocessorConfig."
            )
        pools.setdefault(field, []).append(definition)
    return pools
