"""The ``data-cleaning`` preset: outliers, duplicates and label health in one dataset, judged by checks, and the
dataset without what they flagged (spec §10)."""

__all__ = ["DataCleaningWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.evaluators.quality import DuplicatesConfig, LabelHealthConfig, OutliersConfig
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.data_cleaning._config import DataCleaningConfig


class DataCleaningWorkflow(Preset, Workflow[DataCleaningConfig, ChainResult]):
    """Finds outliers and duplicates in one dataset, judges them against ``health_thresholds``, and removes them.

    The settings expand to this chain, run on the task's one source, ``data``:

    - ``outliers`` (the ``outliers`` evaluator, per box on detection data), ``labels`` (``label-health``),
      ``by-class`` (``classwise-outliers``) and ``dupes`` (``duplicates``);
    - the checks ``image-outliers``, ``target-outliers``, ``classwise``, ``duplicates`` and ``imbalance``, each
      judged against its ``health_thresholds`` entry;
    - ``clean`` (``remove``): the dataset without each flagged image and box, and without each duplicate but the
      first of its group.

    Run as a step of a custom workflow, ``<step>.clean`` reads the cleaned dataset.
    """

    name: ClassVar[str] = "data-cleaning"
    description: ClassVar[str] = "Outlier and duplicate detection for image datasets, and the dataset without them"
    slots: ClassVar[tuple[str, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = (Port("clean", DataType.DATASET),)

    @classmethod
    def chain(cls, config: DataCleaningConfig) -> PresetChain:
        """The evaluators these settings configure, the checks their thresholds judge by, and the removal."""
        limits = config.health_thresholds
        method: Any = (
            config.outlier_method
            if config.outlier_threshold is None
            else (config.outlier_method, config.outlier_threshold)
        )
        evaluators = [
            OutliersConfig(
                name="outliers",
                flags=list(config.outlier_flags),
                outlier_threshold=method,
                cluster_threshold=config.outlier_cluster_threshold,
                cluster_algorithm=config.outlier_cluster_algorithm,
                n_clusters=config.outlier_n_clusters,
                per_target=True,
                stats=config.stats,
            ),
            DuplicatesConfig(
                name="dupes",
                flags=list(config.duplicate_flags) if config.duplicate_flags is not None else None,
                merge_near_duplicates=config.duplicate_merge_near,
                cluster_sensitivity=config.duplicate_cluster_sensitivity,
                cluster_algorithm=config.duplicate_cluster_algorithm,
                n_clusters=config.duplicate_n_clusters,
                stats=config.stats,
            ),
            LabelHealthConfig(name="labels", metadata=config.metadata),
        ]
        steps: list[dict[str, Any]] = [
            {"name": "outliers", "evaluator": "outliers", "input": "data"},
            {"name": "labels", "evaluator": "labels", "input": "data"},
            {"name": "by-class", "combine": "classwise-outliers", "input": "data", "outliers": "outliers"},
            {"name": "dupes", "evaluator": "dupes", "input": "data"},
            {"name": "image-outliers", "check": "outlier-rate", "input": "outliers", "image": limits.image_outliers},
            {
                "name": "target-outliers",
                "check": "target-outlier-rate",
                "input": "outliers",
                "labels": "labels",
                "target": limits.target_outliers,
            },
            {
                "name": "classwise",
                "check": "classwise-outlier-rate",
                "input": "by-class",
                "total": limits.classwise_outliers,
            },
            {
                "name": "duplicates",
                "check": "duplicate-rate",
                "input": "dupes",
                "exact": limits.exact_duplicates,
                "near": limits.near_duplicates,
            },
            {"name": "imbalance", "check": "class-imbalance", "input": "labels", "ratio": limits.class_label_imbalance},
            {
                "name": "clean",
                "transform": "remove",
                "input": "data",
                "plans": {"dupes": {"dup_types": ["exact", "near"], "keep": "first"}, "outliers": {"min_flags": 1}},
            },
        ]
        return PresetChain(steps=steps, evaluators=evaluators)
