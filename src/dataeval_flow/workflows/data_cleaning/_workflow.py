"""The ``data-cleaning`` preset: outliers, duplicates and label health in one dataset, judged by checks, and the
dataset without what they flagged (spec §10)."""

__all__ = ["DataCleaningWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.evaluators.quality import DuplicatesConfig, LabelHealthConfig, OutliersConfig
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.data_cleaning._config import DataCleaningConfig


class DataCleaningWorkflow(Preset, Workflow[DataCleaningConfig, ChainResult]):
    """Finds outliers and duplicates in one dataset, judges them against ``checks``, and removes them.

    The settings expand to this chain, run on the task's one source, ``data``:

    - ``outliers`` (the ``outliers`` evaluator, per box on detection data), ``label-health``,
      ``outliers-by-class`` and ``duplicates``;
    - the checks ``image-outliers``, ``target-outliers``, ``classwise-outliers``, ``image-duplicates`` and
      ``class-imbalance``, each judged
      against its ``checks`` entry;
    - ``clean`` (``remove``): the dataset without each flagged image and box, and without each duplicate but the
      first of its group.

    Run as a step of a custom workflow, ``<step>.clean`` reads the cleaned dataset.
    """

    name: ClassVar[str] = "data-cleaning"
    title: ClassVar[str] = "Data Cleaning"
    description: ClassVar[str] = "Outlier and duplicate detection for image datasets, and the dataset without them."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = (Port("clean", DataType.DATASET),)

    @classmethod
    def chain(cls, config: DataCleaningConfig) -> PresetChain:
        """The evaluators these settings configure, the checks their thresholds judge by, and the removal."""
        limits = config.checks
        outliers, duplicates = config.outliers, config.duplicates
        evaluators = [
            OutliersConfig(
                name="outliers",
                flags=list(outliers.flags),
                outlier_threshold=outliers.outlier_threshold,
                cluster_threshold=outliers.cluster_threshold,
                cluster_algorithm=outliers.cluster_algorithm,
                n_clusters=outliers.n_clusters,
                per_target=True,
                stats=config.stats,
            ),
            DuplicatesConfig(
                name="duplicates",
                flags=list(duplicates.flags) if duplicates.flags is not None else None,
                merge_near_duplicates=duplicates.merge_near_duplicates,
                cluster_sensitivity=duplicates.cluster_sensitivity,
                cluster_algorithm=duplicates.cluster_algorithm,
                n_clusters=duplicates.n_clusters,
                stats=config.stats,
            ),
            LabelHealthConfig(name="label-health", metadata=config.metadata),
        ]
        steps: list[dict[str, Any]] = [
            {"name": "outliers", "evaluator": "outliers", "input": "data"},
            {"name": "label-health", "evaluator": "label-health", "input": "data"},
            {"name": "outliers-by-class", "combine": "outliers-by-class", "input": "data", "outliers": "outliers"},
            {"name": "duplicates", "evaluator": "duplicates", "input": "data"},
            {
                "name": "image-outliers",
                "check": "image-outliers",
                "input": "outliers",
                **limits.image_outliers.model_dump(),
            },
            {
                "name": "target-outliers",
                "check": "target-outliers",
                "input": "outliers",
                "labels": "label-health",
                **limits.target_outliers.model_dump(),
            },
            {
                "name": "classwise-outliers",
                "check": "classwise-outliers",
                "input": "outliers-by-class",
                **limits.classwise_outliers.model_dump(),
            },
            {
                "name": "image-duplicates",
                "check": "image-duplicates",
                "input": "duplicates",
                **limits.image_duplicates.model_dump(),
            },
            {
                "name": "class-imbalance",
                "check": "class-imbalance",
                "input": "label-health",
                **limits.class_imbalance.model_dump(),
            },
            {
                "name": "clean",
                "transform": "remove",
                "input": "data",
                "plans": {
                    "duplicates": {"dup_types": ["exact", "near"], "keep": "first"},
                    "outliers": {"min_flags": 1},
                },
            },
        ]
        return PresetChain(steps=steps, evaluators=evaluators)
