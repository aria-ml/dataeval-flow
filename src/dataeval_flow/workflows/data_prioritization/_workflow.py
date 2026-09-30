"""The ``data-prioritization`` preset: each pool ranked against a reference, after optional cleaning, and the top of
each ranking kept (spec §10.9)."""

__all__ = ["DataPrioritizationWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.evaluators.quality import DuplicatesConfig, OutliersConfig
from dataeval_flow.evaluators.scope import PrioritizeConfig
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.data_prioritization._config import (
    DataPrioritizationCleaningConfig,
    DataPrioritizationConfig,
)


class DataPrioritizationWorkflow(Preset, Workflow[DataPrioritizationConfig, ChainResult]):
    """Ranks each pool against the reference, after dropping outliers and duplicates when ``cleaning`` is set, and
    keeps the top of each ranking.

    The task's first source is ``reference``; every later one is an element of ``pools``. The settings expand to:

    - with ``cleaning`` set, ``reference-outliers`` and ``pool-outliers`` (``outliers``), ``reference-dupes`` and
      ``pool-dupes`` (``duplicates``), then ``reference-clean`` and ``pool-clean`` (``remove``): each source without
      its outliers, and without each duplicate but the first of its group;
    - ``rank`` (``prioritize``): each pool, cleaned or not, ranked against the reference;
    - ``selected`` (``select``): the first ``n``, or ``fraction``, of each pool's ranking; all of it when neither is
      set.

    Each pool step runs once per pool. Run as a step of a custom workflow, ``<step>.selected`` reads the selection, a
    list keyed by pool.
    """

    name: ClassVar[str] = "data-prioritization"
    title: ClassVar[str] = "Data Prioritization"
    description: ClassVar[str] = (
        "Ranks each pool against a reference for labeling, after optional cleaning, and keeps the top"
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = (
        "reference",
        InputSlot.model_validate({"name": "pools", "list": True}),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("selected", DataType.DATASET),)

    @classmethod
    def chain(cls, config: DataPrioritizationConfig) -> PresetChain:
        """The cleaning steps ``cleaning`` configures, the ranking, and the selection."""
        evaluators: list[Any] = [
            PrioritizeConfig(
                name="rank",
                method=config.method,
                k=config.k,
                c=config.c,
                n_init=config.n_init,
                max_cluster_size=config.max_cluster_size,
                order=config.order,
                policy=config.policy,
                num_bins=config.num_bins,
            )
        ]
        steps: list[dict[str, Any]] = []
        reference, pools = "reference", "pools"
        if config.cleaning is not None:
            evaluators += _cleaning_evaluators(config.cleaning, config)
            steps += _cleaning_steps("reference", "reference", config.cleaning)
            steps += _cleaning_steps("pool", "pools", config.cleaning)
            reference, pools = "reference-clean", "pool-clean"
        amount: dict[str, Any] = {"n": config.n} if config.n is not None else {"fraction": config.fraction or 1.0}
        steps += [
            {"name": "rank", "evaluator": "rank", "input": [pools, reference]},
            {"name": "selected", "transform": "select", "input": pools, "ranking": "rank", **amount},
        ]
        return PresetChain(steps=steps, evaluators=evaluators)


def _cleaning_evaluators(cleaning: DataPrioritizationCleaningConfig, config: DataPrioritizationConfig) -> list[Any]:
    """The ``outliers`` and ``dupes`` entries the cleaning steps name, with the cleaning block's settings."""
    method: Any = (
        cleaning.outlier_method
        if cleaning.outlier_threshold is None
        else (cleaning.outlier_method, cleaning.outlier_threshold)
    )
    return [
        OutliersConfig(
            name="outliers", flags=list(cleaning.outlier_flags), outlier_threshold=method, stats=config.stats
        ),
        DuplicatesConfig(
            name="dupes",
            flags=list(cleaning.duplicate_flags) if cleaning.duplicate_flags is not None else None,
            merge_near_duplicates=cleaning.duplicate_merge_near,
            stats=config.stats,
        ),
    ]


def _cleaning_steps(prefix: str, source: str, cleaning: DataPrioritizationCleaningConfig) -> list[dict[str, Any]]:
    """``source`` without its outliers and duplicates, as steps ``<prefix>-outliers``, ``-dupes`` and ``-clean``."""
    dup_types = ["exact"] if cleaning.duplicate_exact_only else ["exact", "near"]
    return [
        {"name": f"{prefix}-outliers", "evaluator": "outliers", "input": source},
        {"name": f"{prefix}-dupes", "evaluator": "dupes", "input": source},
        {
            "name": f"{prefix}-clean",
            "transform": "remove",
            "input": source,
            "plans": {
                f"{prefix}-dupes": {"dup_types": dup_types, "keep": "first"},
                f"{prefix}-outliers": {"min_flags": 1},
            },
        },
    ]
