"""The per-source bundle an evaluator's ``run`` receives."""

__all__ = ["Inputs"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval.core import ClusterResult, StatsResult
    from numpy.typing import NDArray

    from dataeval_flow.stats import ResolvedStatsPolicy


@dataclass(frozen=True)
class Inputs:
    """What one source was turned into for this run. Only the kinds the run asked for are set.

    ``stats_policy`` travels with ``stats`` because the views an evaluator reads (for
    outliers, ``outliers_from``) are the policy's to name. ``embeddings`` is set whenever
    ``clusters`` is, since clusters are built from it and DataEval may read both. Later
    phases add ``metadata`` and ``labels`` with the evaluators that read them.
    """

    source: str
    stats: "StatsResult | None" = None
    stats_policy: "ResolvedStatsPolicy | None" = None
    clusters: "ClusterResult | None" = None
    embeddings: "NDArray[Any] | None" = None
