"""The per-source bundle an evaluator's ``run`` receives."""

__all__ = ["EvaluatorInputs"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval.core import ClusterResult, StatsResult
    from numpy.typing import NDArray

    from dataeval_flow._stats import ResolvedStatsPolicy


@dataclass(frozen=True)
class EvaluatorInputs:
    """What Flow prepared from one source for an evaluator's run. Only the kinds the run wants are set.

    :meth:`Evaluator.run` receives one per source, in the order the task names the sources. Flow builds them after
    applying the source's view; an evaluator only reads them. They carry ``stats`` (with the stats policy they
    were measured under) and ``clusters`` (with the embeddings they were built from) only: a run that wants another
    kind fails, saying no producer exists for it.
    """

    source: str
    """The source's name, as the task names it."""
    stats: "StatsResult | None" = None
    """DataEval's image statistics for the source, when the run wants ``stats``."""
    stats_policy: "ResolvedStatsPolicy | None" = None
    """The stats policy ``stats`` was measured under, set with ``stats``.

    It travels with the statistics because the views an evaluator reads are the policy's to name. An evaluator may
    read two of its attributes, each a tuple of views: ``outliers_from``, the views outlier tests read, and
    ``factors_from``, the views metadata factors read. A view is ``None`` for the whole image, whose columns in
    ``stats`` are named by the statistic alone, or a band group or ``"background"``, whose columns are named
    ``<view>_<statistic>``. ``stats`` may also hold columns another reader of the source had measured, so an outliers
    evaluator keeps only the columns of the views ``outliers_from`` names. The policy's other attributes are
    internal.
    """
    clusters: "ClusterResult | None" = None
    """DataEval's clusters over the source's embeddings, when the run wants ``clusters``."""
    embeddings: "NDArray[Any] | None" = None
    """The source's embeddings, one row per item, set with ``clusters``: clusters are built from them, and DataEval
    may read both."""
