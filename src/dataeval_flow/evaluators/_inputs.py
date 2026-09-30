"""The per-source bundle an evaluator's ``run`` receives."""

__all__ = ["EvaluatorInputs"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    from dataeval import Metadata, Ontology
    from dataeval.core import ClusterResult, StatsResult
    from numpy.typing import NDArray

    from dataeval_flow._policy import ResolvedPolicy
    from dataeval_flow._stats import ResolvedStatsPolicy


@dataclass(frozen=True)
class EvaluatorInputs:
    """What Flow prepared from one source for an evaluator's run. Only the kinds the run wants are set.

    :meth:`Evaluator.run` receives one per source, in the order the task names the sources. Flow builds them after
    applying the source's view; an evaluator only reads them. Each input kind the run wants sets its fields:
    ``stats`` sets ``stats`` and ``stats_policy``; ``clusters`` sets ``clusters`` and ``embeddings``;
    ``embeddings`` sets ``embeddings``; ``metadata`` sets ``metadata`` and ``metadata_policy``; ``labels`` sets
    ``labels`` and ``index2label``. ``ontology`` and ``ontology_source`` are the task's, set on every source whatever
    the run wants. ``label_source`` is the source's, set on every source whatever the run wants.
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
    """The source's embeddings from the task's extractor, one row per item, when the run wants ``embeddings`` or
    ``clusters``: clusters are built from them, and DataEval may read both."""
    metadata: "Metadata | None" = None
    """DataEval's ``Metadata`` for the source, built under the task's metadata policy, when the run wants
    ``metadata``."""
    metadata_policy: "ResolvedPolicy | None" = None
    """The metadata policy ``metadata`` was built under, set with ``metadata``; ``None`` where the task names none and
    DataEval's defaults applied. An evaluator may read its ``factor_source``: how the bias statistics read each factor.
    Its other attributes are internal."""
    labels: "NDArray[np.intp] | None" = None
    """The source's class labels, as its metadata reads them, when the run wants ``labels``: one per item for an
    image-classification dataset, one per target for a detection dataset, and empty for a dataset whose targets carry
    none."""
    index2label: "Mapping[int, str] | None" = None
    """The source's class names by label, set with ``labels``; empty where the dataset declares none."""
    ontology: "Ontology | None" = None
    """The ontology the task's ``ontology:`` names, resolved; ``None`` where the task names none. The same on every
    source."""
    ontology_source: str | None = None
    """How the task named ``ontology``: the ``ontologies:`` entry's name, the resolved path, ``inline`` or
    ``concepts``; set with ``ontology``."""
    label_source: "str | Sequence[str] | None" = None
    """Where the source's labels came from, such as ``filepath`` or ``annotations``; one per operand for a merged
    source, and ``None`` where the source does not say."""
