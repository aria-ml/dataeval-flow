"""Per-dataset and per-run context a workflow or evaluator executes with."""

__all__ = ["DatasetContext", "ResolvedOntology", "WorkflowContext"]

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dataeval import Metadata, Ontology
    from dataeval.core import ClusterResult, StatsResult
    from dataeval.flags import ImageStats
    from dataeval.protocols import AnnotatedDataset
    from numpy.typing import NDArray

    from dataeval_flow._cache import DatasetCache
    from dataeval_flow._policy import ResolvedPolicy
    from dataeval_flow._stats import ResolvedStatsPolicy
    from dataeval_flow.config._schemas import ViewOperation
    from dataeval_flow.config.extractors._base import ExtractorConfig


@dataclass
class DatasetContext:
    """One of a task's sources, as Flow resolved it: the dataset, its view, and the task's extractor.

    A :class:`WorkflowContext` holds one per source. Read a source through the context's methods: they apply the
    view and use the cache.
    """

    name: str
    """The source's name, as the task names it."""
    dataset: "AnnotatedDataset[Any]"
    """The dataset the source reads, before the source's view is applied."""
    extractor: "ExtractorConfig | None" = None
    """The task's extractor, or ``None`` when the task names none."""
    transforms: Callable | None = None
    """The preprocessing the extractor applies, or ``None`` when it names no preprocessor."""
    view_operations: "Sequence[ViewOperation] | None" = None
    """The operations of the source's view, or ``None`` when it reads through none."""
    batch_size: int | None = None
    """The extractor's batch size, or ``None`` for DataEval's global default."""
    label_source: "str | Sequence[str] | None" = None
    """Where the labels came from: one value, or one per operand where a merged corpus reads more than one
    provenance."""
    value_range: "tuple[float, float] | None" = None
    """The interval the dataset's imagery occupies, as its dataset config declares it, or ``None``."""
    channel_groups: "Mapping[str, tuple[int, ...]] | None" = None
    """Named band groups this dataset declares, taken from the dataset config.

    Read by the stats policy, which selects the groups it measures from these.
    """
    cache: "DatasetCache | None" = None
    """The source's cache, or ``None`` to compute without one."""


@dataclass(frozen=True)
class ResolvedOntology:
    """The label space a task was configured with, resolved before the dataset was read.

    Read :attr:`error` rather than expecting an exception. An ontology problem degrades a
    ``data-coverage`` run to a skip reason and leaves label, metadata and gap analysis
    running, so resolving earlier must not abort the task.
    """

    ontology: "Ontology | None"
    """The loaded ontology, or ``None`` when it failed to load."""
    source: str
    """Where it came from. Loaded: the pool entry's name, the resolved path, ``inline`` or ``concepts``. Failed: the
    workflow's ``ontology`` value as written, as a string."""
    error: str | None = None
    """Why it failed to load, or ``None``."""


@dataclass
class WorkflowContext:
    """What a workflow's ``run`` reads: the task's sources, and cached access to what each yields.

    Flow builds one per task and hands it to :meth:`Workflow.run`. Read a source through :meth:`dataset`,
    :meth:`stats`, :meth:`embeddings`, :meth:`clusters`, :meth:`metadata` and :meth:`labels`, which apply the
    source's view and the task's extractor and policies, and cache what they compute. The fields hold what Flow
    resolved for the task; build a context by hand only to call a workflow's ``run`` directly.
    """

    dataset_contexts: "Mapping[str, DatasetContext]" = field(default_factory=dict)
    """Each source's :class:`DatasetContext`, keyed by source name in the order the task names them."""
    batch_size: int | None = None
    """The batch size of the task's extractor, or ``None`` for DataEval's global default."""
    metadata_policy: "ResolvedPolicy | None" = None
    """How factors become codes, resolved and checked before the dataset was read.

    Carried on the context rather than read off the parameters, because resolving it
    needs the pipeline the policy pool lives on and the data root its descriptor is
    relative to — neither of which a workflow has.  None where the caller built a context
    directly, which takes DataEval's defaults.
    """
    ontology: "ResolvedOntology | None" = None
    """The label space this task names, resolved before the dataset was read.

    Set here rather than in the workflow, for the same reason as :attr:`metadata_policy`:
    resolving a name needs the pipeline holding the pool, resolving a path needs the data
    root, and a workflow has neither. ``None`` when the caller built a context directly or
    configured no ontology. The workflow then reads its own parameters.
    """
    stats_policy: "ResolvedStatsPolicy | None" = None
    """What to measure and which views each consumer reads, resolved before the run.

    Set here rather than in the workflow, for the same reason as :attr:`metadata_policy`:
    resolving a name needs the pipeline holding the pool, and resolving its bands needs the
    datasets. ``None`` when the caller built a context directly or named no policy, in which
    case a workflow measures the whole image with its own flags.
    """

    @property
    def sources(self) -> list[str]:
        """The names of the sources this run reads, in the order the task named them.

        Returns
        -------
        list[str]
            Source names, in task order.
        """
        return list(self.dataset_contexts)

    def _source(self, source: str) -> DatasetContext:
        """The named source's per-dataset context, or raise a listing of the real ones."""
        try:
            return self.dataset_contexts[source]
        except KeyError:
            raise KeyError(f"No source {source!r} in this run; its sources are {self.sources}.") from None

    def _viewed(self, dc: DatasetContext) -> "AnnotatedDataset[Any]":
        """*dc*'s dataset with its view applied, or the dataset itself where it declared none."""
        from dataeval_flow._view import build_view

        return build_view(dc.dataset, list(dc.view_operations)) if dc.view_operations else dc.dataset

    def dataset(self, source: str) -> "AnnotatedDataset[Any]":
        """The source's dataset with its view applied — what every other method here reads.

        Parameters
        ----------
        source : str
            One of :attr:`sources`.

        Returns
        -------
        AnnotatedDataset
            The dataset after the task's ``view:`` operations, or the dataset itself where
            the task declared none.

        Raises
        ------
        KeyError
            When *source* is not one of :attr:`sources`.
        """
        return self._viewed(self._source(source))

    @contextmanager
    def _cached(self, source: str) -> "Iterator[AnnotatedDataset[Any]]":
        """Yield the source's dataset with its cache active, where the source names one."""
        from dataeval_flow._cache import active_cache, selection_repr

        dc = self._source(source)
        dataset = self._viewed(dc)
        with ExitStack() as stack:
            if dc.cache is not None:
                stack.enter_context(active_cache(dc.cache, selection_repr(dataset)))
            yield dataset

    def stats(self, source: str, flags: "ImageStats | None" = None) -> "StatsResult":
        """Image statistics for the source, cached by dataset, view and what is measured.

        Without a declared stats policy (the task's ``stats:`` field), measures *flags* —
        every family by default — directly over the whole image. With a declared policy,
        the policy's own ``measure`` is always what gets computed; *flags* is instead
        *checked* against it, over the whole image, so a caller naming a family the policy
        does not measure gets a ``ValueError`` rather than a result missing that family.
        Passing no *flags* against a declared policy skips that check and reads
        whatever the policy measures.

        Parameters
        ----------
        source : str
            One of :attr:`sources`.
        flags : ImageStats or None, optional
            The families this caller needs. Without a declared stats policy, measured
            directly (every family when omitted). With a declared policy, checked against
            its whole-image ``measure`` rather than requested; the check is skipped when
            omitted.

        Returns
        -------
        StatsResult
            DataEval's stats result for :meth:`dataset` — under the declared policy's own
            ``measure`` where the task names one, or *flags* otherwise.

        Raises
        ------
        KeyError
            When *source* is not one of :attr:`sources`.
        ValueError
            When a declared stats policy does not measure *flags* over the whole image.
        """
        from dataeval.flags import ImageStats

        from dataeval_flow._cache import get_or_compute_stats
        from dataeval_flow._stats import stats_policy_for

        if flags is None:
            policy = stats_policy_for(self, derive_flags=ImageStats.ALL)
        else:
            # `duplicate_flags` is `stats_policy_for`'s vehicle for "check this against a
            # declared policy's whole-image measure, or request it directly where none is
            # declared" — reused here rather than adding a second check path.
            policy = stats_policy_for(
                self,
                duplicate_flags=flags,
                duplicate_declaration="the `flags` passed to `WorkflowContext.stats`",
            )
        with self._cached(source) as dataset:
            return get_or_compute_stats(policy, dataset=dataset, value_range=self._source(source).value_range)

    def embeddings(self, source: str) -> "NDArray[np.float32]":
        """The task's extractor applied to every item in the source, cached.

        Parameters
        ----------
        source : str
            One of :attr:`sources`.

        Returns
        -------
        NDArray[np.float32]
            One row of embeddings per item in :meth:`dataset`.

        Raises
        ------
        KeyError
            When *source* is not one of :attr:`sources`.
        ValueError
            When the task named no extractor for *source*.
        """
        from dataeval_flow._cache import get_or_compute_embeddings

        dc = self._source(source)
        if dc.extractor is None:
            raise ValueError(f"Source {source!r} has no extractor; name one on the task with `extractor:`.")
        with self._cached(source) as dataset:
            return get_or_compute_embeddings(dataset, dc.extractor, dc.transforms, dc.batch_size)

    def clusters(
        self, source: str, *, algorithm: "Literal['kmeans', 'hdbscan'] | None" = None, n_clusters: int | None = None
    ) -> "ClusterResult":
        """Clusters over the source's embeddings, cached.

        Parameters
        ----------
        source : str
            One of :attr:`sources`.
        algorithm : {"kmeans", "hdbscan"} or None, optional
            The clustering algorithm to use. DataEval's own default when omitted.
        n_clusters : int or None, optional
            A target cluster count, for either algorithm. For ``"kmeans"``, the exact number
            to find; defaults to ``int(sqrt(samples))`` when omitted. For ``"hdbscan"``, an
            adaptive hint: it sets ``min_cluster_size`` from the sample count to steer toward
            roughly this many clusters, rather than a hard target.

        Returns
        -------
        ClusterResult
            DataEval's cluster result over :meth:`embeddings`.

        Raises
        ------
        KeyError
            When *source* is not one of :attr:`sources`.
        ValueError
            When the task named no extractor for *source*.
        """
        from dataeval.types import ClusterConfigMixin

        from dataeval_flow._cache import get_or_compute_cluster_result

        dc = self._source(source)
        embeddings = self.embeddings(source)
        chosen = algorithm or ClusterConfigMixin.model_fields["cluster_algorithm"].default
        with self._cached(source):
            return get_or_compute_cluster_result(
                embeddings,
                algorithm=chosen,
                n_clusters=n_clusters,
                extractor_config=dc.extractor,
                transforms=dc.transforms,
                batch_size=dc.batch_size,
            )

    def metadata(self, source: str) -> "Metadata":
        """The source's metadata, cached under this run's metadata policy.

        Parameters
        ----------
        source : str
            One of :attr:`sources`.

        Returns
        -------
        Metadata
            DataEval's ``Metadata`` for :meth:`dataset`, built under the task's
            ``metadata:`` policy — DataEval's own defaults where the task names none.

        Raises
        ------
        KeyError
            When *source* is not one of :attr:`sources`.
        """
        from dataeval_flow._cache import get_or_compute_metadata

        with self._cached(source) as dataset:
            return get_or_compute_metadata(dataset, self.metadata_policy)

    def labels(self, source: str) -> "NDArray[np.intp]":
        """Each item's class label, as :meth:`metadata` reads them.

        Parameters
        ----------
        source : str
            One of :attr:`sources`.

        Returns
        -------
        NDArray[np.intp]
            One class label per item, at the metadata's label level.

        Raises
        ------
        KeyError
            When *source* is not one of :attr:`sources`.
        ValueError
            When the metadata's view sits above its label level, where there is no label
            per row to return.
        """
        return np.asarray(self.metadata(source).class_labels)
