"""Embeddings convenience builder wrapping DataEval."""

__all__ = [
    "build_embeddings",
    "build_extractor",
    "claim_fitter",
    "fit_identity",
    "is_stateful_extractor",
    "mark_fitted",
    "new_extractor_scope",
    "node_embeddings",
    "reuse_within_task",
    "shared_extractor_scope",
]

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from dataeval import Embeddings
from dataeval.protocols import AnnotatedDataset

from dataeval_flow.config.extractors._base import _InstanceExtractorConfig
from dataeval_flow.config.extractors._registry import get_extractor

_logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dataeval_flow.config.extractors._base import ExtractorConfig
    from dataeval_flow.workflows._context import DatasetContext

# The task scope: one entry per extractor identity (the instance every source shares, and
# the selection it is fitted on), and the stateful results computed without a seed.
_task_scope: ContextVar["_Scope | None"] = ContextVar("dataeval_flow_task_scope", default=None)

# The stateful extractors this process has said it will not cache without a seed; said once each.
_unseeded_noted: set[str] = set()

T = TypeVar("T")


@dataclass(frozen=True)
class _Fitter:
    """The selection a shared stateful extractor is fitted on, and what fitting it there takes."""

    # Names the selection: its dataset's cache identity and its selection key.
    key: str
    dataset: AnnotatedDataset[Any]
    batch_size: int | None


class _SharedExtractor:
    """A stateful extractor every source in a task shares, and the selection it is fitted on.

    The fitter is claimed when a source first asks for embeddings, before anything is built or
    read from the cache, so every source's cache key can name it. The extractor is built on first
    use, which may come after the fitter's embeddings were served from the cache; ``fitted``
    records whether it has seen the fitter's data yet.
    """

    __slots__ = ("extractor", "fitted", "fitter")

    def __init__(self) -> None:
        self.extractor: Callable | None = None
        self.fitter: _Fitter | None = None
        self.fitted = False


class _Scope:
    """One task's shared stateful extractors, and the results they computed without a seed."""

    __slots__ = ("entries", "results")

    def __init__(self) -> None:
        self.entries: dict[str, _SharedExtractor] = {}
        # Without a seed every fit learns a different vocabulary, so a stateful result is only valid beside
        # results from the same fit: it is reused within this task and dropped with it.
        self.results: dict[str, Any] = {}


def is_stateful_extractor(extractor_config: "ExtractorConfig") -> bool:
    """Whether this extractor's output depends on data it has already seen: its extractor's ``stateful``.

    BoVW derives a visual vocabulary from the images it is first given and then describes
    everything else in it, so two instances built from one config are not interchangeable:
    each clusters its own codebook and the histograms they produce share no basis. The
    pretrained extractors carry their representation with them and have no such state.
    An extractor given as an object is used as it is.
    """
    if isinstance(extractor_config, _InstanceExtractorConfig):
        return False
    return get_extractor(extractor_config.model).stateful


def new_extractor_scope() -> _Scope:
    """A fresh scope for :func:`shared_extractor_scope` to hold: a task matrix keeps one per ``sources`` value."""
    return _Scope()


@contextmanager
def shared_extractor_scope(scope: "_Scope | None" = None) -> Iterator[None]:
    """Build and fit each stateful extractor once for everything inside this scope.

    A task compares sources with each other, so they have to be described the same way.
    The first source to ask for embeddings is the fitter: the extractor is fitted on its data
    and describes the rest in that fit. The orchestrator opens one scope per task. A task matrix
    passes the `scope` its runs over one ``sources`` value share; a scope opened inside an open one joins it.
    """
    if scope is None and _task_scope.get() is not None:
        yield
        return
    token = _task_scope.set(scope if scope is not None else _Scope())
    try:
        yield
    finally:
        _task_scope.reset(token)


def _fit_seed() -> int | None:
    """The seed a stateful extractor fits under: DataEval's, which BoVW's k-means reads. None when unseeded.

    A fit is reproducible only under a seed. Without one, refitting on the same data learns a
    different vocabulary, so nothing but the fit itself can name what it produced.
    """
    from dataeval.config import get_seed

    return get_seed()


def _fitting_batch(batch_size: int | None) -> int | None:
    """The batch size that picks the images a stateful extractor fits on: `batch_size`, else DataEval's global.

    A stateful extractor fits on the first batch it is given, so the batch size decides which
    images fit it. None when neither is set, where the embedding pass itself refuses to run.
    """
    if batch_size is not None:
        return batch_size
    from dataeval.config import get_batch_size

    try:
        return get_batch_size()
    except ValueError:
        return None


def reuse_within_task(key: str, compute: Callable[[], T], model: str) -> T:
    """Compute an unseeded stateful result, reusing it within this task and never beyond it.

    Without a seed each fit learns a different vocabulary, so a result can only be served beside
    results from the fit that computed it. It is kept for the task that computed it; outside a
    task nothing is kept, so each call computes afresh.
    """
    if model not in _unseeded_noted:
        _unseeded_noted.add(model)
        _logger.warning(
            "Embeddings from the '%s' extractor are recomputed for every task, because no seed is set and each "
            "unseeded fit learns a different vocabulary. To cache and reuse them, set the pipeline's `seed:`, or "
            "call `dataeval.config.set_seed` before `run()`.",
            model,
        )
    scope = _task_scope.get()
    if scope is None:
        return compute()
    if key not in scope.results:
        scope.results[key] = compute()
    return scope.results[key]


def _shared_entry(
    extractor_config: "ExtractorConfig", transforms: Callable | None, *, create: bool
) -> "_SharedExtractor | None":
    """This scope's entry for a stateful extractor, or None outside a scope or for a stateless one."""
    scope = _task_scope.get()
    if scope is None or not is_stateful_extractor(extractor_config):
        return None
    key = _shared_key(extractor_config, transforms)
    return scope.entries.setdefault(key, _SharedExtractor()) if create else scope.entries.get(key)


def claim_fitter(
    extractor_config: "ExtractorConfig",
    transforms: Callable | None,
    selection: str,
    dataset: AnnotatedDataset[Any],
    batch_size: int | None,
) -> str:
    """The selection a stateful extractor describing `selection` is fitted on, claiming it if none is yet.

    Called when `selection` asks for embeddings, before anything is built or read from the
    cache. Within a scope the first selection to ask claims the fitter, with the dataset and
    batch size that fit the extractor on it; every later one is described in that fit. Outside
    a scope nothing is shared, so a selection fits itself.

    Callers fold the result into their cache key: the extractor config alone cannot tell a
    vocabulary fitted on the reference from one fitted anywhere else.
    """
    entry = _shared_entry(extractor_config, transforms, create=True)
    if entry is None:
        return selection
    if entry.fitter is None:
        entry.fitter = _Fitter(key=selection, dataset=dataset, batch_size=batch_size)
    return entry.fitter.key


def fit_identity(
    extractor_config: "ExtractorConfig", transforms: Callable | None, selection: str, batch_size: int | None
) -> str | None:
    """Name the fit of a stateful extractor describing `selection`, as a cache key suffix; None without a seed.

    A fit is what the extractor learned: from which data (the selection it is fitted on, the
    claimed fitter within a task), from which of its images (the batch size that picks the
    first batch) and under which seed. Refitting reproduces it only when all three match. Without
    a seed no refit does, so nothing names the fit, and its results are kept only for the task
    that computed them (:func:`reuse_within_task`).

    `batch_size` is the one `selection` is embedded with; within a task the fitter's own is used.
    Claims nothing: call :func:`claim_fitter` first where embeddings are requested.
    """
    seed = _fit_seed()
    if seed is None:
        return None
    entry = _shared_entry(extractor_config, transforms, create=False)
    if entry is not None and entry.fitter is not None:
        selection, batch_size = entry.fitter.key, entry.fitter.batch_size
    return f"fitted_on={selection}|batch={_fitting_batch(batch_size)}|seed={seed}"


def build_embeddings(
    dataset: AnnotatedDataset[Any],
    extractor_config: "ExtractorConfig",
    transforms: Callable | None = None,
    batch_size: int | None = None,
    *,
    selection: str | None = None,
) -> Embeddings:
    """Build Embeddings from dataset and extractor config.

    Creates the appropriate extractor based on the config's model type and
    wraps it in a DataEval Embeddings instance.

    Parameters
    ----------
    dataset : MaiteDataset
        Input dataset.
    extractor_config : ExtractorConfig
        Extractor configuration with model type and params.
    transforms : Callable | None
        Preprocessing transforms to apply before encoding.
        Only used by extractor types that accept it (onnx, torch, uncertainty).
    selection : str | None
        The selection `dataset` is, as :func:`claim_fitter` named it. A shared stateful
        extractor that has not seen its fitter's data yet is fitted on it first, so it never
        fits on `dataset` unless `dataset` is the fitter.

    Returns
    -------
    Embeddings
        DataEval Embeddings instance (implements FeatureExtractor).

    Notes
    -----
    A caller that caches a stateful extractor's embeddings must claim a fitter first and pass
    `selection`, as ``get_or_compute_embeddings`` does, or its cache key cannot name the fit.
    Called with no `selection`, it claims no fitter: a shared extractor fits on the first data it
    embeds.
    """

    extractor = build_extractor(extractor_config, transforms)
    if selection is not None:
        _fit_on_fitter(extractor, extractor_config, transforms, selection)
    return Embeddings(dataset, extractor=extractor, batch_size=batch_size)


def _fit_on_fitter(
    extractor: Callable, extractor_config: "ExtractorConfig", transforms: Callable | None, selection: str
) -> None:
    """Fit a shared stateful extractor on its fitter's data before it describes `selection`.

    When the fitter's embeddings came from the cache, the extractor has not seen the fitter's
    data, and describing another selection would fit it there instead. So it is first given the
    fitter's first batch, exactly as the fitter's own embedding pass would give it, and the result
    is discarded: a stateful extractor fits on the first data it is given. When `selection` is
    the fitter, the pass about to run fits it on that very data, and :func:`mark_fitted` records it.

    Refitting reproduces the fit behind the cached embeddings only under the seed that fitted them,
    which their key names. Without a seed they are never cached beyond the fit that computed them
    (:func:`reuse_within_task`), so the fitter's embeddings never come from a cache, and this runs
    only when the fitter's own pass in this task did not complete.
    """
    entry = _shared_entry(extractor_config, transforms, create=False)
    if entry is None or entry.fitter is None or entry.fitted or entry.fitter.key == selection:
        return
    fitter = entry.fitter
    _logger.info(
        "Fitting the shared %s extractor on %s before describing %s", extractor_config.model, fitter.key, selection
    )
    embeddings = Embeddings(fitter.dataset, extractor=extractor, batch_size=fitter.batch_size)
    embeddings[: embeddings.batch_size]  # computing the first batch fits the extractor; the batch is discarded
    entry.fitted = True


def mark_fitted(extractor_config: "ExtractorConfig", transforms: Callable | None, selection: str) -> None:
    """Record that `selection`'s embedding pass succeeded, which fits the shared extractor when it is the fitter."""
    entry = _shared_entry(extractor_config, transforms, create=False)
    if entry is not None and entry.fitter is not None and entry.fitter.key == selection:
        entry.fitted = True


def _shared_key(extractor_config: "ExtractorConfig", transforms: Callable | None) -> str:
    """Identity of an extractor built from this config and preprocessing."""
    return f"{extractor_config.model_dump_json()}|{transforms!r}"


def build_extractor(extractor_config: "ExtractorConfig", transforms: Callable | None = None) -> Callable:
    """Build a standalone extractor (not wrapped in Embeddings).

    Used for workflows that need to apply the extractor separately from embedding extraction
    (e.g. to extract metadata features for evaluation).

    Parameters
    ----------
    extractor_config : ExtractorConfig
        Extractor configuration with model type and params.
    transforms : Callable | None
        Preprocessing transforms to apply before encoding.
        Only used by extractor types that accept it (onnx, torch, uncertainty).

    Returns
    -------
    Callable
        A callable extractor function that takes a dataset and returns extracted features.
    """
    if isinstance(extractor_config, _InstanceExtractorConfig):
        # Given as an object: used as it is, never registered, never shared through the scope.
        return extractor_config.extractor

    _logger.debug("Building %s extractor", extractor_config.model)

    extractor = get_extractor(extractor_config.model)()
    entry = _shared_entry(extractor_config, transforms, create=True) if extractor.stateful else None
    if entry is None:
        return extractor.build(extractor_config, transforms)
    if entry.extractor is None:
        entry.extractor = extractor.build(extractor_config, transforms)
    return entry.extractor


def node_embeddings(context: "DatasetContext", extract: "Callable[[], Any]") -> "NDArray[Any]":
    """`context`'s embeddings: rows this run already extracted for it, rows sliced from an ancestor that has them,
    or `extract()`'s, remembered for its subsets (data-splitting spec §5.3).

    Only a config extractor's feature rows are remembered or sliced: an instance has no stable key, and a model's
    rows can be detections.
    """
    import numpy as np

    key = _settings_key(context)
    if key is None:
        return extract()
    memo = context.embedded
    if memo is not None and key in memo:
        return memo[key]
    rows = _sliced(context, key)
    if rows is not None:
        return rows
    rows = np.asarray(extract())
    if memo is not None:
        memo[key] = rows
    return rows


def _settings_key(context: "DatasetContext") -> str | None:
    """What makes two contexts' rows the same: the extractor's config, its preprocessing and batch size; ``None``
    where rows must not be remembered."""
    from dataeval_flow._cache import _extractor_config_key, _is_instance_extractor
    from dataeval_flow._predictions import runs_model

    extractor = context.extractor
    if extractor is None or _is_instance_extractor(extractor) or runs_model(extractor):
        return None
    return f"{_extractor_config_key(extractor)}|{context.transforms!r}|{context.batch_size}"


def _sliced(context: "DatasetContext", key: str) -> "NDArray[Any] | None":
    """`context`'s rows from the nearest ancestor this run extracted them for under `key`; ``None`` where none did."""
    import numpy as np

    link = context.parent
    if link is None:
        return None
    parent = link.context
    rows = parent.embedded.get(key) if parent.embedded is not None else None
    if rows is None:
        rows = _sliced(parent, key)
    return None if rows is None else rows[np.asarray(link.indices, dtype=np.intp)]
