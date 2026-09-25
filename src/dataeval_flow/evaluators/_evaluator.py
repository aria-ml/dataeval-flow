"""The evaluator base class: its config, the DataEval class it wraps, and the one call."""

__all__ = ["Evaluator"]

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from dataeval_flow._input_spec import InputKind
from dataeval_flow._kind import bind_implementation
from dataeval_flow.evaluators._base import EvaluatorConfig

if TYPE_CHECKING:
    from dataeval_flow.evaluators._inputs import EvaluatorInputs

ConfigT = TypeVar("ConfigT", bound="EvaluatorConfig[Any]")
OutputT = TypeVar("OutputT")


class Evaluator(ABC, Generic[ConfigT, OutputT]):
    """One DataEval evaluator made runnable from config: Flow prepares what it reads, and keeps what it returns.

    A DataEval evaluator (a ``dataeval.types.Evaluator``, such as ``dataeval.quality.Outliers``) computes on the
    data it is handed. A Flow evaluator names one, takes its arguments from a pipeline entry, has Flow prepare
    what it reads from each of a task's sources, and returns DataEval's output, which Flow keeps in an
    :class:`EvaluatorResult`. ``Evaluator`` and :class:`EvaluatorConfig` share their names with DataEval's
    ``dataeval.types.Evaluator`` and ``dataeval.types.EvaluatorConfig`` on purpose: a Flow evaluator stands for its
    DataEval evaluator, and its config holds that evaluator's arguments. An evaluator judges
    nothing: its result has no findings and no health status; verdicts belong to workflows.

    Subclassing
    -----------
    Parameterize ``Evaluator`` with the evaluator's config class and the DataEval output class it returns,
    ``class BrightnessEvaluator(Evaluator[BrightnessConfig, OutliersOutput[Any]])``, which binds ``config_type``
    when the class is defined. The arguments must be given to ``Evaluator`` itself: an abstract base of your own
    may take them for its subclasses, but a generic one (``class Shared(Evaluator[C, O])``, subclassed as
    ``Shared[BrightnessConfig, OutliersOutput[Any]]``) is refused. Then define:

    - ``name: ClassVar[str]``: the type id. It must equal the config's ``type`` default and the entry-point name.
    - ``description: ClassVar[str]``: one line, which ``dataeval-flow evaluators`` prints.
    - ``dataeval_class: ClassVar[type]``: the DataEval evaluator class it wraps.
    - ``dataeval_methods: ClassVar[Mapping[InputKind, str]]``: the DataEval method :meth:`run` calls for each
      input kind its config reads, e.g. ``{InputKind.STATS: "from_stats"}``. Flow calls neither of these two; they
      state which DataEval API the evaluator depends on, readable without running it.
    - :meth:`run`: the call to DataEval, the only place besides ``dataeval_methods`` that names a DataEval method.

    A concrete evaluator without ``name``, ``description``, ``dataeval_class`` or ``dataeval_methods``, not
    parameterized, or whose config is not parameterized with a result class, raises ``TypeError`` when the class is
    defined. Register the class under the ``dataeval_flow.evaluators`` entry-point group, named by ``name``. Flow
    loads it on the first registry lookup and leaves it out, logging why, when it fails to import, is not an
    ``Evaluator``, its entry-point name, ``name`` and the config's ``type`` default disagree, its config declares
    no ``inputs``, or another evaluator has its name.

    For each task that runs the evaluator, Flow builds an instance with no arguments, applies each source's view,
    prepares every input kind the config wants under that source's cache, and calls :meth:`run` once with one
    :class:`EvaluatorInputs` per source. Flow prepares ``stats`` (with the stats policy they were measured under)
    and ``clusters`` (with the embeddings they were built from); a run that wants another kind fails, saying no
    producer exists for it. Flow guarantees that:

    - ``config`` is an instance of ``config_type``, validated when it was built or loaded;
    - the task meets ``config.inputs``, so ``inputs`` holds one entry per source the evaluator takes;
    - an exception raised while preparing the inputs, in :meth:`run`, or while serializing or recording its output
      becomes a failed result of the config's result class that records the error;
    - the output :meth:`run` returns becomes the result's ``output`` as it is, its ``data()`` is serialized for
      ``to_dict()`` and ``export()``, and its ``meta()`` is recorded in the envelope.

    Examples
    --------
    An evaluator that flags images whose brightness is an outlier, with the config and result classes it needs:

    >>> from collections.abc import Mapping, Sequence
    >>> from typing import Any, ClassVar
    >>> from dataeval.flags import ImageStats
    >>> from dataeval.quality import Outliers, OutliersOutput
    >>> from dataeval_flow import InputKind, InputSpec, SourceCount
    >>> from dataeval_flow.config import StatsConfigMixin
    >>> from dataeval_flow.evaluators import Evaluator, EvaluatorConfig, EvaluatorInputs, EvaluatorResult
    >>> class BrightnessResult(EvaluatorResult[OutliersOutput[Any]]):
    ...     pass
    >>> class BrightnessConfig(EvaluatorConfig[BrightnessResult], StatsConfigMixin):
    ...     type: str = "example.brightness"
    ...     inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE)
    ...
    ...     def stats_request(self) -> dict[str, Any]:
    ...         return {"outlier_flags": ImageStats.VISUAL_BRIGHTNESS}
    >>> class BrightnessEvaluator(Evaluator[BrightnessConfig, OutliersOutput[Any]]):
    ...     name: ClassVar[str] = "example.brightness"
    ...     description: ClassVar[str] = "Images whose brightness is an outlier."
    ...     dataeval_class: ClassVar[type] = Outliers
    ...     dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.STATS: "from_stats"}
    ...
    ...     def run(self, config: BrightnessConfig, inputs: Sequence[EvaluatorInputs]) -> OutliersOutput[Any]:
    ...         (source,) = inputs
    ...         assert source.stats is not None
    ...         outliers = Outliers(flags=ImageStats.VISUAL_BRIGHTNESS, outlier_threshold="zscore")
    ...         return outliers.from_stats(source.stats)

    Register it in the plugin's ``pyproject.toml``:

    .. code-block:: toml

        [project.entry-points."dataeval_flow.evaluators"]
        "example.brightness" = "my_package:BrightnessEvaluator"

    Once the plugin is installed, it runs like a built-in:

    >>> from dataeval_flow import run
    >>> result = run(BrightnessConfig(), dataset)  # doctest: +SKIP
    >>> result.output.aggregate_by_item()  # doctest: +SKIP
    """

    name: ClassVar[str]
    description: ClassVar[str]
    config_type: ClassVar["type[EvaluatorConfig[Any]]"]
    dataeval_class: ClassVar[type]
    dataeval_methods: ClassVar[Mapping[InputKind, str]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``config_type`` from the type arguments, and require identity on a concrete evaluator."""
        super().__init_subclass__(**kwargs)
        bind_implementation(cls, Evaluator, extra=("dataeval_class", "dataeval_methods"))

    @abstractmethod
    def run(self, config: ConfigT, inputs: "Sequence[EvaluatorInputs]") -> OutputT:
        """Call DataEval on the prepared inputs and return its output.

        Parameters
        ----------
        config : ConfigT
            This entry's settings, an instance of ``config_type``.
        inputs : Sequence[EvaluatorInputs]
            One per source, in the order the task names them, each holding the input kinds the config wants.

        Returns
        -------
        OutputT
            The output object DataEval returned, unchanged.

        Raises
        ------
        Exception
            Anything, to fail the run: Flow records the error on a failed result of the config's result class.
        """
