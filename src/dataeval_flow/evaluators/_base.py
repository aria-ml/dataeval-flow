"""Evaluator framework base: the config base every evaluator shares.

Imports nothing heavier than pydantic. ``config._schemas`` imports every evaluator's
config, and this module sits beneath all of them.
"""

__all__ = ["EvaluatorConfig"]

from typing import Any, ClassVar, Generic, TypeVar

from pydantic import ConfigDict

from dataeval_flow._kind import KindConfig, bind_result_type

R = TypeVar("R")


class EvaluatorConfig(KindConfig, Generic[R]):
    """The settings of one evaluator entry, and the result class its evaluator returns.

    Each evaluator has one config class. A pipeline's ``evaluators:`` entry is validated with the config class
    its ``type`` names, and :func:`~dataeval_flow.run` takes an instance directly. Every entry has a ``name``,
    which tasks reference it by and which defaults to its ``type``.

    Unknown keys are refused. Each field is a DataEval argument spelled as DataEval spells it, so a misspelling
    fails the config load rather than silently running DataEval's default.

    Subclassing
    -----------
    Parameterize ``EvaluatorConfig`` with the evaluator's result class, an :class:`EvaluatorResult` subclass. That
    binds ``result_type``: the class of every result a run of this config returns, a failed run's included, and
    the type :func:`~dataeval_flow.run` returns for it. An evaluator whose config is not parameterized with a
    result class raises ``TypeError`` when the evaluator class is defined. Then define:

    - ``type``: a ``str`` field whose default is the evaluator's type id. A validator refuses any other value,
      and the JSON schema states it as a ``const``.
    - ``inputs``: a ``ClassVar[InputSpec]`` naming what the evaluator reads and how many sources a task gives it.
      A task that does not meet it is refused when the pipeline config loads, and again before a run.
    - The DataEval arguments, as pydantic fields, each with ``Field(description=...)``. The JSON schema, the TUI
      and the interactive CLI show the descriptions.

    Flow prepares, for each source, the input kinds :meth:`wanted_kinds` returns: ``inputs.required`` unless you
    override it to add the optional kinds a setting switches on. Override :meth:`check_inputs` when a setting
    limits the source count beyond ``inputs.sources``. An evaluator that reads ``stats`` may override
    :meth:`stats_request` to measure only the families it reads, and mix in
    :class:`~dataeval_flow.config.StatsConfigMixin` to measure under a stats policy the pipeline names. One that
    reads ``clusters`` declares the fields Flow clusters with: ``cluster_algorithm`` (``"kmeans"``,
    ``"hdbscan"``, or ``None`` for DataEval's default) and ``n_clusters`` (``int | None``).

    A config has no entry point of its own: Flow finds it through its evaluator's ``config_type`` (see
    :class:`Evaluator`).

    Examples
    --------
    The config of :class:`Evaluator`'s example, with its result class:

    >>> from typing import Any, ClassVar
    >>> from dataeval.flags import ImageStats
    >>> from dataeval.quality import OutliersOutput
    >>> from dataeval_flow import InputKind, InputSpec, SourceCount
    >>> from dataeval_flow.config import StatsConfigMixin
    >>> from dataeval_flow.evaluators import EvaluatorConfig, EvaluatorResult
    >>> class BrightnessResult(EvaluatorResult[OutliersOutput[Any]]):
    ...     pass
    >>> class BrightnessConfig(EvaluatorConfig[BrightnessResult], StatsConfigMixin):
    ...     type: str = "example.brightness"
    ...     inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE)
    ...
    ...     def stats_request(self) -> dict[str, Any]:
    ...         return {"outlier_flags": ImageStats.VISUAL_BRIGHTNESS}
    >>> BrightnessConfig().name
    'example.brightness'

    A pipeline entry for it:

    .. code-block:: yaml

        evaluators:
          - name: brightness
            type: example.brightness
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``result_type`` from the result class this config was parameterized with."""
        super().__pydantic_init_subclass__(**kwargs)
        bind_result_type(cls)

    def stats_request(self) -> dict[str, Any]:
        """The statistics this evaluator reads, when its inputs include ``stats``.

        Where the task names no stats policy, Flow measures what this requests. Where it names one, Flow measures
        what the policy says, and checks the policy against the flags this names for a consumer. Override it to
        request, and check, only the families the evaluator reads.

        Returns
        -------
        dict[str, ImageStats]
            ``ImageStats`` flags keyed by the consumer that reads them: ``outlier_flags``, ``duplicate_flags`` or
            ``factor_flags``, which are measured without a policy and checked against one; or ``derive_flags``,
            which is measured without a policy and checks a policy against nothing. The default,
            ``{"derive_flags": ImageStats.ALL}``, measures every family.
        """
        from dataeval.flags import ImageStats

        return {"derive_flags": ImageStats.ALL}
