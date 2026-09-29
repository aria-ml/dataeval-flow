"""The check: a step that judges Outputs against thresholds, and makes findings (spec §9.1, §9.2)."""

__all__ = ["Check", "CheckConfig", "CheckContext"]

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

from dataeval_flow._kind import bind_implementation, is_abstract
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import InlineStep, StepConfig, StepKind
from dataeval_flow.workflows._base import Finding


class CheckConfig(StepConfig):
    """A check step's settings: the addresses its ports read, and its thresholds.

    Type each threshold ``float | None``. ``None`` switches that judgment off: the finding is still made, as
    ``info``.
    """


CheckConfigT = TypeVar("CheckConfigT", bound=CheckConfig)


@dataclass(frozen=True)
class CheckContext:
    """The run a check belongs to."""

    task: str
    """The task running the chain."""
    step: str
    """This step's name."""


class Check(InlineStep, ABC, Generic[CheckConfigT]):
    """A step that judges what evaluators found: it reads Outputs and makes findings, each ``ok``, ``info`` or
    ``warning``.

    A chain's health rolls up over its checks' findings. A ``warning`` counts toward ``--fail-on-warning``. A check
    whose input holds nothing is never skipped: the engine reports one ``info`` finding in its place, titled
    ``title`` and briefed "not assessed", saying which input produced nothing and why.

    Subclassing
    -----------
    Parameterize ``Check`` with its config class, ``class Rate(Check[RateConfig])``, which binds ``config_type``.
    Then declare:

    - ``name: ClassVar[str]``: the type id a step names under ``check:``. It equals the entry-point name.
    - ``description: ClassVar[str]``: one line.
    - ``title: ClassVar[str]``: the title of the finding the check makes.
    - ``inputs``: ``ClassVar[tuple[Port, ...]]`` of ``output`` ports, each naming the Output classes it takes, and
      each a field of the config. A port declared ``is_list`` takes a whole list and judges it at once, such as a
      worst case across splits; it receives the elements that exist, and ``.elements`` names every key.
    - :meth:`run`.

    Its one output, ``findings``, is fixed. Register the class under the ``dataeval_flow.checks`` entry-point
    group, named by ``name``.

    Examples
    --------
    >>> from collections.abc import Mapping
    >>> from typing import Any, ClassVar
    >>> from dataeval.quality import DuplicatesOutput
    >>> from dataeval_flow.steps import Check, CheckConfig, CheckContext, DataType, Port
    >>> from dataeval_flow.workflows import Finding
    >>> class GroupsConfig(CheckConfig):
    ...     input: str
    ...     most: float | None = 0.0
    >>> class Groups(Check[GroupsConfig]):
    ...     name: ClassVar[str] = "duplicate-groups"
    ...     description: ClassVar[str] = "Warns when a Dataset holds more duplicate groups than `most`."
    ...     title: ClassVar[str] = "Duplicate groups"
    ...     inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(DuplicatesOutput,)),)
    ...
    ...     def run(self, config: GroupsConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
    ...         count = len(inputs["input"].value.data())
    ...         judged = config.most is not None
    ...         severity = ("warning" if count > config.most else "ok") if judged else "info"
    ...         return [Finding(severity=severity, title=self.title, brief=f"{count} groups")]
    >>> Groups.output_ports()[0].type
    <DataType.FINDINGS: 'findings'>
    """

    kind: ClassVar[StepKind] = "check"
    title: ClassVar[str]
    outputs: ClassVar[tuple[Port, ...]] = (Port("findings", DataType.FINDINGS),)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``config_type``, require identity on a concrete check, and refuse an input that is no Output."""
        super().__init_subclass__(**kwargs)
        bind_implementation(cls, Check, extra=("inputs", "title"), with_result=False)
        if is_abstract(cls):
            return
        for port in cls.inputs:
            if port.type is not DataType.OUTPUT:
                raise TypeError(f"{cls.__name__}'s input `{port.name}` carries {port.type}: a check reads Outputs.")

    @abstractmethod
    def run(self, config: CheckConfigT, inputs: Mapping[str, Any], context: CheckContext) -> Sequence[Finding]:
        """Judge the inputs, and return the findings, in the order a report lists them.

        Parameters
        ----------
        config : CheckConfigT
            The step's settings: its addresses and thresholds.
        inputs : Mapping[str, Any]
            By input port name: a node for a port fed one address (``.value`` is the Output, ``.items`` how many
            items the Datasets it was computed on hold), a list of nodes for a port fed several, or the keyed list
            itself for a port declared ``is_list``.
        context : CheckContext
            The run this step belongs to.

        Returns
        -------
        Sequence[Finding]
            The findings; none where there is nothing to report, as a rate with nothing flagged may choose.
        """
