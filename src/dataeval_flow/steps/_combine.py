"""The combine: a step that makes an Output from Outputs, and from Datasets' derived data (spec §9.3)."""

__all__ = ["Combine", "CombineConfig", "CombineContext"]

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from dataeval_flow._kind import bind_implementation, is_abstract
from dataeval_flow.steps._port import DataType
from dataeval_flow.steps._step import InlineStep, StepConfig, StepKind

if TYPE_CHECKING:
    from dataeval import Metadata

    from dataeval_flow._blocks import Block


class CombineConfig(StepConfig):
    """A combine step's settings: the addresses its ports read, and its own arguments."""


CombineConfigT = TypeVar("CombineConfigT", bound=CombineConfig)


@dataclass(frozen=True)
class CombineContext:
    """What a combine may read besides its inputs: the run it belongs to, and derived data for its Dataset inputs."""

    task: str
    """The task running the chain."""
    step: str
    """This step's name."""
    derive_metadata: "Callable[[Any], Metadata]"
    """Metadata of a Dataset node under this step's metadata policy, cached on the node."""
    derive_stats: "Callable[[Any], Any]"
    """Image statistics of a Dataset node, per image and not per target, under this step's stats policy, or every
    statistic where it names none, at the Dataset's value range: DataEval's stats result, whose ``stats`` maps each
    statistic to one value per image, beside ``image_count``. Cached as evaluators' statistics are. Only the
    requested statistics are returned, whatever else the cache holds."""


class Combine(InlineStep, ABC, Generic[CombineConfigT]):
    """A step that reads Outputs and makes an Output: a union, a pivot, or a merge of what evaluators found.

    A combine judges nothing. A check reads what it makes.

    Subclassing
    -----------
    Parameterize ``Combine`` with the config class, ``class Pivot(Combine[PivotConfig])``, which binds
    ``config_type``. Then declare:

    - ``name: ClassVar[str]``: the type id a step names under ``combine:``. It equals the entry-point name.
    - ``description: ClassVar[str]``: one line.
    - ``inputs`` and ``outputs``: ``ClassVar[tuple[Port, ...]]``. Each input port is a field of the config. Every
      output port is an ``output`` port naming the class it gives, so what reads it is type-checked at load. An
      input may be a Dataset port with ``derives``, for a combine that reads a node's derived data, such as its
      labels; :attr:`CombineContext.derive_metadata` reads it.
    - :meth:`run`.
    - optionally :meth:`section`, the step's report section.

    ``same_node`` and :meth:`bound_problem` work as they do for a :class:`~dataeval_flow.steps.Transform`. Register
    the class under the ``dataeval_flow.combines`` entry-point group, named by ``name``.

    Examples
    --------
    >>> from collections.abc import Mapping
    >>> from typing import Any, ClassVar
    >>> from pydantic import BaseModel
    >>> from dataeval.quality import DuplicatesOutput
    >>> from dataeval_flow.steps import Combine, CombineConfig, CombineContext, DataType, Port
    >>> class GroupCount(BaseModel):
    ...     groups: int
    >>> class CountConfig(CombineConfig):
    ...     input: str
    >>> class CountGroups(Combine[CountConfig]):
    ...     name: ClassVar[str] = "count-groups"
    ...     description: ClassVar[str] = "Counts duplicate groups."
    ...     inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(DuplicatesOutput,)),)
    ...     outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(GroupCount,)),)
    ...
    ...     def run(
    ...         self, config: CountConfig, inputs: Mapping[str, Any], context: CombineContext
    ...     ) -> Mapping[str, Any]:
    ...         return {"output": GroupCount(groups=len(inputs["input"].value.data()))}
    >>> CountGroups.config_type is CountConfig
    True
    """

    kind: ClassVar[StepKind] = "combine"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``config_type``, require identity and ports on a concrete combine, and refuse a non-Output output."""
        super().__init_subclass__(**kwargs)
        bind_implementation(cls, Combine, extra=("inputs", "outputs"), with_result=False)
        if is_abstract(cls):
            return
        for port in cls.outputs:
            if port.type is not DataType.OUTPUT:
                raise TypeError(f"{cls.__name__}'s output `{port.name}` carries {port.type}: a combine makes Outputs.")

    @abstractmethod
    def run(self, config: CombineConfigT, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        """Make this step's outputs.

        Parameters
        ----------
        config : CombineConfigT
            The step's settings.
        inputs : Mapping[str, Any]
            By input port name: a node (``.value`` is the Output or Dataset) for a port fed one address, a list of
            nodes for a port fed several, or the keyed list itself for a port declared ``is_list``.
        context : CombineContext
            The run this step belongs to.

        Returns
        -------
        Mapping[str, Any]
            Each Output, by output port name.
        """

    def section(self, record: Any) -> list["Block"]:  # noqa: ARG002
        """This step's report section body, given one run's :class:`~dataeval_flow.steps.StepResult`, whose
        ``output`` is the Output made. The chain report shows it, and captures thumbnails of the items it names. By
        default, nothing."""
        return []
