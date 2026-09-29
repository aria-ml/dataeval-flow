"""The step base, and the transform: a step that makes Datasets from Datasets and outputs."""

__all__ = ["Step", "StepKind", "Transform", "TransformConfig", "TransformContext", "port_addresses"]

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from dataeval_flow._kind import bind_implementation
from dataeval_flow.steps._address import Address, parse_address
from dataeval_flow.steps._port import Port

if TYPE_CHECKING:
    from dataeval import Metadata

    from dataeval_flow._blocks import Block
    from dataeval_flow._result import LabelSpaceRecord
    from dataeval_flow.config._models import PipelineConfig

StepKind = Literal["evaluator", "workflow", "transform", "combine", "check"]


class Step:
    """Anything a workflow can chain: it declares typed ports, a config, and a kind.

    The engine reads only this: each step's name, kind, config class and ports. How a step runs depends on its
    kind. An :class:`~dataeval_flow.evaluators.Evaluator` calls DataEval on data Flow derives from its input
    Datasets, and a :class:`Transform` makes Datasets.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    kind: ClassVar[StepKind]
    config_type: ClassVar[type[BaseModel]]

    @classmethod
    def input_ports(cls) -> tuple[Port, ...]:
        """The ports this step reads, in the order its config names them."""
        return tuple(getattr(cls, "inputs", ()))

    @classmethod
    def output_ports(cls) -> tuple[Port, ...]:
        """The ports this step gives."""
        return tuple(getattr(cls, "outputs", ()))


class TransformConfig(BaseModel):
    """A transform step's settings: the addresses its ports read, and its own arguments.

    Each input port is a field of the same name holding an address (``str``), several (``list[str]``), or a
    mapping keyed by addresses (``dict[str, ...]``). Unknown keys are refused, so a misspelled setting fails the load.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


ConfigT = TypeVar("ConfigT", bound=TransformConfig)


@dataclass(frozen=True)
class TransformContext:
    """What a transform may read besides its inputs: the run it belongs to, and derived data for its nodes."""

    task: str
    """The task running the chain."""
    step: str
    """This step's name."""
    output_dir: Path | None = None
    """Where the run writes files, or ``None`` when it writes none."""
    pipeline: "PipelineConfig | None" = None
    """The pipeline the task belongs to, for resolving pool names."""
    data_dir: Path | None = None
    """The root that relative paths resolve against."""
    derive_metadata: "Callable[[Any], Metadata] | None" = None
    """Metadata of a Dataset node under this step's metadata policy, cached on the node."""
    lineage: "Callable[[str], Sequence[Any]] | None" = None
    """The lineage records a Dataset node descends from, nearest first."""


class Transform(Step, ABC, Generic[ConfigT]):
    """A step that makes Datasets: from Datasets, and from Outputs computed on them.

    Subclassing
    -----------
    Parameterize ``Transform`` with the config class, ``class Keep(Transform[KeepConfig])``, which binds
    ``config_type``. Then declare:

    - ``name: ClassVar[str]``: the type id a step names under ``transform:``. It equals the entry-point name.
    - ``description: ClassVar[str]``: one line.
    - ``inputs`` and ``outputs``: ``ClassVar[tuple[Port, ...]]``. Each input port is a field of the config.
    - :meth:`run`.

    Override :meth:`output_kinds` when an output's Dataset kind differs from ``input``'s, and :meth:`digest` when
    the output depends on data the settings do not name, such as a removal plan. ``same_node`` names output ports
    whose Outputs must have been computed on the ``input`` Dataset. :meth:`empty_outputs` names outputs its
    settings leave empty, and :meth:`destinations` the directories a run writes.

    Register the class under the ``dataeval_flow.transforms`` entry-point group, named by ``name``.

    Examples
    --------
    >>> from collections.abc import Mapping
    >>> from typing import Any, ClassVar
    >>> from dataeval_flow.steps import DataType, Port, Transform, TransformConfig, TransformContext
    >>> class KeepConfig(TransformConfig):
    ...     input: str
    >>> class Keep(Transform[KeepConfig]):
    ...     name: ClassVar[str] = "keep"
    ...     description: ClassVar[str] = "Hands its input on unchanged."
    ...     inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    ...     outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)
    ...
    ...     def run(
    ...         self, config: KeepConfig, inputs: Mapping[str, Any], context: TransformContext
    ...     ) -> Mapping[str, Any]:
    ...         return {"output": inputs["input"].value}
    >>> Keep.config_type is KeepConfig
    True
    """

    kind: ClassVar[StepKind] = "transform"
    inputs: ClassVar[tuple[Port, ...]]
    outputs: ClassVar[tuple[Port, ...]]
    same_node: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``config_type`` from the type argument, and require identity and ports on a concrete transform."""
        super().__init_subclass__(**kwargs)
        bind_implementation(cls, Transform, extra=("inputs", "outputs"), with_result=False)

    @abstractmethod
    def run(self, config: ConfigT, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        """Make this step's outputs.

        Parameters
        ----------
        config : ConfigT
            The step's settings.
        inputs : Mapping[str, Any]
            By input port name: a node (``.value`` is the Dataset or Output) for a port fed one address, a list of
            nodes for a port fed several, or the keyed list itself for a port declared ``is_list``.
        context : TransformContext
            The run this step belongs to.

        Returns
        -------
        Mapping[str, Any]
            By output port name: a Dataset, a ``dict[str, Dataset]`` for a list output, or a record.
        """

    def output_kinds(
        self,
        config: ConfigT,  # noqa: ARG002
        input_kinds: Mapping[str, str | None],
    ) -> Mapping[str, str | None]:
        """Each Dataset output's kind, from its inputs' kinds; by default, the kind of ``input``.

        Raise ``ValueError`` naming the problem when an input's kind is one this step cannot take: preflight turns it
        into a config error before any step runs.
        """
        kind = input_kinds.get("input")
        return {port.name: kind for port in self.output_ports() if port.type == "dataset"}

    @classmethod
    def output_keys(cls, config: ConfigT) -> Mapping[str, tuple[str, ...]]:  # noqa: ARG003
        """For each list output whose keys the settings fix, those keys, e.g. ``kfold``'s ``"0"``..``"k-1"``."""
        return {}

    @classmethod
    def resolved(cls, config: ConfigT, pipeline: "PipelineConfig") -> ConfigT:  # noqa: ARG003
        """These settings with every pool name replaced by what it names, checked at load.

        The engine keys, checks and runs the resolved settings, so editing a pool entry a step names changes the
        step's output key. Raise ``ValueError`` naming the problem when a name is unknown. By default, `config`.
        """
        return config

    def digest(
        self,
        config: ConfigT,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],  # noqa: ARG002
    ) -> str | None:
        """What the outputs resolved from the data that the settings do not name, or ``None``: spec §5.7."""
        return None

    @classmethod
    def empty_outputs(cls, config: ConfigT) -> frozenset[str]:  # noqa: ARG003
        """Outputs these settings leave empty, which a step may not name."""
        return frozenset()

    @classmethod
    def destinations(cls, config: ConfigT, *, task: str, step: str) -> tuple[str, ...]:  # noqa: ARG003
        """Directories under the run's output a run of this step writes to."""
        return ()

    def label_space(
        self,
        config: ConfigT,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],  # noqa: ARG002
        *,
        address: str,  # noqa: ARG002
    ) -> list["LabelSpaceRecord"]:
        """The label-space records this step's relabelling adds to the result; `address` is its output's."""
        return []

    def section(self, record: Any) -> list["Block"]:  # noqa: ARG002
        """This step's report section body, given its :class:`~dataeval_flow.steps.StepResult`."""
        return []


def port_addresses(config: BaseModel, port: Port) -> tuple[Address, ...]:
    """The addresses `config` gives input `port`: one, a list, or a mapping's keys; ``()`` when unset."""
    value = getattr(config, port.name, None)
    if value is None:
        return ()
    if isinstance(value, str):
        return (parse_address(value),)
    if isinstance(value, Mapping):
        return tuple(parse_address(key) for key in value)
    return tuple(parse_address(item) for item in value)
