"""Ports: what a step takes and gives, typed so the engine connects steps without running them."""

__all__ = ["DATASET_KINDS", "DataType", "Port"]

import builtins
from dataclasses import dataclass
from enum import StrEnum

from dataeval_flow._input_spec import InputKind, SourceCount

DATASET_KINDS: tuple[str, ...] = (
    "image_only",
    "classification",
    "object_detection",
    "segmentation",
    "multiobject_tracking",
)
"""The Dataset kinds DataEval detects from a datum: ``dataeval.utils.data.DatasetKind`` less ``any_target``."""


class DataType(StrEnum):
    """What flows along a port."""

    DATASET = "dataset"
    OUTPUT = "output"
    EXPORT = "export"
    WORKFLOW_RESULT = "workflow_result"
    FINDINGS = "findings"


@dataclass(frozen=True)
class Port:
    """One input or output of a step."""

    # ``builtins.type``: past the ``type`` field below, a bare ``type`` names the field, not the builtin.
    name: str
    """The config field that feeds an input port, such as ``input`` or ``plans``, or an output's name."""
    type: DataType
    """What flows along it."""
    classes: tuple[builtins.type, ...] = ()
    """For an ``output`` port, the output classes it accepts or produces. Empty accepts any."""
    kinds: frozenset[str] | None = None
    """For a ``dataset`` port, the Dataset kinds it accepts or produces, from :data:`DATASET_KINDS`. ``None`` accepts
    any."""
    is_list: bool = False
    """Whether the port takes or gives a whole keyed list rather than one item."""
    count: SourceCount | None = None
    """For a Dataset port fed several addresses, how many it allows."""
    derives: frozenset[InputKind] = frozenset()
    """For an evaluator's Dataset port, the kinds Flow derives from it."""

    def accepts_class(self, cls: builtins.type | None) -> bool:
        """Whether an output of class `cls` may flow into this port."""
        if not self.classes:
            return True
        return cls is not None and issubclass(cls, self.classes)
