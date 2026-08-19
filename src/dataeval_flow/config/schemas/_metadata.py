"""Result metadata schema for JATIC compliance [IR-3-H-12]."""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ResultMetadata(BaseModel):
    """Base metadata envelope for workflow results.

    Contains JATIC-required fields (version, timestamp, tool info,
    dataset identifiers).
    """

    version: str = "1.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_id: str | Sequence[str] = ""
    label_source: str | None = None
    model_id: str | None = None
    preprocessor_id: str | None = None
    selection_id: str | None = None
    source_descriptions: Sequence[str] = ()
    resolved_config: dict[str, Any] = Field(default_factory=dict)
    tool: str = "dataeval-flow"
    tool_version: str = ""
    execution_time_s: float | None = None

    #: How each metadata factor was encoded, and how well that encoding fits.
    #: Per factor: its type and level, the ``encoding`` applied (edges or
    #: vocabulary, who chose them, and how they were placed), the name each code
    #: reads as, and the ``fit`` this run's rows made against it — counts,
    #: occupied spans, and declared bins nothing reached.  ``None`` for workflows
    #: that build no metadata.  Recorded because the encoding decides what every
    #: evaluator reads, and it is otherwise reported only in logs no envelope
    #: references.
    metadata_binning: dict[str, Any] | None = None

    #: Fingerprint of the encoding every factor was read under, or ``None`` where
    #: the workflow built no metadata or its splits did not share one.  Comparing
    #: two runs is only sound if each can say which cuts produced it: without this,
    #: a bias score that moved is unattributable between *the override worked* and
    #: *the data changed*, which are the two readings a reader is trying to tell
    #: apart.  Same digest and same data means the numbers are comparable.
    encoding_digest: str | None = None

    #: Library diagnostics raised while the workflow ran — the decisions
    #: DataEval made on the caller's behalf and the ranges it could not resolve.
    #: Empty when the run raised none.
    diagnostics: Sequence[str] = ()
