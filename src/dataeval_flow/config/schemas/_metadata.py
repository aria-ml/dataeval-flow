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
