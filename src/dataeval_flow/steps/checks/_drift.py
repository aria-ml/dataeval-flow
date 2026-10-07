"""The drift check: judges a drift detector's verdict, whole or chunk by chunk (spec §10.11)."""

__all__ = [
    "EmbeddingDivergenceCheck",
    "EmbeddingDivergenceConfig",
    "DriftCheck",
    "DriftConfig",
    "DriftThresholds",
    "evaluator_heading",
]

from collections.abc import Mapping
from typing import Any, ClassVar, Self

import polars as pl
from dataeval.shift import DriftOutput
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_flow._step_title import step_title
from dataeval_flow.evaluators.shift import DivergenceOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.workflows._base import Finding


class DriftThresholds(BaseModel):
    """When a drift verdict warns: on drift, or, chunk by chunk, on how many chunks drift and how many in a row."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warn_on_drift: bool = Field(
        default=True,
        description=("Unchunked, and per class: drift is a warning, or info when false."),
    )
    chunk_percent: float | None = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description=(
            "Chunked: the share of drifted chunks, in percent, past which the finding warns; `null` judges none."
        ),
    )
    consecutive_chunks: int | None = Field(
        default=2,
        ge=0,
        description=(
            "Chunked: the longest run of drifted chunks past which the finding warns, so 2 warns on three in a row "
            "and 0 on any drifted chunk; `null` judges none."
        ),
    )


class DriftConfig(CheckConfig, DriftThresholds):
    """A `drift` step's input, its thresholds, and what its finding is titled."""

    input: str = Field(description="A drift evaluator's Output.")
    subject: str | None = Field(
        default=None,
        description=(
            "What the finding is titled. Unset: the evaluator's title, followed by its entry's name where that "
            "differs from its type, such as `Drift (MMD) · mmd_chunked`."
        ),
    )


def evaluator_heading(entry: Any) -> str:
    """An evaluator entry's title, with its name beside it where the name is not its type."""
    title = step_title("evaluator", entry.type)
    return title if entry.name == entry.type else f"{title} · {entry.name}"


class DriftCheck(Check[DriftConfig]):
    """``drift``: warns on drift, or, chunked, when enough chunks drift or enough drift in a row."""

    name: ClassVar[str] = "drift"
    description: ClassVar[str] = (
        "Judges a `drift-*` detector's output: warns when it finds drift, whole or chunk by chunk."
    )
    title: ClassVar[str] = "Drift"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(DriftOutput,)),)

    def run(self, config: DriftConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """One finding: the verdict, or the chunks' verdicts."""
        node = inputs["input"]
        output = node.value
        title = config.subject or (evaluator_heading(node.config) if node.config is not None else self.title)
        chunks = output.details["drifted"].to_list() if isinstance(output.details, pl.DataFrame) else []
        if not chunks:
            severity = ("warning" if config.warn_on_drift else "info") if output.drifted else "ok"
            return [Finding(severity=severity, title=title, brief="drift" if output.drifted else "no drift")]
        drifted, longest, run = sum(map(bool, chunks)), 0, 0
        for flag in chunks:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        percent = 100.0 * drifted / len(chunks)
        judged = config.chunk_percent is not None or config.consecutive_chunks is not None
        warns = (config.chunk_percent is not None and drifted > 0 and percent > config.chunk_percent) or (
            config.consecutive_chunks is not None and longest > config.consecutive_chunks
        )
        severity = ("warning" if warns else "info" if drifted else "ok") if judged else "info"
        return [
            Finding(
                severity=severity,
                title=title,
                brief=f"{drifted}/{len(chunks)} chunks drifted",
                description=f"{drifted}/{len(chunks)} chunks drifted ({percent:.0f}%) | max consecutive: {longest}",
            )
        ]


class EmbeddingDivergenceConfig(CheckConfig):
    """A `embedding-divergence` step's input, and the bands of divergence that warn and inform."""

    input: str = Field(description="A `divergence` Output.")
    warning: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=("The divergence above which the finding warns; `null` never warns."),
    )
    info: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "The divergence above which the finding is `info`, at or below which it is `ok`. Unset, it is 0.4 times "
            "`warning`; `null` has no `info` band. Must not exceed `warning`."
        ),
    )

    @model_validator(mode="after")
    def _info_under_warning(self) -> Self:
        if self.warning is None:
            return self
        if "info" not in self.model_fields_set:
            # derived, so it stays out of model_fields_set; rounded, because 0.4 * 0.2 is 0.08000000000000002
            object.__setattr__(self, "info", round(0.4 * self.warning, 10))
        elif self.info is not None and self.info > self.warning:
            raise ValueError(f"`info` ({self.info}) must not exceed `warning` ({self.warning}).")
        return self


class EmbeddingDivergenceCheck(Check[EmbeddingDivergenceConfig]):
    """``embedding-divergence``: whether two sources' embeddings sit too far apart, judging one `divergence` Output."""

    name: ClassVar[str] = "embedding-divergence"
    description: ClassVar[str] = "Judges `divergence`'s output: warns when two sources' embeddings sit too far apart."
    title: ClassVar[str] = "Embedding Divergence"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(DivergenceOutput,)),)

    def run(self, config: EmbeddingDivergenceConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The divergence, banded high, moderate or low."""
        data = inputs["input"].value.data()
        value, method = float(data["divergence"]), data["method"]
        if config.warning is None and config.info is None:
            brief = f"divergence {value:.4f} ({method})"
            return [Finding(severity="info", title=self.title, brief=brief, description=f"Divergence {value:.4f}.")]
        if config.warning is not None and value > config.warning:
            severity, level = "warning", "high"
        elif config.info is not None and value > config.info:
            severity, level = "info", "moderate"
        else:
            severity, level = "ok", "low"
        brief = f"{level} divergence: {value:.4f} ({method})"
        return [
            Finding(severity=severity, title=self.title, brief=brief, description=f"{brief[0].upper()}{brief[1:]}.")
        ]
