"""The comparison a matrix's report opens with (task-matrix spec §7.1)."""

__all__ = ["comparison_blocks"]

from typing import TYPE_CHECKING

from dataeval_flow._blocks import Block, Paragraph

if TYPE_CHECKING:
    from dataeval_flow._matrix._result import MatrixResult


def comparison_blocks(result: "MatrixResult", *, detailed: bool) -> list[Block]:  # noqa: ARG001 - Task 5 reads it
    """Each run's number, label and status, one line each."""
    return [Paragraph(text=f"Run {run.number} · {run.label}: {run.status}") for run in result.runs]
