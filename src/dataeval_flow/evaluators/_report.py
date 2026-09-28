"""An evaluator result's report body: DataEval's output as it came, as report blocks."""

__all__ = ["ROW_LIMIT", "output_blocks", "render_result_body", "serialized_of", "table_blocks"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from dataeval_flow._blocks import Block, Column, Paragraph, Section, Table, Tree
from dataeval_flow._blocks._draw import flow_repr
from dataeval_flow._blocks._text import Frame, render_text
from dataeval_flow._result import failure_section

if TYPE_CHECKING:
    from dataeval_flow.evaluators._result import EvaluatorResult

# Rows a table shows on the console before the rest are elided; ``result.txt`` shows them all.
ROW_LIMIT = 20
# Widest a cell renders before it is cut with an ellipsis.
_CELL_WIDTH = 40
# Array values shown before the rest are elided.
_ARRAY_HEAD = 10


def render_result_body(result: "EvaluatorResult[Any]", *, detailed: bool) -> list[str]:
    """The result's report body as text: the output for a success, or ``FAILED`` plus each error otherwise."""
    blocks = (
        output_blocks(serialized_of(result), detailed=detailed) if result.success else [failure_section(result.errors)]
    )
    return render_text(blocks, Frame(indent="  ", depth=1))


def serialized_of(result: "EvaluatorResult[Any]") -> dict[str, Any]:
    """DataEval's output as JSON, as the result serializes it; empty for a failed run."""
    return result._serialized or {}  # noqa: SLF001 - the result's JSON form has no public attribute by design


def output_blocks(output: dict[str, Any], *, detailed: bool) -> list[Block]:
    """Serialized DataEval output under an ``OUTPUT`` section, its extras in a section of their own."""
    blocks = _shape_blocks(output, detailed=detailed)
    extras = output.get("extras")
    if extras:
        blocks.append(Section(title="Extras", blocks=_mapping_blocks(extras, detailed=detailed)))
    return [Section(title="Output", brief=_brief(output) or None, blocks=blocks)]


def table_blocks(columns: Sequence[str], rows: Sequence[dict[str, Any]], *, limit: int | None) -> list[Block]:
    """Table rows as left-aligned columns, eliding rows past *limit* and cutting wide cells."""
    if not rows:
        return [Paragraph(text="(no rows)")]
    shown = rows if limit is None else rows[:limit]
    table = Table(
        columns=[Column(key=column, header=column, align="left") for column in columns],
        rows=[{column: _cell(row.get(column)) for column in columns} for row in shown],
    )
    if limit is None or len(rows) <= limit:
        return [table]
    more = f"… and {len(rows) - limit} more rows (run with -v, or read result.txt, to see them all)"
    return [table, Paragraph(text=more)]


def _brief(output: dict[str, Any]) -> str:
    shape = output.get("shape")
    if shape == "table":
        count = len(output["rows"])
        return f"{count} row{'s' if count != 1 else ''}"
    if shape == "array":
        count = len(output["data"])
        return f"{count} value{'s' if count != 1 else ''}"
    return ""


def _shape_blocks(value: Any, *, detailed: bool) -> list[Block]:
    shape = value.get("shape") if isinstance(value, dict) else None
    if shape == "table":
        return table_blocks(value["columns"], value["rows"], limit=None if detailed else ROW_LIMIT)
    if shape == "array":
        return [_array(value["data"])]
    if shape == "mapping":
        return _mapping_blocks(value["data"], detailed=detailed)
    return [Tree(value=value)]


def _mapping_blocks(data: dict[str, Any], *, detailed: bool) -> list[Block]:
    """Each nested table as its own section; the entries around them kept together in one tree."""
    blocks: list[Block] = []
    pending: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict) and value.get("shape") == "table":
            if pending:
                blocks.append(Tree(value=pending if detailed else _elided(pending)))
                pending = {}
            limit = None if detailed else ROW_LIMIT
            blocks.append(Section(title=key, blocks=table_blocks(value["columns"], value["rows"], limit=limit)))
        else:
            pending[key] = value
    if pending:
        blocks.append(Tree(value=pending if detailed else _elided(pending)))
    return blocks


def _array(values: Sequence[Any]) -> Paragraph:
    head = flow_repr(list(values[:_ARRAY_HEAD]))
    more = f" … and {len(values) - _ARRAY_HEAD} more" if len(values) > _ARRAY_HEAD else ""
    return Paragraph(text=f"{len(values)} values: {head}{more}")


def _elided(value: Any) -> Any:
    """`value` with every list longer than `_ARRAY_HEAD` cut to its head and a count of the rest, at any depth."""
    if isinstance(value, dict):
        return {key: _elided(item) for key, item in value.items()}
    if isinstance(value, list):
        head = [_elided(item) for item in value[:_ARRAY_HEAD]]
        return [*head, f"… and {len(value) - _ARRAY_HEAD} more"] if len(value) > _ARRAY_HEAD else head
    return value


def _cell_repr(value: Any) -> str:
    """Render a cell value like :func:`flow_repr`, but keep a contiguous int list as its items,
    with no ``range(...)`` collapse. A table cell shows the actual values."""
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {_cell_repr(v)}" for k, v in value.items())
        return "{" + inner + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_cell_repr(v) for v in value) + "]"
    return str(value)


def _cell(value: Any) -> str:
    text = "" if value is None else _cell_repr(value)
    return text if len(text) <= _CELL_WIDTH else text[: _CELL_WIDTH - 1] + "…"
