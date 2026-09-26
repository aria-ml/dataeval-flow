"""Look up a finding's report blocks by type.

A workflow test states the evidence a finding should carry: ``fields(finding)["Distance"]`` or
``tables(finding)[0]``, rather than how its blocks happen to nest.
"""

from collections.abc import Iterator, Sequence
from typing import TypeVar

from dataeval_flow._blocks import Block, BulletList, Code, Fields, Paragraph, Section, Table
from dataeval_flow._blocks._models import Cell, Scalar
from dataeval_flow._blocks._text import DEFAULT_WIDTH, Frame, render_text
from dataeval_flow.workflows import Finding
from dataeval_flow.workflows._result import finding_section

_B = TypeVar("_B")


def walk(blocks: Sequence[Block]) -> Iterator[Block]:
    """Every block in *blocks* in document order, descending into sections."""
    for block in blocks:
        yield block
        if isinstance(block, Section):
            yield from walk(block.blocks)


def blocks_of(finding: Finding, kind: type[_B]) -> list[_B]:
    """Every block of *kind* in the finding, at any depth. The description is not a block."""
    return [block for block in walk(finding.blocks) if isinstance(block, kind)]


def tables(finding: Finding) -> list[Table]:
    """Every table in the finding."""
    return blocks_of(finding, Table)


def sections(finding: Finding) -> list[Section]:
    """Every nested section in the finding."""
    return blocks_of(finding, Section)


def paragraphs(finding: Finding) -> list[str]:
    """Every paragraph's text."""
    return [block.text for block in blocks_of(finding, Paragraph)]


def bullets(finding: Finding) -> list[str]:
    """Every bullet, across every list."""
    return [item for block in blocks_of(finding, BulletList) for item in block.items]


def codes(finding: Finding) -> list[str]:
    """Every code block's text."""
    return [block.text for block in blocks_of(finding, Code)]


def fields(finding: Finding) -> dict[str, Scalar]:
    """Every labelled value, across every ``Fields`` block; a later label replaces an earlier one."""
    return {label: value for block in blocks_of(finding, Fields) for label, value in block.items}


def column(table: Table, key: str) -> list[Cell]:
    """The table's cells under *key*, top to bottom; ``None`` where a row has none."""
    return [row.get(key) for row in table.rows]


def rendered(finding: Finding, width: int = DEFAULT_WIDTH) -> str:
    """The finding's detail section as the text report draws it."""
    return "\n".join(render_text([finding_section(finding)], Frame(width=width, indent="  ", depth=1)))
