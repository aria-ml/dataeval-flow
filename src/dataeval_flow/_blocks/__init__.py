"""Report blocks: what a report says, as typed data that each renderer draws its own way.

Workflows build their findings from these blocks, and Flow renders them as text or HTML. Blocks carry
data, never drawings: a bar column holds numbers and a :class:`Distribution` holds counts, so every
renderer can choose how to draw them.

The package is internal. What is public is a finding's blocks as ``results.json`` holds them, one
object per block with its ``type`` tag.
"""

from dataeval_flow._blocks._models import (
    Asset,
    Block,
    BulletList,
    Cell,
    Code,
    Column,
    Distribution,
    Fields,
    Flag,
    ItemRef,
    Paragraph,
    Proportion,
    Quantiles,
    Scalar,
    Section,
    Summary,
    SummaryItem,
    Table,
    Tree,
    Verdict,
)

__all__ = [
    "Asset",
    "Block",
    "BulletList",
    "Cell",
    "Code",
    "Column",
    "Distribution",
    "Fields",
    "Flag",
    "ItemRef",
    "Paragraph",
    "Proportion",
    "Quantiles",
    "Scalar",
    "Section",
    "Summary",
    "SummaryItem",
    "Table",
    "Tree",
    "Verdict",
]
