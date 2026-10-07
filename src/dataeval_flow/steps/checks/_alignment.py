"""The `label-mergeability` check: legacy data-coverage's Label Alignment finding, with the Relabel stanza that
conforms a dataset to the vocabulary (coverage spec §3.4)."""

__all__ = ["LabelMergeabilityCheck", "LabelMergeabilityConfig", "relabel_stanza", "yaml_scalar"]

import json
from collections.abc import Mapping
from typing import Any, ClassVar

import yaml
from pydantic import Field

from dataeval_flow._blocks import Block, Cell, Code, Column, Fields, Paragraph, Table
from dataeval_flow.evaluators.scope import LabelAlignmentOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity
from dataeval_flow.workflows._base import Finding

# A collapse is usually deliberate, so `lossy` informs rather than warns; `partial` warns because Relabel drops a
# class.
_SEVERITY: dict[str, Severity] = {"lossless": "ok", "lossy": "info", "partial": "warning"}

_DESCRIBED = {
    "lossless": "Every class carries over one-to-one.",
    "lossy": "Every class carries over, but two or more collapse into a single concept.",
    "partial": "At least one class cannot carry over and is dropped by Relabel.",
}


def yaml_scalar(value: str) -> str:
    """A YAML scalar for *value*, quoted only where a plain scalar would not round-trip.

    Checked by round-tripping rather than by matching a character set. A label may contain
    a metacharacter, but it may also be a plain word that YAML resolves to a non-string:
    ``0``, ``on``, ``null``, or a date. A config that parses back to an int key never
    matches the class it names, and ``Relabel`` then drops that class silently.
    """
    try:
        safe = yaml.safe_load(f"[{value}]") == [value]
    except yaml.YAMLError:
        safe = False
    return value if safe else json.dumps(value)


def relabel_stanza(paste_remap: dict[str, str], target_vocabulary: list[str]) -> str:
    """The alignment as a view operation that can be pasted into a config."""
    lines = [
        "      - type: Relabel",
        "        params:",
        "          class_remap:",
    ]
    lines.extend(
        f"            {yaml_scalar(source)}: {yaml_scalar(target)}" for source, target in sorted(paste_remap.items())
    )
    targets = ", ".join(yaml_scalar(t) for t in target_vocabulary)
    lines.append(f"          target: [{targets}]")
    return "\n".join(lines)


class LabelMergeabilityConfig(CheckConfig):
    """A `label-mergeability` step's input. It has no thresholds: severity follows the mergeability DataEval reports."""

    input: str = Field(description="A `label-alignment` Output.")


class LabelMergeabilityCheck(Check[LabelMergeabilityConfig]):
    """``label-mergeability``: whether a Dataset's classes carry over to an ontology's vocabulary, with the Relabel
    stanza that conforms it. Lossless is ok, lossy informs, partial warns; a target label several concepts share
    warns."""

    name: ClassVar[str] = "label-mergeability"
    description: ClassVar[str] = "Whether a Dataset's classes carry over to an ontology's vocabulary, with the stanza."
    title: ClassVar[str] = "Label Mergeability"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(LabelAlignmentOutput,)),)

    def run(self, config: LabelMergeabilityConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The mergeability, what is dropped or not covered, the correspondences, and the stanza."""
        al = inputs["input"].value.alignment
        severity: Severity = "warning" if al.ambiguous_labels else _SEVERITY.get(al.mergeability, "info")
        blocks: list[Block] = []
        if al.unaligned_source:
            blocks.append(Paragraph(text=f"Dropped: {', '.join(al.unaligned_source)}."))
        if al.unaligned_target:
            blocks.append(Paragraph(text=f"Concepts this dataset does not cover: {', '.join(al.unaligned_target)}."))
        if al.ambiguous_labels:
            blocks.append(
                Paragraph(
                    text=(
                        f"{len(al.ambiguous_labels)} target label(s) name more than one concept "
                        f"({', '.join(al.ambiguous_labels)}). The stanza below cannot be used until the "
                        "ontology is fixed, because the index such a label takes is undetermined."
                    )
                )
            )
        if al.correspondences:
            columns = [
                Column(key="source", header="Source"),
                Column(key="relation", header="Relation"),
                Column(key="target", header="Target"),
                Column(key="confidence", header="Confidence"),
                Column(key="matcher", header="Matcher"),
            ]
            rows: list[dict[str, Cell]] = [
                {
                    "source": c.source,
                    "relation": c.relation,
                    "target": c.target_label,
                    "confidence": round(c.confidence, 3),
                    "matcher": c.matcher,
                }
                for c in al.correspondences
            ]
            blocks.append(Table(columns=columns, rows=rows))
        if al.paste_remap:
            blocks += [
                Paragraph(text="To conform a dataset to this vocabulary, add to its view:"),
                # Printed exactly as given: the leading spaces nest it under a view's `operations:`.
                Code(text=relabel_stanza(al.paste_remap, al.target_vocabulary), language="yaml"),
                Paragraph(
                    text=(
                        "Datasets merged together must pass the identical `target`, or their integer "
                        "labels denote different classes."
                    )
                ),
            ]
        if al.label_space_digest:
            blocks.append(Fields(items=[("Label space", al.label_space_digest)]))
        return [
            Finding(
                severity=severity,
                title=self.title,
                description=f"Mergeability: {al.mergeability}. {_DESCRIBED.get(al.mergeability, '')}",
                blocks=blocks,
            )
        ]
