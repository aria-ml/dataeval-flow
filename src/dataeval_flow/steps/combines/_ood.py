"""`ood-union`: where OOD detectors agree on a test source's images, and where they don't (ood-detection spec
§5.2)."""

__all__ = ["OODUnionOutput", "OODUnionCombine", "OODUnionConfig", "union_blocks", "union_of"]

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

import numpy as np
from dataeval.shift import OODOutput
from pydantic import BaseModel, Field

from dataeval_flow._blocks import Block, Cell, Column, ItemRef, Paragraph, Section, Table
from dataeval_flow._input_spec import SourceCount
from dataeval_flow._tables import table_limits
from dataeval_flow.evaluators.shift._report import derived_threshold
from dataeval_flow.steps._combine import Combine, CombineConfig, CombineContext
from dataeval_flow.steps._port import DataType, Port


class OODUnionOutput(BaseModel):
    """Each flagged test image's place among the OOD detectors, flagged by every one, by some or by one alone, with
    its agreement score."""

    source: str | None = Field(description="The test Dataset, whose items the image indices name.")
    detectors: list[str] = Field(description="The steps whose Outputs were combined, in input order.")
    left_out: list[str] = Field(
        description=(
            "Detectors whose derived threshold is not positive, which take no part in the scores, the groups or the "
            "union."
        )
    )
    images: int = Field(description="The test images.")
    assessed: int = Field(description="The test images every detector taking part assessed.")
    union: list[int] = Field(description="Images any detector taking part flagged.")
    mutual: list[int] = Field(description="Images every detector taking part flagged.")
    partial: list[int] = Field(
        description="Images more than one detector flagged, but not every one; empty with one or two detectors."
    )
    unique: dict[str, list[int]] = Field(
        description="By detector, the images it alone flagged; empty with one detector."
    )
    scores: list[float | None] = Field(
        description=(
            "Each test image's agreement score: the mean, over the detectors taking part that scored it, of its score "
            "over the detector's threshold; `null` where none did."
        )
    )
    thresholds: dict[str, float | None] = Field(
        description=(
            "Each detector's threshold for that score, derived from its flags: the highest score it did not flag, or "
            "the lowest where it flagged every image it assessed."
        )
    )
    flagged_detections: list[int] | None = Field(
        description=(
            "Per test image, the most of its detections any one detector reading detection rows flagged; `null` where "
            "no detector read them."
        )
    )


def union_of(nodes: Sequence[Any]) -> OODUnionOutput:
    """The detectors' Outputs, one per node, combined into groups, scores and thresholds."""
    first = nodes[0]
    count = len(first.value.is_ood)
    thresholds = {
        node.step: derived_threshold(list(node.value.instance_score), list(node.value.is_ood)) for node in nodes
    }
    taking = [node for node in nodes if (thresholds[node.step] or 0.0) > 0]
    flags = {node.step: np.asarray(node.value.is_ood, dtype=bool) for node in taking}
    scored = {node.step: np.isfinite(np.asarray(node.value.instance_score, dtype=float)) for node in taking}
    votes = np.sum(list(flags.values()), axis=0) if flags else np.zeros(count, dtype=int)
    many = len(taking)
    union = [int(index) for index in np.flatnonzero(votes > 0)]
    mutual = [index for index in union if votes[index] == many]
    partial = [index for index in union if 1 < votes[index] < many]
    unique = {step: [i for i in union if flag[i] and votes[i] == 1] if many > 1 else [] for step, flag in flags.items()}
    ratios = [
        np.where(
            scored[node.step],
            np.asarray(node.value.instance_score, dtype=float) / cast(float, thresholds[node.step]),
            np.nan,
        )
        for node in taking
    ]
    scores: list[float | None] = []
    for index in range(count):
        values = [ratio[index] for ratio in ratios if np.isfinite(ratio[index])]
        scores.append(float(np.mean(values)) if values else None)
    assessed = int(np.logical_and.reduce(list(scored.values())).sum()) if scored else count
    # By step name: `in` would compare nodes by value, and two Outputs' arrays have no single truth value.
    taken = {node.step for node in taking}
    readers = [node for node in taking if getattr(node.value, "rows", None)]
    flagged: list[int] | None = None
    if readers:
        most = np.zeros(count, dtype=int)
        for node in readers:
            hit = [row["image"] for row in node.value.rows["detections"] if row["is_ood"]]
            most = np.maximum(most, np.bincount(np.asarray(hit, dtype=int), minlength=count))
        flagged = [int(value) for value in most]
    return OODUnionOutput(
        source=first.computed_on[-1].address if first.computed_on else None,
        detectors=[node.step for node in nodes],
        left_out=[node.step for node in nodes if node.step not in taken],
        images=count,
        assessed=assessed,
        union=union,
        mutual=mutual,
        partial=partial,
        unique=unique,
        scores=scores,
        thresholds=thresholds,
        flagged_detections=flagged,
    )


def union_blocks(union: OODUnionOutput) -> list[Block]:
    """The flagged images, most out of distribution first: those every detector flagged, then, folded away, those
    some did and those one alone did. Each image appears once."""
    blocks: list[Block] = []
    if union.left_out:
        names = ", ".join(f"`{name}`" for name in union.left_out)
        blocks.append(Paragraph(text=f"Left out, with a threshold that is not positive: {names}."))
    if not union.union:
        return [*blocks, Paragraph(text="No image was flagged.")]
    if len(union.detectors) - len(union.left_out) == 1:
        return [*blocks, *_images(union.union, union)]
    blocks.extend(_images(union.mutual, union) or [Paragraph(text="No image was flagged by every detector.")])
    if union.partial:
        count = f"{len(union.partial)} image(s)"
        blocks.append(
            Section(title="Some detectors agree", brief=count, reference=True, blocks=_images(union.partial, union))
        )
    for step, indices in union.unique.items():
        if indices:
            count = f"{len(indices)} unique image(s)"
            blocks.append(Section(title=step, brief=count, reference=True, blocks=_images(indices, union)))
    return blocks


def _images(indices: Sequence[int], union: OODUnionOutput) -> list[Block]:
    """`indices`, most out of distribution first: each image's thumbnail, item and agreement score, and its flagged
    detections where a detector read them. At most ``result: max_rows``, with a paragraph counting the rest."""
    ranked = sorted(indices, key=lambda index: (-(union.scores[index] or 0.0), index))
    if not ranked:
        return []
    limits = table_limits()
    counted = union.flagged_detections
    rows: list[dict[str, Cell]] = []
    for index in ranked[: limits.rows]:
        row: dict[str, Cell] = {
            "image": ItemRef(source=union.source or "", index=index),
            "item": index,
            "score": union.scores[index] or 0.0,
        }
        if counted is not None:
            row["detections"] = counted[index]
        rows.append(row)
    columns = [
        Column(key="image", kind="image"),
        Column(key="item", header="Item"),
        Column(key="score", header="Score", format="{:.2f}x"),
        *([Column(key="detections", header="Flagged detections")] if counted is not None else []),
    ]
    blocks: list[Block] = [Table(columns=columns, rows=rows, preview=limits.preview)]
    if limits.rows is not None and len(ranked) > limits.rows:
        blocks.append(
            Paragraph(
                text=f"{len(ranked):,} images; the {limits.rows:,} most out of distribution are listed, and every one "
                "is in the JSON."
            )
        )
    return blocks


class OODUnionConfig(CombineConfig):
    """An `ood-union` step's input: the OOD Outputs of one comparison of a test source with a reference."""

    input: str | list[str] = Field(
        description="Each detector's OOD Output, every one computed on the same reference and test source."
    )


class OODUnionCombine(Combine[OODUnionConfig]):
    """``ood-union``: groups each test image the OOD detectors flagged as flagged by every one, by some, or by one
    alone, and scores their agreement."""

    name: ClassVar[str] = "ood-union"
    title: ClassVar[str] = "OOD Union"
    description: ClassVar[str] = "Groups each flagged image as flagged by every OOD detector, by some, or by one alone."
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.OUTPUT, classes=(OODOutput,), count=SourceCount.ONE_OR_MORE),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(OODUnionOutput,)),)
    shared_datasets: ClassVar[tuple[str, ...]] = ("input",)

    def run(self, config: OODUnionConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:  # noqa: ARG002
        """The detectors' Outputs, combined."""
        return {"output": union_of(list(inputs["input"]))}

    def section(self, record: Any) -> list[Block]:
        """The flagged images, each once."""
        union = record.output
        return union_blocks(union) if isinstance(union, OODUnionOutput) else []
