"""`conform`: relabel a Dataset onto an ontology by the remap its label alignment derived (spec §6.2)."""

__all__ = ["ConformConfig", "ConformTransform"]

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from dataeval_flow._alignment import LabelAlignmentOutput, PastedRemap
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext

if TYPE_CHECKING:
    from dataeval.data import Relabel

_ORDER = {"lossless": 0, "lossy": 1, "partial": 2}


class ConformConfig(TransformConfig):
    """A `conform` step's settings: its input, the alignment of it, what loss it may accept, and overrides."""

    input: str = Field(description="The Dataset to relabel.")
    alignment: str = Field(description="A `label-alignment` step computed on `input`.")
    allow: Literal["lossless", "lossy", "partial"] = Field(
        default="lossless",
        description="The most loss accepted: `lossy` lets classes collapse, `partial` drops unaligned classes.",
    )
    class_remap: dict[str, str] = Field(
        default_factory=dict,
        description="Overrides: a source class to a target concept, by id or by an unambiguous label.",
    )


class ConformTransform(Transform[ConformConfig]):
    """``conform``: ``Relabel(class_remap, target=ontology)`` from the alignment, refused beyond ``allow``.

    The engine makes one instance per invocation, so the remap :meth:`run` derives is the one :meth:`digest`,
    :meth:`details` and :meth:`label_space` read, and the Relabel it applies is the one :meth:`details` counts the
    drops of.
    """

    name: ClassVar[str] = "conform"
    description: ClassVar[str] = (
        "Relabels a Dataset onto an ontology by its label alignment, refusing loss beyond `allow`."
    )
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.DATASET),
        Port("alignment", DataType.OUTPUT, classes=(LabelAlignmentOutput,)),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)
    same_node: ClassVar[tuple[str, ...]] = ("alignment",)

    _pasted: PastedRemap
    _relabel: "Relabel"

    def run(
        self,
        config: ConformConfig,
        inputs: Mapping[str, Any],
        context: TransformContext,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        """The input, relabelled; raises naming an override for a class the input does not have, or the loss when it
        is beyond ``allow``."""
        from dataeval.data import Relabel, View

        from dataeval_flow._alignment import effective_mergeability, pasted_remap

        node = inputs["input"]
        classes = list(dict(node.value.metadata.get("index2label", {})).values())
        unknown = [source for source in config.class_remap if source not in classes]
        if unknown:
            raise ValueError(
                f"`class_remap` overrides {', '.join(f'`{source}`' for source in unknown)}, which `{node.address}` "
                f"does not have: its classes are {', '.join(classes)}."
            )
        found: LabelAlignmentOutput = inputs["alignment"].value
        remap = self._remap(config, found)
        unaligned = [name for name in found.alignment.unaligned_source if name not in remap]
        mergeability = effective_mergeability(remap, unaligned)
        if _ORDER[mergeability] > _ORDER[config.allow]:
            raise ValueError(
                f"The alignment is {mergeability}, beyond `allow: {config.allow}`: {self._loss(remap, unaligned)}. "
                f"Set `allow: {mergeability}` to accept it, or settle classes with `class_remap:`."
            )
        self._pasted = pasted_remap(found.ontology, remap)
        self._relabel = Relabel(
            remap, target=found.ontology, on_unmatched="drop" if config.allow == "partial" else "raise"
        )
        return {"output": View(inputs["input"].value, self._relabel)}

    @staticmethod
    def _remap(config: ConformConfig, found: LabelAlignmentOutput) -> dict[str, str]:
        """The derived remap by concept id, with each override's target resolved to one concept id."""
        ontology = found.ontology
        remap = dict(found.alignment.class_remap)
        for source, target in config.class_remap.items():
            if target in ontology.ids:
                remap[source] = target
                continue
            matches = [cid for cid in ontology.ids if ontology.concept(cid).label == target]
            if len(matches) != 1:
                problem = (
                    "names no concept"
                    if not matches
                    else f"is the label of {len(matches)} concepts; write the concept id"
                )
                raise ValueError(f"`class_remap: {source}: {target}` {problem}.")
            remap[source] = matches[0]
        return remap

    @staticmethod
    def _loss(remap: Mapping[str, str], unaligned: list[str]) -> str:
        parts = [
            f"{', '.join(sorted(sources))} collapse onto {target}" for target, sources in _collapses(remap).items()
        ]
        if unaligned:
            parts.append(f"{', '.join(sorted(unaligned))} align to nothing")
        return "; ".join(parts)

    def digest(
        self,
        config: ConformConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],  # noqa: ARG002
    ) -> str:
        """The remap applied and the target vocabulary."""
        found: LabelAlignmentOutput = inputs["alignment"].value
        payload = {"remap": self._pasted.by_id, "target": list(found.ontology.ids)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def details(
        self,
        config: ConformConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],  # noqa: ARG002
    ) -> dict[str, Any]:
        """Collapses, and the classes and items the Relabel :meth:`run` applied dropped."""
        labels = self._pasted.labels
        collapses = _collapses(self._pasted.by_id)
        return {
            "remap": dict(self._pasted.remap),
            "collapses": {labels.get(target, target): sorted(sources) for target, sources in collapses.items()},
            "dropped_classes": sorted(self._relabel.dropped.values()),
            "dropped_items": len(self._relabel.dropped_indices),
        }

    def label_space(
        self,
        config: ConformConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],  # noqa: ARG002
        *,
        address: str,
    ) -> list[Any]:
        """One record of the remap applied to `address`, digested as data-coverage digests its alignment."""
        from dataeval_flow._result import LabelSpaceRecord

        found: LabelAlignmentOutput = inputs["alignment"].value
        pasted = self._pasted
        return [
            LabelSpaceRecord(
                source=address,
                ontology=found.ontology_source,
                ontology_digest=pasted.ontology_digest,
                class_remap=dict(pasted.remap),
                target=list(pasted.vocabulary),
                digest=pasted.digest,
            )
        ]

    def section(self, record: Any) -> list[Any]:
        """What was collapsed, and what was dropped."""
        from dataeval_flow._blocks import Fields, Paragraph

        details = record.details or {}
        blocks: list[Any] = [
            Fields(
                items=[("Classes", len(details.get("remap", {}))), ("Items dropped", details.get("dropped_items", 0))]
            )
        ]
        for target, sources in (details.get("collapses") or {}).items():
            blocks.append(Paragraph(text=f"{', '.join(f'`{s}`' for s in sources)} collapse onto `{target}`."))
        if details.get("dropped_classes"):
            blocks.append(
                Paragraph(
                    text=f"Dropped, aligning to nothing: {', '.join(f'`{c}`' for c in details['dropped_classes'])}."
                )
            )
        return blocks


def _collapses(remap: Mapping[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for source, target in remap.items():
        grouped.setdefault(target, []).append(source)
    return {target: sources for target, sources in grouped.items() if len(sources) > 1}
