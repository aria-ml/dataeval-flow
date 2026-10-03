"""`wrap`: a DataEval dataset wrapper that changes a Dataset's kind, such as DetectionCrops."""

__all__ = ["WrapConfig", "WrapTransform"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext

if TYPE_CHECKING:
    from dataeval_flow._blocks import Block

_WRAPPERS: dict[str, tuple[str, str]] = {"DetectionCrops": ("object_detection", "classification")}
"""Each wrapper's input kind and output kind."""


class WrapConfig(TransformConfig):
    """A `wrap` step's settings: which wrapper, its keyword arguments, and what to do with a Dataset of another
    kind."""

    input: str = Field(description="The Dataset to wrap.")
    wrapper: Literal["DetectionCrops"] = Field(
        description="The DataEval wrapper: `DetectionCrops` crops each detection into an item."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="The wrapper's keyword arguments, such as `padding`."
    )
    other_kinds: Literal["refuse", "pass"] = Field(
        default="refuse",
        description=(
            "A Dataset the wrapper does not take: `refuse` it before the run, or `pass` it on unchanged, keeping "
            "its kind, so a chain can wrap detection data and read anything else as it is."
        ),
    )


class WrapTransform(Transform[WrapConfig]):
    """``wrap``: ``DetectionCrops`` turns each detection into a classification item. With ``other_kinds: pass`` a
    Dataset of another kind is handed on unchanged (coverage spec §5.1). Video wrappers come later."""

    name: ClassVar[str] = "wrap"
    title: ClassVar[str] = "Wrap"
    description: ClassVar[str] = "Wraps a Dataset in a DataEval wrapper that changes its kind, such as DetectionCrops."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def output_kinds(self, config: WrapConfig, input_kinds: Mapping[str, str | None]) -> Mapping[str, str | None]:
        """The wrapper's output kind, after checking it takes the input's; another kind, passed, keeps its own."""
        takes, makes = _WRAPPERS[config.wrapper]
        kind = input_kinds.get("input")
        if kind is not None and kind != takes:
            if config.other_kinds == "pass":
                return {"output": kind}
            raise ValueError(f"{config.wrapper} takes a {takes} Dataset, but its input is {kind}.")
        if kind is None and config.other_kinds == "pass":
            return {"output": None}
        return {"output": makes}

    def run(
        self,
        config: WrapConfig,
        inputs: Mapping[str, Any],
        context: TransformContext,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        """The input, wrapped; or, with `other_kinds: pass`, handed on as it is when it is another kind. A made node
        carries no kind, so the input's is detected again here, from one datum."""
        from dataeval import data

        from dataeval_flow._chain._preflight import detect_kind

        dataset = inputs["input"].value
        takes, _ = _WRAPPERS[config.wrapper]
        if config.other_kinds == "pass" and detect_kind(dataset) != takes:
            return {"output": dataset}
        return {"output": getattr(data, config.wrapper)(dataset, **config.params)}

    def details(
        self,
        config: WrapConfig,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Whether it wrapped, the input's kind, and for crops how many were made and how many boxes were dropped."""
        from dataeval_flow._chain._preflight import detect_kind

        made, dataset = outputs["output"], inputs["input"].value
        if made is dataset:
            return {"wrapped": False, "kind": detect_kind(dataset)}
        return {
            "wrapped": True,
            "kind": _WRAPPERS[config.wrapper][0],
            "items": len(made),
            "dropped": int(made.n_dropped),
        }

    def section(self, record: Any) -> list["Block"]:
        """What it did, in a sentence."""
        from dataeval_flow._blocks import Paragraph

        details = record.details or {}
        if not details.get("wrapped", True):
            kind = details.get("kind") or "empty"
            return [Paragraph(text=f"Passed through: the input is {kind}.")]
        return [
            Paragraph(
                text=f"Cropped {details.get('items', 0)} detections into items; {details.get('dropped', 0)} were too "
                "small or degenerate to embed."
            )
        ]
