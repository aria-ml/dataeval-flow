"""`wrap`: a DataEval dataset wrapper that changes a Dataset's kind, such as DetectionCrops."""

__all__ = ["WrapConfig", "WrapTransform"]

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import Field

from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext

_WRAPPERS: dict[str, tuple[str, str]] = {"DetectionCrops": ("object_detection", "classification")}
"""Each wrapper's input kind and output kind."""


class WrapConfig(TransformConfig):
    """A `wrap` step's settings: which wrapper, and its keyword arguments."""

    input: str = Field(description="The Dataset to wrap.")
    wrapper: Literal["DetectionCrops"] = Field(
        description="The DataEval wrapper: `DetectionCrops` crops each detection into an item."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="The wrapper's keyword arguments, such as `padding`."
    )


class WrapTransform(Transform[WrapConfig]):
    """``wrap``: ``DetectionCrops`` turns each detection into a classification item. Video wrappers come later."""

    name: ClassVar[str] = "wrap"
    title: ClassVar[str] = "Wrap"
    description: ClassVar[str] = "Wraps a Dataset in a DataEval wrapper that changes its kind, such as DetectionCrops."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def output_kinds(self, config: WrapConfig, input_kinds: Mapping[str, str | None]) -> Mapping[str, str | None]:
        """The wrapper's output kind, after checking it takes the input's."""
        takes, makes = _WRAPPERS[config.wrapper]
        kind = input_kinds.get("input")
        if kind is not None and kind != takes:
            raise ValueError(f"{config.wrapper} takes a {takes} Dataset, but its input is {kind}.")
        return {"output": makes}

    def run(
        self,
        config: WrapConfig,
        inputs: Mapping[str, Any],
        context: TransformContext,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        """The input, wrapped."""
        from dataeval import data

        return {"output": getattr(data, config.wrapper)(inputs["input"].value, **config.params)}
