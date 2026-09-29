"""`merge`: several Datasets concatenated into one, in the order named."""

__all__ = ["MergeConfig", "MergeTransform"]

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import Field

from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext


class MergeConfig(TransformConfig):
    """A `merge` step's inputs: two or more Datasets sharing one label vocabulary."""

    input: list[str] = Field(min_length=2, description="The Datasets to concatenate, in order.")


class MergeTransform(Transform[MergeConfig]):
    """``merge``: DataEval's ``merge_datasets``; the inputs must share ``index2label``, which ``conform`` gives them."""

    name: ClassVar[str] = "merge"
    description: ClassVar[str] = "Concatenates Datasets that share a label vocabulary, in order."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(
        self,
        config: MergeConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],
        context: TransformContext,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        """The inputs, concatenated."""
        from dataeval.data import merge_datasets

        return {"output": merge_datasets(*[node.value for node in inputs["input"]])}
