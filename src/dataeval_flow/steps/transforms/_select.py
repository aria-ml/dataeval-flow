"""`select`: the first items of a Prioritize ranking computed on the same Dataset."""

__all__ = ["SelectConfig", "SelectTransform"]

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from dataeval.scope import PrioritizeOutput
from pydantic import Field, model_validator

from dataeval_flow._chain._identity import indices_digest
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext


class SelectConfig(TransformConfig):
    """A `select` step's settings: its input, the ranking of it, and how many to keep."""

    input: str = Field(description="The Dataset to select from.")
    ranking: str = Field(description="A `scope.prioritize` step computed on `input`.")
    n: int | None = Field(default=None, ge=1, description="How many items to keep.")
    fraction: float | None = Field(default=None, gt=0.0, le=1.0, description="The share of items to keep, rounded up.")

    @model_validator(mode="after")
    def _one_amount(self) -> "SelectConfig":
        if (self.n is None) == (self.fraction is None):
            raise ValueError("A `select` step takes `n:` or `fraction:`, exactly one.")
        return self


class SelectTransform(Transform[SelectConfig]):
    """``select``: ``Indices(ranking.indices[:n])`` over the Dataset the ranking was computed on."""

    name: ClassVar[str] = "select"
    description: ClassVar[str] = "Keeps the top of a Prioritize ranking of the same Dataset."
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.DATASET),
        Port("ranking", DataType.OUTPUT, classes=(PrioritizeOutput,)),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)
    same_node: ClassVar[tuple[str, ...]] = ("ranking",)

    def run(
        self,
        config: SelectConfig,
        inputs: Mapping[str, Any],
        context: TransformContext,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        """The ranking's first `n` items, in ranked order."""
        from dataeval.data import Indices, View

        order = np.asarray(inputs["ranking"].value.indices)
        count = config.n if config.n is not None else math.ceil((config.fraction or 0.0) * len(order))
        return {"output": View(inputs["input"].value, Indices([int(index) for index in order[:count]]))}

    def digest(
        self,
        config: SelectConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],
    ) -> str:
        """The chosen indices, in order."""
        return indices_digest(outputs["output"].resolve_indices())
