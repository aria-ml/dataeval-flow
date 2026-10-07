"""`view`: DataEval view operations over a Dataset, as a source's view applies them."""

__all__ = ["ViewTransform", "ViewTransformConfig", "root_indices"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field, model_validator

from dataeval_flow._view import root_indices
from dataeval_flow.config._schemas._view import ViewOperation
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig

_GENERIC = (None, "any_target", "image_only")


class ViewTransformConfig(TransformConfig):
    """A `view` step's settings: its input, and the operations to apply, inline or by a `views:` name."""

    input: str = Field(description="The Dataset to view.")
    operations: list[ViewOperation] | None = Field(
        default=None, description="The `dataeval.data` operations to apply, in order, as a `views:` entry writes them."
    )
    view: str | None = Field(default=None, description="A `views:` entry whose operations to apply instead.")

    @model_validator(mode="after")
    def _one_source_of_operations(self) -> "ViewTransformConfig":
        if (self.operations is None) == (self.view is None):
            raise ValueError("A `view` step takes `operations:` or `view:`, exactly one.")
        return self


class ViewTransform(Transform[ViewTransformConfig]):
    """``view``: DataEval view operations, such as ``ClassFilter``, ``Relabel``, ``Limit`` or ``Indices``."""

    name: ClassVar[str] = "view"
    title: ClassVar[str] = "View"
    description: ClassVar[str] = "Applies DataEval view operations, such as ClassFilter, Relabel, Limit or Indices."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    @classmethod
    def resolved(cls, config: ViewTransformConfig, pipeline: "PipelineConfig") -> ViewTransformConfig:
        """The config with a `view:` name replaced by that view's operations."""
        if config.view is None:
            return config
        view = next((entry for entry in pipeline.views or () if entry.name == config.view), None)
        if view is None:
            raise ValueError(f"`view: {config.view}` names a view that `views:` does not define.")
        return config.model_copy(update={"operations": list(view.operations), "view": None})

    def output_kinds(
        self, config: ViewTransformConfig, input_kinds: Mapping[str, str | None]
    ) -> Mapping[str, str | None]:
        """The input's kind, after checking that each operation's ``requires`` takes it."""
        from dataeval_flow._view import build_operations

        kind = input_kinds.get("input")
        if kind is not None:
            for operation in build_operations(config.operations or ()):
                required = getattr(operation, "requires", None)
                if required not in _GENERIC and required != kind:
                    raise ValueError(
                        f"`{type(operation).__name__}` needs a {required} Dataset, but its input is {kind}."
                    )
        return {"output": kind}

    def run(
        self,
        config: ViewTransformConfig,
        inputs: Mapping[str, Any],
        context: TransformContext,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        """The input, viewed through the operations."""
        from dataeval_flow._view import build_view

        return {"output": build_view(inputs["input"].value, list(config.operations or ()))}

    def details(
        self,
        config: ViewTransformConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """The view's items as indices into the dataset at the bottom of its views; none where it keeps every item of
        its input in order (data-splitting spec §5.4)."""
        from dataeval.data import View

        output = outputs["output"]
        if not isinstance(output, View) or list(output.resolve_indices()) == list(range(len(inputs["input"].value))):
            return None
        return {"indices": root_indices(output)}
