"""`export`: a chain's Dataset written to disk, and recorded in the result (spec §6.3)."""

__all__ = ["ExportRecord", "ExportStepConfig", "ExportTransform"]

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import StepSkipped, Transform, TransformConfig, TransformContext

_SEGMENT = r"^[^/\\]+$"


class ExportStepConfig(TransformConfig):
    """An `export` step's settings: the format, what to do about an occupied destination, and where."""

    input: str = Field(description="The Dataset to write. Object detection only.")
    format: Literal["coco", "yolo", "huggingface_vision", "visdrone"] = Field(
        default="coco", description="The format to write."
    )
    mode: Literal["error", "replace", "append"] = Field(
        default="error",
        description="What to do when the destination holds a dataset: refuse, replace it, or add to it.",
    )
    ontology: str | None = Field(
        default=None, description="The ontology to record in the provenance, by `ontologies:` name."
    )
    to: str | None = Field(
        default=None,
        pattern=_SEGMENT,
        description="The directory under `<output>/datasets/`. Defaults to `<task>.<step>`.",
    )


class ExportRecord(BaseModel):
    """What an export step wrote."""

    path: str = Field(description="The directory written.")
    format: str = Field(description="The format written.")
    mode: str = Field(description="The mode it was written in.")
    items: int = Field(description="How many images it holds.")
    provenance: dict[str, Any] = Field(description="The provenance written beside it, in `provenance.json`.")


class ExportTransform(Transform[ExportStepConfig]):
    """``export``: write a Dataset under the run's output directory, and record where. Skipped without one."""

    name: ClassVar[str] = "export"
    description: ClassVar[str] = "Writes an object-detection Dataset to disk as COCO, YOLO or another datamaite format."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET, kinds=frozenset({"object_detection"})),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.EXPORT),)

    @classmethod
    def destinations(cls, config: ExportStepConfig, *, task: str, step: str) -> tuple[str, ...]:
        """``to``, or ``<task>.<step>``."""
        return (config.to or f"{task}.{step}",)

    def output_kinds(
        self,
        config: ExportStepConfig,  # noqa: ARG002
        input_kinds: Mapping[str, str | None],  # noqa: ARG002
    ) -> Mapping[str, str | None]:
        """An export makes no Dataset."""
        return {}

    def run(self, config: ExportStepConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        """Write the input, or skip when the run writes no files."""
        from dataeval_flow._export import write_node, write_source

        if context.output_dir is None:
            raise StepSkipped("the run has no output directory")
        node = inputs["input"]
        (to,) = self.destinations(config, task=context.task, step=context.step)
        dest = context.output_dir / "datasets" / to
        common = {
            "name": to,
            "format": config.format,
            "mode": config.mode,
            "ontology_owner": config,
            "dest": dest,
            "data_dir": context.data_dir,
        }
        if node.step is None and node.source is not None and context.pipeline is not None:
            path, provenance, items = write_source(node.source, config=context.pipeline, **common)
        else:
            lineage = context.lineage(node.address) if context.lineage is not None else []
            path, provenance, items = write_node(
                node.value, config=context.pipeline, lineage=lineage, task=context.task, step=context.step, **common
            )
        record = ExportRecord(
            path=str(path), format=config.format, mode=config.mode, items=items, provenance=provenance
        )
        return {"output": record}

    def section(self, record: Any) -> list[Any]:
        """Where the Dataset went, in which format, and how many images."""
        from dataeval_flow._blocks import Fields

        output = record.output
        return [Fields(items=[("Path", output.path), ("Format", output.format), ("Images", output.items)])]
