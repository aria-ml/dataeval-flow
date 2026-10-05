"""Export configuration schema — write a source out as a dataset."""

__all__ = ["ExportConfig"]

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ONE_DIRECTORY_SEGMENT = r"^(?!\.\.?$)[^/\\]+$"
"""What :func:`one_directory_segment` accepts, as a JSON schema pattern, so an editor flags what loading refuses.

It is written only into the schema: the validator refuses at load, with a message saying why.
"""


class ExportConfig(BaseModel):
    """A named dataset to write, and the format to write it in.

    Declare an export to take a conformed dataset out of the tool. It is written to
    ``output/datasets/<name>/`` when the run has an output directory, beside
    ``output/results/``. An export names a source, not a task, so it writes whether or
    not any task reads that source.

    YAML example::

        exports:
          - name: conformed_dataset
            source: merged
            format: coco
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str = Field(
        description="Identifier for the export, and the directory it is written to.",
        json_schema_extra={"pattern": ONE_DIRECTORY_SEGMENT},
    )
    source: str = Field(description="Reference to a source name.")
    format: Literal["coco", "yolo", "huggingface_vision", "visdrone"] = Field(
        default="coco",
        description=(
            "Output format, from datamaite's object-detection writers. Pinned rather than "
            "left open so a typo is a config error and not a failure after the walk."
        ),
    )
    mode: Literal["error", "replace", "append"] = Field(
        default="error",
        description=(
            "What to do when the destination already holds a dataset. The default refuses, "
            "so a re-run cannot overwrite a dataset somebody is using. `replace` clears it "
            "first; `append` writes into it and may leave stale files behind."
        ),
    )
    ontology: str | None = Field(
        default=None,
        description=(
            "Label space the emitted dataset's labels are read under, by name under the "
            "top-level `ontologies:` key or as a path. Written into the dataset's "
            "provenance so the emitted dataset carries the same digest as the run and the "
            "label-space run. An export names no workflow, so it cannot inherit one."
        ),
    )

    @field_validator("name")
    @classmethod
    def _one_directory_segment(cls, value: str) -> str:
        """Refuse a name that is not a single safe directory segment."""
        return one_directory_segment(value, what="Export name")


def one_directory_segment(value: str, *, what: str) -> str:
    """Refuse `value` unless it is a single safe directory segment; `what` names it in the message.

    The value becomes a directory under the run's output. A separator in it would write the
    dataset somewhere the caller never named, and `.` or `..` would name the directory holding
    every other dataset, or the run's whole output, which `mode: replace` then clears.
    """
    if not value:
        raise ValueError(f"{what} must not be empty. Give it a plain name, such as 'conformed_dataset'.")
    if "/" in value or "\\" in value:
        raise ValueError(
            f"{what} '{value}' must be one directory segment and cannot contain '/' or '\\'. "
            "It names a directory under the run's output, not a path to write to."
        )
    if value in {".", ".."}:
        raise ValueError(
            f"{what} '{value}' names a relative path rather than a directory. "
            "Give it a plain name, such as 'conformed_dataset'."
        )
    return value
