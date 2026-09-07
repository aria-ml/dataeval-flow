"""Export configuration schema — write a source out as a dataset."""

__all__ = ["ExportConfig"]

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExportConfig(BaseModel):
    """A named dataset to write, and the format to write it in.

    Declare an export to take a conformed corpus out of the tool. It is written to
    ``output/datasets/<name>/`` when the run has an output directory, beside
    ``output/results/``. An export names a source, not a task, so it writes whether or
    not any task reads that source.

    YAML example::

        exports:
          - name: conformed_corpus
            source: merged
            format: coco
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    name: str = Field(description="Identifier for the export, and the directory it is written to.")
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
            "so a re-run cannot overwrite a corpus somebody is using. `replace` clears it "
            "first; `append` writes into it and may leave stale files behind."
        ),
    )
    ontology: str | None = Field(
        default=None,
        description=(
            "Label space the emitted dataset's labels are read under, by name under the "
            "top-level `ontologies:` key or as a path. Written into the dataset's "
            "provenance so the emitted corpus carries the same digest as the run and the "
            "audit. An export names no workflow, so it cannot inherit one."
        ),
    )

    @field_validator("name")
    @classmethod
    def _one_directory_segment(cls, value: str) -> str:
        """Refuse a name that is not a single safe directory segment.

        The name becomes a directory under the run's output. A separator or a `..` in it
        would write the corpus somewhere the caller never named.
        """
        if not value:
            raise ValueError("Export name must not be empty. Give it a plain name, such as 'conformed_corpus'.")
        if "/" in value or "\\" in value:
            raise ValueError(
                f"Export name '{value}' must be one directory segment and cannot contain '/' or '\\'. "
                "It names a directory under the run's output, not a path to write to."
            )
        if value in {".", ".."}:
            raise ValueError(
                f"Export name '{value}' names a relative path rather than a directory. "
                "Give it a plain name, such as 'conformed_corpus'."
            )
        return value
