"""Task configuration schema."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    GetJsonSchemaHandler,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

AutoBinMethod = Literal["uniform_width", "uniform_count", "clusters"]
FactorSource = Literal["coded", "values", "auto"]

TaskKind = Literal["workflow", "evaluator"]


class TaskConfig(BaseModel):
    """Task/workflow configuration schema.

    Tasks reference sources (dataset+selection bundles) and an optional
    extractor (model+preprocessor+batch_size bundle) by name.

    A config file names what a task runs under ``workflow:`` or ``evaluator:``, so the
    file says which kind of task it is. Loaded, either lands in ``workflow``, and ``kind``
    records which key the file used. Running a task reads ``kind`` only to look the name
    up in ``workflows:`` or ``evaluators:``; saved, the task is written back under its key.
    """

    name: str = Field(description="Identifier for the task: the key of its result, and what `--task` selects.")
    workflow: str = Field(
        description=(
            "What this task runs, by name: an entry of `workflows:` or of `evaluators:`, as `kind` says. "
            "A config file names an evaluator under `evaluator:` instead."
        )
    )
    kind: TaskKind = Field(
        default="workflow",
        description=(
            "Whether `workflow` names a workflow or an evaluator, read from the key the config file used. "
            "A result carries the same value."
        ),
    )
    enabled: bool = Field(default=True, description="Whether this task is included when running the pipeline.")
    sources: str | Sequence[str] = Field(
        description="The sources this task reads, by name: one, or several in the order the workflow reads them."
    )
    extractor: str | None = Field(
        default=None, description="The extractor this task embeds with, by name. Leave unset when it needs none."
    )

    @property
    def source_names(self) -> list[str]:
        """The sources this task reads, as a list whether one or several were named."""
        return [self.sources] if isinstance(self.sources, str) else list(self.sources)

    @model_validator(mode="before")
    @classmethod
    def _read_target_key(cls, data: Any) -> Any:
        """Read the file's ``workflow:`` or ``evaluator:`` into ``workflow`` and ``kind``.

        An empty key (``null`` in YAML, or a TUI form's unset field) counts as absent.
        """
        if not isinstance(data, Mapping):
            return data
        name = data.get("name", "<unnamed>")
        workflow, evaluator = data.get("workflow"), data.get("evaluator")
        if workflow and evaluator:
            raise ValueError(f"Task '{name}' names both a workflow and an evaluator; a task runs exactly one.")
        if not workflow and not evaluator:
            raise ValueError(
                f"Task '{name}' names neither a workflow nor an evaluator; set `workflow:` or `evaluator:`."
            )
        fields = {key: value for key, value in data.items() if key != "evaluator"}
        if evaluator:
            if fields.get("kind", "evaluator") != "evaluator":
                raise ValueError(f"Task '{name}' names an evaluator, so its kind cannot be '{fields['kind']}'.")
            fields.update(workflow=evaluator, kind="evaluator")
        return fields

    @model_serializer(mode="wrap")
    def _write_target_key(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Write the task as a config file names it: under ``workflow:`` or ``evaluator:``, with no ``kind``."""
        data = handler(self)
        data.pop("kind", None)
        if self.kind == "evaluator" and "workflow" in data:
            data["evaluator"] = data.pop("workflow")
        return data

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Describe the file's shape: exactly one of ``workflow:`` and ``evaluator:``, and no ``kind``."""
        schema = handler.resolve_ref_schema(handler(core_schema))
        properties = schema["properties"]
        properties.pop("kind", None)
        properties["workflow"]["description"] = "The workflow this task runs, by its name in `workflows:`."
        properties["evaluator"] = {
            "type": "string",
            "title": "Evaluator",
            "description": "The evaluator this task runs, by its name in `evaluators:`. Set this or `workflow`.",
        }
        schema["required"] = [key for key in schema["required"] if key != "workflow"]
        schema["oneOf"] = [{"required": ["workflow"]}, {"required": ["evaluator"]}]
        return schema
