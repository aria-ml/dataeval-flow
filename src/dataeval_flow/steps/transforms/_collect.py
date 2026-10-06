"""`collect`: single Datasets gathered into one list, keyed by name, each handed on as it is."""

__all__ = ["CollectConfig", "CollectTransform", "collect_keys"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import Field, model_validator

from dataeval_flow._input_spec import SourceCount
from dataeval_flow.steps._address import Address, parse_address
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext

if TYPE_CHECKING:
    from dataeval_flow._blocks import Block


class CollectConfig(TransformConfig):
    """A `collect` step's inputs, one or more single Datasets, and the key each takes in the list."""

    input: list[str] = Field(min_length=1, description="The Datasets to gather, in order.")
    keys: list[str] | None = Field(
        default=None,
        description=(
            "Each element's key, one per input. Unset takes each from its address: `[key]` where written, else its "
            "output's name, else its step's."
        ),
    )

    @model_validator(mode="after")
    def _one_key_per_input(self) -> Self:
        if self.keys is not None and len(self.keys) != len(self.input):
            raise ValueError(
                f"`keys` names {len(self.keys)} elements, but `input` gathers {len(self.input)}: name one key per "
                "input."
            )
        if self.keys is not None and not all(self.keys):
            raise ValueError("`keys` holds an empty key: name each element.")
        seen: dict[str, str] = {}
        for text, key in zip(self.input, collect_keys(self), strict=True):
            if key in seen:
                raise ValueError(
                    f"`keys` names `{key}` twice."
                    if self.keys is not None
                    else f"`{seen[key]}` and `{text}` both take key `{key}`: name each element with `keys:`."
                )
            seen[key] = text
        return self


def collect_keys(config: CollectConfig) -> tuple[str, ...]:
    """Each element's key: `config.keys`, or each input address's `[key]`, else its output's name, else its name."""
    if config.keys is not None:
        return tuple(config.keys)
    return tuple(_key(parse_address(text)) for text in config.input)


def _key(address: Address) -> str:
    return address.key or address.output or address.name


class CollectTransform(Transform[CollectConfig]):
    """``collect``: its inputs as one list, keyed by name; each element is its input, unchanged."""

    name: ClassVar[str] = "collect"
    title: ClassVar[str] = "Collect"
    description: ClassVar[str] = "Gathers Datasets into one list, keyed by name, each as it is."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET, count=SourceCount.ONE_OR_MORE),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET, is_list=True),)

    @classmethod
    def output_keys(cls, config: CollectConfig) -> Mapping[str, tuple[str, ...]]:
        """The list's keys, fixed by the settings."""
        return {"output": collect_keys(config)}

    def run(
        self,
        config: CollectConfig,
        inputs: Mapping[str, Any],
        context: TransformContext,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        """The inputs, as one list keyed by name."""
        return {"output": dict(zip(collect_keys(config), (node.value for node in inputs["input"]), strict=True))}

    def details(
        self,
        config: CollectConfig,
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],  # noqa: ARG002
    ) -> dict[str, Any]:
        """Each key, and the address it holds."""
        return {"elements": dict(zip(collect_keys(config), config.input, strict=True))}

    def section(self, record: Any) -> list["Block"]:
        """What it gathered, in a sentence."""
        from dataeval_flow._blocks import Paragraph

        elements: dict[str, str] = (record.details or {}).get("elements") or {}
        listing = ", ".join(f"`{key}` holds `{address}`" for key, address in elements.items())
        return [Paragraph(text=f"Gathered {len(elements)} Dataset{'s' if len(elements) != 1 else ''}: {listing}.")]
