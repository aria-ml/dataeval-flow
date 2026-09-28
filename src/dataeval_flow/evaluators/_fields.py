"""What the built-in evaluator configs share without sharing a base: Flow's own fields, and DataEval's arguments."""

__all__ = ["FLOW_FIELDS", "dataeval_arguments", "require"]

from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

# Fields a built-in evaluator config holds for Flow: the entry's identity, the policies and ontology it names, and
# the chunking it asks for. Every other field is a DataEval argument, spelled as DataEval spells it.
FLOW_FIELDS: frozenset[str] = frozenset({"name", "type", "stats", "metadata", "ontology", "chunking"})


def dataeval_arguments(config: BaseModel) -> dict[str, Any]:
    """The DataEval arguments `config` sets: each field that is not one of Flow's and is not ``None``.

    An unset field is left out, so DataEval's own default applies.
    """
    values = {name: getattr(config, name) for name in type(config).model_fields if name not in FLOW_FIELDS}
    return {name: value for name, value in values.items() if value is not None}


def require(value: T | None, what: str, source: str) -> T:
    """`value`, or an error naming the source and the input its producer did not supply."""
    if value is None:
        raise ValueError(f"Source '{source}' arrived without {what}; its producer did not run.")
    return value
