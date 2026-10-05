"""The ``factor-leakage`` evaluator: the raw values of named metadata factors two sources hold."""

__all__ = ["FactorLeakageEvaluator"]

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

from dataeval import Metadata

from dataeval_flow._input_spec import InputKind
from dataeval_flow._policy import strip_row_level
from dataeval_flow.evaluators._core import execution
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.quality._config import FactorLeakageConfig
from dataeval_flow.evaluators.quality._result import FactorLeakageOutput

# Columns DataEval's per-item frame holds that are not factors.
_BOOKKEEPING = frozenset(
    {"level", "item_index", "target_index", "class_label", "score", "box", "item_id", "instance_index"}
)


def raw_values(metadata: Metadata, name: str, source: str) -> list[Any]:
    """The values `name` holds, one per item that has one, read whatever the policy excludes: the per-item frame keeps
    an excluded column, raw, even on Metadata loaded from Flow's cache."""
    frame = metadata.rows_at(metadata.item_level)
    columns = [column for column in frame.columns if column not in _BOOKKEEPING and not column.endswith("#")]
    if name in columns:
        found = name
    else:
        like = [column for column in columns if strip_row_level(column) == strip_row_level(name)]
        if not like:
            held = ", ".join(sorted(columns)) or "none"
            raise ValueError(f"Source '{source}' has no factor '{name}'; its metadata holds {held}.")
        found = like[0]
    return frame[found].drop_nulls().to_list()


class FactorLeakageEvaluator(Evaluator[FactorLeakageConfig, FactorLeakageOutput]):
    """``factor-leakage``: the raw values of named factors two sources hold, with each value's item count per side."""

    name: ClassVar[str] = "factor-leakage"
    title: ClassVar[str] = "Factor Leakage"
    description: ClassVar[str] = "The raw values of named metadata factors each of two sources holds."
    dataeval_class: ClassVar[Any] = Metadata
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.METADATA: "rows_at"}
    reads_factors: ClassVar[bool] = False

    def run(self, config: FactorLeakageConfig, inputs: Sequence[EvaluatorInputs]) -> FactorLeakageOutput:
        """Each named factor's values in both sources, counted per side."""
        sides = [(prepared.source, require(prepared.metadata, "metadata", prepared.source)) for prepared in inputs]
        started, clock = datetime.now(UTC), time.monotonic()
        factors: dict[str, dict[str, list[int]]] = {}
        for name in config.factors:
            counts = [Counter(raw_values(metadata, name, source)) for source, metadata in sides]
            values = dict.fromkeys([*counts[0], *counts[1]])
            factors[name] = {str(value): [counts[0][value], counts[1][value]] for value in values}
        meta = execution("dataeval_flow.factor_leakage", started, time.monotonic() - clock, {"factors": config.factors})
        return FactorLeakageOutput(
            {
                "sources": [source for source, _ in sides],
                "items": [int(metadata.item_count) for _, metadata in sides],
                "factors": factors,
            },
            meta,
        )
