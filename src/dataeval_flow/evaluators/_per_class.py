"""A run per class or class group: the inputs sliced by key, the evaluator called once per key (spec §5.9)."""

__all__ = ["PerClassOutput", "require_one_label_per_item", "run_per_class", "serialize_per_class", "split_keys"]

import dataclasses
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from dataeval_flow._result import failure_message
from dataeval_flow.evaluators._core import execution
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators._serialize import serialize_output

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dataeval_flow.evaluators._evaluator import Evaluator
    from dataeval_flow.steps._by import ByConfig


class PerClassOutput:
    """What an evaluate step with `by:` returns: each key's Output, and each key left out with why.

    ``outputs`` holds the evaluator's own Output per key, in key order: classes in ascending class index, or groups
    in the order written. ``skipped`` holds each key left out, with why. ``label`` says what a key is, ``"class"`` or
    ``"group"``.
    """

    def __init__(self, outputs: Mapping[str, Any], skipped: Mapping[str, str], *, label: str, meta: Any) -> None:
        self.outputs: dict[str, Any] = dict(outputs)
        self.skipped: dict[str, str] = dict(skipped)
        self.label = label
        self._meta = meta

    def data(self) -> dict[str, Any]:
        """Each key's Output, and each key skipped with why."""
        return {"classes": dict(self.outputs), "skipped": dict(self.skipped)}

    def meta(self) -> Any:
        """The run's ``ExecutionMetadata``."""
        return self._meta


def require_one_label_per_item(datasets: Mapping[str, Any], inputs: Sequence[EvaluatorInputs]) -> None:
    """Refuse an input without one label per item: `by: class` keys items, so each needs exactly one class."""
    from dataeval_flow._chain._preflight import detect_kind

    for (name, dataset), prepared in zip(datasets.items(), inputs, strict=True):
        kind = detect_kind(dataset)
        labels = prepared.labels
        if kind != "classification" or labels is None or len(labels) != len(dataset):
            raise ValueError(
                f"`by: class` needs one label per item (image classification), and `{name}` is "
                f"{kind if kind not in (None, 'classification') else 'unlabelled'}."
            )


def split_keys(inputs: Sequence[EvaluatorInputs], by: "ByConfig") -> tuple[dict[str, list[Any]], dict[str, str]]:
    """Each key's item mask per input, in key order; and each key left out, with why."""
    first = inputs[0]
    names = {int(index): name for index, name in (first.index2label or {}).items()}
    for other in inputs[1:]:
        for index, name in (other.index2label or {}).items():
            if int(index) in names and names[int(index)] != name:
                raise ValueError(
                    f"`{other.source}` names class {index} `{name}`, `{first.source}` `{names[int(index)]}`: "
                    "conform it first."
                )
    # `require_one_label_per_item` has refused any input without labels.
    labels = [cast("NDArray[np.intp]", prepared.labels) for prepared in inputs]
    present = sorted({int(label) for own in labels for label in np.unique(own)})
    skipped: dict[str, str] = {}
    members: dict[str, list[int]] = {}
    if by.class_.groups is None:
        members = {names.get(index, str(index)): [index] for index in present}
    else:
        reverse = {name: index for index, name in names.items()}
        for group, listed in by.class_.groups.items():
            members[group] = []
            for member in listed:
                if isinstance(member, str) and member not in reverse:
                    raise ValueError(
                        f"Group `{group}` names class `{member}`, which `{first.source}`'s index2label does not have."
                    )
                members[group].append(reverse[member] if isinstance(member, str) else member)
        grouped = {index for indices in members.values() for index in indices}
        skipped.update({names.get(index, str(index)): "in no group" for index in present if index not in grouped})
    masks: dict[str, list[Any]] = {}
    for key, indices in members.items():
        per_input = [np.isin(own, indices) for own in labels]
        short = [
            (prepared.source, count)
            for prepared, mask in zip(inputs, per_input, strict=True)
            if (count := int(mask.sum())) < by.class_.min_items
        ]
        if short:
            source, count = short[0]
            noun = "item" if count == 1 else "items"
            skipped[key] = f"{count} {noun} in `{source}`, fewer than `min_items` {by.class_.min_items}"
        else:
            masks[key] = per_input
    return masks, skipped


def run_per_class(
    evaluator: "Evaluator[Any, Any]", config: Any, inputs: Sequence[EvaluatorInputs], by: "ByConfig"
) -> PerClassOutput:
    """Call `evaluator` once per key, on each input's items of that key.

    A key whose run raises is skipped with its error, and the other keys still run; when every key raises, the first
    error is raised.
    """
    started, clock = datetime.now(UTC), time.monotonic()
    masks, skipped = split_keys(inputs, by)
    outputs: dict[str, Any] = {}
    errors: list[Exception] = []
    for key, per_input in masks.items():
        sliced = [_sliced(prepared, mask) for prepared, mask in zip(inputs, per_input, strict=True)]
        try:
            outputs[key] = evaluator.run(config, sliced)
        except Exception as error:  # noqa: BLE001 - a key it cannot run, such as a class too small, is skipped
            errors.append(error)
            skipped[key] = failure_message(error)
    if errors and not outputs:
        raise errors[0]
    meta = execution(f"{evaluator.name} by {by.label}", started, time.monotonic() - clock, {"by": by.model_dump()})
    return PerClassOutput(outputs, skipped, label=by.label, meta=meta)


def _sliced(prepared: EvaluatorInputs, mask: Any) -> EvaluatorInputs:
    """`prepared` narrowed to the items `mask` selects: its embeddings and labels."""
    updates = {name: value[mask] for name in ("embeddings", "labels") if (value := getattr(prepared, name)) is not None}
    return dataclasses.replace(prepared, **updates)


def serialize_per_class(output: PerClassOutput, extras: Sequence[str] = ()) -> dict[str, Any]:
    """Each key's Output as the evaluator's own JSON, and each skipped key with why."""
    return {
        "shape": "per_class",
        "key": output.label,
        "classes": {key: serialize_output(inner, extras=extras) for key, inner in output.outputs.items()},
        "skipped": dict(output.skipped),
    }
