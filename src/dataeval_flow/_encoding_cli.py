"""Extract the encoding descriptor a result was computed under, for review and commit.

The record lives in the result's metadata envelope, so the artifact can be written out
from an archived ``result.json`` without re-running the task.
"""

__all__ = ["write_encoding"]

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger: logging.Logger = logging.getLogger(__name__)


def _rendered(name: str, record: Any) -> str | None:
    """One task's descriptor as bytes, or None where it has none to write."""
    from dataeval_flow._binning import descriptor_from_record

    try:
        return json.dumps(descriptor_from_record(record), sort_keys=True)
    except ValueError:
        _logger.debug("Task %r records no descriptor", name, exc_info=True)
        return None


@dataclass(frozen=True)
class _Disagreeing:
    """A matrix task whose runs were encoded differently, so no one descriptor describes it."""

    task: str
    runs: list[str]

    def problem(self) -> str:
        return (
            f"Matrix task {self.task!r}: {', '.join(self.runs)} were encoded differently, "
            "so no one descriptor describes it."
        )


def agreed(records: Sequence[tuple[str, Any]]) -> tuple[Any | None, list[str]]:
    """The record every one with a descriptor agrees on, and the names of those that differ.

    Two records agree when their rendered descriptors are equal; a record with none is skipped, as across tasks.
    ``(None, [])`` when none has a descriptor.
    """
    rendered = {name: text for name, record in records if record and (text := _rendered(name, record)) is not None}
    if not rendered:
        return None, []
    if len(set(rendered.values())) == 1:
        return dict(records)[next(iter(rendered))], []
    return None, sorted(rendered)


def _binning_records(document: Any) -> dict[str, Any]:
    """Every task's binning record in a result file, keyed by task name.

    ``result.json`` is a mapping of task name to that task's result, so a file may hold
    several runs over one dataset.  Tasks that built no metadata have no record here. A matrix task
    has one record where its runs agree, and a ``_Disagreeing`` where they do not.
    """
    if not isinstance(document, dict):
        return {}
    records: dict[str, Any] = {}
    for name, result in document.items():
        if not isinstance(result, dict):
            continue
        if result.get("kind") == "matrix":
            runs = [
                (f"run {run.get('number')}", ((run.get("result") or {}).get("metadata") or {}).get("metadata_binning"))
                for run in result.get("runs") or []
            ]
            record, differing = agreed(runs)
            if differing:
                records[name] = _Disagreeing(name, differing)
            elif record:
                records[name] = record
        elif isinstance(result.get("metadata"), dict) and result["metadata"].get("metadata_binning"):
            records[name] = result["metadata"]["metadata_binning"]
    return records


def _unless_disagreeing(record: Any) -> Any:
    """*record*, or the refusal where it is a matrix whose runs were encoded differently."""
    if isinstance(record, _Disagreeing):
        raise ValueError(record.problem())
    return record


def _select(records: dict[str, Any], task: str | None) -> Any:
    """The one record to write, or a message saying why there is not one."""
    if not records:
        raise ValueError(
            "This result records no encodings. Only a result that built metadata records one: a custom "
            "workflow or preset whose steps read metadata.",
        )
    if task is not None:
        if task not in records:
            raise ValueError(f"No task named {task!r} recorded an encoding. Available: {sorted(records)}.")
        return _unless_disagreeing(records[task])
    if len(records) == 1:
        return _unless_disagreeing(next(iter(records.values())))
    for chosen in records.values():
        _unless_disagreeing(chosen)

    # Tasks whose record yields no descriptor are skipped: they give no information about
    # whether the rest agree, and refusing on them would hide every task that has an
    # encoding.
    chosen, differing = agreed(list(records.items()))
    if chosen is not None:
        return chosen
    if not differing:
        raise ValueError(
            "This result records no encodings. Only a result that built metadata records one: a custom "
            "workflow or preset whose steps read metadata.",
        )
    raise ValueError(
        f"Tasks {differing} were encoded differently, so no one descriptor describes "
        "this result. Name the one you want with --task.",
    )


def write_encoding(result: Path, output: Path | None = None, task: str | None = None) -> int:
    """Write (or print) the encoding descriptor held in a result file.

    Parameters
    ----------
    result : Path
        A ``result.json`` written by a run.
    output : Path | None
        Where to write the descriptor.  Printed to stdout when omitted.
    task : str | None
        Which task's encoding to take, where a result holds several that differ.

    Returns
    -------
    int
        Process exit status: 0 on success, 1 with a message on any failure.
    """
    from dataeval_flow._binning import descriptor_from_record

    try:
        document = json.loads(Path(result).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _logger.error("Cannot read %s: %s", result, exc)
        return 1

    try:
        descriptor = descriptor_from_record(_select(_binning_records(document), task))
    except ValueError as exc:
        _logger.error("%s", exc)
        return 1

    # Sorted keys and a fixed indent, matching what DataEval writes, so the same encoding
    # produces the same bytes and a change to one factor reads as a change to one factor.
    text = json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
        return 0

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(text, encoding="utf-8")
    _logger.info(
        "Wrote %s (%d factors). Commit it, and reference it from a metadata policy's `encoding`.",
        output,
        len(descriptor.get("factors", {})),
    )
    return 0
