"""Extract the encoding descriptor a result was computed under.

Stage six of the lifecycle — lock the encoding in and commit it — reached from a result
somebody already has.  The record lives in the envelope, so the artifact is obtainable from
an archived ``result.json`` weeks later, which is the case where pinning an encoding
actually matters and the one where re-running to get it is what you are trying to avoid.
"""

__all__ = ["write_encoding"]

import json
import logging
from pathlib import Path
from typing import Any

_logger: logging.Logger = logging.getLogger(__name__)


def _binning_records(document: Any) -> dict[str, Any]:
    """Every task's binning record in a result file, keyed by task name.

    ``result.json`` is a mapping of task name to that task's result, so a file may hold
    several runs over one dataset.  Tasks that built no metadata simply have nothing here.
    """
    if not isinstance(document, dict):
        return {}
    return {
        name: result["metadata"]["metadata_binning"]
        for name, result in document.items()
        if isinstance(result, dict)
        and isinstance(result.get("metadata"), dict)
        and result["metadata"].get("metadata_binning")
    }


def _select(records: dict[str, Any], task: str | None) -> Any:
    """The one record to write, or a message saying why there is not one."""
    if not records:
        raise ValueError(
            "This result records no encodings. Only a workflow that builds metadata "
            "produces one — data-analysis, data-cleaning, data-coverage or ood-detection.",
        )
    if task is not None:
        if task not in records:
            raise ValueError(f"No task named {task!r} recorded an encoding. Available: {sorted(records)}.")
        return records[task]
    if len(records) == 1:
        return next(iter(records.values()))

    from dataeval_flow.binning import descriptor_from_record

    rendered = {name: json.dumps(descriptor_from_record(record), sort_keys=True) for name, record in records.items()}
    if len(set(rendered.values())) == 1:
        return next(iter(records.values()))
    raise ValueError(
        f"Tasks {sorted(records)} were encoded differently, so no one descriptor describes "
        "this result. Name the one you want with --task.",
    )


def write_encoding(result: Path, output: Path | None = None, task: str | None = None) -> int:
    """Write (or print) the encoding descriptor held in a result file.

    Parameters
    ----------
    result : Path
        A ``result.json`` written by a run.
    output : Path | None
        Where to write the descriptor.  Printed to stdout when omitted, so the command
        composes with a pipe as readily as it writes a file.
    task : str | None
        Which task's encoding to take, where a result holds several that differ.

    Returns
    -------
    int
        Process exit status: 0 on success, 1 with a message on any failure.
    """
    from dataeval_flow.binning import descriptor_from_record

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
