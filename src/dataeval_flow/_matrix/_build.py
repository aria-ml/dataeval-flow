"""A task's matrix expanded into its runs, each a copy of the pipeline holding only the run's task, with the run's
overrides re-validated (task-matrix spec §3.5)."""

__all__ = ["MatrixError", "MatrixRunPlan", "expand_matrix"]

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from dataeval_flow._matrix._keys import Target, as_written, at, canonical, reads, set_override, walk
from dataeval_flow.config._schemas._matrix import grid_runs, run_label, show_value

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config._schemas._task import TaskConfig


class MatrixError(ValueError):
    """Every reason a task's matrix can't run, found at once."""


@dataclass(frozen=True)
class MatrixRunPlan:
    """One run of a matrix.

    Attributes
    ----------
    number
        The run's number, from 1 across the matrix.
    label
        Its grid's keys and values, ``key=value, key=value``.
    values
        The values it sets, by key as written.
    task
        The matrix task with the run's sources and extractor, and no matrix.
    pipeline
        A copy of the pipeline holding only `task`, with the run's entries, validated.
    """

    number: int
    label: str
    values: dict[str, Any]
    task: "TaskConfig"
    pipeline: "PipelineConfig"


def _dispatchers() -> dict[str, Callable[[Any], Any]]:
    from dataeval_flow.config._models import _evaluator_entry, _extractor_entry, _workflow_entry

    return {"evaluators": _evaluator_entry, "workflows": _workflow_entry, "extractors": _extractor_entry}


def _text(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return "; ".join(str(detail["msg"]).removeprefix("Value error, ") for detail in error.errors())
    return str(error)


def _message(task: "TaskConfig", problems: Sequence[str]) -> str:
    return f"Task '{task.name}' has a matrix that can't run:\n" + "\n".join(f"- {problem}" for problem in problems)


def expand_matrix(task: "TaskConfig", pipeline: "PipelineConfig") -> list[MatrixRunPlan]:
    """`task`'s runs, in order; ``[]`` for a task with no matrix.

    Raises :class:`MatrixError` naming every key that names nothing or a setting a matrix may not vary, every grid
    whose keys overlap, every run that does not validate or sets a pool entry it does not read, and every pair of runs
    that set the same settings.
    """
    if task.matrix is None:
        return []
    runs = grid_runs(task.matrix)
    targets, paths = _targets(runs, task, pipeline)
    problems: list[str] = []
    plans: list[MatrixRunPlan] = []
    seen: dict[str, int] = {}
    for number, (_, pairs) in enumerate(runs, start=1):
        label = run_label(pairs)
        try:
            plan, signature = _build(number, label, pairs, targets, paths, task, pipeline)
        except ValueError as error:
            problems.append(f"run {number} ({label}): {_text(error)}")
            continue
        if signature in seen:
            problems.append(f"runs {seen[signature]} and {number} ({label}) set the same settings; drop one")
            continue
        seen[signature] = number
        plans.append(plan)
    if problems:
        raise MatrixError(_message(task, problems))
    return plans


def _targets(
    runs: Sequence[tuple[int, Sequence[tuple[str, Any]]]], task: "TaskConfig", pipeline: "PipelineConfig"
) -> tuple[dict[str, Target], dict[str, tuple[str | int, ...]]]:
    """Each key's target, and the dump keys and list indices its path walks through a pool entry.

    Raises :class:`MatrixError` naming every key that names nothing or a setting a matrix may not vary, and every grid
    whose keys overlap, before any run is built.
    """
    problems: list[str] = []
    targets: dict[str, Target] = {}
    paths: dict[str, tuple[str | int, ...]] = {}
    for key in dict.fromkeys(key for _, pairs in runs for key, _ in pairs):
        try:
            targets[key] = target = canonical(key, task, pipeline)
            if target.pool != "task":
                root = next(entry for entry in getattr(pipeline, target.pool) if entry.name == target.entry)
                paths[key], _ = walk(root, target.path, _where(key, target, root))
        except ValueError as error:
            problems.append(str(error))
    if not problems:
        problems = _overlaps(runs, targets)
    if problems:
        raise MatrixError(_message(task, problems))
    return targets, paths


def _overlaps(runs: Sequence[tuple[int, Sequence[tuple[str, Any]]]], targets: dict[str, Target]) -> list[str]:
    """One problem per pair of keys in one grid where one names the other's setting or a part of it."""
    problems: list[str] = []
    grids = dict.fromkeys(grid for grid, _ in runs)
    for grid in grids:
        keys = list(dict.fromkeys(key for index, pairs in runs if index == grid for key, _ in pairs))
        for position, first in enumerate(keys):
            for second in keys[position + 1 :]:
                a, b = targets[first], targets[second]
                if a == b:
                    problems.append(f"`{first}` and `{second}` name the same setting; keep one")
                elif a.covers(b) or b.covers(a):
                    outer, inner = (first, second) if a.covers(b) else (second, first)
                    problems.append(
                        f"`{outer}` and `{inner}` are in one grid, and `{inner}` sets part of `{outer}`; vary them in "
                        "separate grids"
                    )
    return problems


def _known(pipeline: "PipelineConfig", field: str, value: Any) -> None:
    if field == "sources":
        names = [value] if isinstance(value, str) else value
        if not isinstance(names, list | tuple) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"`sources` value {show_value(value)} is not a source name or a list of them")
        defined = {source.name for source in pipeline.sources or ()}
        missing = [name for name in names if name not in defined]
        if missing:
            raise ValueError(f"`sources` value {show_value(value)} names no source `{missing[0]}`")
    elif value is not None and not isinstance(value, str):
        raise ValueError(f"`extractor` value {show_value(value)} is not an extractor name or null")
    elif value is not None and value not in {extractor.name for extractor in pipeline.extractors or ()}:
        raise ValueError(f"`extractor` value {show_value(value)} names no extractor `{value}`")


def _build(
    number: int,
    label: str,
    pairs: Sequence[tuple[str, Any]],
    targets: dict[str, Target],
    paths: dict[str, tuple[str | int, ...]],
    task: "TaskConfig",
    pipeline: "PipelineConfig",
) -> tuple[MatrixRunPlan, str]:
    from dataeval_flow.config._schemas._task import TaskConfig

    task_data = task.model_dump()
    task_data.pop("matrix", None)
    dumps: dict[tuple[str, str | None], dict[str, Any]] = {}
    walked: list[tuple[str, Target, tuple[str | int, ...]]] = []
    for key, value in pairs:
        target = targets[key]
        if target.pool == "task":
            _known(pipeline, target.path[0], value)
            task_data[target.path[0]] = value
            walked.append((key, target, ()))
            continue
        root = next(entry for entry in getattr(pipeline, target.pool) if entry.name == target.entry)
        dump = dumps.setdefault((target.pool, target.entry), as_written(root))
        set_override(dump, root, paths[key], value)
        walked.append((key, target, paths[key]))
    run_task = TaskConfig.model_validate(task_data)
    fields = {name: getattr(pipeline, name) for name in type(pipeline).model_fields}
    dispatch = _dispatchers()
    for (pool, name), dump in dumps.items():
        fields[pool] = [dispatch[pool](dump) if entry.name == name else entry for entry in fields[pool]]
    fields["tasks"] = [run_task]
    run_pipeline = type(pipeline).model_validate(fields)
    read = reads(run_task, run_pipeline)
    for key, target, _ in walked:
        if target.pool != "task" and (target.pool, target.entry) not in read:
            raise ValueError(
                f"the run reads no {target.pool[:-1]} `{target.entry}`, so `{key}` would change nothing; give it a "
                "grid of its own with the runs that read it"
            )
    settings = []
    for _, target, keys in walked:
        if target.pool == "task":
            value = run_task.source_names if target.path == ("sources",) else run_task.extractor
        else:
            value = at(next(e for e in getattr(run_pipeline, target.pool) if e.name == target.entry), keys)
        settings.append((str(target), to_jsonable_python(value, fallback=str)))
    signature = json.dumps(sorted(settings), sort_keys=True, default=str)
    return MatrixRunPlan(number, label, dict(pairs), run_task, run_pipeline), signature


def _where(key: str, target: Target, root: Any) -> str:
    """How a refusal names the entry a key walks: a bare key's by its type (``quality``), as the task's own
    entry; a pool-qualified or ``steps.`` key's by pool and name (``evaluator 'dupes'``)."""
    if key.split(".")[0] in ("evaluators", "workflows", "extractors", "steps"):
        return f"{target.pool[:-1]} '{target.entry}'"
    return str(getattr(root, "type", None) or f"{target.pool[:-1]} '{target.entry}'")
