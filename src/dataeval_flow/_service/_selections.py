"""Selections over a run: a profile's bin, range, categories or missing values, or a table's flagged rows, resolved
exactly and served a page at a time.

A selection is its definition: the task and field it reads, the scope, and a predicate. Its id is the hash of that
definition, saved under the run's ``selections/`` directory, so the same request always gives the same id. Members are
read from what the run wrote, a profile's rows under ``results/profiles/`` or a result's table, which never change once
written, so every page of a selection is read from the same rows and the pages together hold each member once.
Nothing is measured again.
"""

from __future__ import annotations

__all__ = ["SelectionError", "SelectionRequest", "Selections", "flagged", "matching"]

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow.evaluators.quality._profile import as_json

if TYPE_CHECKING:
    from dataeval_flow._service._store import RunStore

SCHEMA = 1
_ID = re.compile(r"[0-9a-f]{16}")
_KEYS = ["item", "target"]
_KIND = ("body", "predicate", "kind")


class _Predicate(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bin(_Predicate):
    """One bin of a number's histogram, by its position: closed on the left, and the last on the right too."""

    kind: Literal["bin"]
    index: int = Field(ge=0)


class Range(_Predicate):
    """The finite values between two bounds, either left out for no bound."""

    kind: Literal["range"]
    min: float | None = None
    max: float | None = None
    include_min: bool = True
    include_max: bool = False


class Category(_Predicate):
    """The rows holding any of these values; ``1``, ``"1"`` and ``true`` are three values."""

    kind: Literal["category"]
    values: list[str | bool | int | float | None] = Field(min_length=1)


class Other(_Predicate):
    """The rows holding a value the profile did not name, its other group."""

    kind: Literal["other"]


class Missing(_Predicate):
    """The rows with no value."""

    kind: Literal["missing"]


class NonFinite(_Predicate):
    """The rows holding NaN or an infinity."""

    kind: Literal["non_finite"]


class Rows(_Predicate):
    """A table's rows, each a flag on an item or a box, optionally only those of one metric."""

    kind: Literal["rows"]
    metric: str | None = None


class SelectionRequest(BaseModel):
    """What to select: a field of a ``profile`` task with a predicate over its rows, or a ``rows`` predicate over the
    table a task (or one of its steps) wrote, such as an ``outliers`` table."""

    model_config = ConfigDict(extra="forbid")
    task: str = Field(description="The task whose result is selected from.")
    step: str | None = Field(default=None, description="A step of the task's chain, for a table a step wrote.")
    field: str | None = Field(default=None, description="The profile field, by name; unset for a `rows` predicate.")
    scope: Literal["image", "target"] | None = Field(
        default=None, description="`image` or `target`; needed only where the field has both."
    )
    origin: Literal["computed", "supplied"] | None = Field(
        default=None,
        description="`computed` or `supplied`; needed only where a statistic and a metadata field share it.",
    )
    predicate: Annotated[Bin | Range | Category | Other | Missing | NonFinite | Rows, Field(discriminator="kind")]


class SelectionError(Exception):
    """A selection this run cannot answer, with the HTTP status that says why: 422 for a request the run's pipeline
    cannot answer, 409 for a result not written yet, 404 for an unknown selection."""

    def __init__(self, status: int, message: str, loc: Sequence[str] = ("body",)) -> None:
        super().__init__(message)
        self.status = status
        self.loc = list(loc)
        """Where in the request a 422 is, as Pydantic locates a field it refuses."""


def matching(rows: pl.DataFrame, column: str, predicate: Mapping[str, Any], field: Mapping[str, Any]) -> pl.DataFrame:
    """The profile rows of `column` the predicate holds, by item and target: a bin as its histogram counts it, a range
    as its bounds say and never a non-finite value, a category or the other group by each value's JSON."""
    kind, value, numeric = predicate["kind"], pl.col(column), field.get("type") == "numeric"
    if kind == "missing":
        return _sorted(rows.filter(value.is_null()))
    if kind in ("bin", "range", "non_finite") and not numeric:
        raise SelectionError(422, f"`{kind}` selects numbers, and this field is {field.get('type')}", _KIND)
    if kind in ("category", "other") and numeric:
        raise SelectionError(422, f"`{kind}` selects categories, and this field is numeric", _KIND)
    if kind == "non_finite":
        return _sorted(rows.filter(value.is_not_null() & ~value.is_finite()))
    if kind == "category":
        return _sorted(rows.filter(value.is_in([as_json(item) for item in predicate["values"]])))
    if kind == "other":
        named = [as_json(entry["value"]) for entry in field.get("values") or []]
        return _sorted(rows.filter(value.is_not_null() & ~value.is_in(named)))
    low, high, include_low, include_high = _bounds(predicate, field)
    condition = value.is_finite()
    if low is not None:
        condition &= value >= low if include_low else value > low
    if high is not None:
        condition &= value <= high if include_high else value < high
    return _sorted(rows.filter(condition))


def _bounds(predicate: Mapping[str, Any], field: Mapping[str, Any]) -> tuple[Any, Any, bool, bool]:
    """A range's bounds as given, or a bin's edges: closed on the left, and the last bin on the right too."""
    if predicate["kind"] == "range":
        return (
            predicate.get("min"),
            predicate.get("max"),
            predicate.get("include_min", True),
            predicate.get("include_max", False),
        )
    edges = (field.get("histogram") or {}).get("edges") or []
    number = predicate["index"]
    if not 0 <= number < len(edges) - 1:
        message = f"This field's histogram has {max(len(edges) - 1, 0)} bins, so no bin {number}"
        raise SelectionError(422, message, ("body", "predicate", "index"))
    return edges[number], edges[number + 1], True, number == len(edges) - 2


def _sorted(rows: pl.DataFrame) -> pl.DataFrame:
    return rows.sort(_KEYS, nulls_last=False)


def flagged(rows: Sequence[Mapping[str, Any]], metric: str | None, scope: str | None) -> list[Mapping[str, Any]]:
    """A table's rows, in its order: those of `metric` where one is named, and of the scope where one is named, a row
    with no ``target_index`` being an image's."""
    kept = []
    for row in rows:
        on_target = row.get("target_index") is not None
        if metric is not None and row.get("metric_name") != metric:
            continue
        if scope is not None and on_target != (scope == "target"):
            continue
        kept.append(row)
    return kept


class Selections:
    """The selections of the runs in `store`."""

    def __init__(self, store: RunStore) -> None:
        self._store = store

    def create(self, run_id: str, request: SelectionRequest) -> dict[str, Any]:
        """Resolve `request` against the run, save its definition under its id, and summarize its members."""
        definition = self._define(run_id, request)
        text = json.dumps(definition, sort_keys=True, separators=(",", ":"))
        selection_id = hashlib.sha256(f"{run_id}:{text}".encode()).hexdigest()[:16]
        path = self._path(run_id, selection_id)
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(text)
            temporary.replace(path)
        return self._summary(selection_id, definition, self._members(run_id, definition))

    def page(self, run_id: str, selection_id: str, offset: int, limit: int) -> dict[str, Any]:
        """The selection's summary and a page of its members, in a fixed order."""
        definition = self._load(run_id, selection_id)
        members = self._members(run_id, definition)
        page = {"offset": offset, "limit": limit, "members": members[offset : offset + limit]}
        return self._summary(selection_id, definition, members) | page

    def view(self, run_id: str, selection_id: str, *, parents: bool) -> dict[str, Any]:
        """The selection's images as a new source over the same data: the source's view, then an ``Indices`` keeping
        them. Merged into the run's pipeline, a later run reads exactly these images."""
        definition = self._load(run_id, selection_id)
        members = self._members(run_id, definition)
        if not parents and any(member["target"] is not None for member in members):
            raise SelectionError(
                422,
                "This selection holds boxes, not images. Ask for the images that hold them with `parents=true`; "
                "keeping only these boxes, or removing them, is not supported yet.",
                ("query", "parents"),
            )
        pipeline = self._store.get(run_id)["pipeline"]
        source = next(entry for entry in pipeline.get("sources") or [] if entry.get("name") == definition["source"])
        views = {entry["name"]: entry for entry in pipeline.get("views") or []}
        operations = list(views[source["view"]]["operations"]) if source.get("view") else []
        name = f"{definition['source']}-{selection_id}"
        indices = sorted({member["index"] for member in members})
        operations.append({"type": "Indices", "params": {"indices": indices}})
        return {
            "views": [{"name": name, "operations": operations}],
            "sources": [{**source, "name": name, "view": name}],
        }

    def _define(self, run_id: str, request: SelectionRequest) -> dict[str, Any]:
        """The request with its scope and source resolved, checked against what the run wrote."""
        predicate = request.predicate.model_dump()
        output = self._output(run_id, request.task, request.step)
        if predicate["kind"] == "rows":
            if output.get("shape") != "table" or "item_index" not in (output.get("columns") or []):
                raise SelectionError(422, f"Task '{request.task}' wrote no table of items to select rows from", _KIND)
            source = self._row_source(run_id, request.task)
            return self._reproducible(
                run_id,
                {
                    "schema": SCHEMA,
                    "task": request.task,
                    "step": request.step,
                    "field": None,
                    "scope": request.scope,
                    "source": source,
                    "predicate": predicate,
                },
            )
        data = output.get("data") or {}
        fields = [field for field in data.get("fields") or [] if field.get("name") == request.field]
        if not fields:
            raise SelectionError(422, f"Task '{request.task}' profiled no field '{request.field}'", ("body", "field"))
        fields = [field for field in fields if request.scope in (None, field["scope"])]
        fields = [field for field in fields if request.origin in (None, field["origin"])]
        if len({field["scope"] for field in fields}) > 1:
            message = f"Field '{request.field}' was profiled per image and per box: name a `scope`"
            raise SelectionError(422, message, ("body", "scope"))
        if len(fields) > 1:
            message = f"Field '{request.field}' is a statistic and a metadata field: name an `origin`"
            raise SelectionError(422, message, ("body", "origin"))
        if not fields or fields[0].get("type") == "unsupported":
            raise SelectionError(422, f"Field '{request.field}' has no rows to select there", ("body", "field"))
        (field,) = fields
        definition = {
            "schema": SCHEMA,
            "task": request.task,
            "step": None,
            "field": request.field,
            "scope": field["scope"],
            "origin": field["origin"],
            "source": data.get("source"),
            "predicate": predicate,
        }
        return self._reproducible(run_id, definition)

    def _reproducible(self, run_id: str, definition: dict[str, Any]) -> dict[str, Any]:
        """`definition`, once the run is known to have drawn the source's items alike in every task, so its rows' items
        resolve, and its predicate is known to be answerable, such as a bin the histogram has."""
        from dataeval_flow._sources import drawn_per_task
        from dataeval_flow.config._loader import load_config

        if reason := drawn_per_task(load_config(self._store.directory(run_id) / "pipeline.json"), definition["source"]):
            raise SelectionError(422, reason, ("body", "task"))
        self._members(run_id, definition)
        return definition

    def _members(self, run_id: str, definition: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Every member of the selection, by item and target, in a fixed order."""
        source, predicate = definition["source"], definition["predicate"]
        output = self._output(run_id, definition["task"], definition["step"])
        if predicate["kind"] == "rows":
            rows = flagged(output.get("rows") or [], predicate.get("metric"), definition["scope"])
            return [
                {"source": source, "index": row["item_index"], "target": row.get("target_index"), **row} for row in rows
            ]
        field = next(
            entry
            for entry in output["data"]["fields"]
            if (entry["name"], entry["scope"], entry["origin"])
            == (definition["field"], definition["scope"], definition["origin"])
        )
        path = (
            self._store.directory(run_id)
            / "results"
            / "profiles"
            / definition["task"]
            / f"{definition['scope']}.parquet"
        )
        if not path.is_file():
            raise SelectionError(409, f"Task '{definition['task']}' kept no {definition['scope']} rows")
        rows = pl.read_parquet(path, columns=[*_KEYS, field["column"]])
        chosen = matching(rows, field["column"], predicate, field)
        decode = field["type"] != "numeric"
        return [
            {
                "source": source,
                "index": item,
                "target": target,
                "value": json.loads(value) if decode and value else value,
            }
            for item, target, value in chosen.iter_rows()
        ]

    def _output(self, run_id: str, task: str, step: str | None) -> dict[str, Any]:
        """The serialized output of the task, or of one step of its chain."""
        from dataeval_flow._service._app import _results
        from dataeval_flow.config._schemas._export import one_directory_segment

        run = self._store.get(run_id)
        if task not in {entry.get("name") for entry in run["pipeline"].get("tasks") or []}:
            raise SelectionError(422, f"This run's pipeline has no task '{task}'", ("body", "task"))
        try:
            one_directory_segment(task, what="task name")
        except ValueError as error:
            raise SelectionError(422, f"Task '{task}' keeps no rows to select: {error}", ("body", "task")) from None
        result = _results(self._store.directory(run_id), run).get(task)
        if result is None:
            raise SelectionError(409, f"Task '{task}' has written no result yet")
        if step is not None:
            steps = result.get("steps") or {}
            result = steps.get(step)
            if result is None:
                raise SelectionError(422, f"Task '{task}' has no step '{step}'", ("body", "step"))
            if derived := [name for name in result.get("inputs") or [] if name in steps]:
                raise SelectionError(
                    422,
                    f"Step '{step}' reads {', '.join(derived)}, which the chain derived from the source: its rows "
                    "count items within that Dataset, not within the source. Select from a step that reads a workflow "
                    "input.",
                    ("body", "step"),
                )
        output = result.get("output")
        if not isinstance(output, dict):
            raise SelectionError(409, f"Task '{task}' wrote no output to select from")
        return output

    def _row_source(self, run_id: str, task: str) -> str:
        """The one source a table's rows index into."""
        entry = next(entry for entry in self._store.get(run_id)["pipeline"]["tasks"] if entry.get("name") == task)
        sources = entry.get("sources")
        names = [sources] if isinstance(sources, str) else list(sources or [])
        if len(names) != 1:
            message = f"Task '{task}' reads {len(names)} sources; select rows of a one-source task"
            raise SelectionError(422, message, ("body", "task"))
        return names[0]

    def _summary(
        self, selection_id: str, definition: Mapping[str, Any], members: list[dict[str, Any]]
    ) -> dict[str, Any]:
        targets = {(member["index"], member["target"]) for member in members if member["target"] is not None}
        return {
            "schema": SCHEMA,
            "id": selection_id,
            "definition": definition,
            "source": definition["source"],
            "scope": definition["scope"],
            "total": len(members),
            "images": len({member["index"] for member in members}),
            "targets": len(targets),
        }

    def _load(self, run_id: str, selection_id: str) -> dict[str, Any]:
        path = self._path(run_id, selection_id)
        if not path.is_file():
            raise SelectionError(404, "Unknown selection")
        return json.loads(path.read_text())

    def _path(self, run_id: str, selection_id: str) -> Path:
        if not _ID.fullmatch(selection_id):
            raise SelectionError(404, "Unknown selection")
        return self._store.directory(run_id) / "selections" / f"{selection_id}.json"
