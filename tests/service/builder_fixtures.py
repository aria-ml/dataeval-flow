"""The builder contract fixtures Studio copies (builder spec §8): each request, and what the service answers.

Regenerate after changing a contract: ``uv run python -m tests.service.builder_fixtures``. A test fails while a
checked-in fixture differs from what the service answers now, apart from versions, fingerprints, IDs and times.
"""

import json
import re
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "builder"
_COMMAND = "uv run python -m tests.service.builder_fixtures"
_SAMPLE_STEPS = {
    ("evaluator", "duplicates"),
    ("check", "image-duplicates"),
    ("transform", "conform"),
    ("transform", "remove"),
}
_VARYING = {
    "flow_version", "dataeval_version", "schema_fingerprint", "actual", "id", "run_id", "created_at", "started_at",
    "finished_at", "timestamp", "elapsed_s",
}  # fmt: skip
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_STALE = "sha256:" + "0" * 64


def normalized(value: Any) -> Any:
    """`value` with what differs between machines and runs replaced, to compare a fixture with a fresh answer."""
    if isinstance(value, dict):
        return {key: "<varies>" if key in _VARYING else normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, str):
        return _UUID.sub("<run-id>", value)
    return value


def exchange(client: Any, method: str, path: str, body: Any = None, *, shown: Any = None) -> dict[str, Any]:
    """One request and the service's answer, with `shown` recorded in place of a body too large to keep."""
    response = client.request(method, path, json=body)
    return {
        "request": {"method": method, "path": path, "body": body if shown is None else shown},
        "status": response.status_code,
        "response": response.json(),
    }


def fixture_pipeline() -> dict[str, Any]:
    """A small pipeline over the fixture dataset: a duplicates evaluator judged by a check, in a custom workflow."""
    return {
        "datasets": [{"name": "fixture", "format": "coco", "path": "fixture"}],
        "sources": [{"name": "data", "dataset": "fixture"}],
        "evaluators": [{"name": "dupes", "type": "duplicates"}],
        "workflows": [
            {
                "name": "review",
                "inputs": ["data"],
                "steps": [
                    {"name": "dupes", "evaluator": "dupes", "input": "data"},
                    {"name": "judge", "check": "image-duplicates", "input": "dupes"},
                ],
            }
        ],
        "tasks": [{"name": "review", "workflow": "review", "sources": "data"}],
    }


def _capabilities(client: Any) -> dict[str, Any]:
    answer = exchange(client, "GET", "/v1/capabilities")
    catalog = answer["response"]["steps"]
    catalog["steps"] = [entry for entry in catalog["steps"] if (entry["kind"], entry["type"]) in _SAMPLE_STEPS]
    return answer


def _schema_changed(client: Any) -> dict[str, Any]:
    body = {"pipeline": fixture_pipeline(), "expected_schema_fingerprint": _STALE}
    return exchange(client, "POST", "/v1/validate", body)


def _too_large(client: Any) -> dict[str, Any]:
    body = {"pipeline": fixture_pipeline(), "padding": "x" * 1_048_576}
    return exchange(client, "POST", "/v1/validate", body, shown="<a pipeline padded past 1 MiB>")


_ONTOLOGY = {
    "name": "vehicles",
    "concepts": [
        {"id": "Vehicle", "label": "Vehicle", "synonyms": ["car", "truck"]},
        {"id": "Person", "label": "Person", "synonyms": ["person", "pedestrian"]},
    ],
}
_SPLIT = {"name": "s", "transform": "split", "input": "data", "test_frac": 0.2, "val_frac": 0}
_DUPES = {"name": "dupes", "evaluator": "dupes", "input": "data"}


def _steps(pipeline: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    pipeline["workflows"][0]["steps"] = steps
    return pipeline


def _resolve(client: Any, pipeline: dict[str, Any]) -> dict[str, Any]:
    return exchange(client, "POST", "/v1/resolve", {"pipeline": pipeline})


def _resolve_valid(client: Any) -> dict[str, Any]:
    return _resolve(client, fixture_pipeline())


def _unknown_address() -> dict[str, Any]:
    pipeline = fixture_pipeline()
    pipeline["workflows"][0]["steps"][1]["input"] = "missing"
    return pipeline


def _resolve_unknown_address(client: Any) -> dict[str, Any]:
    return _resolve(client, _unknown_address())


def _resolve_split_unavailable(client: Any) -> dict[str, Any]:
    steps = [_SPLIT, {"name": "dupes", "evaluator": "dupes", "input": "s.train"}]
    return _resolve(client, _steps(fixture_pipeline(), steps))


def _resolve_shared_evaluator(client: Any) -> dict[str, Any]:
    steps = [_DUPES, _SPLIT, {"name": "dupes-train", "evaluator": "dupes", "input": "s.train"}]
    return _resolve(client, _steps(fixture_pipeline(), steps))


def _resolve_list_slot(client: Any) -> dict[str, Any]:
    pipeline = fixture_pipeline()
    pipeline["sources"].append({"name": "more", "dataset": "fixture"})
    pipeline["workflows"][0]["inputs"] = [{"name": "all", "list": True}]
    pipeline["tasks"][0]["sources"] = ["data", "more"]
    steps = [
        {"name": "dupes", "evaluator": "dupes", "input": "all"},
        {"name": "judge", "check": "image-duplicates", "input": "dupes[more]"},
    ]
    return _resolve(client, _steps(pipeline, steps))


def _resolve_nested_preset(client: Any) -> dict[str, Any]:
    pipeline = fixture_pipeline()
    outliers = {"flags": ["pixel"], "outlier_threshold": "zscore"}
    pipeline["workflows"].append({"name": "quality", "type": "quality", "outliers": outliers})
    return _resolve(client, _steps(pipeline, [{"name": "q", "workflow": "quality", "input": "data"}]))


def _resolve_remove_plans(client: Any) -> dict[str, Any]:
    pipeline = fixture_pipeline()
    pipeline["evaluators"].append({"name": "outs", "type": "outliers"})
    steps = [
        _DUPES,
        {"name": "outs", "evaluator": "outs", "input": "data"},
        {"name": "clean", "transform": "remove", "input": "data", "plans": {"dupes": {}, "outs": {}}},
    ]
    return _resolve(client, _steps(pipeline, steps))


def _resolve_wrong_dataset(client: Any) -> dict[str, Any]:
    pipeline = fixture_pipeline()
    pipeline["evaluators"].append({"name": "align", "type": "label-alignment", "ontology": "vehicles"})
    pipeline["ontologies"] = [_ONTOLOGY]
    steps = [
        {"name": "few", "transform": "view", "input": "data", "operations": [{"type": "Limit", "params": {"size": 2}}]},
        {"name": "aligned", "evaluator": "align", "input": "few"},
        {"name": "c", "transform": "conform", "input": "data", "alignment": "aligned"},
    ]
    return _resolve(client, _steps(pipeline, steps))


def _resolve_forward_address(client: Any) -> dict[str, Any]:
    steps = [{"name": "judge", "check": "image-duplicates", "input": "dupes"}, _DUPES]
    return _resolve(client, _steps(fixture_pipeline(), steps))


def _resolve_matrix(client: Any) -> dict[str, Any]:
    pipeline = _steps(fixture_pipeline(), [_SPLIT, _DUPES])
    pipeline["tasks"][0]["matrix"] = {"steps.s.test_frac": [0.2, 0.3]}
    return _resolve(client, pipeline)


def _resolve_unattached_workflow(client: Any) -> dict[str, Any]:
    pipeline = fixture_pipeline()
    draft = {
        "name": "draft",
        "inputs": ["data"],
        "steps": [{"name": "j", "check": "image-duplicates", "input": "nothing"}],
    }
    pipeline["workflows"].append(draft)
    return _resolve(client, pipeline)


def _resolve_too_many_nodes(client: Any) -> dict[str, Any]:
    from dataeval_flow._service import _contract

    limit = _contract.MAX_RESOLVED_NODES
    _contract.MAX_RESOLVED_NODES = 1
    try:
        answer = _resolve(client, fixture_pipeline())
    finally:
        _contract.MAX_RESOLVED_NODES = limit
    note = "Answered with `max_resolved_nodes` lowered to 1, to show the error's shape; the service's limit is 1000."
    return {"note": note, **answer}


def _validate_located_422(client: Any) -> dict[str, Any]:
    return exchange(client, "POST", "/v1/validate", {"pipeline": _unknown_address()})


_QUALITY_ENTRY = {"name": "q", "type": "quality", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
_PLUGIN_ENTRIES = {
    "dataeval_flow.workflows": [
        ("example.count", "tests.example_plugin:CountWorkflow"),
        ("example.refusing", "tests.example_plugin:RefusingWorkflow"),
    ],
    "dataeval_flow.combines": [("example.items", "tests.example_plugin:ItemsCombine")],
    "dataeval_flow.checks": [("example.at-least", "tests.example_plugin:AtLeastCheck")],
    "dataeval_flow.evaluators": [("example.brightness", "tests.example_plugin:BrightnessEvaluator")],
    "dataeval_flow.extractors": [("example.mean", "tests.example_plugin:MeanExtractor")],
    "dataeval_flow.image_transforms": [("example.Invert", "tests.example_plugin:Invert")],
}  # as tests/test_example_plugin.py::EXAMPLES


@contextmanager
def _example_plugin() -> Iterator[None]:
    """Serve the example plugin's entry points for one exchange (as tests/conftest.py::plugins does)."""
    from importlib.metadata import EntryPoint

    import dataeval_flow._registry as registry_module
    from dataeval_flow.config.extractors._registry import EXTRACTORS
    from dataeval_flow.config.image_transforms._registry import IMAGE_TRANSFORMS
    from dataeval_flow.evaluators._registry import EVALUATORS
    from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS
    from dataeval_flow.workflows._registry import WORKFLOWS

    registries = [WORKFLOWS, EVALUATORS, EXTRACTORS, IMAGE_TRANSFORMS, TRANSFORMS, COMBINES, CHECKS]

    def entry_points(*, group: str) -> list[EntryPoint]:
        return [EntryPoint(name=name, value=value, group=group) for name, value in _PLUGIN_ENTRIES.get(group, [])]

    original = registry_module.entry_points
    registry_module.entry_points = entry_points
    for registry in registries:
        registry._reset()
    try:
        yield
    finally:
        registry_module.entry_points = original
        for registry in registries:
            registry._reset()


def _with_preset(entry: dict[str, Any], *, evaluators: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """`fixture_pipeline()` plus a preset entry run by a task named `clean`."""
    pipeline = fixture_pipeline()
    pipeline["evaluators"].extend(evaluators or [])
    pipeline["workflows"].append(entry)
    pipeline["tasks"].append({"name": "clean", "workflow": entry["name"], "sources": "data"})
    return pipeline


def _decompose(client: Any, pipeline: dict[str, Any], **request: Any) -> dict[str, Any]:
    body = {"pipeline": pipeline, "operation": "copy_definition", "target_pointer": "/workflows/1", **request}
    return exchange(client, "POST", "/v1/presets/decompose", body)


def _decompose_copy_quality(client: Any) -> dict[str, Any]:
    pipeline = _with_preset(
        _QUALITY_ENTRY, evaluators=[{"name": "qc-outliers", "type": "outliers", "flags": ["visual"]}]
    )
    return _decompose(client, pipeline, new_name="qc", reassign_tasks=["clean"])


def _decompose_copy_audit(client: Any) -> dict[str, Any]:
    audit = {"name": "a", "type": "audit", "outliers": _QUALITY_ENTRY["outliers"]}
    return _decompose(client, _with_preset(audit), new_name="audit_copy")


def _decompose_copy_splits(client: Any) -> dict[str, Any]:
    return _decompose(client, _with_preset({"name": "s", "type": "splits", "folds": 2}), new_name="splits_copy")


def _sweeping() -> dict[str, Any]:
    pipeline = _with_preset(_QUALITY_ENTRY)
    pipeline["tasks"][1]["matrix"] = {"outliers.outlier_threshold": ["zscore", "iqr"]}
    return pipeline


def _decompose_copy_keeps_matrix(client: Any) -> dict[str, Any]:
    return _decompose(client, _sweeping(), new_name="qc")


def _decompose_matrix_refused(client: Any) -> dict[str, Any]:
    return _decompose(client, _sweeping(), new_name="qc", reassign_tasks=["clean"])


def _decompose_inline_unavailable(client: Any) -> dict[str, Any]:
    pipeline = _with_preset(_QUALITY_ENTRY)
    pipeline["workflows"].append(
        {"name": "outer", "inputs": ["data"], "steps": [{"name": "c", "workflow": "q", "input": "data"}]}
    )
    return _decompose(client, pipeline, operation="inline_step", target_pointer="/workflows/2/steps/0")


def _decompose_not_a_preset(client: Any) -> dict[str, Any]:
    return _decompose(client, fixture_pipeline(), target_pointer="/workflows/0", new_name="qc")


def _decompose_name_taken(client: Any) -> dict[str, Any]:
    return _decompose(client, _with_preset(_QUALITY_ENTRY), new_name="review")


def _decompose_plugin_gap(client: Any) -> dict[str, Any]:
    with _example_plugin():
        return _decompose(client, _with_preset({"name": "c", "type": "example.count"}), new_name="count_copy")


def _decompose_plugin_preflight(client: Any) -> dict[str, Any]:
    with _example_plugin():
        return _decompose(client, _with_preset({"name": "r", "type": "example.refusing"}), new_name="refusing_copy")


def _decompose_several_reasons(client: Any) -> dict[str, Any]:
    pipeline = _with_preset({"name": "c", "type": "example.count"})
    pipeline["tasks"][1]["matrix"] = {"minimum": [1, 2]}
    with _example_plugin():
        return _decompose(client, pipeline, new_name="count_copy", reassign_tasks=["clean"])


def _decompose_reassign_unknown_task(client: Any) -> dict[str, Any]:
    return _decompose(client, _with_preset(_QUALITY_ENTRY), new_name="qc", reassign_tasks=["clean", "nope"])


GENERATORS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "capabilities-sample": _capabilities,
    "schema-changed": _schema_changed,
    "document-too-large": _too_large,
    "resolve-valid": _resolve_valid,
    "resolve-unknown-address": _resolve_unknown_address,
    "resolve-split-unavailable": _resolve_split_unavailable,
    "resolve-shared-evaluator": _resolve_shared_evaluator,
    "resolve-list-slot": _resolve_list_slot,
    "resolve-nested-preset": _resolve_nested_preset,
    "resolve-remove-plans": _resolve_remove_plans,
    "resolve-wrong-dataset": _resolve_wrong_dataset,
    "resolve-forward-address": _resolve_forward_address,
    "resolve-matrix": _resolve_matrix,
    "resolve-unattached-workflow": _resolve_unattached_workflow,
    "resolve-too-many-nodes": _resolve_too_many_nodes,
    "validate-located-422": _validate_located_422,
    "decompose-copy-quality": _decompose_copy_quality,
    "decompose-copy-audit": _decompose_copy_audit,
    "decompose-copy-splits": _decompose_copy_splits,
    "decompose-copy-keeps-matrix": _decompose_copy_keeps_matrix,
    "decompose-matrix-refused": _decompose_matrix_refused,
    "decompose-inline-unavailable": _decompose_inline_unavailable,
    "decompose-not-a-preset": _decompose_not_a_preset,
    "decompose-name-taken": _decompose_name_taken,
    "decompose-plugin-gap": _decompose_plugin_gap,
    "decompose-plugin-preflight": _decompose_plugin_preflight,
    "decompose-several-reasons": _decompose_several_reasons,
    "decompose-reassign-unknown-task": _decompose_reassign_unknown_task,
}


def generate(directory: Path) -> dict[str, dict[str, Any]]:
    """Every fixture, answered by a fresh service over the fixture dataset written under `directory`."""
    from fastapi.testclient import TestClient

    from dataeval_flow._service._app import create_app
    from tests.service.conftest import write_dataset

    with TestClient(create_app(write_dataset(directory / "data"), directory / "out")) as client:
        fixtures = {name: make(client) for name, make in GENERATORS.items()}
        fixtures.update(_activity_fixtures(client))
        return fixtures


def _finished(store: Any, run_id: str, timeout: float = 300) -> None:
    end = time.monotonic() + timeout
    while store.get(run_id)["status"] not in {"succeeded", "failed", "cancelled", "interrupted"}:
        if time.monotonic() > end:
            raise TimeoutError(f"run {run_id} did not finish")
        time.sleep(0.2)


def _activity_fixtures(client: Any) -> dict[str, dict[str, Any]]:
    """One real run of the fixture pipeline as the activity routes answer it, and a run from before activity."""
    from dataeval_flow._activity import ACTIVITY_FILE, read_events, snapshot

    store = client.app.state.store
    run_id = client.post("/v1/runs", json={"pipeline": fixture_pipeline()}).json()["id"]
    _finished(store, run_id)
    events = read_events(store.directory(run_id) / ACTIVITY_FILE)
    running = next(
        event["sequence"] for event in events if event["kind"] == "step_state" and event["step"]["status"] == "running"
    )
    path = f"/v1/runs/{run_id}/activity"
    legacy = store.create({"tasks": ["review"]}, fixture_pipeline())  # as a run queued before activity was recorded
    return {
        "activity-final": exchange(client, "GET", path),
        "activity-running": {
            "note": "What the route answers at the first running step's sequence, folded from this run's own events.",
            "request": {"method": "GET", "path": path, "body": None},
            "status": 200,
            "response": snapshot(events[:running], run_id, store.get(run_id)["schema_fingerprint"]),
        },
        "activity-events": exchange(client, "GET", f"/v1/runs/{run_id}/events?after={running}&limit=100"),
        "activity-unavailable": exchange(client, "GET", f"/v1/runs/{legacy['id']}/activity"),
        "events-after-the-end": exchange(client, "GET", f"/v1/runs/{run_id}/events?after={len(events) + 1}"),
    }


def main() -> None:
    """Write every fixture under `FIXTURES`, with the versions that answered it and the command that wrote it."""
    import dataeval

    from dataeval_flow import __version__

    FIXTURES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        for name, answer in generate(Path(scratch)).items():
            document = {
                "generated_by": _COMMAND,
                "flow_version": __version__,
                "dataeval_version": dataeval.__version__,
                **answer,
            }
            (FIXTURES / f"{name}.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
