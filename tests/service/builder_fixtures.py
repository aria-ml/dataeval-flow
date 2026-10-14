"""The builder contract fixtures Studio copies (builder spec §8): each request, and what the service answers.

Regenerate after changing a contract: ``uv run python -m tests.service.builder_fixtures``. A test fails while a
checked-in fixture differs from what the service answers now, apart from versions, fingerprints, IDs and times.
"""

import json
import tempfile
from collections.abc import Callable
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
_STALE = "sha256:" + "0" * 64


def normalized(value: Any) -> Any:
    """`value` with what differs between machines and runs replaced, to compare a fixture with a fresh answer."""
    if isinstance(value, dict):
        return {key: "<varies>" if key in _VARYING else normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item) for item in value]
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


GENERATORS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "capabilities-sample": _capabilities,
    "schema-changed": _schema_changed,
    "document-too-large": _too_large,
}


def generate(directory: Path) -> dict[str, dict[str, Any]]:
    """Every fixture, answered by a fresh service over the fixture dataset written under `directory`."""
    from fastapi.testclient import TestClient

    from dataeval_flow._service._app import create_app
    from tests.service.conftest import write_dataset

    with TestClient(create_app(write_dataset(directory / "data"), directory / "out")) as client:
        return {name: make(client) for name, make in GENERATORS.items()}


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
