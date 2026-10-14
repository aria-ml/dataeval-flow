"""The checked-in builder fixtures are what the service answers now (builder spec §8)."""

import json

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from tests.service.builder_fixtures import FIXTURES, generate, normalized

pytestmark = pytest.mark.optional

_REGENERATE = "regenerate with: uv run python -m tests.service.builder_fixtures"


def test_the_checked_in_fixtures_are_what_the_service_answers(tmp_path) -> None:
    fresh = generate(tmp_path)
    stored = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in FIXTURES.glob("*.json")}
    assert sorted(stored) == sorted(fresh), _REGENERATE
    for name, exchange in fresh.items():
        kept = {key: stored[name][key] for key in exchange}
        assert normalized(kept) == normalized(exchange), f"{name} is stale; {_REGENERATE}"
