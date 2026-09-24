"""The evaluator catalog names every registered evaluator and every parameter each one takes."""

from pathlib import Path

import pytest

from dataeval_flow.evaluator import get_evaluator, list_evaluators

_CATALOG = Path(__file__).resolve().parents[1] / "docs" / "source" / "reference" / "evaluators.md"
_NAMES = [entry["name"] for entry in list_evaluators()]


def _section(name: str) -> str:
    text = _CATALOG.read_text(encoding="utf-8")
    heading = f"### `{name}`"
    assert heading in text, f"the catalog has no section for {name}"
    return text.split(heading, 1)[1].split("\n#", 1)[0]


@pytest.mark.parametrize("name", _NAMES)
def test_every_evaluator_has_a_section(name: str):
    assert _section(name).strip()


@pytest.mark.parametrize("name", _NAMES)
def test_every_parameter_is_listed_in_its_section(name: str):
    section = _section(name)
    fields = get_evaluator(name).params_schema.model_fields
    missing = [field for field in fields if f"| `{field}` |" not in section]
    assert not missing, f"{name}: catalog is missing {missing}"
