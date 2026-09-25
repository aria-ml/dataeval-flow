"""The evaluator catalog names every registered evaluator and every parameter each one takes."""

from pathlib import Path

import pytest

from dataeval_flow.evaluators import get_evaluator, list_evaluators

_CATALOG = Path(__file__).resolve().parents[1] / "docs" / "source" / "reference" / "evaluators.md"
_NAMES = [cls.name for cls in list_evaluators()]


def _section(name: str) -> str:
    text = _CATALOG.read_text(encoding="utf-8")
    heading = f"### `{name}`"
    assert heading in text, f"the catalog has no section for {name}"
    return text.split(heading, 1)[1].split("\n#", 1)[0]


@pytest.mark.parametrize("name", _NAMES)
def test_every_evaluator_has_a_section(name: str):
    assert _section(name).strip()


# Every entry's identity, which the catalog describes once rather than per evaluator.
_IDENTITY = {"name", "type"}


@pytest.mark.parametrize("name", _NAMES)
def test_every_parameter_is_listed_in_its_section(name: str):
    section = _section(name)
    fields = set(get_evaluator(name).config_type.model_fields) - _IDENTITY
    missing = [field for field in sorted(fields) if f"| `{field}` |" not in section]
    assert not missing, f"{name}: catalog is missing {missing}"
