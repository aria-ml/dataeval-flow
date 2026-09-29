"""The transform catalog names every built-in transform and every setting each one takes."""

from pathlib import Path

import pytest

from dataeval_flow.steps import get_transform
from dataeval_flow.steps._registry import _BUILTINS

_CATALOG = Path(__file__).resolve().parents[1] / "docs" / "source" / "reference" / "transforms.md"
_NAMES = sorted(_BUILTINS)


def _section(name: str) -> str:
    text = _CATALOG.read_text(encoding="utf-8")
    heading = f"### `{name}`"
    assert heading in text, f"the catalog has no section for {name}"
    return text.split(heading, 1)[1].split("\n#", 1)[0]


@pytest.mark.parametrize("name", _NAMES)
def test_every_transform_has_a_section(name: str):
    assert _section(name).strip()


@pytest.mark.parametrize("name", _NAMES)
def test_every_setting_is_listed_in_its_section(name: str):
    section = _section(name)
    fields = set(get_transform(name).config_type.model_fields)
    missing = [field for field in sorted(fields) if f"| `{field}` |" not in section]
    assert not missing, f"{name}: catalog is missing {missing}"


def test_every_transform_is_in_the_table_at_a_glance():
    glance = _CATALOG.read_text(encoding="utf-8").split("## At a glance", 1)[1].split("\n## ", 1)[0]
    missing = [name for name in _NAMES if f"| `{name}` |" not in glance]
    assert not missing, f"the table at a glance is missing {missing}"
