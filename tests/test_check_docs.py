"""The Check and Combine Catalog names every built-in check and combine, and every setting each one takes."""

from pathlib import Path

import pytest

from dataeval_flow.steps import get_check, get_combine
from dataeval_flow.steps._registry import _CHECK_BUILTINS, _COMBINE_BUILTINS

_CATALOG = Path(__file__).resolve().parents[1] / "docs" / "source" / "reference" / "checks.md"
_STEPS = [("check", name) for name in sorted(_CHECK_BUILTINS)] + [
    ("combine", name) for name in sorted(_COMBINE_BUILTINS)
]


def _section(name: str) -> str:
    text = _CATALOG.read_text(encoding="utf-8")
    heading = f"### `{name}`"
    assert heading in text, f"the catalog has no section for {name}"
    return text.split(heading, 1)[1].split("\n#", 1)[0]


@pytest.mark.parametrize(("kind", "name"), _STEPS)
def test_every_setting_is_listed_in_its_section(kind: str, name: str) -> None:
    impl = get_check(name) if kind == "check" else get_combine(name)
    section = _section(name)
    missing = [field for field in sorted(impl.config_type.model_fields) if f"| `{field}` |" not in section]
    assert not missing, f"{name}: catalog is missing {missing}"


@pytest.mark.parametrize(("kind", "name"), _STEPS)
def test_every_step_is_in_the_table_at_a_glance(kind: str, name: str) -> None:
    glance = _CATALOG.read_text(encoding="utf-8").split("## At a glance", 1)[1].split("\n## ", 1)[0]
    assert f"| `{name}` | {kind} |" in glance
