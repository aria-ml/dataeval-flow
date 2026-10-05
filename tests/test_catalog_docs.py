"""The reference pages describe every built-in step and preset in one template (naming spec §7, §9.2)."""

import importlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.test_naming_conventions import _MINIMAL, _PRESETS

_REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "source" / "reference"
_CHAIN_KINDS = ("evaluator", "transform", "combine", "check")
_PY_CLASS = re.compile(r"\{py:class\}`~?([\w.]+)`")


def _page(name: str) -> str:
    return (_REFERENCE / name).read_text(encoding="utf-8")


def _section(page: str, heading: str) -> str:
    """The text under the one `heading` line on `page`, up to the next line starting with `#`."""
    text = _page(page)
    assert text.count(f"\n{heading}\n") == 1, f"{page} has no single `{heading}` heading"
    return text.split(f"\n{heading}\n", 1)[1].split("\n#", 1)[0]


def _bullet(section: str, label: str) -> str:
    """The text of the `- **label:**` bullet, continuation lines included; empty when there is none."""
    match = re.search(rf"^- \*\*{label}:\*\* (.*(?:\n  .*)*)", section, re.M)
    return match.group(1) if match else ""


def _types(text: str) -> set[str]:
    """Every backticked name in `text`: the step types and presets a cross-link bullet names."""
    return set(re.findall(r"`([a-z0-9-]+)`", text))


def _chain(cls: type) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    """The steps `cls` builds from `_MINIMAL`, and each evaluator entry's type by entry name."""
    chain = cls.chain(cls.config_type.model_validate(_MINIMAL[cls.name]))
    steps = [step for step in chain.steps if isinstance(step, Mapping)]
    assert len(steps) == len(chain.steps), f"{cls.name}'s chain holds a step that is not a mapping"
    return steps, {entry.name: entry.type for entry in chain.evaluators or ()}


def _preset_section(cls: type) -> str:
    return _section("presets.md", f"## `{cls.name}`")


@pytest.mark.parametrize("cls", _PRESETS, ids=lambda cls: cls.name)
def test_every_preset_has_a_section_in_the_preset_catalog(cls: type) -> None:
    assert _preset_section(cls).strip()


@pytest.mark.parametrize("cls", _PRESETS, ids=lambda cls: cls.name)
def test_a_preset_section_follows_the_template(cls: type) -> None:
    section = _preset_section(cls)
    first = section.strip().split("\n\n", 1)[0].replace("\n", " ")
    assert first == cls.description, f"`{cls.name}` opens with {first!r}, not its registry description"
    marks = ["- **Answers:**", "- **Reads:**", "- **Makes:**", "**Chain**", "**Settings** ("]
    if "checks" in cls.config_type.model_fields:
        marks.append("**Checks**")
    marks.append("```yaml")
    positions = [section.find(mark) for mark in marks]
    missing = [mark for mark, position in zip(marks, positions, strict=True) if position == -1]
    assert not missing, f"`{cls.name}`'s section lacks {missing}"
    assert positions == sorted(positions), f"`{cls.name}`'s section is out of the template's order"


@pytest.mark.parametrize("cls", _PRESETS, ids=lambda cls: cls.name)
def test_a_preset_section_lists_every_setting(cls: type) -> None:
    section = _preset_section(cls)
    keys = [field.alias or name for name, field in cls.config_type.model_fields.items() if name not in {"name", "type"}]
    missing = [key for key in keys if f"| `{key}` |" not in section]
    assert not missing, f"`{cls.name}`'s settings table is missing {missing}"


@pytest.mark.parametrize(
    "cls", [cls for cls in _PRESETS if "checks" in cls.config_type.model_fields], ids=lambda cls: cls.name
)
def test_a_preset_section_lists_every_check_it_runs(cls: type) -> None:
    section = _preset_section(cls)
    checks = cls.config_type.model_validate(_MINIMAL[cls.name]).checks  # pyright: ignore[reportAttributeAccessIssue]
    missing = [key for key in checks.model_dump(by_alias=True) if f"`{key}`" not in section]
    assert not missing, f"`{cls.name}`'s checks table is missing {missing}"


@pytest.mark.parametrize("cls", _PRESETS, ids=lambda cls: cls.name)
def test_a_preset_section_lists_its_chain(cls: type) -> None:
    section = _preset_section(cls)
    steps, _ = _chain(cls)
    missing = [step["name"] for step in steps if f"| `{step['name']}` |" not in section]
    assert not missing, f"`{cls.name}`'s chain table is missing {missing}"


@pytest.mark.parametrize("page", ["evaluators.md", "transforms.md", "checks.md", "presets.md"])
def test_every_class_a_page_names_imports(page: str) -> None:
    missing = []
    for path in sorted(set(_PY_CLASS.findall(_page(page)))):
        module, _, name = path.rpartition(".")
        try:
            getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            missing.append(path)
    assert not missing, f"{page} names classes nothing imports: {missing}"
