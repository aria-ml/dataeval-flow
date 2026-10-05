"""The reference pages describe every built-in step and preset in one template (naming spec §7, §9.2)."""

import importlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow.evaluators import get_evaluator
from dataeval_flow.steps import get_check, get_combine, get_transform, list_steps
from dataeval_flow.steps._catalog import StepCatalogEntry
from tests.test_naming_conventions import _MINIMAL, _PRESETS

_REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "source" / "reference"
_CHAIN_KINDS = ("evaluator", "transform", "combine", "check")
_PY_CLASS = re.compile(r"\{py:class\}`~?([\w.]+)`")

# Each kind's catalog page.
_PAGES = {"evaluator": "evaluators.md", "transform": "transforms.md", "combine": "combines.md", "check": "checks.md"}
_GET = {"evaluator": get_evaluator, "transform": get_transform, "combine": get_combine, "check": get_check}
_CATALOG = [entry for entry in list_steps(plugins=False).steps if entry.kind in _PAGES]
# The kinds whose pages follow the entry template; each joins as its page is rewritten.
_TEMPLATED = {"combine", "check", "evaluator", "transform"}
# An evaluator entry's identity, which its page describes once rather than per entry; no other kind has this.
_IDENTITY = {"name", "type"}


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


def _id(entry: StepCatalogEntry) -> str:
    return f"{entry.kind}:{entry.type}"


def _entry(entry: StepCatalogEntry) -> str:
    return _section(_PAGES[entry.kind], f"### `{entry.type}`")


def _makes(entry: StepCatalogEntry) -> set[str]:
    return {cls for port in entry.outputs for cls in port.classes}


def _reads(entry: StepCatalogEntry) -> set[str]:
    return {cls for port in entry.inputs for cls in port.classes}


def _judges(check: StepCatalogEntry) -> set[str]:
    """The evaluators and combines whose Output `check` reads."""
    return {e.type for e in _CATALOG if e.kind in ("evaluator", "combine") and _makes(e) & _reads(check)}


def _judged_by(entry: StepCatalogEntry) -> set[str]:
    """The checks that read `entry`'s Output."""
    return {check.type for check in _CATALOG if check.kind == "check" and _makes(entry) & _reads(check)}


def _preset_section(cls: type) -> str:
    return _section("presets.md", f"## `{cls.name}`")


def _used_in() -> dict[str, set[str]]:
    """Each step type, and the presets whose chain, built from `_MINIMAL`, runs it."""
    used: dict[str, set[str]] = {}
    for cls in _PRESETS:
        steps, entries = _chain(cls)
        for step in steps:
            kind = next(kind for kind in _CHAIN_KINDS if kind in step)
            type_id = str(entries.get(step[kind], step[kind]) if kind == "evaluator" else step[kind])
            used.setdefault(type_id, set()).add(cls.name)
    return used


_USED_IN = _used_in()


@pytest.mark.parametrize("entry", _CATALOG, ids=_id)
def test_every_step_has_one_entry_on_its_kinds_page(entry: StepCatalogEntry) -> None:
    assert _entry(entry).strip()


@pytest.mark.parametrize("entry", _CATALOG, ids=_id)
def test_every_setting_is_listed_in_its_entry(entry: StepCatalogEntry) -> None:
    section = _entry(entry)
    fields = set(_GET[entry.kind](entry.type).config_type.model_fields) - (
        _IDENTITY if entry.kind == "evaluator" else set()
    )
    missing = [field for field in sorted(fields) if f"| `{field}` |" not in section]
    assert not missing, f"`{entry.type}`: {_PAGES[entry.kind]} is missing {missing}"


@pytest.mark.parametrize("entry", _CATALOG, ids=_id)
def test_every_step_is_in_its_pages_table_at_a_glance(entry: StepCatalogEntry) -> None:
    glance = _page(_PAGES[entry.kind]).split("## At a glance", 1)[1].split("\n## ", 1)[0]
    assert f"| `{entry.type}` |" in glance, f"{_PAGES[entry.kind]}'s table at a glance lacks `{entry.type}`"


@pytest.mark.parametrize("entry", [entry for entry in _CATALOG if entry.kind in _TEMPLATED], ids=_id)
def test_an_entry_follows_the_template(entry: StepCatalogEntry) -> None:
    section = _entry(entry)
    first = section.strip().split("\n\n", 1)[0].replace("\n", " ")
    assert first == entry.description, f"`{entry.type}` opens with {first!r}, not its registry description"

    cross = {"check": ["- **Judges:**"], "transform": []}.get(entry.kind, ["- **Judged by:**"])
    marks = ["- **Reads:**", "- **Makes:**", "**Settings** (", *cross, "- **Used in:**", "```yaml"]
    positions = [section.find(mark) for mark in marks]
    missing = [mark for mark, position in zip(marks, positions, strict=True) if position == -1]
    assert not missing, f"`{entry.type}` lacks {missing}"
    assert positions == sorted(positions), f"`{entry.type}`'s parts are out of the template's order"

    reads = _bullet(section, "Reads")
    unread = [port.port for port in entry.inputs if f"`{port.port}`" not in reads]
    assert not unread, f"`{entry.type}`'s Reads leaves out {unread}"
    if len(entry.outputs) > 1:
        makes = _bullet(section, "Makes")
        unmade = [port.port for port in entry.outputs if f"`{port.port}`" not in makes]
        assert not unmade, f"`{entry.type}`'s Makes leaves out {unmade}"

    if entry.kind == "check":
        judges = _types(_bullet(section, "Judges"))
        assert judges == _judges(entry), f"`{entry.type}`'s Judges disagrees with its ports"
    elif entry.kind != "transform":
        judged = _types(_bullet(section, "Judged by"))
        assert judged == _judged_by(entry), f"`{entry.type}`'s Judged by disagrees with the checks' ports"
    unlisted = _USED_IN.get(entry.type, set()) - _types(_bullet(section, "Used in"))
    assert not unlisted, f"`{entry.type}`'s Used in leaves out {sorted(unlisted)}"


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


@pytest.mark.parametrize("page", [*_PAGES.values(), "presets.md"])
def test_every_class_a_page_names_imports(page: str) -> None:
    missing = []
    for path in sorted(set(_PY_CLASS.findall(_page(page)))):
        module, _, name = path.rpartition(".")
        try:
            getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            missing.append(path)
    assert not missing, f"{page} names classes nothing imports: {missing}"


def _questions() -> set[str]:
    return set(re.findall(r"^## (.+)$", _page("index.md"), re.M))


@pytest.mark.parametrize("name", sorted({entry.type for entry in _CATALOG} | {cls.name for cls in _PRESETS}))
def test_every_step_and_preset_is_in_the_question_index(name: str) -> None:
    assert f"`{name}`" in _page("index.md"), f"reference/index.md does not name `{name}`"


@pytest.mark.parametrize("page", ["evaluators.md", "combines.md", "checks.md"])
def test_a_catalog_groups_its_entries_under_the_indexs_questions(page: str) -> None:
    questions, text = _questions(), _page(page)
    for heading in re.finditer(r"^### `([a-z0-9-]+)`$", text, re.M):
        group = re.findall(r"^## (.+)$", text[: heading.start()], re.M)[-1]
        assert group in questions, f"{page}: `{heading[1]}` sits under {group!r}, which reference/index.md does not ask"


@pytest.mark.parametrize("cls", _PRESETS, ids=lambda cls: cls.name)
def test_a_preset_links_the_question_it_answers(cls: type) -> None:
    answers = _bullet(_preset_section(cls), "Answers")
    assert re.search(r"\]\(index\.md#[a-z0-9-]+\)", answers), f"`{cls.name}`'s Answers links no question"
