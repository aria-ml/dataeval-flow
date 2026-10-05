"""The reference pages describe every built-in step and preset in one template (naming spec §7, §9.2)."""

import importlib
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

from dataeval_flow.evaluators import get_evaluator
from dataeval_flow.steps import get_check, get_combine, get_transform, list_steps
from dataeval_flow.steps._catalog import StepCatalogEntry, _class_path
from tests.test_naming_conventions import _MINIMAL, _PRESETS

_REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "source" / "reference"
_CHAIN_KINDS = ("evaluator", "transform", "combine", "check")
_PY_CLASS = re.compile(r"\{py:\w+\}`~?([\w.]+)`")
# A cross-link whose text is a backticked name: `[`x`](page.md#y)`.
_LINK = re.compile(r"\[`([^`]+)`\]\([^)#\s]*\.md#([^)\s]+)\)")

# Each kind's catalog page.
_PAGES = {"evaluator": "evaluators.md", "transform": "transforms.md", "combine": "combines.md", "check": "checks.md"}
_GET = {"evaluator": get_evaluator, "transform": get_transform, "combine": get_combine, "check": get_check}
_CATALOG = [entry for entry in list_steps(plugins=False).steps if entry.kind in _PAGES]
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


def _part(section: str, start: str, end: str) -> str:
    """The text of `section` from its `start` mark up to the next `end` mark."""
    return section.split(start, 1)[1].split(end, 1)[0]


def _chain(cls: type) -> list[tuple[str, str, str]]:
    """Each step `cls` builds from `_MINIMAL`: its name, kind and type, an evaluator's type read from its entry."""
    chain = cls.chain(cls.config_type.model_validate(_MINIMAL[cls.name]))
    steps = [step for step in chain.steps if isinstance(step, Mapping)]
    assert len(steps) == len(chain.steps), f"{cls.name}'s chain holds a step that is not a mapping"
    entries = {entry.name: entry.type for entry in chain.evaluators or ()}
    rows = []
    for step in steps:
        kind = next(kind for kind in _CHAIN_KINDS if kind in step)
        type_id = entries.get(step[kind], step[kind]) if kind == "evaluator" else step[kind]
        rows.append((str(step["name"]), kind, str(type_id)))
    return rows


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
        for _, _, type_id in _chain(cls):
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


@pytest.mark.parametrize("entry", _CATALOG, ids=_id)
def test_an_entry_follows_the_template(entry: StepCatalogEntry) -> None:
    section = _entry(entry)
    first = section.strip().split("\n\n", 1)[0].replace("\n", " ")
    assert first == entry.description, f"`{entry.type}` opens with {first!r}, not its registry description"
    settings = f"**Settings** ({{py:class}}`~{_class_path(_GET[entry.kind](entry.type).config_type)}`)"
    assert settings in section, f"`{entry.type}`'s settings do not name its config class as {settings}"

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
    used = _types(_bullet(section, "Used in"))
    unlisted = _USED_IN.get(entry.type, set()) - used
    assert not unlisted, f"`{entry.type}`'s Used in leaves out {sorted(unlisted)}"
    assert used <= {cls.name for cls in _PRESETS}, f"`{entry.type}`'s Used in names {sorted(used)}, not only presets"


@pytest.mark.parametrize("cls", _PRESETS, ids=lambda cls: cls.name)
def test_every_preset_has_a_section_in_the_preset_catalog(cls: type) -> None:
    assert _preset_section(cls).strip()


@pytest.mark.parametrize("cls", _PRESETS, ids=lambda cls: cls.name)
def test_a_preset_section_follows_the_template(cls: type) -> None:
    section = _preset_section(cls)
    first = section.strip().split("\n\n", 1)[0].replace("\n", " ")
    assert first == cls.description, f"`{cls.name}` opens with {first!r}, not its registry description"
    marks = ["- **Answers:**", "- **Reads:**", "- **Makes:**", "**Chain**", "**Settings** ("]
    checks = cls.config_type.model_fields.get("checks")
    if checks:
        marks.append("**Checks**")
    marks.append("```yaml")
    positions = [section.find(mark) for mark in marks]
    missing = [mark for mark, position in zip(marks, positions, strict=True) if position == -1]
    assert not missing, f"`{cls.name}`'s section lacks {missing}"
    assert positions == sorted(positions), f"`{cls.name}`'s section is out of the template's order"

    settings = f"**Settings** ({{py:class}}`~{_class_path(cls.config_type)}`)"
    assert settings in section, f"`{cls.name}`'s settings do not name its config class as {settings}"
    if checks:
        named = f"({{py:class}}`~{_class_path(checks.annotation)}`)"
        assert named in _part(section, "**Checks**", "\n"), f"`{cls.name}`'s checks do not name their class as {named}"


@pytest.mark.parametrize("cls", _PRESETS, ids=lambda cls: cls.name)
def test_a_preset_section_lists_every_setting(cls: type) -> None:
    fields = cls.config_type.model_fields
    table = _part(_preset_section(cls), "**Settings** (", "**Checks**" if "checks" in fields else "```yaml")
    keys = [field.alias or name for name, field in fields.items() if name not in {"name", "type"}]
    missing = [key for key in keys if f"| `{key}` |" not in table]
    assert not missing, f"`{cls.name}`'s settings table is missing {missing}"


@pytest.mark.parametrize(
    "cls", [cls for cls in _PRESETS if "checks" in cls.config_type.model_fields], ids=lambda cls: cls.name
)
def test_a_preset_section_lists_every_check_it_runs(cls: type) -> None:
    table = _part(_preset_section(cls), "**Checks**", "```yaml")
    checks = cls.config_type.model_validate(_MINIMAL[cls.name]).checks  # pyright: ignore[reportAttributeAccessIssue]
    missing = [key for key in checks.model_dump(by_alias=True) if f"| [`{key}`](checks.md#{key}) |" not in table]
    assert not missing, f"`{cls.name}`'s checks table is missing {missing}"


@pytest.mark.parametrize("cls", _PRESETS, ids=lambda cls: cls.name)
def test_a_preset_section_lists_its_chain(cls: type) -> None:
    table = _part(_preset_section(cls), "**Chain**", "**Settings** (")
    rows = [f"| `{name}` | {kind} | [`{type_id}`]({_PAGES[kind]}#{type_id}) |" for name, kind, type_id in _chain(cls)]
    missing = [row for row in rows if row not in table]
    assert not missing, f"`{cls.name}`'s chain table is missing {missing}"


@pytest.mark.parametrize("page", [*_PAGES.values(), "presets.md"])
def test_every_object_a_page_names_imports(page: str) -> None:
    missing = []
    for path in sorted(set(_PY_CLASS.findall(_page(page)))):
        module, _, name = path.rpartition(".")
        try:
            getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError):
            missing.append(path)
    assert not missing, f"{page} names classes or functions nothing imports: {missing}"


@pytest.mark.parametrize("page", [*_PAGES.values(), "presets.md", "index.md"])
def test_a_cross_link_names_its_target(page: str) -> None:
    wrong = [link[0] for link in _LINK.finditer(_page(page)) if link[1] != link[2]]
    assert not wrong, f"{page} links a name to another name's anchor: {wrong}"


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


# The step vocabulary and the distinctions the names rest on (naming spec §7.4).
_TERMS = [
    "Brief",
    "Chain",
    "Check",
    "Combine",
    "Element",
    "Evaluation Split",
    "Finding",
    "Key",
    "Metadata",
    "Not Assessed",
    "Output",
    "Port",
    "Severity",
    "Step",
    "Subject",
    "Test Source",
    "Transform",
]


@pytest.mark.parametrize("term", _TERMS)
def test_the_glossary_defines(term: str) -> None:
    assert re.search(rf"^{re.escape(term)}$", _page("glossary.md"), re.M), f"the glossary has no entry {term!r}"
