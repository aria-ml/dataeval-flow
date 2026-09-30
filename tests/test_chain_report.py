"""A chain's report: each finding beside the evidence it judged, the other steps, then a table of every step.

Measured on the toy datasets: data-cleaning on 24 toy images finds one image outlier, of class b, and one exact
duplicate pair, and judges the two balanced classes; `target-outliers` finds nothing on classification data.
"""

from typing import Any
from unittest.mock import patch

import pytest

from dataeval_flow import run_tasks
from dataeval_flow._blocks import Block, Fields, Paragraph, Section, Summary, Table
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult
from tests.chain_toys import CountGroups, GroupLimit, chain_pipeline, register_toys, run_chain_task, run_toy_chain
from tests.evaluator_toys import ToyImages

pytestmark = pytest.mark.usefixtures("toys")

_DUPES = {"name": "dupes", "evaluator": "dupes", "input": "a"}
_COUNT = {"name": "count", "combine": "toy-count-groups", "input": "dupes"}
_JUDGE = {"name": "judge", "check": "toy-at-most", "input": "count", "most": 0}
_CAMS = {"s1": ToyImages(), "s2": ToyImages(seed=1)}


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _cleaning() -> ChainResult:
    workflow = {"name": "cleaning", "type": "data-cleaning", "outlier_method": "zscore"}
    config = chain_pipeline(
        workflows=[{**workflow, "outlier_flags": ["pixel", "visual"]}],
        tasks=[{"name": "t", "workflow": "cleaning", "sources": ["src"]}],
        datasets={"src": ToyImages(count=24)},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def _chain(*steps: dict[str, Any], datasets: dict[str, Any] | None = None, lists: bool = False) -> ChainResult:
    datasets = datasets or {"src": ToyImages()}
    inputs = [{"name": "a", "list": True}] if lists else ["a"]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": inputs, "steps": list(steps)}],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    return result


def _top(result: ChainResult) -> list[Block]:
    return result._document(detailed=True).blocks


def _outline(result: ChainResult) -> list[tuple[str, str | None, str | None]]:
    """Each top-level section's title, brief and severity, in order."""
    return [(block.title, block.brief, block.severity) for block in _top(result) if isinstance(block, Section)]


def _section(result: ChainResult, title: str) -> Section:
    (section,) = [block for block in _top(result) if isinstance(block, Section) and block.title == title]
    return section


def _evidence(section: Section) -> list[str]:
    """What a finding's section shows of the evidence: each "From" section's title, and each pointer line."""
    return [
        block.title if isinstance(block, Section) else block.text
        for block in section.blocks
        if (isinstance(block, Section) and block.title.startswith("From "))
        or (isinstance(block, Paragraph) and block.text.startswith("Evidence: "))
    ]


def test_data_cleaning_puts_each_finding_beside_the_evidence_it_judged() -> None:
    result = _cleaning()
    assert _outline(result) == [
        ("Summary", None, None),
        ("Image Outliers", "1 images (4.2%)", "warning"),
        ("Classwise Outliers", "worst: b (8.3%), 1/1 classes over 3.0%", "warning"),
        ("Duplicates", "2 exact (8.3%), 0 near (0.0%)", "warning"),
        ("Label Distribution", "2 classes, 24 items, imbalance 1.0:1", "info"),
        ("Remove · clean", None, None),
        ("Steps", None, None),
        ("Configuration", None, None),
    ]
    assert _evidence(_section(result, "Image Outliers")) == ["From Outliers"]
    # `classwise` reads `by-class`, a combine with nothing to show, which read `outliers`: shown already.
    assert _evidence(_section(result, "Classwise Outliers")) == ["Evidence: Outliers, under Image Outliers."]
    assert _evidence(_section(result, "Duplicates")) == ["From Duplicates · dupes"]
    assert _evidence(_section(result, "Label Distribution")) == ["From Label Health · labels"]


def test_a_finding_s_evidence_follows_its_own_blocks() -> None:
    duplicates = _section(_cleaning(), "Duplicates")
    *own, evidence = duplicates.blocks
    assert own == [Paragraph(text="1 exact duplicate groups, 0 near-duplicate groups found.")]
    assert isinstance(evidence, Section)
    assert (evidence.title, evidence.brief) == ("From Duplicates · dupes", None)
    assert isinstance(evidence.blocks[0], Table)


def test_the_header_counts_every_step_in_one_line() -> None:
    assert _cleaning()._report_body(detailed=True)[0] == Fields(items=[("Steps", "10 ran")])
    with patch.object(CountGroups, "run", side_effect=RuntimeError("no count")):
        failed = _chain(_DUPES, _COUNT, _JUDGE, {"name": "after", "transform": "toy-keep", "input": "a"})
    assert failed._report_body(detailed=True)[0] == Fields(items=[("Steps", "4 (3 ran, 1 failed)")])


def test_the_steps_table_says_what_each_step_is_what_it_read_and_why_it_made_nothing() -> None:
    steps = _section(_cleaning(), "Steps")
    assert steps.reference
    (table,) = steps.blocks
    assert isinstance(table, Table)
    assert [column.header for column in table.columns] == ["Step", "Title", "Type", "Status", "Reads", "Note"]
    cells = ("step", "title", "type", "status", "reads", "note")
    assert [tuple(row[key] for key in cells) for row in table.rows] == [
        ("outliers", "Outliers", "outliers", "ok", "`data` (src)", ""),
        ("labels", "Label Health", "label-health", "ok", "`data` (src)", ""),
        ("by-class", "Outliers by Class", "classwise-outliers", "ok", "`data` (src)\n`outliers`", ""),
        ("dupes", "Duplicates", "duplicates", "ok", "`data` (src)", ""),
        ("image-outliers", "Image Outliers", "outlier-rate", "ok", "`outliers`", ""),
        ("target-outliers", "Target Outliers", "target-outlier-rate", "ok", "`outliers`\n`labels`", "no findings"),
        ("classwise", "Classwise Outliers", "classwise-outlier-rate", "ok", "`by-class`", ""),
        ("duplicates", "Duplicates", "duplicate-rate", "ok", "`dupes`", ""),
        ("imbalance", "Label Distribution", "class-imbalance", "ok", "`labels`", ""),
        ("clean", "Remove", "remove", "ok", "`data` (src)\n`dupes`\n`outliers`", ""),
    ]


def test_the_html_draws_one_card_per_finding() -> None:
    page = _cleaning().to_html()
    assert page.count('<details class="card ') == 4
    assert page.count('<details class="card warning"') == 3


_MIXED = [
    {"name": "few", "transform": "toy-first", "input": "a", "n": 4},
    {"name": "dupes", "evaluator": "dupes", "input": "few"},
    {"name": "boom", "transform": "toy-explode", "input": "few"},
    {"name": "after", "transform": "toy-keep", "input": "boom"},
]


def test_a_chain_with_no_checks_keeps_a_section_per_step() -> None:
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a"], "steps": _MIXED}], evaluators=[DuplicatesConfig(name="dupes")]
    )
    result = ChainResult.from_run("w", run_toy_chain(config, "w", ["src"]))
    # `boom` failed, so the Summary holds the health line though no step made a finding.
    assert _outline(result) == [
        ("Summary", None, None),
        ("toy-first · few", None, None),
        ("Duplicates · dupes", None, None),
        ("toy-explode · boom", "failed", None),
        ("toy-keep · after", "skipped", None),
        ("Steps", None, None),
    ]
    assert result._report_body(detailed=True)[0] == Fields(items=[("Steps", "4 (2 ran, 1 failed, 1 skipped)")])
    (table,) = _section(result, "Steps").blocks
    assert isinstance(table, Table)
    assert [(row["step"], row["reads"], row["note"]) for row in table.rows] == [
        ("few", "`a` (src)", ""),
        ("dupes", "`few` ← `a` (src)", ""),
        ("boom", "`few` ← `a` (src)", "RuntimeError: boom on few"),
        ("after", "`boom`", "needs `boom`, which failed"),
    ]


def test_a_not_assessed_finding_carries_its_reason_and_its_failed_evidence_shows_the_failure() -> None:
    with patch.object(CountGroups, "run", side_effect=RuntimeError("no count")):
        result = _chain(_DUPES, _COUNT, _JUDGE)
    assert _outline(result) == [
        ("Summary", None, None),
        ("Group count", "not assessed", "info"),
        ("Steps", None, None),
        ("Configuration", None, None),
    ]
    finding = _section(result, "Group count")
    reason, count, dupes = finding.blocks
    assert reason == Paragraph(text="Not assessed: `count` failed: RuntimeError: no count.")
    assert isinstance(count, Section)
    assert (count.title, count.brief) == ("From toy-count-groups · count", "failed")
    assert count.blocks == [Section(title="Failed", blocks=[Paragraph(text="RuntimeError: no count")])]
    assert isinstance(dupes, Section)
    assert (dupes.title, dupes.brief) == ("From Duplicates · dupes", None)


def test_a_check_run_once_per_element_shows_each_element_s_findings_beside_that_element_s_evidence() -> None:
    result = _chain(_DUPES, _COUNT, _JUDGE, datasets=_CAMS, lists=True)
    assert _outline(result) == [
        ("Summary", None, None),
        ("Group count [s1]", "1 groups", "warning"),
        ("Group count [s2]", "1 groups", "warning"),
        ("Steps", None, None),
        ("Configuration", None, None),
    ]
    assert _evidence(_section(result, "Group count [s1]")) == ["From Duplicates · dupes [s1]"]
    assert _evidence(_section(result, "Group count [s2]")) == ["From Duplicates · dupes [s2]"]


def test_an_element_no_finding_read_stays_with_the_other_steps() -> None:
    judged = GroupLimit.run

    def only_s1(self: GroupLimit, config: Any, inputs: Any, context: Any) -> Any:
        return [] if inputs["input"].address.endswith("[s2]") else judged(self, config, inputs, context)

    with patch.object(GroupLimit, "run", only_s1):
        result = _chain(_DUPES, _COUNT, _JUDGE, datasets=_CAMS, lists=True)
    assert _outline(result) == [
        ("Summary", None, None),
        ("Group count [s1]", "1 groups", "warning"),
        ("Duplicates · dupes", None, None),
        ("Steps", None, None),
        ("Configuration", None, None),
    ]
    assert _evidence(_section(result, "Group count [s1]")) == ["From Duplicates · dupes [s1]"]
    rest = _section(result, "Duplicates · dupes")
    assert [block.title for block in rest.blocks if isinstance(block, Section)] == ["[s2]"]
    (table,) = _section(result, "Steps").blocks
    assert isinstance(table, Table)
    assert table.rows[2]["note"] == "[s2] no findings"


def test_a_failed_check_is_listed_among_the_other_steps_with_its_failure() -> None:
    from dataeval_flow.steps.checks import DuplicateRateCheck

    with patch.object(DuplicateRateCheck, "run", side_effect=RuntimeError("boom")):
        result = _cleaning()
    assert _outline(result) == [
        ("Summary", None, None),
        ("Image Outliers", "1 images (4.2%)", "warning"),
        ("Classwise Outliers", "worst: b (8.3%), 1/1 classes over 3.0%", "warning"),
        ("Label Distribution", "2 classes, 24 items, imbalance 1.0:1", "info"),
        ("Duplicates · dupes", None, None),
        ("Duplicates · duplicates", "failed", None),
        ("Remove · clean", None, None),
        ("Steps", None, None),
        ("Configuration", None, None),
    ]
    assert _section(result, "Duplicates · duplicates").blocks == [
        Section(title="Failed", blocks=[Paragraph(text="RuntimeError: boom")])
    ]


def test_a_skipped_check_is_listed_among_the_other_steps_with_its_reason() -> None:
    with patch.object(GroupLimit, "run", side_effect=RuntimeError("boom")):
        result = _chain(_DUPES, _COUNT, {**_JUDGE, "optional": True})
    assert _outline(result) == [
        ("Duplicates · dupes", None, None),
        ("Group count · judge", "skipped", None),
        ("Steps", None, None),
        ("Configuration", None, None),
    ]
    assert _section(result, "Group count · judge").blocks == [Paragraph(text="Skipped: failed: RuntimeError: boom")]


def test_a_check_with_a_failed_element_lists_that_element_among_the_other_steps() -> None:
    judged = GroupLimit.run

    def fails_on_s2(self: GroupLimit, config: Any, inputs: Any, context: Any) -> Any:
        if inputs["input"].address.endswith("[s2]"):
            raise RuntimeError("boom")
        return judged(self, config, inputs, context)

    with patch.object(GroupLimit, "run", fails_on_s2):
        result = _chain(_DUPES, _COUNT, _JUDGE, datasets=_CAMS, lists=True)
    assert _outline(result) == [
        ("Summary", None, None),
        ("Group count [s1]", "1 groups", "warning"),
        ("Duplicates · dupes", None, None),
        ("Group count · judge", "failed", None),
        ("Steps", None, None),
        ("Configuration", None, None),
    ]
    assert _section(result, "Group count · judge").blocks == [
        Section(
            title="[s2]",
            brief="failed",
            blocks=[Section(title="Failed", blocks=[Paragraph(text="RuntimeError: boom")])],
        )
    ]


def _spliced() -> ChainResult:
    """data-cleaning run as step `cleaning` of a custom workflow, whose spliced steps' names are the widest."""
    tidy = {"name": "tidy", "type": "data-cleaning", "outlier_method": "zscore", "outlier_flags": ["pixel", "visual"]}
    kept = {"name": "kept", "transform": "toy-keep", "input": "cleaning.clean"}
    workflow = {
        "name": "w",
        "inputs": ["data"],
        "steps": [{"name": "cleaning", "workflow": "tidy", "input": "data"}, kept],
    }
    config = chain_pipeline(
        workflows=[tidy, workflow],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": ToyImages(count=24)},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def test_no_line_of_a_data_cleaning_report_is_wider_than_the_report() -> None:
    for result in (_cleaning(), _spliced()):
        text = result.report()
        assert "  STEPS" in text
        assert max(len(line) for line in text.splitlines()) <= 80


def test_a_wrapped_reads_cell_breaks_before_a_lineage_arrow() -> None:
    result = _chain(
        {"name": "k", "transform": "toy-keep", "input": "a"},
        {"name": "few", "transform": "toy-first", "input": "k", "n": 4},
        {"name": "dupes", "evaluator": "dupes", "input": "few"},
    )
    lines = result.report(width=60).splitlines()
    start = lines.index("  STEPS") + 2
    assert lines[start : lines.index("  CONFIGURATION") - 2] == [
        "  Step   Title       Type        Status  Reads          Note",
        "  -----  ----------  ----------  ------  -------------  ----",
        "  k      toy-keep    toy-keep    ok      `a` (src)",
        "",
        "  few    toy-first   toy-first   ok      `k`",
        "                                         ← `a` (src)",
        "",
        "  dupes  Duplicates  duplicates  ok      `few` ← `k`",
        "                                         ← `a` (src)",
    ]


def test_the_steps_panel_renders_what_a_step_read_as_code() -> None:
    page = _cleaning().to_html()
    assert '<td class="left" data-value="`dupes`"><code>dupes</code></td>' in page
    assert ">`dupes`<" not in page


def test_a_chain_whose_required_step_failed_says_so_in_its_health_line_though_nothing_warns() -> None:
    result = _chain(_DUPES, _COUNT, {**_JUDGE, "most": 5}, {"name": "boom", "transform": "toy-explode", "input": "a"})
    assert result.health["status"] == "failed"
    text = result.report()
    assert "  Health: failed [!!] — step `boom` failed" in text.splitlines()
    assert "All checks passed" not in text
    page = result.to_html()
    assert '<h1>w</h1><span class="badge failed">failed: boom</span>' in page
    assert '<span class="badge ok">passed</span>' not in page


def test_a_chain_whose_check_failed_says_so_in_its_health_line_beside_its_warnings() -> None:
    from dataeval_flow.steps.checks import DuplicateRateCheck

    with patch.object(DuplicateRateCheck, "run", side_effect=RuntimeError("boom")):
        result = _cleaning()
    assert "  Health: failed [!!] — step `duplicates` failed; 2 warning(s) to review" in result.report().splitlines()
    assert '<h1>Data Cleaning</h1><span class="badge failed">failed: duplicates</span>' in result.to_html()


def test_a_failed_chain_with_no_findings_still_has_a_health_line() -> None:
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a"], "steps": _MIXED}], evaluators=[DuplicatesConfig(name="dupes")]
    )
    result = ChainResult.from_run("w", run_toy_chain(config, "w", ["src"]))
    summary = _section(result, "Summary")
    assert summary.blocks == [
        Paragraph(text="No findings to report."),
        Summary(items=[], warnings=0, failed=["boom"]),
    ]
    assert '<p class="health failed">Step <code>boom</code> failed</p>' in result.to_html()
