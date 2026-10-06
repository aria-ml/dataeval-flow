"""Renderers show the verdict a result counted once, and fold the sections it marks as reference (spec §9.1, §9.6)."""

from dataeval_flow._binning_report import binning_blocks
from dataeval_flow._blocks import Section, Summary, SummaryItem, Tree
from dataeval_flow._blocks._html import render_html
from dataeval_flow._blocks._text import render_text
from dataeval_flow.steps import ChainMetadata, ChainResult, Finding
from tests.workflow_toys import count_result

_WARNING = SummaryItem(label="Duplicates", value="3 groups", severity="warning")


def _result(*findings: Finding, config: dict[str, object] | None = None) -> ChainResult:
    return count_result(*findings, metadata=ChainMetadata(resolved_config=config or {}))


def test_the_text_health_line_states_the_count_the_result_made() -> None:
    assert render_text([Summary(items=[_WARNING], warnings=2)])[-1] == (
        "Health: 2 warning(s) [!!] — review flagged findings"
    )


def test_the_html_health_line_states_the_count_the_result_made() -> None:
    fragment = render_html([Summary(items=[_WARNING], warnings=2)])
    assert '<p class="health warning">2 warnings — review the flagged findings</p>' in fragment


def test_the_text_health_line_of_a_failed_run_names_its_failed_steps_then_its_warnings() -> None:
    assert render_text([Summary(items=[_WARNING], warnings=2, failed=["clean"])])[-1] == (
        "Health: failed [!!] — step `clean` failed; 2 warning(s) to review"
    )
    assert render_text([Summary(items=[], warnings=0, failed=["a", "b"])]) == [
        "Health: failed [!!] — steps `a`, `b` failed"
    ]


def test_the_html_health_line_and_badge_of_a_failed_run_name_its_failed_steps() -> None:
    fragment = render_html([Summary(items=[_WARNING], warnings=2, failed=["clean"])])
    assert '<p class="health failed">Step <code>clean</code> failed; 2 warnings to review</p>' in fragment
    alone = render_html([Summary(items=[], warnings=0, failed=["a", "b"])])
    assert alone == '<p class="health failed">Steps <code>a</code>, <code>b</code> failed</p>'
    report = Section(
        title="Run", blocks=[Section(title="Summary", blocks=[Summary(items=[], warnings=0, failed=["a"])])]
    )
    assert '<span class="badge failed">failed: a</span></header>' in render_html([report])


def test_the_report_header_badge_states_the_count_its_summary_carries() -> None:
    report = Section(
        title="Run",
        blocks=[
            Section(title="Summary", blocks=[Summary(items=[_WARNING], warnings=3)]),
            Section(title="Duplicates", brief="3 groups", severity="warning"),
        ],
    )
    assert '<span class="badge warning">3 warnings</span></header>' in render_html([report])


def test_a_report_whose_summary_counts_no_warnings_passes() -> None:
    item = SummaryItem(label="Labels", severity="ok")
    report = Section(title="Run", blocks=[Section(title="Summary", blocks=[Summary(items=[item], warnings=0)])])
    assert '<span class="badge ok">passed</span></header>' in render_html([report])


def test_a_workflow_result_s_summary_carries_the_count_it_made() -> None:
    result = _result(Finding(title="a", severity="warning"), Finding(title="b", severity="ok"))
    (section,) = result._summary_blocks()
    assert isinstance(section, Section)
    (summary,) = section.blocks
    assert isinstance(summary, Summary)
    assert summary.warnings == 1


def test_a_section_marked_as_reference_folds_away_whatever_its_title() -> None:
    report = Section(
        title="Run", blocks=[Section(title="Settings used", reference=True, blocks=[Tree(value={"a": 1})])]
    )
    assert '<details class="panel"><summary><h2>Settings used</h2></summary>' in render_html([report])


def test_a_section_titled_configuration_but_not_marked_does_not_fold() -> None:
    report = Section(title="Run", blocks=[Section(title="Configuration", blocks=[Tree(value={"a": 1})])])
    assert '<details class="panel">' not in render_html([report])


def test_a_result_marks_its_configuration_as_reference() -> None:
    document = _result(Finding(title="a"), config={"seed": 1})._document(detailed=True)
    last = document.blocks[-1]
    assert isinstance(last, Section)
    assert (last.title, last.reference) == ("Configuration", True)


def test_the_metadata_factors_section_is_marked_as_reference() -> None:
    (section,) = binning_blocks(None, ["a diagnostic"])
    assert isinstance(section, Section)
    assert (section.title, section.reference) == ("Metadata Factors", True)


def test_a_section_s_mark_is_left_out_of_the_json_while_false() -> None:
    assert "reference" not in Section(title="x").model_dump(mode="json")
    assert Section(title="x", reference=True).model_dump(mode="json")["reference"] is True


def test_a_finding_leaves_an_unset_step_out_of_its_json_and_keeps_a_set_one() -> None:
    assert "step" not in Finding(title="x").model_dump(mode="json")
    assert Finding(title="x", step="judge").model_dump(mode="json")["step"] == "judge"


def test_a_finding_keeps_its_fields_in_the_serialization_schema() -> None:
    properties = Finding.model_json_schema(mode="serialization")["properties"]
    assert {"title", "severity"} <= set(properties)
