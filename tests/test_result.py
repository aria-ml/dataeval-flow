"""The shared result shape: both kinds render and serialize the same way, and differ only inside."""

import json

import pytest

from dataeval_flow import Result, ResultMetadata
from dataeval_flow._result import results_html
from dataeval_flow.evaluators import EvaluatorResult
from dataeval_flow.evaluators._result import EvaluatorMetadata
from dataeval_flow.workflows import WorkflowReport, WorkflowResult
from tests.test_blocks_html import _well_formed
from tests.workflow_toys import ToyCountMetadata, ToyCountOutput, ToyCountRaw, ToyCountResult


def _output() -> ToyCountOutput:
    return ToyCountOutput(
        raw=ToyCountRaw(dataset_size=3),
        report=WorkflowReport(summary="Items counted."),
    )


def _workflow(*, success: bool = True) -> ToyCountResult:
    return ToyCountResult(
        type="test.count",
        success=success,
        output=_output() if success else None,
        metadata=ToyCountMetadata(),
        errors=[] if success else ["boom"],
    )


def _evaluator(*, success: bool = True) -> EvaluatorResult[object]:
    return EvaluatorResult(
        type="duplicates",
        success=success,
        output=object() if success else None,
        serialized={"shape": "mapping", "data": {"groups": 0}} if success else None,
        metadata=EvaluatorMetadata(evaluator="duplicates"),
        errors=[] if success else ["boom", "bang"],
    )


def test_the_short_page_gives_the_verdict_the_full_page_gives():
    """The short form has no finding cards, so its badge comes from the summary's lines."""
    from dataeval_flow.steps import Finding

    result = _workflow()
    assert result.output is not None
    result.output.report.findings = [Finding(title="Duplicates", severity="warning", brief="3 groups")]
    for detailed in (True, False):
        assert '<span class="badge warning">1 warning</span></header>' in result.to_html(detailed=detailed)


def test_the_page_shows_the_thumbnails_its_result_carries():
    """One run's results share one page, and each shows the thumbnails its own result captured."""
    from dataeval_flow._blocks import Asset, Column, ItemRef, Table
    from dataeval_flow.steps import Finding

    ref = ItemRef(source="train", index=4)
    result = _workflow()
    result.output.report.findings = [
        Finding(title="Outliers", blocks=[Table(columns=[Column(key="i", kind="image")], rows=[{"i": ref}])])
    ]
    assert '<span class="item">4</span>' in result.to_html()
    result.assets = [Asset(item=ref, media_type="image/webp", width=4, height=4, data="QUJD")]
    thumbnail = '<img src="data:image/webp;base64,QUJD" alt="train 4">'
    assert thumbnail in result.to_html()
    assert thumbnail in results_html([_evaluator(), result])


def test_the_page_titles_a_report_s_own_sections_as_it_titles_its_findings():
    """Summary, Configuration, Output and Failed, in title case; the text report capitalizes every section alike."""
    from dataeval_flow.steps import Finding

    result = _workflow()
    result.output.report.findings = [Finding(title="Duplicates", severity="warning", brief="3 groups")]
    result.metadata.resolved_config = {"seed": 1}
    assert '<details class="panel"><summary><h2>Configuration</h2></summary>' in result.to_html()
    assert '<section class="section"><h2>Summary</h2>' in result.to_html(detailed=False)
    assert '<section class="section"><h2>Output' in _evaluator().to_html()
    assert '<section class="section"><h2>Failed</h2>' in _workflow(success=False).to_html()
    assert "  CONFIGURATION" in result.report().splitlines()


def test_a_page_of_several_results_shows_each_its_own_thumbnails():
    """Each task draws its own random view, so one source's index may be two images: each report shows its own."""
    from dataeval_flow._blocks import Asset, Column, ItemRef, Table
    from dataeval_flow.steps import Finding

    ref = ItemRef(source="train", index=4)
    results = []
    for data in ("QUFB", "QkJC"):
        result = _workflow()
        result.output.report.findings = [
            Finding(title="Outliers", blocks=[Table(columns=[Column(key="i", kind="image")], rows=[{"i": ref}])])
        ]
        result.assets = [Asset(item=ref, media_type="image/webp", width=4, height=4, data=data)]
        results.append(result)
    page = results_html(results)
    assert page.index("base64,QUFB") < page.index('id="r2"') < page.index("base64,QkJC")


def test_the_base_cannot_be_built_on_its_own():
    with pytest.raises(TypeError, match="abstract"):
        Result(type="r", success=False, metadata=ResultMetadata())  # type: ignore[abstract]


@pytest.mark.parametrize(("make", "kind"), [(_workflow, "workflow"), (_evaluator, "evaluator")])
class TestOneShape:
    def test_each_kind_subclasses_result(self, make, kind):
        result = make()
        assert isinstance(result, Result)
        assert result.kind == kind

    def test_type_names_what_ran(self, make, kind):
        result = make()
        assert result.type in {"test.count", "duplicates"}
        assert not hasattr(result, "name")

    def test_a_dict_leads_with_the_kind_and_the_envelope(self, make, kind):
        payload = make().to_dict()
        assert list(payload)[:2] == ["kind", "metadata"]
        assert payload["kind"] == kind

    def test_json_export_is_the_dict(self, make, kind):
        result = make()
        assert json.loads(result.export()) == result.to_dict()

    def test_assets_close_the_dict_where_there_are_any(self, make, kind):
        from dataeval_flow._blocks import Asset, ItemRef

        result = make()
        assert result.assets == []
        assert "assets" not in result.to_dict()
        asset = Asset(item=ItemRef(source="s", index=4), media_type="image/webp", width=16, height=16, data="UklG")
        result.assets = [asset]
        payload = result.to_dict()
        assert list(payload)[-1] == "assets"
        assert payload["assets"] == [asset.model_dump(mode="json")]
        assert json.loads(result.export())["assets"][0]["item"] == {"source": "s", "index": 4}

    def test_a_failed_run_reports_its_errors(self, make, kind):
        lines = make(success=False).report().splitlines()
        at = lines.index("  FAILED")
        assert lines[at + 2] == "  boom"

    def test_the_report_defaults_to_eighty_columns(self, make, kind):
        lines = make().report().splitlines()
        assert lines[1] == "=" * 80
        assert all(len(line) <= 80 for line in lines)

    def test_the_report_draws_at_the_width_asked_for(self, make, kind):
        lines = make().report(width=60).splitlines()
        assert lines[1] == "=" * 60
        assert all(len(line) <= 60 for line in lines), max(lines, key=len)

    def test_the_html_page_is_the_report(self, make, kind):
        result = make()
        page = result.to_html()
        assert page.startswith("<!doctype html>")
        heading = {"workflow": "test.count", "evaluator": "Duplicates"}[kind]
        assert f"<h1>{heading}</h1>" in page
        assert _well_formed(page)
        assert page.count("<script>") == 1

    def test_a_failed_run_s_page_shows_its_errors(self, make, kind):
        page = make(success=False).to_html()
        assert "<h2>Failed</h2>" in page
        assert "<p>boom</p>" in page

    def test_an_empty_configuration_draws_no_section(self, make, kind):
        result = make()
        result.metadata.resolved_config = {}
        assert "CONFIGURATION" not in result.report()

    def test_the_configuration_leaves_out_settings_that_are_none_but_the_json_keeps_them(self, make, kind):
        result = make()
        result.metadata.resolved_config = {"seed": 1, "device": None, "nested": {"limit": None, "kept": 2}}
        text = result.report()
        assert "seed: 1" in text
        assert "kept: 2" in text
        assert "device" not in text
        assert "limit" not in text
        assert result.to_dict()["metadata"]["resolved_config"]["device"] is None

    def test_the_configuration_keeps_a_none_it_was_told_was_set_through_lists_as_through_mappings(self, make, kind):
        result = make()
        result.metadata.resolved_config = {"sources": [{"x": None, "y": None, "z": 1}], "w": None}
        result._set_nulls = frozenset({("sources", 0, "x")})
        text = result.report()
        assert "x: None" in text
        assert "y" not in text.split("CONFIGURATION")[1]
        assert "w:" not in text.split("CONFIGURATION")[1]
        result.metadata.resolved_config = {"sources": [{"x": None}]}
        result._set_nulls = frozenset()
        assert "x: None" not in result.report()

    def test_a_configuration_of_only_none_draws_no_section(self, make, kind):
        result = make()
        result.metadata.resolved_config = {"device": None}
        assert "CONFIGURATION" not in result.report()

    def test_a_non_json_value_in_the_configuration_renders_as_its_text(self, make, kind):
        from pathlib import Path

        result = make()
        result.metadata.resolved_config = {"sources": [{"path": Path("data/train")}]}
        assert "path: data/train" in result.report()

    def test_a_width_below_forty_is_refused(self, make, kind):
        with pytest.raises(ValueError, match="at least 40"):
            make().report(width=39)

    def test_a_failed_runs_output_raises_naming_each_error(self, make, kind):
        failed = make(success=False)
        with pytest.raises(RuntimeError) as raised:
            _ = failed.output
        assert all(error in str(raised.value) for error in failed.errors)

    def test_a_failed_runs_dict_is_its_kind_envelope_and_errors(self, make, kind):
        payload = make(success=False).to_dict()
        assert set(payload) == {"kind", "metadata", "errors"}
        assert payload["errors"] == make(success=False).errors

    def test_a_success_must_carry_its_output(self, make, kind):
        result = make()
        with pytest.raises(ValueError, match="output"):
            type(result)(type=result.type, success=True, metadata=result.metadata)

    def test_a_failure_cannot_carry_an_output(self, make, kind):
        result = make()
        with pytest.raises(ValueError, match="output"):
            type(result)(type=result.type, success=False, output=result.output, metadata=result.metadata)

    def test_output_raises_whenever_the_run_is_not_a_success(self, make, kind):
        """The guard reads ``success``, so a result marked failed never hands back what it holds."""
        result = make()
        result.success = False
        with pytest.raises(RuntimeError, match="did not complete"):
            _ = result.output


def test_isinstance_narrows_to_the_types_own_result():
    result: Result = _workflow()
    assert isinstance(result, ToyCountResult)
    assert isinstance(result, WorkflowResult)
    assert result.output.raw.dataset_size == 3
    assert isinstance(result.metadata, ToyCountMetadata)


def test_only_a_workflow_carries_health():
    assert "health" in _workflow().to_dict()
    assert "health" not in _evaluator().to_dict()
    assert not hasattr(_evaluator(), "health")


def test_a_failed_workflow_has_no_findings_and_a_failed_health():
    failed = _workflow(success=False)
    assert failed.findings == []
    assert failed.health["status"] == "failed"
    assert failed.warning_count == 0


def test_a_failed_result_of_a_class_carries_that_class_metadata():
    failed = ToyCountResult.failed(type="test.count", errors=["boom"])
    assert isinstance(failed, ToyCountResult)
    assert isinstance(failed.metadata, ToyCountMetadata)
    assert not failed.success
    assert failed.errors == ["boom"]


def test_every_argument_is_keyword_only():
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        WorkflowResult("test.count", True, _output(), ResultMetadata())  # type: ignore[misc]


def test_every_result_of_a_run_shares_one_page():
    page = results_html([_workflow(), _evaluator()])
    assert "<title>dataeval-flow results</title>" in page
    assert page.count("<h1>") == 2
    assert _well_formed(page)


def test_a_run_of_one_task_is_titled_by_its_report():
    assert "<title>test.count</title>" in results_html([_workflow()])


def test_a_run_with_no_report_to_show_still_writes_a_page():
    page = results_html([])
    assert "<title>dataeval-flow results</title>" in page
    assert _well_formed(page)


def test_a_nulled_setting_of_an_alias_keyed_model_is_tracked_under_its_alias() -> None:
    from pydantic import BaseModel, ConfigDict, Field

    from dataeval_flow._orchestrator import _nulls_of

    class Block(BaseModel):
        warning: float | None = 1.0

    class Keyed(BaseModel):
        model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)
        image_outliers: Block = Field(default_factory=Block, alias="image-outliers")

    class Plain(BaseModel):
        image_outliers: Block = Field(default_factory=Block)

    assert _nulls_of(Keyed.model_validate({"image-outliers": {"warning": None}}), ("w",)) == {
        ("w", "image-outliers", "warning")
    }
    assert _nulls_of(Plain(image_outliers=Block(warning=None)), ("w",)) == {("w", "image_outliers", "warning")}
