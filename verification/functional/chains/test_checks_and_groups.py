"""TC-19-4 and TC-19-5 — judging what a chain found (checks, combines, groups, presets), and per-class steps."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult, CustomWorkflowConfig, StepEntry
from verification.functional.audit._helpers import section
from verification.functional.chains._toys import Images, shifted_sources, yaml_pipeline

pytestmark = pytest.mark.required

_JUDGED = """
evaluators:
  - {name: dupes, type: duplicates}
  - {name: labels, type: label-health}
workflows:
  - name: w
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - {name: labels, evaluator: labels, input: data}
      - {name: dup_check, check: image-duplicates, input: dupes%s}
      - {name: imb, check: class-imbalance, input: labels%s}
%s
tasks:
  - {name: t, workflow: w, sources: [src]}
"""
_GROUPS = """    groups:
      - {heading: "Is the data clean?", checks: [image-duplicates]}"""


def _run(text: str, datasets: dict, *, extractor: bool = False) -> ChainResult:
    result = run_tasks(yaml_pipeline(text, datasets, extractor=extractor))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _judged(exact_near: str = "", imbalance: str = "", groups: str = "") -> str:
    return _JUDGED % (exact_near, imbalance, groups)


class TestChecks:
    def test_evaluators_alone_judge_nothing(self) -> None:
        text = _judged().replace("      - {name: dup_check, check: image-duplicates, input: dupes}\n", "")
        text = text.replace("      - {name: imb, check: class-imbalance, input: labels}\n", "")
        result = _run(text, {"src": Images()})
        assert result.findings == []
        assert result.health["status"] == "ok"
        assert "No findings to report." in result.report()

    def test_a_check_judges_an_evaluator_s_output_against_its_thresholds(self) -> None:
        result = _run(_judged(), {"src": Images(9)})
        by_step = {finding.step: finding for finding in result.findings}
        assert (by_step["dup_check"].severity, by_step["dup_check"].title) == ("warning", "Image Duplicates")
        assert by_step["dup_check"].brief == "2 exact (22.2%), 0 near (0.0%)"
        assert by_step["imb"].severity == "info"
        assert result.steps["dup_check"].kind == "check"

    def test_a_threshold_written_beside_the_step_changes_the_severity(self) -> None:
        result = _run(_judged(imbalance=", warning: 1.0"), {"src": Images(9)})
        assert {finding.step: finding.severity for finding in result.findings}["imb"] == "warning"
        assert result.health["warnings"] == 2

    def test_a_null_threshold_judges_nothing_and_the_finding_is_info(self) -> None:
        result = _run(_judged(exact_near=", exact: null, near: null"), {"src": Images(9)})
        assert {finding.step: finding.severity for finding in result.findings}["dup_check"] == "info"
        assert result.health["status"] == "ok"

    def test_a_chain_s_health_is_warning_where_a_finding_is_a_warning(self) -> None:
        result = _run(_judged(), {"src": Images(9)})
        assert result.health == {"status": "warning", "warnings": 1, "findings": 2, "failed_steps": []}
        assert result.warning_count == 1
        assert "Health: 1 warning(s)" in result.report()

    def test_the_report_gives_each_finding_a_section_holding_the_evidence_it_judged(self) -> None:
        report = _run(_judged(), {"src": Images(9)}).report()
        heading = next(line for line in report.splitlines() if line.strip().startswith("IMAGE DUPLICATES"))
        assert "2 exact (22.2%), 0 near (0.0%)" in heading
        assert "From Duplicates · dupes" in report
        assert "From Label Health · labels" in report

    def test_a_threshold_a_check_does_not_take_is_refused_when_the_config_loads(self) -> None:
        with pytest.raises(ValidationError, match="bogus"):
            yaml_pipeline(_judged(exact_near=", bogus: 1"), {"src": Images()})

    def test_a_check_over_an_input_that_made_nothing_says_not_assessed_instead_of_passing(self) -> None:
        text = """
evaluators:
  - {name: dupes, type: duplicates}
workflows:
  - name: w
    inputs: [train, {name: evals, list: true, empty: no evaluation split given}]
    steps:
      - {name: d, evaluator: dupes, input: evals}
      - {name: dc, check: image-duplicates, input: d}
tasks:
  - {name: t, workflow: w, sources: [src]}
"""
        result = _run(text, {"src": Images()})
        assert result.steps["d"].status == "skipped"
        assert result.steps["d"].reason == "no evaluation split given"
        assert result.steps["dc"].status == "ok"
        assert [(f.severity, f.brief) for f in result.findings] == [("info", "not assessed")]
        assert result.health["status"] == "ok"


class TestCombines:
    def test_a_combine_makes_an_output_a_check_reads(self) -> None:
        text = """
evaluators:
  - {name: out, type: outliers, flags: [pixel], outlier_threshold: zscore}
workflows:
  - name: w
    inputs: [data]
    steps:
      - {name: out, evaluator: out, input: data}
      - {name: by_class, combine: outliers-by-class, input: data, outliers: out}
      - {name: worst, check: class-outliers, input: by_class, warning: 1.0}
tasks:
  - {name: t, workflow: w, sources: [src]}
"""
        result = _run(text, {"src": Images(40)})
        assert result.steps["by_class"].kind == "combine"
        (finding,) = result.findings
        assert (finding.step, finding.title, finding.severity) == ("worst", "Class Outliers", "warning")
        assert finding.brief.startswith("worst: b (")

    def test_a_check_refuses_an_input_of_a_kind_it_does_not_judge(self) -> None:
        text = _judged().replace("check: image-duplicates, input: dupes", "check: image-duplicates, input: labels")
        with pytest.raises(ValidationError, match="labels"):
            yaml_pipeline(text, {"src": Images()})


class TestGroups:
    def test_groups_put_findings_under_headings_with_a_status_line(self) -> None:
        report = _run(_judged(groups=_GROUPS), {"src": Images(9)}).report()
        assert "IS THE DATA CLEAN?" in report
        heading = next(line for line in report.splitlines() if "IS THE DATA CLEAN?" in line)
        assert heading.rstrip().endswith("1 warning")
        assert "Image Duplicates" in section(report, "Is the data clean?")

    def test_findings_of_check_types_no_group_names_follow_the_groups(self) -> None:
        report = _run(_judged(groups=_GROUPS), {"src": Images(9)}).report()
        assert report.index("IS THE DATA CLEAN?") < report.index("CLASS IMBALANCE")

    def test_a_group_naming_a_check_no_step_runs_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="group 'H' names check `image-duplicates`, which no step runs"):
            CustomWorkflowConfig(
                name="w",
                inputs=["a"],
                steps=[StepEntry(name="d", evaluator="dupes", input="a")],
                groups=[{"heading": "H", "checks": ["image-duplicates"]}],  # type: ignore[list-item]
            )

    def test_groups_change_only_the_report_and_give_no_verdict(self) -> None:
        grouped = _run(_judged(groups=_GROUPS), {"src": Images(9)})
        plain = _run(_judged(), {"src": Images(9)})
        assert grouped.verdict is None
        assert [(f.step, f.severity) for f in grouped.findings] == [(f.step, f.severity) for f in plain.findings]
        assert grouped.health == plain.health


_PRESET_AS_STEP = """
evaluators:
  - {name: labels, type: label-health}
workflows:
  - name: tidy
    type: quality
    outliers: {flags: [pixel], outlier_threshold: zscore}
  - name: w
    inputs: [data]
    steps:
      - {name: cleaning, workflow: tidy, input: data}
      - {name: after, evaluator: labels, input: cleaning.clean}
tasks:
  - {name: t, workflow: w, sources: [src]}
"""


class TestPresetAsAStep:
    def test_a_preset_runs_its_whole_chain_under_the_step_s_name(self) -> None:
        result = _run(_PRESET_AS_STEP, {"src": Images()})
        inner = [name for name in result.steps if name.startswith("cleaning/")]
        assert {"cleaning/outliers", "cleaning/duplicates", "cleaning/image-outliers", "cleaning/clean"} <= set(inner)
        assert result.steps["cleaning/clean"].type == "remove"

    def test_a_later_step_reads_the_preset_s_declared_output(self) -> None:
        result = _run(_PRESET_AS_STEP, {"src": Images()})
        # The preset removed the planted duplicate and the white outlier.
        assert len(result.steps["cleaning/clean"].output) == 10
        assert result.steps["after"].output.data()["item_count"] == 10

    def test_the_preset_s_findings_count_toward_the_chain_s_health(self) -> None:
        result = _run(_PRESET_AS_STEP, {"src": Images()})
        assert {f.step for f in result.findings} >= {"cleaning/image-outliers", "cleaning/image-duplicates"}
        assert result.health["status"] == "warning"

    @pytest.mark.parametrize("address", ["cleaning", "cleaning.duplicates"])
    def test_only_a_declared_output_can_be_read_and_always_by_name(self, address: str) -> None:
        with pytest.raises(ValidationError, match="cleaning"):
            yaml_pipeline(_PRESET_AS_STEP.replace("cleaning.clean", address), {"src": Images()})


class TestPerClassSteps:
    _TEXT = """
evaluators:
  - {name: mmd, type: drift-mmd}
workflows:
  - name: w
    inputs: [reference, test]
    steps:
      - {name: by_class, evaluator: mmd, input: [reference, test], by: class}
      - {name: judged, check: drift, input: by_class, by: class}
      - name: by_group
        evaluator: mmd
        input: [reference, test]
        by: {class: {groups: {everything: [a, b]}, min_items: 2}}
tasks:
  - {name: t, workflow: w, sources: [reference, test], extractor: flat}
"""

    def test_by_class_runs_the_step_once_per_class_and_keeps_every_run_in_one_output(self) -> None:
        result = _run(self._TEXT, shifted_sources(40), extractor=True)
        output = result.steps["by_class"].output
        assert type(output).__name__ == "PerClassOutput"
        assert list(output.outputs) == ["a", "b"]
        assert output.skipped == {}
        assert all(run.drifted for run in output.outputs.values())

    def test_a_check_with_by_class_judges_each_key_and_rolls_the_findings_into_one(self) -> None:
        result = _run(self._TEXT, shifted_sources(40), extractor=True)
        (finding,) = result.findings
        assert finding.step == "judged"
        assert finding.severity == "warning"
        assert finding.title.endswith("by class")
        assert finding.brief == "2/2 classes warn"

    def test_groups_key_each_named_group_instead_of_each_class(self) -> None:
        result = _run(self._TEXT, shifted_sources(40), extractor=True)
        assert list(result.steps["by_group"].output.outputs) == ["everything"]
