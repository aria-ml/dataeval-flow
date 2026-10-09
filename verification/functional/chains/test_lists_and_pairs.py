"""TC-19-5 — lists: list inputs, a step run once per element, keys and addresses, `pairs:` and `empty:`."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from verification.functional.chains._toys import Images, yaml_pipeline

pytestmark = pytest.mark.required

_CAMERAS = """
evaluators:
  - {name: lh, type: label-health}
  - {name: dupes, type: duplicates}
workflows:
  - name: w
    inputs: [reference, {name: cameras, list: true}]
    steps:
%s
tasks:
  - {name: t, workflow: w, sources: [ref, cam1, cam2, cam3]}
"""
_DATA = {
    "ref": Images(12, planted=False),
    "cam1": Images(12, seed=1, planted=False),
    "cam2": Images(9, seed=2, planted=False),
    "cam3": Images(14, seed=3, planted=False),
}


def _cameras(steps: str) -> str:
    return _CAMERAS % steps


def _run(text: str, datasets: dict | None = None) -> ChainResult:
    result = run_tasks(yaml_pipeline(text, datasets or _DATA))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


class TestListInputs:
    def test_the_last_input_binds_every_remaining_source_keyed_by_source_name(self) -> None:
        result = _run(_cameras("      - {name: per_camera, evaluator: lh, input: cameras}"))
        elements = result.steps["per_camera"].elements
        assert list(elements) == ["cam1", "cam2", "cam3"]
        assert [elements[key].output.data()["item_count"] for key in elements] == [12, 9, 14]
        assert {r.name for r in result.metadata.lineage} >= {"cameras[cam1]", "cameras[cam2]", "cameras[cam3]"}

    def test_a_step_can_read_one_element_by_its_key(self) -> None:
        result = _run(_cameras('      - {name: just_two, evaluator: lh, input: "cameras[cam2]"}'))
        assert result.steps["just_two"].output.data()["item_count"] == 9

    def test_a_single_dataset_beside_a_list_is_repeated_for_each_element(self) -> None:
        # Each element of `cameras` is compared with the same reference: a duplicates search across the two.
        result = _run(_cameras("      - {name: against_ref, evaluator: dupes, input: [reference, cameras]}"))
        assert list(result.steps["against_ref"].elements) == ["cam1", "cam2", "cam3"]

    def test_a_check_run_once_per_element_makes_findings_grouped_under_the_element_s_key(self) -> None:
        steps = """      - {name: per_camera, evaluator: lh, input: cameras}
      - {name: imbalance, check: class-imbalance, input: per_camera}"""
        result = _run(_cameras(steps))
        assert [finding.step for finding in result.findings] == [
            "imbalance[cam1]",
            "imbalance[cam2]",
            "imbalance[cam3]",
        ]
        report = result.report()
        assert all(f"\n  {key.upper()}\n" in report for key in ("cam1", "cam2", "cam3"))

    def test_collect_gathers_named_elements_into_a_new_list(self) -> None:
        steps = """      - {name: some, transform: collect, input: [reference, "cameras[cam1]"]}
      - {name: each, evaluator: lh, input: some}"""
        result = _run(_cameras(steps))
        assert list(result.steps["each"].elements) == ["reference", "cam1"]


class TestListAddresses:
    def test_a_key_the_task_does_not_bind_is_refused_when_the_config_loads(self) -> None:
        text = _cameras('      - {name: s, evaluator: lh, input: "cameras[cam9]"}')
        with pytest.raises(ValidationError, match=r"`cameras\[cam9\]`, but `cameras` has elements cam1, cam2, cam3"):
            yaml_pipeline(text, _DATA)

    def test_a_fold_outside_the_folds_the_step_makes_is_refused_when_the_config_loads(self) -> None:
        text = """
evaluators:
  - {name: lh, type: label-health}
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: kf, transform: kfold, input: a, folds: 3}
      - {name: s, evaluator: lh, input: "kf.train[5]"}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        with pytest.raises(ValidationError, match=r"`kf\.train\[5\]`, but `kf\.train` has elements 0, 1, 2"):
            yaml_pipeline(text, {"a": Images(30)})

    def test_an_address_with_a_key_left_unquoted_in_flow_style_fails_to_parse(self) -> None:
        with pytest.raises(yaml.YAMLError):
            yaml_pipeline(_cameras("      - {name: s, evaluator: lh, input: cameras[cam1]}"), _DATA)

    def test_a_step_with_several_outputs_must_be_addressed_by_one_of_them(self) -> None:
        text = """
evaluators:
  - {name: lh, type: label-health}
workflows:
  - name: w
    inputs: [a]
    steps:
      - {name: sp, transform: split, input: a, test_frac: 0.2}
      - {name: s, evaluator: lh, input: sp}
tasks:
  - {name: t, workflow: w, sources: [a]}
"""
        with pytest.raises(
            ValidationError, match=r"step 'sp' has outputs test, train, val: name one, such as `sp\.test`"
        ):
            yaml_pipeline(text, {"a": Images(30)})

    def test_only_the_last_input_may_be_a_list(self) -> None:
        text = _cameras("      - {name: s, evaluator: lh, input: reference}").replace(
            "inputs: [reference, {name: cameras, list: true}]", "inputs: [{name: cameras, list: true}, reference]"
        )
        with pytest.raises(ValidationError, match="Only the last input may be a list"):
            yaml_pipeline(text, _DATA)

    def test_a_task_whose_sources_do_not_fit_the_inputs_is_refused_when_the_config_loads(self) -> None:
        text = _cameras("      - {name: s, evaluator: lh, input: reference}").replace(
            "sources: [ref, cam1, cam2, cam3]", "sources: [ref]"
        )
        with pytest.raises(ValidationError, match="takes one source for 'reference' and at least one for 'cameras'"):
            yaml_pipeline(text, _DATA)


class TestPairs:
    def test_pairs_runs_a_step_once_per_unordered_pair_of_a_list_in_list_order(self) -> None:
        result = _run(_cameras("      - {name: pairwise, evaluator: dupes, input: cameras, pairs: true}"))
        assert list(result.steps["pairwise"].elements) == ["cam1_vs_cam2", "cam1_vs_cam3", "cam2_vs_cam3"]

    def test_a_list_of_one_element_gives_no_pair_and_one_record_saying_so(self) -> None:
        text = _cameras("      - {name: pairwise, evaluator: dupes, input: cameras, pairs: true}").replace(
            "sources: [ref, cam1, cam2, cam3]", "sources: [ref, cam1]"
        )
        result = _run(text)
        assert result.steps["pairwise"].status == "skipped"
        assert "holds one element, so it has no pair" in result.steps["pairwise"].reason


class TestEmptyLists:
    _TEXT = """
evaluators:
  - {name: lh, type: label-health}
workflows:
  - name: w
    inputs: [train, {name: evals, list: true%s}]
    steps:
      - {name: counts, evaluator: lh, input: evals}
      - {name: imbalance, check: class-imbalance, input: counts}
tasks:
  - {name: t, workflow: w, sources: [train]}
"""

    def test_a_list_input_with_empty_may_bind_no_source(self) -> None:
        result = _run(self._TEXT % ", empty: no evaluation split given", {"train": Images()})
        assert result.steps["counts"].status == "skipped"
        assert result.steps["counts"].reason == "no evaluation split given"

    def test_a_check_over_an_empty_list_reports_the_reason_as_not_assessed(self) -> None:
        result = _run(self._TEXT % ", empty: no evaluation split given", {"train": Images()})
        assert result.steps["imbalance"].status == "ok"
        (finding,) = result.findings
        assert (finding.severity, finding.brief) == ("info", "not assessed")
        assert "no evaluation split given" in (finding.description or "")
        assert result.health["status"] == "ok"

    def test_a_list_input_without_empty_needs_at_least_one_source(self) -> None:
        with pytest.raises(ValidationError, match="at least one for 'evals'"):
            yaml_pipeline(self._TEXT % "", {"train": Images()})

    def test_empty_on_an_input_that_is_not_a_list_is_refused(self) -> None:
        text = self._TEXT.replace("inputs: [train,", "inputs: [{name: train, empty: nope},")
        with pytest.raises(ValidationError, match="binds one source, so it takes no `empty:`"):
            yaml_pipeline(text % "", {"train": Images()})
