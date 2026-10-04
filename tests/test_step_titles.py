"""Every step type's friendly title, and the banner and step headings that carry it."""

from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from dataeval_flow import run, run_task, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._report import step_heading
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators import Evaluator
from dataeval_flow.evaluators._registry import EVALUATORS
from dataeval_flow.steps import ChainResult, list_steps
from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS
from dataeval_flow.steps._result import StepResult
from dataeval_flow.workflows._registry import WORKFLOWS
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages
from tests.workflow_toys import register_count

_TITLES = {
    "balance": "Balance",
    "diversity": "Diversity",
    "parity": "Parity",
    "duplicates": "Duplicates",
    "label-health": "Label Health",
    "outliers": "Outliers",
    "factor-leakage": "Factor Leakage",
    "factor-triage": "Factor Triage",
    "content-digest": "Content Digest",
    "metadata-summary": "Metadata Summary",
    "coverage": "Coverage",
    "label-alignment": "Label Alignment",
    "completeness": "Completeness",
    "label-reconciliation": "Label Reconciliation",
    "ontology-validation": "Ontology Validation",
    "prioritize": "Prioritization",
    "representation": "Representation",
    "drift-domain-classifier": "Drift (Domain Classifier)",
    "drift-kneighbors": "Drift (K-Neighbors)",
    "drift-mmd": "Drift (MMD)",
    "drift-univariate": "Drift (Univariate)",
    "drift-wasserstein": "Drift (Wasserstein)",
    "ood-domain-classifier": "OOD (Domain Classifier)",
    "ood-kneighbors": "OOD (K-Neighbors)",
    "divergence": "Divergence",
    "conform": "Conform",
    "export": "Export",
    "kfold": "K-Fold Split",
    "merge": "Merge",
    "remove": "Remove",
    "select": "Select",
    "split": "Split",
    "view": "View",
    "wrap": "Wrap",
    "classwise-outliers": "Outliers by Class",
    "ood-union": "OOD Agreement",
    "factor-deviation": "OOD Sample Metadata Deviations",
    "factor-gaps": "Factor Gaps",
    "factor-predictors": "OOD Factor Predictors",
    "data-analysis": "Data Analysis",
    "data-cleaning": "Data Cleaning",
    "data-coverage": "Data Coverage",
    "data-prioritization": "Data Prioritization",
    "data-splitting": "Data Splitting",
    "drift-monitoring": "Drift Monitoring",
    "label-space": "Label Space",
    "metadata-triage": "Metadata Triage",
    "ood-detection": "OOD Detection",
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def test_every_built_in_step_declares_its_title() -> None:
    registries = (EVALUATORS, TRANSFORMS, COMBINES, WORKFLOWS)
    found = {cls.name: cls.title for registry in registries for cls in registry.list(plugins=False)}
    assert found == _TITLES


def test_a_check_keeps_the_title_of_its_finding() -> None:
    titles = {cls.name: cls.title for cls in CHECKS.list(plugins=False)}
    assert titles["duplicate-rate"] == "Duplicates"
    assert titles["classwise-outlier-rate"] == "Classwise Outliers"


def test_a_plugin_class_without_a_title_takes_its_id() -> None:
    class Bare(Evaluator):  # type: ignore[type-arg]
        name: ClassVar[str] = "bare"
        description: ClassVar[str] = "Has no title."

    assert Bare.title == "bare"


def test_the_catalog_carries_each_title() -> None:
    catalog = {(entry.kind, entry.type): entry.title for entry in list_steps(plugins=False).steps}
    assert catalog[("evaluator", "label-health")] == "Label Health"
    assert catalog[("transform", "kfold")] == "K-Fold Split"
    assert catalog[("combine", "classwise-outliers")] == "Outliers by Class"
    assert catalog[("check", "classwise-outlier-rate")] == "Classwise Outliers"
    assert catalog[("workflow", "data-cleaning")] == "Data Cleaning"


_CLEAN = {"name": "clean", "type": "data-cleaning", "outlier_method": "zscore", "outlier_flags": ["pixel"]}


def _banner(report: str) -> list[str]:
    lines = report.splitlines()
    rules = [i for i, line in enumerate(lines) if line.startswith("=") or set(line) == {"="}]
    return [line.strip() for line in lines[rules[0] + 1 : rules[1]]]


def _first_line(report: str) -> str:
    """The envelope's first line: the first content line under the banner's closing rule."""
    lines = report.splitlines()
    rules = [i for i, line in enumerate(lines) if set(line) == {"="}]
    return " ".join(lines[rules[1] + 1].split())


def _run(config: Any, task: str = "t") -> Any:
    result = run_tasks(config)[task]
    assert result.success, result.errors
    return result


def test_a_preset_task_is_headed_by_its_title_and_named_in_the_envelope() -> None:
    config = chain_pipeline(
        workflows=[_CLEAN],
        tasks=[{"name": "t", "workflow": "clean", "sources": ["src"]}],
        datasets={"src": ToyImages()},
    )
    report = _run(config).report()
    assert _banner(report) == ["DATA CLEANING"]
    assert _first_line(report) == "Workflow: clean (data-cleaning)"


def test_an_evaluator_task_is_headed_by_its_title_and_named_in_the_envelope() -> None:
    config = chain_pipeline(
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "dupes", "sources": ["src"], "kind": "evaluator"}],
    )
    result = _run(config)
    assert _banner(result.report()) == ["DUPLICATES"]
    assert _first_line(result.report()) == "Evaluator: dupes (duplicates)"
    assert "<title>Duplicates — dupes</title>" in result.to_html()


def test_a_custom_workflow_is_headed_by_its_name() -> None:
    workflow = {
        "name": "mine",
        "inputs": ["a"],
        "steps": [{"name": "dupes", "evaluator": "dupes", "input": "a"}],
    }
    config = chain_pipeline(
        workflows=[workflow],
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "mine", "sources": ["src"]}],
    )
    result = _run(config)
    assert isinstance(result, ChainResult)
    assert _banner(result.report()) == ["MINE"]
    assert _first_line(result.report()) == "Workflow: mine (custom workflow)"
    assert "<title>mine</title>" in result.to_html()


def test_a_legacy_workflow_keeps_its_summary_in_the_body(plugins: dict[str, list[tuple[str, str]]]) -> None:
    register_count(plugins)
    config = chain_pipeline(
        workflows=[{"name": "tc", "type": "test.count"}],
        tasks=[{"name": "t", "workflow": "tc", "sources": ["src"]}],
    )
    result = _run(config)
    lines = result.report().splitlines()
    assert _banner(result.report()) == ["TEST.COUNT"]
    assert _first_line(result.report()) == "Workflow: tc (test.count)"
    assert lines[lines.index("  Source:    src (src_data)") + 2] == "  Items counted."


def test_a_failed_result_is_headed_the_same_way() -> None:
    config = chain_pipeline(
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "dupes", "sources": ["src"], "kind": "evaluator"}],
    )
    result = _run(config)
    failed = type(result).failed(type="duplicates", errors=["boom"])
    assert _banner(failed.report()) == ["DUPLICATES"]
    assert _first_line(failed.report()) == "Evaluator: duplicates"


def test_the_html_header_holds_the_title_and_the_provenance_opens_with_what_ran() -> None:
    config = chain_pipeline(
        workflows=[_CLEAN],
        tasks=[{"name": "t", "workflow": "clean", "sources": ["src"]}],
        datasets={"src": ToyImages()},
    )
    html = _run(config).to_html()
    assert "<h1>Data Cleaning</h1>" in html
    assert 'class="facts"' not in html
    assert '<dl class="fields provenance"><dt>Workflow</dt><dd>clean (data-cleaning)</dd>' in html
    assert "<title>Data Cleaning — clean</title>" in html


def _record(name: str, type_id: str, kind: str) -> StepResult:
    return StepResult(name=name, kind=kind, type=type_id, inputs=[], status="ok")  # type: ignore[arg-type]


def test_a_step_is_headed_by_its_type_title_and_named_where_it_differs() -> None:
    assert step_heading(_record("outliers", "outliers", "evaluator")) == "Outliers"
    assert step_heading(_record("dupes", "duplicates", "evaluator")) == "Duplicates · dupes"
    assert step_heading(_record("clean", "remove", "transform")) == "Remove · clean"
    assert step_heading(_record("clean", "data-cleaning", "workflow")) == "Data Cleaning · clean"
    assert step_heading(_record("x", "not-registered", "transform")) == "not-registered · x"


def test_run_of_a_config_with_the_default_entry_name_names_the_id_alone() -> None:
    result = run(DataCleaningConfig(outlier_method="zscore", outlier_flags=["pixel"]), ToyImages())
    assert _banner(result.report()) == ["DATA CLEANING"]
    assert _first_line(result.report()) == "Workflow: data-cleaning"
    assert "<title>Data Cleaning</title>" in result.to_html()


def test_a_preset_that_fails_in_a_run_keeps_its_entry_in_the_envelope() -> None:
    config = chain_pipeline(
        workflows=[_CLEAN],
        tasks=[{"name": "t", "workflow": "clean", "sources": ["src"]}],
        datasets={"src": ToyImages()},
    )
    with patch.object(TRANSFORMS.get("remove"), "run", side_effect=RuntimeError("boom")):
        result = run_tasks(config)["t"]
    assert not result.success
    assert _banner(result.report()) == ["DATA CLEANING"]
    assert _first_line(result.report()) == "Workflow: clean (data-cleaning)"


def test_a_preset_task_refused_before_it_runs_keeps_its_entry_in_the_envelope() -> None:
    config = chain_pipeline(workflows=[_CLEAN], datasets={"a": ToyImages(), "b": ToyImages()})
    result = run_task(TaskConfig(name="t", workflow="clean", sources=["a", "b"]), config)
    assert not result.success
    assert _banner(result.report()) == ["DATA CLEANING"]
    assert _first_line(result.report()) == "Workflow: clean (data-cleaning)"


def test_a_custom_workflow_named_like_a_preset_is_still_a_custom_workflow() -> None:
    workflow = {"name": "data-cleaning", "inputs": ["a"], "steps": [{"name": "d", "evaluator": "dupes", "input": "a"}]}
    config = chain_pipeline(
        workflows=[workflow],
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "data-cleaning", "sources": ["src"]}],
    )
    report = _run(config).report()
    assert _banner(report) == ["DATA-CLEANING"]
    assert _first_line(report) == "Workflow: data-cleaning (custom workflow)"
