"""A matrix's report: a row per run, a column per finding, failed runs' errors, and each run's report at full detail
(task-matrix spec §7.1, §7.2)."""

from typing import Any

import pytest

from dataeval_flow import MatrixResult, ResultMetadata, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._matrix._result import MatrixRun
from dataeval_flow._matrix._table import comparison_table
from dataeval_flow.workflows import Finding
from tests.chain_toys import chain_pipeline

_CLEANING = {"name": "cleaning", "type": "data-cleaning", "outlier_method": "zscore", "outlier_flags": ["pixel"]}


@pytest.fixture(autouse=True)
def _fresh_cache() -> Any:
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(matrix: Any) -> MatrixResult:
    config = chain_pipeline(
        workflows=[_CLEANING],
        tasks=[{"name": "t", "workflow": "cleaning", "sources": "src", "matrix": matrix}],
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, MatrixResult)
    return result


def test_the_table_has_a_row_per_run_and_a_column_per_finding() -> None:
    result = _run([{"outlier_threshold": [1.0, 3.0]}, {"outlier_method": ["iqr"]}])
    table = comparison_table(result)
    headers = [column.header for column in table.columns]
    assert headers[:4] == ["#", "outlier_threshold", "outlier_method", "Health"]
    assert "Image Outliers" in headers or any("Outliers" in header for header in headers[4:])
    assert [row["number"] for row in table.rows] == [1, 2, 3]
    key = table.columns[2].key
    assert [row[key] for row in table.rows] == ["", "", "iqr"]  # a key its grid does not set is blank
    finding_keys = [column.key for column in table.columns[4:]]
    assert all(str(table.rows[0][key]).startswith(("[!!]", "[..]", "[ok]", "—")) for key in finding_keys)


def test_a_one_value_matrix_gives_a_one_row_table() -> None:
    result = _run({"outlier_threshold": [3.0]})
    assert len(comparison_table(result).rows) == 1


def test_a_short_report_is_the_table_and_a_full_one_adds_each_run() -> None:
    result = _run({"outlier_threshold": [1.0, 3.0]})
    short, full = result.report(detailed=False), result.report(detailed=True)
    assert "matrix of 2 runs" in short
    assert "Run 1 · outlier_threshold=1.0" not in short
    # the text report upper-cases a section title
    assert "RUN 1 · OUTLIER_THRESHOLD=1.0" in full
    assert "RUN 2 · OUTLIER_THRESHOLD=3.0" in full


def test_the_html_page_draws_the_table() -> None:
    page = _run({"outlier_threshold": [1.0, 3.0]}).to_html(detailed=True)
    assert "outlier_threshold" in page
    assert "Run 2" in page


def _fake(number: int, findings: list[Finding], *, success: bool = True) -> MatrixRun:
    from dataeval_flow.workflows._base import WorkflowOutput, WorkflowRawOutput, WorkflowReport
    from dataeval_flow.workflows._result import WorkflowResult

    class _Result(WorkflowResult[ResultMetadata, WorkflowOutput]):  # type: ignore[type-arg]
        pass

    output = WorkflowOutput(
        raw=WorkflowRawOutput(dataset_size=1), report=WorkflowReport(summary="s", findings=findings)
    )
    result = (
        _Result(type="toy", success=True, metadata=ResultMetadata(), output=output)
        if success
        else _Result.failed(type="toy", errors=["it broke"])
    )
    return MatrixRun(number=number, label=f"k={number}", values={"k": number}, result=result)


def test_repeated_titles_are_numbered_and_a_failed_run_s_error_is_listed() -> None:
    twice = [Finding(title="Outliers", severity="warning", brief="9"), Finding(title="Outliers", severity="info")]
    result = MatrixResult(type="toy", keys=["k"], runs=[_fake(1, twice), _fake(2, [], success=False)])
    table = comparison_table(result)
    assert [column.header for column in table.columns] == ["#", "k", "Health", "Outliers", "Outliers (2)"]
    assert [table.rows[0][column.key] for column in table.columns[3:]] == ["[!!] 9", "[..]"]
    assert [table.rows[1][column.key] for column in table.columns[3:]] == ["—", "—"]
    assert "Run 2 failed: it broke" in result.report(detailed=False)


def test_a_legacy_workflow_s_findings_fill_the_table() -> None:
    from tests.evaluator_toys import ToyFactors

    # data-splitting still runs its own code, not a chain; 40 items leave every class in each split.
    config = chain_pipeline(
        workflows=[{"name": "split", "type": "data-splitting"}],
        tasks=[{"name": "t", "workflow": "split", "sources": "src", "matrix": {"test_frac": [0.25, 0.5]}}],
        datasets={"src": ToyFactors(count=40)},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, MatrixResult)
    assert result.success, result.errors
    assert len(comparison_table(result).columns) > 3  # #, test_frac and Health, then its findings


def test_an_evaluator_matrix_has_no_finding_columns() -> None:
    config = chain_pipeline(
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[
            {"name": "t", "evaluator": "dupes", "sources": "src", "matrix": {"merge_near_duplicates": [True, False]}}
        ],
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, MatrixResult)
    assert [column.header for column in comparison_table(result).columns] == ["#", "merge_near_duplicates", "Health"]
