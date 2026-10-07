"""A matrix's report: a row per run, a column per finding, failed runs' errors, and each run's report at full detail
(task-matrix spec §7.1, §7.2)."""

from typing import Any

import pytest

from dataeval_flow import MatrixResult, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._ci_reports import markdown_summary
from dataeval_flow._matrix._result import MatrixRun
from dataeval_flow._matrix._table import comparison_table
from dataeval_flow.steps import Finding
from tests.chain_toys import chain_pipeline
from tests.workflow_toys import register_count

_CLEANING = {
    "name": "cleaning",
    "type": "quality",
    "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"},
}


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
    result = _run([{"outliers.outlier_threshold": [1.0, 3.0]}, {"duplicates.merge_near_duplicates": [False]}])
    table = comparison_table(result)
    headers = [column.header for column in table.columns]
    assert headers[:4] == ["#", "outliers.outlier_threshold", "duplicates.merge_near_duplicates", "Health"]
    assert "Image Outliers" in headers or any("Outliers" in header for header in headers[4:])
    assert [row["number"] for row in table.rows] == [1, 2, 3]
    key = table.columns[2].key
    assert [row[key] for row in table.rows] == ["", "", "false"]  # a key its grid does not set is blank
    finding_keys = [column.key for column in table.columns[4:]]
    assert all(str(table.rows[0][key]).startswith(("[!!]", "[..]", "[ok]", "—")) for key in finding_keys)


def test_a_one_value_matrix_gives_a_one_row_table() -> None:
    result = _run({"outliers.outlier_threshold": [3.0]})
    assert len(comparison_table(result).rows) == 1


def test_a_short_report_is_the_table_and_a_full_one_adds_each_run_under_runs() -> None:
    result = _run({"outliers.outlier_threshold": [1.0, 3.0]})
    short, full = result.report(detailed=False), result.report(detailed=True)
    assert "matrix of 2 runs" in short
    assert "Run 1 · outliers.outlier_threshold=1.0" not in short
    # one upper-cased section holds the runs, so each run's heading keeps its values' case
    assert "\n  RUNS\n" in full
    assert "Run 1 · outliers.outlier_threshold=1.0" in full
    assert "Run 2 · outliers.outlier_threshold=3.0" in full


def test_the_health_column_fits_its_header_at_80_columns() -> None:
    result = _run({"outliers.outlier_threshold": [1.0, 3.0]})
    assert [row["health"] for row in comparison_table(result).rows] == ["[!!]", "[!!]"]
    text = result.report(detailed=False)
    # the table is wider than 80 columns, and narrows its columns to their headers, Health's to `Health`
    # check that no row breaks the word warning across two lines
    assert "warning"[:-1] not in text[text.index("#  outliers.outlier_threshold") :]


def test_the_html_page_draws_the_table() -> None:
    page = _run({"outliers.outlier_threshold": [1.0, 3.0]}).to_html(detailed=True)
    assert "outliers.outlier_threshold" in page
    assert "Run 2" in page


def _fake(number: int, findings: list[Finding], *, success: bool = True) -> MatrixRun:
    from dataeval_flow.steps import ChainResult
    from tests.workflow_toys import count_result

    result = count_result(*findings) if success else ChainResult.failed(type="test.count", errors=["it broke"])
    return MatrixRun(number=number, label=f"k={number}", values={"k": number}, result=result)


def test_repeated_titles_are_numbered_and_a_failed_run_s_error_is_listed() -> None:
    twice = [Finding(title="Outliers", severity="warning", brief="9"), Finding(title="Outliers", severity="info")]
    result = MatrixResult(type="toy", keys=["k"], runs=[_fake(1, twice), _fake(2, [], success=False)])
    table = comparison_table(result)
    assert [column.header for column in table.columns] == ["#", "k", "Health", "Outliers", "Outliers (2)"]
    assert [table.rows[0][column.key] for column in table.columns[3:]] == ["[!!] 9", "[..]"]
    assert [table.rows[1][column.key] for column in table.columns[3:]] == ["—", "—"]
    assert [row["health"] for row in table.rows] == ["[!!]", "failed"]
    short = result.report(detailed=False)
    assert "Run 2 failed: it broke" in short
    assert "Health: failed [!!] — run 2 of 2 failed" in short


def test_the_health_line_counts_one_run_and_one_warning_in_the_singular() -> None:
    warned = MatrixResult(type="toy", keys=["k"], runs=[_fake(1, [Finding(title="O", severity="warning")])])
    assert "Health: 1 warning [!!] across 1 run — review flagged findings" in warned.report(detailed=False)
    clean = MatrixResult(type="toy", keys=["k"], runs=[_fake(1, [Finding(title="O", severity="ok")])])
    assert [row["health"] for row in comparison_table(clean).rows] == ["[ok]"]
    assert "Health: ok — 1 run, no warnings" in clean.report(detailed=False)


def test_a_workflow_that_is_not_a_chain_fills_the_table_with_its_findings(
    plugins: dict[str, list[tuple[str, str]]],
) -> None:
    # test.count runs its own code, not a chain: its findings come from the result, not from steps.
    register_count(plugins)
    config = chain_pipeline(
        workflows=[{"name": "count", "type": "test.count"}],
        tasks=[{"name": "t", "workflow": "count", "sources": "src", "matrix": {"minimum": [0, 100]}}],
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, MatrixResult)
    assert result.success, result.errors
    table = comparison_table(result)
    assert [column.header for column in table.columns] == ["#", "minimum", "Health", "Items"]
    assert [row["health"] for row in table.rows] == ["[ok]", "[!!]"]


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
    # an evaluator judges nothing, so its runs ran rather than passed
    assert "Health: ran — 2 runs; an evaluator has no findings to list" in result.report(detailed=False)
    assert "**Health:** ran" in markdown_summary({"t": result})
