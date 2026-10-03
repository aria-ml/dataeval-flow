"""`metadata-summary`: each factor's type, binning, nulls and range or top values, as legacy data-coverage's Metadata
Distribution (coverage spec §6.1)."""

from dataeval_flow import run
from dataeval_flow._blocks import Paragraph, Table
from dataeval_flow._metadata import build_metadata
from dataeval_flow.evaluators.bias import MetadataSummaryConfig, MetadataSummaryOutput
from dataeval_flow.workflows._common import compute_metadata_summary
from tests.evaluator_toys import ToyFactors, ToyImages, output_json


def test_it_summarizes_every_factor_as_legacy_did() -> None:
    result = run(MetadataSummaryConfig(), ToyFactors(count=30))
    assert result.success, result.errors
    assert isinstance(result.output, MetadataSummaryOutput)
    data = result.output.data()
    assert sorted(data["factors"]) == ["angle", "site"]
    assert data["summary"] == compute_metadata_summary(build_metadata(ToyFactors(count=30), None))


def test_its_section_is_the_factor_table() -> None:
    result = run(MetadataSummaryConfig(), ToyFactors(count=30))
    blocks = result._section(output_json(result), ["src"], detailed=False)
    assert blocks is not None
    (table,) = [block for block in blocks if isinstance(block, Table)]
    assert [column.header for column in table.columns] == ["Factor", "Type", "Unique", "Nulls"]
    assert {str(row["factor"]) for row in table.rows} == {"angle", "site"}


def test_with_no_factors_its_section_says_so() -> None:
    result = run(MetadataSummaryConfig(), ToyImages(count=10))
    assert result.success, result.errors
    blocks = result._section(output_json(result), ["src"], detailed=False)
    assert blocks is not None
    assert [block.text for block in blocks if isinstance(block, Paragraph)] == ["No metadata factors available"]


def test_its_rows_read_each_factor_as_legacy_did() -> None:
    from dataeval_flow.evaluators.bias._report import metadata_summary_section

    summary = {
        "site": {"type": "categorical", "unique_values": 3, "null_count": 2},
        "angle": {"type": "continuous", "mean": 12.3456, "null_count": 0},
        "box": {"type": "continuous", "mean": None},
        "bare": {},
    }
    (table,) = metadata_summary_section({"data": {"factors": list(summary), "summary": summary}})
    assert isinstance(table, Table)
    assert [(row["factor"], row["type"], row["unique"], row["nulls"]) for row in table.rows] == [
        ("site", "categorical", 3, 2),
        ("angle", "continuous", "μ=12.35", 0),
        ("box", "continuous", "-", 0),
        ("bare", "unknown", "-", 0),
    ]
