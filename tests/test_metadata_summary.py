"""`factor-summary`: each factor's type, binning, nulls and range or top values, as legacy data-coverage's Metadata
Distribution (coverage spec §6.1)."""

from typing import Any
from unittest.mock import MagicMock

import polars as pl

from dataeval_flow import run
from dataeval_flow._blocks import Paragraph, Table
from dataeval_flow._metadata import build_metadata
from dataeval_flow.evaluators.bias import FactorSummaryConfig, FactorSummaryOutput
from dataeval_flow.workflows._common import compute_metadata_summary
from tests.evaluator_toys import ToyFactors, ToyImages, output_json


def test_it_summarizes_every_factor_as_legacy_did() -> None:
    result = run(FactorSummaryConfig(), ToyFactors(count=30))
    assert result.success, result.errors
    assert isinstance(result.output, FactorSummaryOutput)
    data = result.output.data()
    assert sorted(data["factors"]) == ["angle", "site"]
    assert data["summary"] == compute_metadata_summary(build_metadata(ToyFactors(count=30), None))


def test_its_section_is_the_factor_table() -> None:
    result = run(FactorSummaryConfig(), ToyFactors(count=30))
    blocks = result._section(output_json(result), ["src"], detailed=False)
    assert blocks is not None
    (table,) = [block for block in blocks if isinstance(block, Table)]
    assert [column.header for column in table.columns] == ["Factor", "Type", "Unique", "Nulls"]
    assert {str(row["factor"]) for row in table.rows} == {"angle", "site"}


def test_with_no_factors_its_section_says_so() -> None:
    result = run(FactorSummaryConfig(), ToyImages(count=10))
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


class TestComputeMetadataSummary:
    """Each factor is summarized over the rows at its own level."""

    @staticmethod
    def _metadata(df: pl.DataFrame, factor_info: dict[str, Any]) -> MagicMock:
        metadata = MagicMock()
        metadata.factor_info = factor_info
        metadata.dropped_factors = {}
        metadata.rows_at.return_value = df
        return metadata

    @staticmethod
    def _info(factor_type: str, level: str = "unit", is_binned: bool = False) -> MagicMock:
        info = MagicMock()
        info.factor_type = factor_type
        info.level = level
        info.is_binned = is_binned
        return info

    def test_continuous_factor(self):
        metadata = self._metadata(pl.DataFrame({"width": [100.0, 200.0, 300.0]}), {"width": self._info("continuous")})

        result = compute_metadata_summary(metadata)
        assert result["width"]["type"] == "continuous"
        assert result["width"]["min"] == 100.0
        assert result["width"]["max"] == 300.0

    def test_categorical_factor(self):
        metadata = self._metadata(pl.DataFrame({"color": ["red", "blue", "red"]}), {"color": self._info("categorical")})

        result = compute_metadata_summary(metadata)
        assert result["color"]["type"] == "categorical"
        assert result["color"]["unique_values"] == 2
        assert "top_values" in result["color"]

    def test_factor_not_in_columns(self):
        metadata = self._metadata(pl.DataFrame({"other": [1]}), {"missing_col": self._info("continuous")})

        result = compute_metadata_summary(metadata)
        assert result["missing_col"] == {"type": "continuous", "level": "unit", "is_binned": False}

    def test_categorical_empty_column(self):
        """Categorical factor with empty column — empty value_counts."""
        metadata = self._metadata(
            pl.DataFrame({"color": pl.Series([], dtype=pl.Utf8)}), {"color": self._info("categorical")}
        )

        result = compute_metadata_summary(metadata)
        assert result["color"]["unique_values"] == 0
        assert "top_values" not in result["color"]

    def test_reports_level_and_binning(self):
        """`is_binned` is the only record that a factor reached the evaluators as codes."""
        metadata = self._metadata(
            pl.DataFrame({"area": [1.0, 2.0, 3.0]}),
            {"area": self._info("continuous", level="instance", is_binned=True)},
        )

        result = compute_metadata_summary(metadata)
        assert result["area"]["level"] == "instance"
        assert result["area"]["is_binned"] is True

    def test_reads_each_factor_at_its_own_level(self):
        """A unit factor must not be summarized over the replicated instance rows."""
        frames = {
            "unit": pl.DataFrame({"brightness": [10.0, 20.0]}),
            "instance": pl.DataFrame({"area": [1.0, 1.0, 2.0, 2.0]}),
        }
        metadata = MagicMock()
        metadata.dropped_factors = {}
        metadata.factor_info = {
            "brightness": self._info("continuous", level="unit"),
            "area": self._info("continuous", level="instance"),
        }
        metadata.rows_at.side_effect = lambda level: frames[level]

        result = compute_metadata_summary(metadata)

        # Mean over the 2 image rows, not over the 4 replicated detection rows.
        assert result["brightness"]["mean"] == 15.0
        assert result["area"]["mean"] == 1.5

    def test_reports_dropped_factors(self):
        """Vector-valued stats never became factors; absence alone reads as 'not measured'."""
        metadata = self._metadata(pl.DataFrame({"width": [1.0]}), {"width": self._info("continuous")})
        metadata.dropped_factors = {"histogram": ["multi-dimensional"]}

        result = compute_metadata_summary(metadata)
        assert result["histogram"]["type"] == "dropped"
        assert result["histogram"]["dropped_reasons"] == ["multi-dimensional"]
