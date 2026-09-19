"""Tests for _text_report rendering helpers."""

from __future__ import annotations

import pytest

from dataeval_flow.workflow._text_report import (
    _BAR_MAX,
    _MAX_ENUMERATED,
    _SHAPE_CELLS,
    _WIDTH,
    _brief_value,
    _compact_indices,
    _factor_table,
    _flow_repr,
    _format_value,
    _render_binning_record,
    _render_binning_section,
    _render_chunk_table,
    _render_classwise_table,
    _render_config_section,
    _render_detail_section,
    _render_distribution,
    _render_factor_line,
    _render_key_value,
    _render_pivot_table,
    _render_ratio,
    _render_review_state,
    _render_split_comparability,
    _render_table,
    _section_header,
    _shape_cells,
    _summary_line,
)
from dataeval_flow.workflow.base import Reportable

pytestmark = pytest.mark.required

# ---------------------------------------------------------------------------
# _brief_value
# ---------------------------------------------------------------------------


class TestBriefValue:
    def test_returns_brief_from_dict(self):
        finding = Reportable(report_type="text", title="T", data={"brief": "42%"})
        assert _brief_value(finding) == "42%"

    def test_returns_empty_when_no_brief(self):
        finding = Reportable(report_type="text", title="T", data={"other": 1})
        assert _brief_value(finding) == ""

    def test_returns_empty_for_non_dict(self):
        finding = Reportable(report_type="text", title="T", data="plain text")
        assert _brief_value(finding) == ""

    def test_returns_empty_for_brief_none(self):
        finding = Reportable(report_type="text", title="T", data={"brief": None})
        assert _brief_value(finding) == ""


# ---------------------------------------------------------------------------
# _summary_line
# ---------------------------------------------------------------------------


class TestSummaryLine:
    def test_basic_line(self):
        finding = Reportable(report_type="text", title="Duplicates", data={"brief": "3"})
        line = _summary_line(finding)
        assert "Duplicates" in line
        assert "3" in line

    def test_warning_marker(self):
        finding = Reportable(report_type="text", severity="warning", title="Issue", data={"brief": "5"})
        line = _summary_line(finding)
        assert "[!!]" in line

    def test_no_marker_for_ok(self):
        finding = Reportable(report_type="text", severity="ok", title="Good", data={"brief": "0"})
        line = _summary_line(finding)
        assert "[!!]" not in line


# ---------------------------------------------------------------------------
# _section_header
# ---------------------------------------------------------------------------


class TestSectionHeader:
    def test_basic_header(self):
        lines = _section_header("TITLE")
        assert any("TITLE" in ln for ln in lines)
        assert any("=" * _WIDTH in ln for ln in lines)

    def test_header_with_right_text(self):
        lines = _section_header("TITLE", "42%")
        header_line = [ln for ln in lines if "TITLE" in ln and "42%" in ln]
        assert len(header_line) == 1


# ---------------------------------------------------------------------------
# _render_detail_section
# ---------------------------------------------------------------------------


class TestRenderDetailSection:
    def test_text_data_string(self):
        finding = Reportable(report_type="text", title="Note", data="hello world")
        lines = _render_detail_section(finding)
        assert any("hello world" in li for li in lines)

    def test_non_dict_empty_string(self):
        finding = Reportable(report_type="text", title="Note", data="")
        lines = _render_detail_section(finding)
        # Should not crash, just header
        assert any("NOTE" in li for li in lines)

    def test_with_description(self):
        finding = Reportable(report_type="text", title="T", data={}, description="A description")
        lines = _render_detail_section(finding)
        assert any("A description" in li for li in lines)

    def test_unknown_report_type_passthrough(self):
        finding = Reportable(report_type="image", title="Img", data={"some": "data"})
        lines = _render_detail_section(finding)
        # Should not crash — just returns header
        assert any("IMG" in li for li in lines)

    def test_pivot_table_dispatch(self):
        finding = Reportable(
            report_type="pivot_table",
            title="Classes",
            data={
                "table_headers": ["Class", "Count", "%"],
                "table_data": [
                    {"Class": "cat", "Count": 10, "pct": 50.0},
                    {"Class": "dog", "Count": 10, "pct": 50.0},
                ],
            },
        )
        lines = _render_detail_section(finding)
        text = "\n".join(lines)
        assert "cat" in text
        assert "dog" in text

    def test_table_dispatch(self):
        finding = Reportable(
            report_type="table",
            title="Dist",
            data={"table_data": {"cat": 10, "dog": 5}},
        )
        lines = _render_detail_section(finding)
        text = "\n".join(lines)
        assert "cat" in text

    def test_key_value_dispatch(self):
        finding = Reportable(
            report_type="key_value",
            title="Outliers",
            data={"per_metric": {"brightness": 5, "contrast": 3}},
        )
        lines = _render_detail_section(finding)
        text = "\n".join(lines)
        assert "brightness" in text

    def test_classwise_table_dispatch(self):
        finding = Reportable(
            report_type="classwise_table",
            title="Classwise Drift",
            data={
                "table_rows": [
                    {"Class": "cat", "Distance": 0.3, "PVal": None, "Status": "DRIFT"},
                    {"Class": "dog", "Distance": 0.1, "PVal": None, "Status": "ok"},
                ],
            },
            description="Per-class drift results",
        )
        lines = _render_detail_section(finding)
        text = "\n".join(lines)
        assert "cat" in text
        assert "dog" in text
        assert "Per-class drift results" in text


# ---------------------------------------------------------------------------
# _render_pivot_table
# ---------------------------------------------------------------------------


class TestRenderPivotTable:
    def test_basic_pivot(self):
        data = {
            "table_headers": ["Class", "Count", "%"],
            "table_data": [
                {"Class": "cat", "Count": 10, "pct": 66.7},
                {"Class": "dog", "Count": 5, "pct": 33.3},
            ],
        }
        lines = _render_pivot_table(data)
        text = "\n".join(lines)
        assert "Class" in text
        assert "Count" in text
        assert "cat" in text
        assert "66.7%" in text

    def test_empty_rows(self):
        data = {"table_headers": ["A"], "table_data": []}
        assert _render_pivot_table(data) == []

    def test_empty_headers(self):
        data = {"table_headers": [], "table_data": [{"a": 1}]}
        assert _render_pivot_table(data) == []

    def test_none_value_formatted_as_empty(self):
        data = {
            "table_headers": ["Name", "Val"],
            "table_data": [{"Name": "x", "Val": None}],
        }
        lines = _render_pivot_table(data)
        assert len(lines) > 0


# ---------------------------------------------------------------------------
# _render_chunk_table
# ---------------------------------------------------------------------------


class TestRenderChunkTable:
    def _make_data(self, drifted_indices=None, lower_thresh=0.05, upper_thresh=0.35):
        """Build chunk_table data with 5 chunks."""
        drifted_indices = drifted_indices or set()
        rows = []
        flags = []
        for i in range(5):
            d = i in drifted_indices
            rows.append(
                {
                    "Chunk": f"[{i * 100}:{(i + 1) * 100}]",
                    "Distance": 0.45 if d else 0.15,
                    "UpperThreshold": upper_thresh,
                    "LowerThreshold": lower_thresh,
                    "Status": "DRIFT" if d else "ok",
                }
            )
            flags.append(d)
        return {"table_rows": rows, "drift_flags": flags}

    def test_table_columns_present(self):
        data = self._make_data({0})
        lines = _render_chunk_table(data)
        text = "\n".join(lines)
        assert "Distance" in text
        assert "Status" in text
        assert "DRIFT" in text
        assert "ok" in text

    def test_threshold_scale_present(self):
        data = self._make_data({0})
        lines = _render_chunk_table(data)
        text = "\n".join(lines)
        assert "Threshold" in text
        assert "|(0.3500)" in text

    def test_both_thresholds_shown(self):
        data = self._make_data({0}, lower_thresh=0.05, upper_thresh=0.35)
        lines = _render_chunk_table(data)
        text = "\n".join(lines)
        assert "(0.0500)|" in text
        assert "|(0.3500)" in text
        assert "---" in text  # dashes between thresholds

    def test_bar_characters(self):
        data = self._make_data({0})
        lines = _render_chunk_table(data)
        text = "\n".join(lines)
        assert "\u2588" in text  # filled block
        assert "\u2591" in text  # light shade (remainder)

    def test_empty_rows(self):
        assert _render_chunk_table({"table_rows": [], "drift_flags": []}) == []

    def test_threshold_scale_equal_positions(self):
        """When lower and upper thresholds are identical, scale shows single pipe."""
        data = self._make_data({0}, lower_thresh=0.20, upper_thresh=0.20)
        lines = _render_chunk_table(data)
        text = "\n".join(lines)
        assert "Threshold" in text
        # Both labels around a single pipe (lp == up branch)
        assert "(0.2000)|" in text

    def test_threshold_scale_upper_only(self):
        """When only upper threshold is present, scale shows dashes up to pipe."""
        rows = [
            {
                "Chunk": f"[{i * 100}:{(i + 1) * 100}]",
                "Distance": 0.40 if i == 0 else 0.10,
                "UpperThreshold": 0.35,
                "LowerThreshold": None,
                "Status": "DRIFT" if i == 0 else "ok",
            }
            for i in range(3)
        ]
        lines = _render_chunk_table({"table_rows": rows})
        text = "\n".join(lines)
        assert "Threshold" in text
        assert "|(0.3500)" in text

    def test_threshold_scale_lower_only(self):
        """When only lower threshold is present, scale shows label and pipe."""
        rows = [
            {
                "Chunk": f"[{i * 100}:{(i + 1) * 100}]",
                "Distance": 0.10,
                "UpperThreshold": None,
                "LowerThreshold": 0.05,
                "Status": "ok",
            }
            for i in range(3)
        ]
        lines = _render_chunk_table({"table_rows": rows})
        text = "\n".join(lines)
        assert "Threshold" in text
        assert "(0.0500)|" in text

    def test_no_thresholds_skips_scale_line(self):
        """When both thresholds are None for all rows, no threshold scale line is rendered."""
        rows = [
            {
                "Chunk": f"[{i * 100}:{(i + 1) * 100}]",
                "Distance": 0.15 + i * 0.01,
                "UpperThreshold": None,
                "LowerThreshold": None,
                "Status": "ok",
            }
            for i in range(4)
        ]
        lines = _render_chunk_table({"table_rows": rows})
        text = "\n".join(lines)
        assert "Distance" in text
        assert "Threshold" not in text

    def test_dispatch_from_detail_section(self):
        data = self._make_data({2, 3})
        finding = Reportable(
            report_type="chunk_table",
            title="MMD — Chunks",
            data=data,
            description="2/5 chunks drifted (40%) | max consecutive: 2",
        )
        lines = _render_detail_section(finding)
        text = "\n".join(lines)
        assert "Threshold" in text
        assert "2/5 chunks drifted" in text


# ---------------------------------------------------------------------------
# _render_table
# ---------------------------------------------------------------------------


class TestRenderTable:
    def test_basic_table_with_bars(self):
        data = {"table_data": {"cat": 100, "dog": 50}}
        lines = _render_table(data)
        text = "\n".join(lines)
        assert "cat" in text
        assert "dog" in text
        # Bar for 'cat' should be full
        assert "\u2588" in text

    def test_empty_table_data(self):
        assert _render_table({"table_data": {}}) == []

    def test_custom_headers(self):
        data = {"table_data": {"a": 1}, "table_headers": ("Label", "Qty")}
        lines = _render_table(data)
        text = "\n".join(lines)
        assert "Label" in text
        assert "Qty" in text

    def test_footer_lines(self):
        data = {"table_data": {"x": 10}, "footer_lines": ["Note: something"]}
        lines = _render_table(data)
        text = "\n".join(lines)
        assert "Note: something" in text

    def test_fractional_bars(self):
        """Items with counts that don't divide evenly should show fractional blocks."""
        data = {"table_data": {"a": 100, "b": 37}}
        lines = _render_table(data)
        # 'b' line should exist with some bar characters
        b_lines = [li for li in lines if "b" in li and "37" in li]
        assert len(b_lines) == 1


# ---------------------------------------------------------------------------
# _render_key_value
# ---------------------------------------------------------------------------


class TestRenderKeyValue:
    def test_per_metric_table(self):
        data = {
            "per_metric": {"brightness": 5, "contrast": 3},
            "total_flags": 8,
            "count": 6,
            "multi_metric_subject": "images",
        }
        lines = _render_key_value(data)
        text = "\n".join(lines)
        assert "brightness" in text
        assert "contrast" in text
        assert "Some images trigger multiple metrics" in text

    def test_per_metric_no_multi_trigger(self):
        data = {"per_metric": {"brightness": 5}, "total_flags": 5, "count": 5}
        lines = _render_key_value(data)
        text = "\n".join(lines)
        assert "brightness" in text
        assert "multiple metrics" not in text

    def test_detail_lines(self):
        data = {"detail_lines": ["Line one", "Line two"]}
        lines = _render_key_value(data)
        text = "\n".join(lines)
        assert "Line one" in text
        assert "Line two" in text

    def test_empty_data(self):
        assert _render_key_value({}) == []


# ---------------------------------------------------------------------------
# _render_classwise_table
# ---------------------------------------------------------------------------


class TestRenderClasswiseTable:
    def test_basic_classwise_table(self):
        data = {
            "table_rows": [
                {"Class": "cat", "Distance": 0.30, "PVal": None, "Status": "DRIFT"},
                {"Class": "dog", "Distance": 0.10, "PVal": None, "Status": "ok"},
            ],
        }
        lines = _render_classwise_table(data)
        text = "\n".join(lines)
        assert "Class" in text
        assert "Distance" in text
        assert "Status" in text
        assert "cat" in text
        assert "dog" in text
        assert "DRIFT" in text
        assert "ok" in text

    def test_empty_rows(self):
        assert _render_classwise_table({"table_rows": []}) == []

    def test_missing_rows_key(self):
        assert _render_classwise_table({}) == []

    def test_bar_characters_drift_vs_ok(self):
        data = {
            "table_rows": [
                {"Class": "a", "Distance": 0.5, "PVal": None, "Status": "DRIFT"},
                {"Class": "b", "Distance": 0.5, "PVal": None, "Status": "ok"},
            ],
        }
        lines = _render_classwise_table(data)
        # DRIFT row uses filled block, ok row uses light shade
        a_line = [ln for ln in lines if ln.strip().startswith("a")][0]
        b_line = [ln for ln in lines if ln.strip().startswith("b")][0]
        assert "\u2588" in a_line  # filled block for DRIFT
        assert "\u2591" in b_line  # light shade for ok

    def test_with_pval_column(self):
        data = {
            "table_rows": [
                {"Class": "cat", "Distance": 0.30, "PVal": 0.01, "Status": "DRIFT"},
                {"Class": "dog", "Distance": 0.10, "PVal": 0.85, "Status": "ok"},
            ],
        }
        lines = _render_classwise_table(data)
        text = "\n".join(lines)
        assert "PVal" in text
        assert "0.01" in text
        assert "0.85" in text

    def test_without_pval_column(self):
        data = {
            "table_rows": [
                {"Class": "cat", "Distance": 0.30, "PVal": None, "Status": "DRIFT"},
                {"Class": "dog", "Distance": 0.10, "PVal": None, "Status": "ok"},
            ],
        }
        lines = _render_classwise_table(data)
        text = "\n".join(lines)
        assert "PVal" not in text

    def test_mixed_pval_some_none(self):
        data = {
            "table_rows": [
                {"Class": "cat", "Distance": 0.30, "PVal": 0.02, "Status": "DRIFT"},
                {"Class": "dog", "Distance": 0.10, "PVal": None, "Status": "ok"},
            ],
        }
        lines = _render_classwise_table(data)
        text = "\n".join(lines)
        # PVal header should appear since at least one row has a value
        assert "PVal" in text
        assert "0.02" in text

    def test_negative_distance_uses_abs(self):
        data = {
            "table_rows": [
                {"Class": "neg", "Distance": -0.40, "PVal": None, "Status": "DRIFT"},
            ],
        }
        lines = _render_classwise_table(data)
        text = "\n".join(lines)
        # The displayed distance should be the raw value (-0.40), but bars use abs
        assert "-0.4000" in text
        assert "\u2588" in text

    def test_single_row(self):
        data = {
            "table_rows": [
                {"Class": "only", "Distance": 0.25, "PVal": None, "Status": "ok"},
            ],
        }
        lines = _render_classwise_table(data)
        text = "\n".join(lines)
        assert "only" in text
        assert "0.2500" in text


# ---------------------------------------------------------------------------
# _render_pivot_table — footer_lines
# ---------------------------------------------------------------------------


class TestRenderPivotTableMultiline:
    def test_multiline_cell(self):
        """Cells with newlines expand to multiple rows."""
        data = {
            "table_headers": ["Split", "Factors"],
            "table_data": [
                {"Split": "train", "Factors": "a (0.90)\nb (0.80)\nc (0.70)"},
            ],
        }
        lines = _render_pivot_table(data)
        text = "\n".join(lines)
        assert "a (0.90)" in text
        assert "b (0.80)" in text
        assert "c (0.70)" in text
        # "train" should appear on first sub-line only; continuation lines should be blank-padded
        train_lines = [ln for ln in lines if "train" in ln]
        assert len(train_lines) == 1

    def test_mixed_single_and_multiline(self):
        """Rows with different numbers of sub-lines render correctly."""
        data = {
            "table_headers": ["Split", "High MI", "Low Div"],
            "table_data": [
                {"Split": "train", "High MI": "x (0.90)\ny (0.80)", "Low Div": "d1"},
                {"Split": "val", "High MI": "z (0.70)", "Low Div": "d2\nd3"},
            ],
        }
        lines = _render_pivot_table(data)
        text = "\n".join(lines)
        # All values present
        for val in ("x (0.90)", "y (0.80)", "d1", "z (0.70)", "d2", "d3"):
            assert val in text
        # "train" row should span 2 output lines (max sub-lines = 2)
        # "val" row should also span 2 output lines
        data_lines = [ln for ln in lines if ln.strip() and "---" not in ln and "Split" not in ln]
        assert len(data_lines) == 4  # 2 sub-lines per row * 2 rows

    def test_all_single_line_unchanged(self):
        """Rows without newlines render as before (one output line per row)."""
        data = {
            "table_headers": ["Name", "Val"],
            "table_data": [
                {"Name": "a", "Val": "10"},
                {"Name": "b", "Val": "20"},
            ],
        }
        lines = _render_pivot_table(data)
        data_lines = [ln for ln in lines if ln.strip() and "---" not in ln and "Name" not in ln]
        assert len(data_lines) == 2

    def test_multiline_column_alignment(self):
        """Multi-line cells should be properly aligned across columns."""
        data = {
            "table_headers": ["Split", "Factors"],
            "table_data": [
                {"Split": "train", "Factors": "short\nvery_long_factor_name"},
            ],
        }
        lines = _render_pivot_table(data)
        # Both factor lines should end at the same column (right-aligned for non-first cols)
        factor_lines = [ln for ln in lines if "short" in ln or "very_long" in ln]
        assert len(factor_lines) == 2
        assert len(factor_lines[0]) == len(factor_lines[1])


class TestRenderPivotTableFooter:
    def test_footer_lines(self):
        """Pivot table with footer_lines renders them (lines 196-197)."""
        data = {
            "table_headers": ["Class", "Count"],
            "table_data": [{"Class": "cat", "Count": 10}],
            "footer_lines": ["Note: partial data"],
        }
        lines = _render_pivot_table(data)
        text = "\n".join(lines)
        assert "Note: partial data" in text


# ---------------------------------------------------------------------------
# _render_config_section
# ---------------------------------------------------------------------------


class TestRenderConfigSection:
    def test_empty_config(self):
        """Empty resolved config returns empty list."""
        assert _render_config_section({}) == []

    def test_full_config(self):
        """Full config renders all top-level keys."""
        resolved = {
            "sources": [{"name": "src", "dataset": "ds"}],
            "workflow": {"name": "clean", "type": "data-cleaning"},
            "extractor": {"name": "ext", "model": "onnx"},
        }
        lines = _render_config_section(resolved)
        text = "\n".join(lines)
        assert "CONFIGURATION" in text
        assert "name: src" in text
        assert "dataset: ds" in text
        assert "name: clean" in text
        assert "name: ext" in text

    def test_nested_dicts(self):
        """Nested dicts expand to multi-line when they exceed width."""
        resolved = {
            "workflow": {
                "name": "ood",
                "detectors": [
                    {"method": "kneighbors", "k": 10},
                    {"method": "domain_classifier", "n_folds": 3},
                ],
            },
        }
        lines = _render_config_section(resolved)
        text = "\n".join(lines)
        assert "method: kneighbors" in text
        assert "k: 10" in text
        assert "method: domain_classifier" in text

    def test_int_lists_compacted(self):
        """Contiguous int lists are replaced with range shorthand."""
        resolved = {"workflow": {"indices": [0, 1, 2, 3, 4]}}
        lines = _render_config_section(resolved)
        text = "\n".join(lines)
        assert "range(0, 5)" in text

    def test_source_with_view(self):
        """Sources with view config render properly."""
        resolved = {
            "sources": [
                {
                    "name": "src",
                    "dataset": "ds",
                    "view": "subset",
                    "view_config": {
                        "operations": [{"type": "Limit", "params": {"size": 1000}}],
                    },
                }
            ],
        }
        lines = _render_config_section(resolved)
        text = "\n".join(lines)
        assert "view: subset" in text
        assert "type: Limit" in text
        assert "size: 1000" in text


# ---------------------------------------------------------------------------
# _flow_repr
# ---------------------------------------------------------------------------


class TestFlowRepr:
    def test_scalar(self):
        """Scalars render as str()."""
        assert _flow_repr("hello") == "hello"
        assert _flow_repr(3.14) == "3.14"
        assert _flow_repr(True) == "True"

    def test_dict(self):
        """Dicts render as {k: v} without quotes."""
        assert _flow_repr({"a": 1, "b": 2}) == "{a: 1, b: 2}"

    def test_list(self):
        """Lists render as [v1, v2]."""
        assert _flow_repr(["dim", "pixel"]) == "[dim, pixel]"

    def test_nested(self):
        """Nested structures render inline."""
        result = _flow_repr({"params": {"size": 100}})
        assert result == "{params: {size: 100}}"

    def test_int_list_compacted(self):
        """Contiguous int lists collapse to range()."""
        assert _flow_repr([0, 1, 2, 3, 4]) == "range(0, 5)"

    def test_non_contiguous_int_list(self):
        """Non-contiguous int lists render normally."""
        assert _flow_repr([1, 3, 7]) == "[1, 3, 7]"


# ---------------------------------------------------------------------------
# _format_value
# ---------------------------------------------------------------------------


class TestFormatValue:
    def test_dict_inline(self):
        """Short dict values render inline."""
        lines: list[str] = []
        _format_value(lines, {"threshold": 2.5}, indent=4, max_width=80)
        assert lines == ["    threshold: 2.5"]

    def test_dict_expanded(self):
        """Dict value that exceeds width expands to block style."""
        lines: list[str] = []
        long_val = {"a" * 40: "b" * 40}
        _format_value(lines, {"params": long_val}, indent=0, max_width=50)
        text = "\n".join(lines)
        assert "params:" in text
        assert "a" * 40 in text

    def test_list_inline(self):
        """Short list items render inline."""
        lines: list[str] = []
        _format_value(lines, [{"method": "knn", "k": 5}], indent=4, max_width=80)
        assert lines == ["    - {method: knn, k: 5}"]

    def test_list_expanded(self):
        """Long list items expand to block style."""
        lines: list[str] = []
        _format_value(lines, [{"method": "a" * 60}], indent=4, max_width=40)
        text = "\n".join(lines)
        assert "    -" in text
        assert "method:" in text

    def test_scalar(self):
        """Plain scalar renders with indent."""
        lines: list[str] = []
        _format_value(lines, "hello", indent=4, max_width=80)
        assert lines == ["    hello"]


# ---------------------------------------------------------------------------
# _compact_indices
# ---------------------------------------------------------------------------


class TestCompactIndices:
    def test_empty_list(self):
        """Empty list returns '[]'."""
        assert _compact_indices([]) == "[]"

    def test_single_element(self):
        """Single element returns str(list)."""
        assert _compact_indices([42]) == "[42]"

    def test_contiguous_range(self):
        """Contiguous range collapses to range()."""
        assert _compact_indices([5, 6, 7, 8, 9]) == "range(5, 10)"

    def test_range_with_step(self):
        """Range with step collapses to range(start, stop, step)."""
        assert _compact_indices([0, 2, 4, 6]) == "range(0, 7, 2)"

    def test_non_contiguous(self):
        """Non-contiguous list returns str(list)."""
        result = _compact_indices([1, 3, 7])
        assert result == "[1, 3, 7]"

    def test_zero_step(self):
        """Repeated elements (step=0) returns str(list) (line 505)."""
        result = _compact_indices([5, 5, 5])
        assert result == "[5, 5, 5]"


# ---------------------------------------------------------------------------
# _format_value — non-dict list fallback
# ---------------------------------------------------------------------------


class TestFormatValueListFallback:
    def test_non_dict_list_item_exceeds_width(self):
        """List item that is not a dict and exceeds width triggers fallback (lines 477-478)."""
        lines: list[str] = []
        long_item = "a" * 80
        _format_value(lines, [long_item], indent=4, max_width=40)
        text = "\n".join(lines)
        assert "a" * 80 in text
        assert "    -" in text


# ---------------------------------------------------------------------------
# _render_factor_line — high-cardinality collapse
# ---------------------------------------------------------------------------


def _digitized(values_counts: list[tuple[str, int]], provenance: str = "derived") -> dict[str, object]:
    """A digitized factor in the shape describe_binning emits: policy plus fit."""
    return {
        "type": "categorical",
        "level": "unit",
        "encoding": {"kind": "levels", "levels": [v for v, _ in values_counts], "provenance": provenance},
        "fit": {
            "levels": [{"code": i, "value": v, "count": n} for i, (v, n) in enumerate(values_counts)],
            "empty": [i for i, (_, count) in enumerate(values_counts) if count == 0],
        },
    }


def _bin_labels(edges: list[object]) -> dict[str, str]:
    """The names DataEval hands back for a cut, in the shape describe_binning stores them."""
    labels: dict[str, str] = {}
    for code in range(1, len(edges)):
        low, high = edges[code - 1], edges[code]
        if low == "-inf":
            labels[str(code)] = f"< {high:g}"
        elif high == "inf":
            labels[str(code)] = f">= {low:g}"
        else:
            labels[str(code)] = f"[{low:g}, {high:g})"
    return labels


def _binned(
    edges: list[object],
    occupied: dict[int, tuple[int, float, float]],
    provenance: str = "derived",
    method: str | None = "uniform_width",
    names: dict[str, str] | None = None,
    hist: list[int] | None = None,
    level: str = "unit",
) -> dict[str, object]:
    """A binned factor in the shape describe_binning emits: policy, names, and fit."""
    entry: dict[str, object] = {
        "type": "continuous",
        "level": level,
        "encoding": {"kind": "bins", "edges": edges, "provenance": provenance, "method": method},
        "names": _bin_labels(edges) if names is None else names,
        "fit": {
            "bins": [
                {"code": code, "count": n, "min": lo, "max": hi} for code, (n, lo, hi) in sorted(occupied.items())
            ],
            "empty": [code for code in range(1, len(edges)) if code not in occupied],
        },
    }
    if hist is not None:
        spans = list(occupied.values())
        entry["distribution"] = {
            "histogram": hist,
            "cells": len(hist),
            "quantiles": {"0.0": min(lo for _, lo, _ in spans), "1.0": max(hi for _, _, hi in spans)},
        }
    return entry


def _record(**factors: object) -> dict[str, object]:
    """A binning record holding just the factors a test cares about."""
    return {"factors": dict(factors), "dropped": {}}


class TestRenderFactorLineLevels:
    def test_enumerates_at_threshold(self):
        """A factor with exactly _MAX_ENUMERATED levels still lists every one."""
        info = _digitized([(f"v{i}", i + 1) for i in range(_MAX_ENUMERATED)])
        lines = _render_factor_line("sensor", info)
        assert lines[0] == f"    sensor [categorical @ unit] — {_MAX_ENUMERATED} levels, derived"
        assert len(lines) == _MAX_ENUMERATED + 1
        assert lines[1] == "        v0 (0): n=1"

    def test_small_factor_lists_every_level_with_its_provenance(self):
        lines = _render_factor_line("sensor", _digitized([("a", 15), ("b", 22), ("c", 23)]))
        assert lines == [
            "    sensor [categorical @ unit] — 3 levels, derived",
            "        a (0): n=15",
            "        b (1): n=22",
            "        c (2): n=23",
        ]

    def test_declared_vocabulary_says_so(self):
        """`declared` is the state a reviewer audits for; `derived` is nobody's decision."""
        lines = _render_factor_line("sensor", _digitized([("a", 1)], provenance="declared"))
        assert lines[0] == "    sensor [categorical @ unit] — 1 levels, declared"

    def test_levels_sort_by_value_not_by_code(self):
        """A vocabulary grows append-only, so a late level carries an out-of-order code."""
        info = _digitized([("b", 1), ("c", 2)])
        # `a` arrived after the first structuring and took the next free code.
        info["fit"]["levels"].append({"code": 2, "value": "a", "count": 3})  # type: ignore[index]
        lines = _render_factor_line("sensor", info)
        assert [line.split("(")[0].strip() for line in lines[1:]] == ["a", "b", "c"]

    def test_collapses_above_threshold_with_spread(self):
        info = _digitized([(f"c{i}", 3 + (i % 17)) for i in range(40)])
        lines = _render_factor_line("klass", info)
        assert lines == ["    klass [categorical @ unit] — 40 levels, derived, n=3–19 per level"]

    def test_uniform_population_reports_single_count(self):
        lines = _render_factor_line("klass", _digitized([(f"c{i}", 5) for i in range(20)]))
        assert lines == ["    klass [categorical @ unit] — 20 levels, derived, n=5 per level"]

    def test_identifier_factor_flagged(self):
        """One level per sample is called out instead of dumping every filename."""
        lines = _render_factor_line("file_name", _digitized([(f"{i:05d}.jpg", 1) for i in range(250)]))
        assert lines == ["    file_name [categorical @ unit] — 250 levels, derived (one per sample)"]

    def test_identifier_flag_needs_high_cardinality(self):
        """A genuinely tiny all-singleton factor is still enumerated rather than flagged."""
        lines = _render_factor_line("pair", _digitized([("a", 1), ("b", 1)]))
        assert lines == [
            "    pair [categorical @ unit] — 2 levels, derived",
            "        a (0): n=1",
            "        b (1): n=1",
        ]

    def test_unreadable_fit_renders_the_policy_alone(self):
        """A companion column renamed upstream costs the counts, not the record."""
        info = _digitized([("a", 1)])
        del info["fit"]
        assert _render_factor_line("sensor", info) == ["    sensor [categorical @ unit] — derived"]

    def test_no_encoding_at_all(self):
        lines = _render_factor_line("sensor", {"type": "categorical", "level": "unit"})
        assert lines == ["    sensor [categorical @ unit] — not encoded"]


class TestRenderFactorLineBins:
    def test_names_bins_from_the_record_not_from_their_contents(self):
        """The defect this replaces: a declared cutoff never reached its own label.

        `{"temp_c": [-inf, 0.0, inf]}` used to render as `[-40, -0.3]` — a fact about the
        sample, printed where the decision belonged.
        """
        info = _binned(["-inf", 0.0, "inf"], {1: (30, -40.0, -0.3), 2: (30, 0.1, 25.0)}, "edges", None)
        lines = _render_factor_line("temp_c", info)
        assert lines[0] == "    temp_c [continuous @ unit] — 2 bins, edges declared"
        assert lines[1].startswith("        < 0 ")
        assert lines[2].startswith("        >= 0 ")
        # The observed span is reported beside the name, never as it.
        assert "occupied [-40, -0.3]" in lines[1]

    def test_falls_back_to_the_code_where_the_record_carries_no_name(self):
        """A release without the accessor costs the labels, not the report."""
        info = _binned(["-inf", 0.0, "inf"], {1: (1, -1.0, -0.5)}, "edges", None, names={})
        lines = _render_factor_line("temp_c", info)
        assert lines[1].startswith("        1 ")

    def test_reports_a_declared_bin_nothing_reached(self):
        """An empty declared bin is what a locked policy no longer fitting looks like."""
        info = _binned(["-inf", 0.0, 10.0, "inf"], {3: (60, 12.9, 25.07)}, "edges", None)
        lines = _render_factor_line("temp_c", info)
        assert lines[0] == "    temp_c [continuous @ unit] — 3 bins, edges declared, 2 empty"
        assert "empty" in lines[1]
        assert "occupied" in lines[3]

    def test_derived_cut_names_the_method_that_placed_it(self):
        info = _binned(["-inf", 50.0, "inf"], {1: (30, 1.0, 49.0), 2: (30, 51.0, 99.0)})
        assert _render_factor_line("elevation", info)[0] == (
            "    elevation [continuous @ unit] — 2 bins, derived (uniform_width)"
        )

    def test_interior_bins_render_as_half_open_intervals(self):
        info = _binned(["-inf", 0.0, 10.0, "inf"], {2: (5, 1.0, 9.0)}, "edges", None)
        lines = _render_factor_line("temp_c", info)
        assert lines[2].startswith("        [0, 10) ")

    def test_large_magnitudes_keep_the_digits_that_distinguish_them(self):
        """Four significant figures printed every epoch-millisecond span identically."""
        base = 1787011240000000.0
        info = _binned(["-inf", base, "inf"], {2: (5, base + 1788.0, base + 191686.0)}, "edges", None)
        lines = _render_factor_line("capture_ms", info)
        assert "1.787e+15" not in lines[2]
        assert "occupied [1787011240001788, 1787011240191686]" in lines[2]

    def test_collapses_above_threshold_with_occupied_span(self):
        """Many bins collapse to the count and the span the bins actually covered."""
        edges: list[object] = ["-inf", *[float(i) for i in range(1, 100)], "inf"]
        info = _binned(edges, {i: (2, float(i), i + 0.5) for i in range(1, 101)})
        lines = _render_factor_line("elevation", info)
        assert lines == ["    elevation [continuous @ unit] — 100 bins, derived (uniform_width), [1, 100.5] occupied"]

    def test_unreadable_fit_renders_the_policy_alone(self):
        info = _binned(["-inf", 0.0, "inf"], {1: (1, 0.0, 1.0)}, "edges", None)
        del info["fit"]
        assert _render_factor_line("temp_c", info) == ["    temp_c [continuous @ unit] — edges declared"]


class TestShapeCells:
    """A factor's shape, drawn to one width so two factors can be compared."""

    def test_draws_one_width_whatever_the_record_holds(self):
        assert len(_shape_cells([1, 2])) == _SHAPE_CELLS
        assert len(_shape_cells([1] * 400)) == _SHAPE_CELLS

    def test_a_cell_nothing_reached_renders_blank(self):
        assert " " not in _shape_cells([5] * _SHAPE_CELLS)
        assert _shape_cells([5, 5, 5, 0, *([5] * 16)], cells=20)[3] == " "

    def test_merges_rather_than_samples_so_a_spike_survives(self):
        """Taking every other cell would drop a spike that landed in the skipped one."""
        counts = [0] * 40
        counts[7] = 100
        assert _shape_cells(counts, cells=20)[3] == "\u2588"

    def test_the_default_resolution_keeps_every_recorded_cell(self):
        """The record is forty cells wide and so is the column, so nothing merges away:
        a spike that occupied one recorded cell still occupies exactly one drawn cell."""
        counts = [0] * 40
        counts[7] = 100
        drawn = _shape_cells(counts)
        assert drawn[7] == "\u2588"
        assert drawn.count("\u2588") == 1

    def test_a_flat_record_draws_flat_at_any_width(self):
        """Integer grouping gives one cell two source cells and its neighbour one, so a
        column of identical counts draws ragged — a distortion a reader has no way to tell
        from the data.  Every drawn cell has to cover the same slice of the record."""
        flat = [10] * 40
        for cells in (17, 26, 33, 37):
            assert len(set(_shape_cells(flat, cells))) == 1, f"ragged at {cells}: {_shape_cells(flat, cells)}"

    def test_a_record_nothing_reached_draws_nothing_rather_than_a_floor(self):
        assert _shape_cells([0, 0, 0]) == " " * _SHAPE_CELLS
        assert _shape_cells([]) == " " * _SHAPE_CELLS

    def test_every_populated_cell_keeps_a_mark_however_thin(self):
        """One row beside a thousand is the reason anyone reads the column."""
        assert _shape_cells([1000, *([1] * 19)], cells=20) == "\u2588" + "\u2581" * 19


class TestFactorTable:
    """One fixed-width row per factor, so two factors can be compared at a glance."""

    @staticmethod
    def _height() -> dict[str, object]:
        return _binned(
            ["-inf", 500.0, "inf"],
            {1: (23, 52.0, 223.0), 2: (204, 757.0, 931.0)},
            "count",
            hist=[2, 0, 5, 30],
        )

    def test_renders_a_header_and_one_row_per_encoded_factor(self):
        record = _record(
            height=self._height(),
            width=_binned(["-inf", 900.0, "inf"], {1: (36, 46.0, 493.0)}, "count", hist=[4, 1, 0, 20]),
        )
        rows = _factor_table(record)
        assert rows[0].split() == ["factor", "lvl", "bins", "shape", "(low", "\u2192", "high)", "n/bin", "range"]
        assert [line.split()[0] for line in rows[1:]] == ["height", "width"]

    def test_occupancy_range_counts_a_bin_nothing_reached_as_zero(self):
        """A cut with gaps whose range skipped its own empty bins hid the gap."""
        info = _binned(["-inf", 0.0, 10.0, "inf"], {2: (61, 1.0, 9.0)}, "edges", None, hist=[0, 61, 0])
        row = _factor_table(_record(temp_c=info))[1]
        assert "0\u201361" in row

    def test_shape_is_drawn_from_the_record_not_from_the_cut(self):
        """Two bins over a bimodal column still show both modes."""
        hist = [9, *([0] * 38), 9]
        info = _binned(["-inf", 5.0, "inf"], {1: (9, 0.0, 1.0), 2: (9, 9.0, 10.0)}, "count", hist=hist)
        (row,) = _factor_table(_record(temp_c=info))[1:]
        assert "\u2588" + " " * 38 + "\u2588" in row

    def test_a_declared_edge_list_is_printed_because_the_row_cannot_imply_it(self):
        """A bin *count* places edges uniformly across the span, so bins and range recover
        them.  A verbatim edge list is arbitrary by construction and nothing else says where
        it fell — which is exactly the decision the policy exists to make visible."""
        info = _binned(
            ["-inf", 0.0, 10.0, "inf"],
            {1: (5, -3.0, -1.0), 2: (60, 1.0, 9.0), 3: (5, 11.0, 20.0)},
            "edges",
            None,
            hist=[5, 60, 5],
        )
        assert any("edges: 0, 10" in line for line in _factor_table(_record(temp_c=info)))

    def test_a_count_declared_cut_carries_no_edge_line(self):
        assert not any("edges:" in line for line in _factor_table(_record(height=self._height())))

    def test_a_long_edge_list_is_truncated_rather_than_overrunning_the_report(self):
        edges: list[object] = ["-inf", *[float(i) for i in range(40)], "inf"]
        info = _binned(edges, {1: (5, -1.0, 0.0)}, "edges", None)
        line = next(line for line in _factor_table(_record(elevation=info)) if "edges:" in line)
        assert len(line) <= _WIDTH
        assert line.endswith("more")

    def test_a_digitized_factor_reports_its_levels_and_no_numeric_span(self):
        """A vocabulary has no extremes to report, so the column stays empty rather than
        inventing an ordering for values that have none."""
        row = _factor_table(_record(sensor=_digitized([("a", 15), ("b", 22), ("c", 23)])))[1]
        assert row.startswith("  sensor  unit")
        assert row.endswith("15\u201323")

    def test_a_record_without_a_histogram_falls_back_to_its_buckets(self):
        """A result written before the shape was recorded still draws something true."""
        info = _binned(["-inf", 5.0, "inf"], {1: (1, 0.0, 1.0), 2: (99, 6.0, 9.0)}, "count")
        row = _factor_table(_record(temp=info))[1]
        assert "\u2588" in row
        assert "\u2581" in row

    def test_a_factor_that_was_never_encoded_stays_out_of_the_table(self):
        """Nothing to compare: no cut, no shape, no occupancy."""
        assert _factor_table(_record(sensor={"type": "categorical", "level": "unit"})) == []

    def test_the_shape_column_gives_up_cells_before_the_table_gives_up_its_width(self):
        """A span like `1.145e-06 – 0.07204` is nineteen characters beside a nineteen
        character factor name.  The shape is the one column that can shrink without the
        table losing a fact, and it stays one width across every row while it does."""
        record = _record(
            instance_brightness=self._height(),
            unit_zeros=_binned(
                ["-inf", 0.5, "inf"],
                {1: (4, 1.145e-06, 0.05), 2: (123, 0.06, 0.07204)},
                "count",
                hist=[1] * 40,
            ),
        )
        lines = _factor_table(record)
        assert all(len(line) <= _WIDTH for line in lines), max(len(line) for line in lines)
        # The span is still printed in full: the fit came out of the shape, not the facts.
        assert "1.145e-06 \u2013 0.07204" in lines[2]

    def test_row_carries_level_bin_count_and_observed_span(self):
        (row,) = _factor_table(_record(height=self._height()))[1:]
        assert row.startswith("  height  unit")
        assert row.endswith("  23\u2013204  52 \u2013 931")


class TestBinningRecordDetail:
    """The table is the default read; the full breakdown stays available behind `detailed`."""

    @staticmethod
    def _record_with_height() -> dict[str, object]:
        return _record(
            height=_binned(
                ["-inf", 500.0, "inf"],
                {1: (23, 52.0, 223.0), 2: (204, 757.0, 931.0)},
                "count",
                hist=[2, 0, 5, 30],
            )
        )

    def test_renders_the_table_alone_by_default(self):
        lines = _render_binning_record(self._record_with_height())
        assert any("n/bin" in line for line in lines)
        assert not any(line.startswith("    height [") for line in lines)

    def test_section_passes_the_detail_request_to_every_split(self):
        binning = {"per_split": {"train": self._record_with_height(), "test": self._record_with_height()}}
        assert sum(1 for line in _render_binning_section(binning, detailed=True) if "occupied" in line) == 4
        assert not any("occupied" in line for line in _render_binning_section(binning))

    def test_each_split_is_set_off_from_the_one_above_it(self):
        """Two tables running together read as one table with a stray heading in it."""
        binning = {"per_split": {"train": self._record_with_height(), "test": self._record_with_height()}}
        lines = _render_binning_section(binning)
        assert lines[lines.index("  [test]") - 1] == ""

    def test_every_split_table_shares_one_column_layout(self):
        """Splits drawn at two resolutions cannot be read against each other, which is the one
        thing this section promises about them when they share an encoding — and two tables
        whose columns land in different places do not stack into something readable."""
        narrow = _binned(["-inf", 5.0, "inf"], {1: (4, 1.0, 4.0), 2: (1234, 6.0, 9.0)}, "count", hist=[1] * 40)
        wide = _binned(
            ["-inf", 0.5, "inf"],
            {1: (4, 1.145e-06, 0.05), 2: (123, 0.06, 0.07204)},
            "count",
            hist=[1] * 40,
        )
        binning = {
            "per_split": {
                "train": _record(instance_brightness=narrow),
                "test": _record(instance_brightness=wide),
            }
        }
        headers = [line for line in _render_binning_section(binning) if "n/bin" in line]
        assert len(headers) == 2
        assert len(set(headers)) == 1

    def test_detailed_keeps_the_full_breakdown_under_the_table(self):
        lines = _render_binning_record(self._record_with_height(), detailed=True)
        table_at = next(i for i, line in enumerate(lines) if "n/bin" in line)
        detail_at = next(i for i, line in enumerate(lines) if line.startswith("    height ["))
        assert table_at < detail_at
        assert any("occupied [52, 223]" in line for line in lines)


class TestDiagnostics:
    """Library prose printed into a report that has a width."""

    def test_a_long_diagnostic_wraps_to_the_report_width(self):
        """A diagnostic has no length contract, and the report around it does. Left
        unwrapped, one sentence drags the whole block sideways — and in the documentation,
        where the block scrolls rather than wraps, it takes the factor table with it."""
        message = "dataeval: Declared cuts left bins unused: " + ", ".join(
            f"factor_{i} ({i} of 40 bins hold rows)" for i in range(10)
        )
        lines = _render_binning_section(None, [message])
        assert all(len(line) <= _WIDTH for line in lines), max(len(line) for line in lines)
        body = [line for line in lines if line.startswith("    ")]
        assert len(body) > 1, "a message this long has to occupy more than one line"
        assert body[1].startswith("      "), "continuations indent under the message they belong to"

    def test_a_short_diagnostic_stays_on_one_line(self):
        lines = _render_binning_section(None, ["dataeval: nothing to report"])
        assert "    dataeval: nothing to report" in lines


class TestSplitComparability:
    """Two splits are comparable when the same code means the same thing in both."""

    @staticmethod
    def _split(**factors) -> dict[str, object]:
        return {"factors": {name: {"encoding": enc} for name, enc in factors.items()}, "dropped": {}}

    @staticmethod
    def _bins(*edges) -> dict[str, object]:
        return {"kind": "bins", "edges": list(edges), "provenance": "edges", "method": None}

    @staticmethod
    def _levels(*values) -> dict[str, object]:
        return {"kind": "levels", "levels": list(values), "provenance": "derived"}

    def test_identical_encodings_read_as_comparable(self):
        split = self._split(temp_c=self._bins("-inf", 0.0, "inf"))
        lines = _render_split_comparability({"train": split, "test": split})
        assert any("comparable across them" in line and "NOT" not in line for line in lines)

    def test_a_grown_vocabulary_is_still_comparable(self):
        """Append-only growth leaves every shared code meaning what it meant.

        Reporting this as a disagreement would flag the ordinary case of one split holding
        a level another lacks, which is the false alarm that teaches people to ignore the
        true ones.
        """
        lines = _render_split_comparability(
            {
                "train": self._split(sensor=self._levels("a", "b")),
                "test": self._split(sensor=self._levels("a", "b", "c")),
            }
        )
        assert any("comparable across them" in line and "NOT" not in line for line in lines)

    def test_a_reordered_vocabulary_is_not_comparable(self):
        """Same values, different codes — the one thing append-only ordering prevents."""
        lines = _render_split_comparability(
            {
                "train": self._split(sensor=self._levels("a", "b")),
                "test": self._split(sensor=self._levels("b", "a")),
            }
        )
        assert any("NOT comparable" in line for line in lines)

    def test_different_cuts_are_not_comparable(self):
        lines = _render_split_comparability(
            {
                "train": self._split(temp_c=self._bins("-inf", 0.0, "inf")),
                "test": self._split(temp_c=self._bins("-inf", 5.0, "inf")),
            }
        )
        assert any("NOT comparable" in line for line in lines)

    def test_names_only_the_factors_that_differ(self):
        lines = _render_split_comparability(
            {
                "train": self._split(temp_c=self._bins("-inf", 0.0, "inf"), sensor=self._levels("a")),
                "test": self._split(temp_c=self._bins("-inf", 5.0, "inf"), sensor=self._levels("a")),
            }
        )
        verdict = next(line for line in lines if "NOT comparable" in line)
        assert "temp_c" in verdict
        assert "sensor" not in verdict

    def test_three_splits_that_each_extend_a_third_are_not_comparable(self):
        """Append-only agreement is not transitive, and comparing only to the first misses it.

        `test` and `val` each merely extend `train`, so pairwise-against-the-first says
        comparable — while code 1 means `b` in one and `c` in the other, which is the exact
        false reassurance this section exists to prevent.
        """
        lines = _render_split_comparability(
            {
                "train": self._split(sensor=self._levels("a")),
                "test": self._split(sensor=self._levels("a", "b")),
                "val": self._split(sensor=self._levels("a", "c")),
            }
        )
        assert any("NOT comparable" in line for line in lines)

    def test_a_grown_vocabulary_does_not_claim_one_shared_digest(self):
        """The line must not contradict the `encoding_digest` printed above it.

        Growth appends, so the codes still agree — but the digests differ and the envelope's
        `encoding_digest` is then None. Saying "share one encoding" over that is a report
        disagreeing with its own record.
        """
        train = {**self._split(sensor=self._levels("a", "b")), "encoding_digest": "aaaa"}
        test = {**self._split(sensor=self._levels("a", "b", "c")), "encoding_digest": "bbbb"}
        lines = _render_split_comparability({"train": train, "test": test})

        assert any("comparable" in line and "NOT" not in line for line in lines)
        assert not any("share one encoding" in line for line in lines)

    def test_one_split_says_nothing(self):
        assert _render_split_comparability({"train": self._split(a=self._levels("x"))}) == []

    def test_no_encodings_says_nothing(self):
        assert _render_split_comparability({"train": self._split(), "test": self._split()}) == []

    def test_the_section_carries_the_digest(self):
        section = _render_binning_section({"encoding_digest": "2b6530cc3015f1fe", "factors": {}, "dropped": {}})
        assert any("2b6530cc3015f1fe" in line for line in section)


class TestReviewState:
    """The report leads with how much of the encoding is nobody's decision."""

    @staticmethod
    def _record(unreviewed: list[str], encoded: list[str]) -> dict:
        return {
            "unreviewed": unreviewed,
            "factors": {name: {"encoding": {"kind": "bins"}} for name in encoded},
            "dropped": {},
        }

    def test_counts_and_names_the_unreviewed(self):
        lines = _render_review_state(self._record(["b"], ["a", "b"]))
        assert "1 of 2 factors still derived" in lines[0]
        assert "(b)" in lines[1]

    def test_says_so_when_everything_was_decided(self):
        lines = _render_review_state(self._record([], ["a", "b"]))
        assert lines == ["  Policy:          all 2 factors declared or reviewed"]

    def test_says_nothing_when_no_factor_was_encoded(self):
        assert _render_review_state(self._record([], [])) == []

    def test_says_nothing_for_a_record_written_before_this_was_tracked(self):
        record = self._record([], ["a"])
        del record["unreviewed"]
        assert _render_review_state(record) == []


# ---------------------------------------------------------------------------
# _render_distribution / _render_ratio
# ---------------------------------------------------------------------------


def _binned_counts(counts: list[int], empty: list[int] | None = None) -> dict:
    """A factor entry whose fit holds the given per-bin counts.

    Named distinctly from the module-level ``_binned`` fixture above (which builds a
    factor from explicit edges and per-bucket spans): both live at module scope, and
    reusing a name would silently rebind it out from under the earlier tests.
    """
    bins = [{"code": i + 1, "count": c, "min": i * 10, "max": i * 10 + 9} for i, c in enumerate(counts) if c]
    edges = [i * 10 for i in range(len(counts) + 1)]
    return {
        "type": "continuous",
        "level": "unit",
        "encoding": {"kind": "bins", "provenance": "derived", "edges": edges},
        "fit": {"bins": bins, "empty": empty or []},
    }


def test_a_nonzero_bucket_is_never_blank():
    """ "Empty" against "one sample landed here" is exactly what a bin count is argued from.

    A bar that rounds the second down to the first argues for the wrong answer. This
    isolates the bar cell itself rather than asserting on the whole row — the row's label
    alone would read non-blank regardless of what the bar drew.
    """
    lines = _render_distribution(_binned_counts([1000, 1]))
    # Row layout is `{label:>w}  {bar:<_BAR_MAX}  {count:>5}{note}`. The bucket with
    # count=1 carries no "empty" note, so its row's fixed-width tail (bar + gutter +
    # count) can be sliced from the end regardless of the label column's width.
    row = lines[-1]
    bar_cell = row[-(5 + 2 + _BAR_MAX) : -(5 + 2)]
    assert bar_cell.strip()


def test_an_empty_bin_is_marked_and_drawn_blank():
    lines = "\n".join(_render_distribution(_binned_counts([50, 0, 50], empty=[2])))
    assert "empty" in lines


def test_the_count_is_always_printed():
    lines = "\n".join(_render_distribution(_binned_counts([412, 7])))
    assert "412" in lines
    assert "7" in lines


def test_a_wide_factor_falls_back_to_a_sparkline():
    lines = _render_distribution(_binned_counts([10] * 30))
    # One sparkline line, no per-bucket enumeration above the cap.
    assert len(lines) == 1


def test_a_narrow_factor_gets_both():
    lines = _render_distribution(_binned_counts([10, 20, 30]))
    assert len(lines) > 1


def test_levels_are_charted_too():
    info = {
        "type": "categorical",
        "level": "unit",
        "encoding": {"kind": "levels", "provenance": "derived", "levels": ["a", "b"]},
        "fit": {
            "levels": [
                {"code": 0, "value": "a", "count": 30},
                {"code": 1, "value": "b", "count": 10},
            ],
            "empty": [],
        },
    }
    lines = "\n".join(_render_distribution(info))
    assert "a" in lines
    assert "30" in lines


def test_a_missing_fit_renders_nothing():
    assert _render_distribution({"type": "continuous", "level": "unit"}) == []


def test_the_ratio_bar_shows_both_kinds():
    line = _render_ratio({"numeric": 1842, "text": 58})
    assert "1,842 numeric" in line
    assert "58 text" in line


def test_a_long_sparkline_does_not_run_into_its_count():
    """A factor with more buckets than the pad still needs a gap before `n=`."""
    lines = _render_distribution(_binned_counts([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]))
    assert "n=" in lines[0]
    assert " n=" in lines[0], f"sparkline runs into the count: {lines[0]!r}"


def test_a_single_kind_column_gets_counts_without_an_empty_bar():
    """A bar comparing one kind to nothing renders as an empty track that reads as zero."""
    line = _render_ratio({"text": 200})
    assert line == "200 text"
    assert "░" not in line


def test_two_kinds_still_get_a_bar():
    line = _render_ratio({"numeric": 198, "text": 2})
    assert "\u2588" in line
    assert "\u2591" in line
    assert "198 numeric" in line
    assert "2 text" in line


def test_a_thin_minority_still_shows_in_the_ratio_bar():
    """198 against 2 is 19.8 blocks; rounding it up hides the reason the line is printed."""
    line = _render_ratio({"numeric": 198, "text": 2})
    assert line.count("░") >= 1, f"the 2 text rows vanished: {line!r}"
    assert line.count("█") == 19


def _shaped(quantiles: dict[str, float], hist: list[int], kind: str = "bins") -> dict:
    """A factor entry carrying a recorded distribution."""
    return {
        "type": "discrete",
        "level": "unit",
        "encoding": {"kind": kind, "provenance": "derived", "edges": [0, 1], "levels": ["a"]},
        "distribution": {"quantiles": {str(k): v for k, v in quantiles.items()}, "histogram": hist, "cells": len(hist)},
    }


def test_a_cut_factor_is_drawn_from_its_shape_not_its_bins():
    """The chart exists to judge the cut, so it must not be drawn at that cut."""
    q = {"0.0": -1.0, "0.25": -1.0, "0.5": 30.0, "0.75": 42.0, "1.0": 240.0}
    lines = _render_distribution(_shaped(q, [60, 20, 10, 5, 3, 2, 1, 1]))
    assert len(lines) == 2
    assert "p25 -1" in lines[1]
    assert "│" in lines[1], "the median mark is drawn"


def test_a_skewed_box_never_collapses_to_bare_whiskers():
    """p25 and p75 within a fraction of a cell still leave a visible box."""
    q = {"0.0": 110.0, "0.25": 900.0, "0.5": 1806.0, "0.75": 5508.0, "1.0": 680264.0}
    lines = _render_distribution(_shaped(q, [80, 5, 2, 1, 0, 0, 0, 1]))
    # Where the box collapses onto the median, the median mark is the box. What must not
    # happen is a line of bare whiskers, which reads as broken rather than as skewed.
    assert not lines[1].startswith("├" + "─" * 6 + "┤"), f"the box vanished: {lines[1]!r}"
    assert "│" in lines[1]


def test_a_digitized_factor_keeps_its_vocabulary():
    """A vocabulary is the values themselves, not a partition imposed on them."""
    info = {
        "type": "categorical",
        "level": "unit",
        "encoding": {"kind": "levels", "provenance": "derived", "levels": ["m210", "mavic"]},
        "fit": {
            "levels": [{"code": 0, "value": "m210", "count": 56}, {"code": 1, "value": "mavic", "count": 120}],
            "empty": [],
        },
        "distribution": {
            "quantiles": {"0.0": 0.0, "0.25": 0.0, "0.5": 1.0, "0.75": 1.0, "1.0": 1.0},
            "histogram": [56, 120],
            "cells": 2,
        },
    }
    lines = "\n".join(_render_distribution(info))
    assert "m210" in lines
    assert "p25" not in lines
