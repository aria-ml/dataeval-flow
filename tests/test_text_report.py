"""Tests for _text_report rendering helpers."""

from __future__ import annotations

import pytest

from dataeval_flow.workflow._text_report import (
    _MAX_ENUMERATED,
    _WIDTH,
    _brief_value,
    _compact_indices,
    _flow_repr,
    _format_value,
    _render_binning_section,
    _render_chunk_table,
    _render_classwise_table,
    _render_config_section,
    _render_detail_section,
    _render_factor_line,
    _render_key_value,
    _render_pivot_table,
    _render_review_state,
    _render_split_comparability,
    _render_table,
    _section_header,
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
) -> dict[str, object]:
    """A binned factor in the shape describe_binning emits: policy, names, and fit."""
    return {
        "type": "continuous",
        "level": "unit",
        "encoding": {"kind": "bins", "edges": edges, "provenance": provenance, "method": method},
        "names": _bin_labels(edges) if names is None else names,
        "fit": {
            "bins": [
                {"code": code, "count": n, "min": lo, "max": hi} for code, (n, lo, hi) in sorted(occupied.items())
            ],
            "empty": [code for code in range(1, len(edges)) if code not in occupied],
        },
    }


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


class TestSplitComparability:
    """Two splits binned independently are not comparable, and the report says so."""

    @staticmethod
    def _record(digest: str) -> dict[str, object]:
        return {"encoding_digest": digest, "factors": {}, "dropped": {}}

    def test_matching_digests_read_as_comparable(self):
        lines = _render_split_comparability({"train": self._record("abc"), "test": self._record("abc")})
        assert any("comparable across them" in line and "NOT" not in line for line in lines)

    def test_differing_digests_are_called_out(self):
        lines = _render_split_comparability({"train": self._record("abc"), "test": self._record("def")})
        assert any("NOT comparable" in line for line in lines)

    def test_one_split_says_nothing(self):
        assert _render_split_comparability({"train": self._record("abc")}) == []

    def test_a_missing_digest_says_nothing(self):
        """Silence beats a claim the record cannot support."""
        record = self._record("abc")
        record["encoding_digest"] = None
        assert _render_split_comparability({"train": record, "test": self._record("def")}) == []

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
