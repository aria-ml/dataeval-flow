"""Tests for _text_report rendering helpers."""

from __future__ import annotations

from dataeval_flow.workflow._text_report import (
    _WIDTH,
    _brief_value,
    _render_chunk_table,
    _render_classwise_table,
    _render_detail_section,
    _render_key_value,
    _render_pivot_table,
    _render_table,
    _section_header,
    _summary_line,
)
from dataeval_flow.workflow.base import Reportable

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
