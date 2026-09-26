"""Tables in text: aligned columns, charts drawn from numbers, and a layout that shrinks charts before it overflows."""

import pytest

from dataeval_flow._blocks import Column, Section, Table
from dataeval_flow._blocks._text import Frame, render_text

pytestmark = pytest.mark.required


def _draw(table: Table, width: int = 90, indent: str = "  ") -> list[str]:
    return render_text([table], Frame(width=width, indent=indent, depth=2))


class TestTextColumns:
    def test_the_first_column_aligns_left_and_the_rest_right(self):
        table = Table(
            columns=[Column(key="c", header="Class"), Column(key="n", header="Count")],
            rows=[{"c": "cat", "n": 12}, {"c": "zebra", "n": 3}],
        )
        assert _draw(table) == ["  Class  Count", "  -----  -----", "  cat       12", "  zebra      3"]

    def test_align_overrides_the_default(self):
        table = Table(
            columns=[Column(key="c", header="Class"), Column(key="s", header="Status", align="left")],
            rows=[{"c": "car", "s": "ok"}],
        )
        assert _draw(table)[-1] == "  car    ok"

    def test_a_format_applies_to_numeric_cells_only(self):
        table = Table(
            columns=[Column(key="c", header="c"), Column(key="p", header="%", format="{:.1f}%")],
            rows=[{"c": "a", "p": 12.345}, {"c": "b", "p": "n/a"}],
        )
        assert _draw(table)[2:] == ["  a  12.3%", "  b    n/a"]

    def test_missing_and_none_cells_are_blank(self):
        table = Table(columns=[Column(key="a", header="a"), Column(key="b", header="b")], rows=[{"a": "x"}])
        assert _draw(table)[-1] == "  x"

    def test_a_multi_line_cell_expands_its_row(self):
        table = Table(
            columns=[Column(key="s", header="Split"), Column(key="f", header="Factors")],
            rows=[{"s": "train", "f": "age\nweather"}],
        )
        assert _draw(table)[2:] == ["  train      age", "         weather"]

    def test_a_table_whose_headers_are_all_empty_has_no_header_row(self):
        table = Table(columns=[Column(key="a", align="right"), Column(key="b")], rows=[{"a": "low", "b": 4}])
        assert _draw(table, indent="") == ["low  4"]

    def test_a_table_with_no_rows_draws_nothing(self):
        assert _draw(Table(columns=[Column(key="a", header="a")], rows=[])) == []


class TestBars:
    def test_a_bar_scales_to_the_largest_value_in_eighths(self):
        table = Table(
            columns=[Column(key="c", header="c"), Column(key="n", kind="bar")],
            rows=[{"c": "a", "n": 40}, {"c": "b", "n": 10}],
        )
        lines = _draw(table)
        assert lines[1] == "  -  " + "-" * 30
        assert lines[2] == "  a  " + "█" * 30
        assert lines[3] == "  b  " + "█" * 7 + "▌"

    def test_a_nonzero_value_is_never_blank(self):
        table = Table(
            columns=[Column(key="c", header="c"), Column(key="n", kind="bar")],
            rows=[{"c": "big", "n": 100000}, {"c": "one", "n": 1}, {"c": "none", "n": 0}],
        )
        lines = _draw(table)
        assert lines[3] == "  one   ▏"
        assert lines[4] == "  none"

    def test_a_scale_below_zero_draws_whole_cells_from_zero(self):
        table = Table(
            columns=[Column(key="c", header="c"), Column(key="d", kind="bar")],
            rows=[{"c": "a", "d": 1.0}, {"c": "b", "d": -1.0}],
        )
        lines = _draw(table)
        assert lines[2] == "  a  " + " " * 15 + "█" * 15
        assert lines[3] == "  b  " + "█" * 15

    def test_an_all_negative_column_draws_each_bar_back_to_zero(self):
        """Zero stays on the scale, so the value nearest it draws the shortest bar rather than none."""
        table = Table(
            columns=[Column(key="c", header="c"), Column(key="d", kind="bar")],
            rows=[{"c": "a", "d": -1.0}, {"c": "b", "d": -3.0}],
        )
        lines = _draw(table)
        assert lines[2] == "  a  " + " " * 20 + "█" * 10
        assert lines[3] == "  b  " + "█" * 30

    def test_markers_draw_a_scale_line_under_the_bar(self):
        table = Table(
            columns=[
                Column(key="chunk", header="Chunk"),
                Column(key="d", header="Distance", format="{:.4f}"),
                Column(key="d", kind="bar", format="{:.4f}", markers=[("Threshold", 0.05), ("Threshold", 0.3)]),
            ],
            rows=[{"chunk": "0:100", "d": 0.1}, {"chunk": "100:200", "d": 0.35}],
        )
        scale = _draw(table)[-1]
        # The lower pipe sits under the bar cell where 0.05 falls: cell 4 of 30 over 0-0.35.
        bar_start = len("  ") + len("100:200") + 2 + len("Distance") + 2
        assert scale.startswith("  Threshold")
        assert scale.index("|") == bar_start + 4
        assert scale.endswith("|(0.3000)")
        assert "(0.0500)|" in scale

    def test_a_single_marker_draws_from_the_bar_start(self):
        table = Table(
            columns=[Column(key="c", header="c"), Column(key="d", kind="bar", markers=[("limit", 1.0)])],
            rows=[{"c": "a", "d": 2.0}],
        )
        # 1.0 falls at cell 15 of 30 over 0-2; the label keeps a space before the scale starts.
        assert _draw(table)[-1] == "  limit " + "-" * 15 + "|(1)"


class TestStacked:
    def test_segments_draw_in_series_order_with_a_legend_header(self):
        table = Table(
            columns=[Column(key="r", header="Range"), Column(key="s", kind="stacked", series=["in-dist", "OOD"])],
            rows=[{"r": "0-1", "s": [86.0, 0.0]}, {"r": "1-2", "s": [2.0, 1.0]}, {"r": "2-3", "s": [0.0, 2.0]}],
        )
        lines = _draw(table)
        assert lines[0] == "  Range  █ in-dist  ░ OOD"
        assert lines[2] == "  0-1    " + "█" * 30
        assert lines[3] == "  1-2    █░"
        assert lines[4] == "  2-3    ░"


class TestSparklines:
    def test_a_sparkline_is_drawn_at_the_column_width_and_its_header_cut_to_fit(self):
        table = Table(
            columns=[Column(key="f", header="factor"), Column(key="h", header="shape (low → high)", kind="sparkline")],
            rows=[{"f": "a", "h": [1.0, 2.0, 3.0]}],
        )
        # 6 + 2 + 40 is 48 against 24 of room, so the sparkline gives up 24 cells and draws at 16.
        lines = _draw(table, width=26)
        assert all(len(line) <= 26 for line in lines)
        assert lines[0] == "  factor  shape (low → hig"
        assert lines[1] == "  ------  " + "-" * 16


class TestFitting:
    def test_sparklines_shrink_before_bars_and_never_below_ten(self):
        table = Table(
            columns=[
                Column(key="f", header="f"),
                Column(key="h", kind="sparkline"),
                Column(key="n", kind="bar"),
            ],
            rows=[{"f": "a", "h": [1.0, 2.0], "n": 1}],
        )
        # Natural widths 1 + 40 + 30 plus two gaps is 75; at 40 the sparkline gives up 30
        # cells to reach its floor of ten, and the bar gives up the last 5.
        lines = _draw(table, width=40, indent="")
        assert [len(part) for part in lines[1].split("  ")] == [1, 10, 25]

    def test_a_stacked_bar_keeps_the_width_of_its_legend(self):
        """A column narrower than its legend would push every heading after it off its column."""
        table = Table(
            columns=[
                Column(key="r", header="Range"),
                Column(key="s", kind="stacked", series=["in-dist", "OOD"]),
                Column(key="m", header="note", align="left"),
            ],
            rows=[{"r": "0-1", "s": [3.0, 1.0], "m": "x"}],
        )
        # Natural widths 5 + 30 + 4 plus two gaps is 43; at 26 the bar would give up 17 cells,
        # but it stops at its legend's 16, and the row overflows instead.
        lines = _draw(table, width=26, indent="")
        legend = "█ in-dist  ░ OOD"
        assert [len(part) for part in lines[1].split("  ")] == [5, len(legend), 4]
        assert lines[0] == f"Range  {legend}  note"

    def test_a_table_too_wide_to_shrink_overflows_rather_than_wrapping_cells(self):
        table = Table(
            columns=[Column(key="a", header="a"), Column(key="b", header="b")], rows=[{"a": "x" * 50, "b": "y" * 50}]
        )
        lines = _draw(table, width=60)
        assert len(lines) == 3
        assert len(lines[-1]) > 60


class TestSharedLayout:
    def _table(self, name: str) -> Table:
        return Table(
            columns=[Column(key="f", header="factor"), Column(key="n", header="n")], rows=[{"f": name, "n": 1}]
        )

    def test_tables_with_identical_columns_in_one_top_level_section_share_widths(self):
        section = Section(
            title="METADATA FACTORS",
            blocks=[
                Section(title="[train]", blocks=[self._table("a_long_factor")]),
                Section(title="[test]", blocks=[self._table("x")]),
            ],
        )
        lines = render_text([section], Frame(indent="  ", depth=1))
        rows = [line for line in lines if line.strip().startswith(("a_long_factor", "x "))]
        assert rows[0].index("1") == rows[1].index("1")

    def test_tables_in_different_top_level_sections_keep_their_own_widths(self):
        first = Section(title="one", blocks=[self._table("a_long_factor")])
        second = Section(title="two", blocks=[self._table("x")])
        lines = render_text([first, second], Frame(indent="  ", depth=1))
        assert "  x       1" in lines


class TestDegenerateNumbers:
    def test_an_all_zero_bar_column_draws_blank_bars(self):
        table = Table(columns=[Column(key="c", header="c"), Column(key="n", kind="bar")], rows=[{"c": "a", "n": 0}])
        assert _draw(table)[-1] == "  a"

    def test_markers_on_a_zero_span_scale_still_draw(self):
        table = Table(
            columns=[Column(key="c", header="c"), Column(key="d", kind="bar", markers=[("t", 0.0), ("t", 0.0)])],
            rows=[{"c": "a", "d": 0.0}],
        )
        assert _draw(table)[-1].startswith("  t")

    def test_an_empty_stacked_cell_draws_blank(self):
        table = Table(
            columns=[Column(key="r", header="r"), Column(key="s", kind="stacked", series=["a", "b"])],
            rows=[{"r": "x", "s": [0.0, 0.0]}],
        )
        assert _draw(table)[-1] == "  x"


class TestFormats:
    def test_an_integer_format_on_a_numpy_count_renders(self):
        import numpy as np

        columns = [{"key": "n", "header": "n", "format": "{:,d}"}]
        table = Table.model_validate({"columns": columns, "rows": [{"n": np.int64(1200)}]})
        assert _draw(table)[-1] == "  1,200"

    @pytest.mark.parametrize("template", ["{:d}", "{:.2f} {}", "{missing}"])
    def test_a_format_that_does_not_fit_the_value_falls_back_to_its_text(self, template):
        table = Table(columns=[Column(key="x", header="x", format=template)], rows=[{"x": 2.5}])
        assert _draw(table)[-1] == "  2.5"


class TestNonFiniteNumbers:
    """A NaN or an infinity in the data must not take the report down with it."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_bar_value_draws_blank(self, bad):
        table = Table(
            columns=[Column(key="c", header="c"), Column(key="d", kind="bar")],
            rows=[{"c": "a", "d": bad}, {"c": "b", "d": 2.0}],
        )
        lines = _draw(table)
        assert lines[2] == "  a"
        assert lines[3] == "  b  " + "█" * 30

    def test_a_non_finite_marker_is_left_off_the_scale(self):
        table = Table(
            columns=[
                Column(key="c", header="c"),
                Column(key="d", kind="bar", markers=[("t", float("nan")), ("t", 1.0)]),
            ],
            rows=[{"c": "a", "d": 2.0}],
        )
        assert _draw(table)[-1].endswith("|(1)")

    def test_a_stacked_cell_with_a_non_finite_segment_draws_blank(self):
        table = Table(
            columns=[Column(key="r", header="r"), Column(key="s", kind="stacked", series=["a", "b"])],
            rows=[{"r": "x", "s": [float("nan"), 1.0]}, {"r": "y", "s": [1.0, 1.0]}],
        )
        lines = _draw(table)
        assert lines[2] == "  x"
        assert lines[3].startswith("  y  \u2588")

    def test_a_non_finite_count_in_a_sparkline_draws_as_empty(self):
        table = Table(
            columns=[Column(key="f", header="f"), Column(key="h", kind="sparkline")],
            rows=[{"f": "a", "h": [1.0, float("nan"), 3.0]}],
        )
        assert _draw(table)[-1].startswith("  a  ")
