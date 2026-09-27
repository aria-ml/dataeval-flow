"""Glyph drawing: sparklines resampled by area, box plots and their labels, and compact value and number text."""

from typing import Any

import pytest

from dataeval_flow._blocks import Section, Tree
from dataeval_flow._blocks._draw import (
    box_plot,
    compact_indices,
    flow_repr,
    fmt_num,
    format_value,
    ratio_line,
    shape_cells,
)
from dataeval_flow._blocks._table import SPARKLINE_CELLS
from dataeval_flow._blocks._text import Frame, render_text

pytestmark = pytest.mark.required


def _config(resolved: dict[str, Any]) -> list[str]:
    """The CONFIGURATION section as a report draws it."""
    return render_text([Section(title="CONFIGURATION", blocks=[Tree(value=resolved)])], Frame(indent="  ", depth=1))


class TestShapeCells:
    """A factor's shape, drawn to one width so two factors can be compared."""

    def test_draws_one_width_whatever_the_record_holds(self):
        assert len(shape_cells([1, 2], SPARKLINE_CELLS)) == SPARKLINE_CELLS
        assert len(shape_cells([1] * 400, SPARKLINE_CELLS)) == SPARKLINE_CELLS

    def test_a_cell_nothing_reached_renders_blank(self):
        assert " " not in shape_cells([5] * SPARKLINE_CELLS, SPARKLINE_CELLS)
        assert shape_cells([5, 5, 5, 0, *([5] * 16)], cells=20)[3] == " "

    def test_merges_rather_than_samples_so_a_spike_survives(self):
        """Taking every other cell would drop a spike that landed in the skipped one."""
        counts = [0] * 40
        counts[7] = 100
        assert shape_cells(counts, cells=20)[3] == "\u2588"

    def test_the_default_resolution_keeps_every_recorded_cell(self):
        """The record is forty cells wide and so is the column, so nothing merges away:
        a spike that occupied one recorded cell still occupies exactly one drawn cell."""
        counts = [0] * 40
        counts[7] = 100
        drawn = shape_cells(counts, SPARKLINE_CELLS)
        assert drawn[7] == "\u2588"
        assert drawn.count("\u2588") == 1

    def test_a_flat_record_draws_flat_at_any_width(self):
        """Integer grouping gives one cell two source cells and its neighbour one. A
        column of identical counts then draws ragged, and a reader cannot tell from the
        data. Every drawn cell has to cover the same slice of the record."""
        flat = [10] * 40
        for cells in (17, 26, 33, 37):
            assert len(set(shape_cells(flat, cells))) == 1, f"ragged at {cells}: {shape_cells(flat, cells)}"

    def test_a_record_nothing_reached_draws_nothing_rather_than_a_floor(self):
        assert shape_cells([0, 0, 0], SPARKLINE_CELLS) == " " * SPARKLINE_CELLS
        assert shape_cells([], SPARKLINE_CELLS) == " " * SPARKLINE_CELLS

    def test_every_populated_cell_keeps_a_mark_however_thin(self):
        """The column exists for sparse cells: one populated cell among a thousand must keep its mark."""
        assert shape_cells([1000, *([1] * 19)], cells=20) == "\u2588" + "\u2581" * 19


# ---------------------------------------------------------------------------
# the configuration section
# ---------------------------------------------------------------------------


class TestFmtNum:
    """A number written compactly, without losing the digits that distinguish it."""

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (5715.0, "5715"),
            (13.16, "13.16"),
            (9999.4, "9999"),
            (9999.5, "10000"),
            (19800.0, "19800"),
            (57521.4, "57521"),
            (-680264.0, "-680264"),
            (999999.7, "1000000"),
            (1787000000000000.0, "1787000000000000"),
            (0.00001234, "1.23e-05"),
        ],
    )
    def test_writes_each_magnitude_in_its_shortest_faithful_form(self, value, text):
        """From 9,999.5, where four figures first need an exponent, up to a million, a whole number is shorter."""
        assert fmt_num(value) == text


_FLAT = [1] * 40


def _plot(
    low: float, q1: float, median: float, q3: float, high: float, histogram: list[int] = _FLAT, room: int = 80
) -> list[str] | None:
    return box_plot(histogram, low, q1, median, q3, high, room)


class TestBoxPlot:
    """A histogram over its box, each value beside the mark it names or named outright."""

    def test_quartiles_sit_in_the_whiskers_and_the_median_under_its_mark(self):
        assert _plot(0.0, 25.0, 50.0, 75.0, 100.0) == [
            "████████████████████████████████████████",
            "├───── 25 ██████████│█████████ 75 ─────┤",
            "0                  50                100",
        ]

    def test_a_quartile_equal_to_its_end_is_left_to_the_box_reaching_it(self):
        """SeaDrone's -1 sentinel: p25 is min, so there is no left whisker and nothing to write in it."""
        assert _plot(-1.0, -1.0, 34.9, 46.5, 90.0)[1:] == [
            "███████████████│█████ 46.5 ────────────┤",
            "-1           34.9                     90",
        ]

    def test_a_box_spanning_the_range_carries_no_whisker_labels_and_stays_positional(self):
        assert _plot(0.0, 0.0, 5.0, 10.0, 10.0)[1:] == [
            "████████████████████│███████████████████",
            "0                   5                 10",
        ]

    def test_a_median_mark_on_the_box_edge_survives_the_label_beside_it(self):
        """yspeed's shape: the median and p75 share the box's last cell, and p75 is negative-adjacent."""
        assert _plot(-7.5, -1.0, -0.1, 0.0, 8.6)[1:] == [
            "├─────────── -1 ██│ 0 ─────────────────┤",
            "-7.5            -0.1                 8.6",
        ]

    def test_a_p25_equal_to_the_median_is_labelled_in_both_places(self):
        assert _plot(0.0, 20.0, 20.0, 60.0, 100.0)[1:] == [
            "├─── 20 │███████████████ 60 ───────────┤",
            "0      20                            100",
        ]

    def test_a_box_narrower_than_a_cell_names_its_quartiles(self):
        assert _plot(0.0, 0.2, 0.4, 0.7, 250.0, histogram=[40] + [1] * 39)[1:] == [
            "│──────────────────────────────────────┤",
            "0                                    250",
            "p25 0.2 · p50 0.4 · p75 0.7",
        ]

    def test_a_median_crowding_an_end_names_all_three_even_one_equal_to_that_end(self):
        """The box is plainly visible, but `1` cannot sit beside `0`: the set is named, never split."""
        assert _plot(0.0, 0.0, 1.0, 30.0, 100.0)[1:] == [
            "│████████████──────────────────────────┤",
            "0                                    100",
            "p25 0 · p50 1 · p75 30",
        ]

    def test_a_whisker_too_short_for_its_label_names_all_three(self):
        assert _plot(0.0, 3.0, 40.0, 60.0, 100.0)[1:] == [
            "├███████████████│███████───────────────┤",
            "0                                    100",
            "p25 3 · p50 40 · p75 60",
        ]

    def test_a_record_too_short_to_label_names_its_quartiles(self):
        assert _plot(0.0, 2.0, 4.0, 6.0, 8.0, histogram=[1, 2, 4, 2, 1]) == [
            "▂▄█▄▂",
            "├█│█┤",
            "0   8",
            "p25 2 · p50 4 · p75 6",
        ]

    def test_a_column_with_no_span_labels_its_one_value_once(self):
        assert _plot(5.0, 5.0, 5.0, 5.0, 5.0)[2:] == ["5", "p25 5 · p50 5 · p75 5"]

    def test_a_range_too_wide_for_the_room_names_all_five(self):
        lines = _plot(1.787e15, 1.7872e15, 1.7875e15, 1.7879e15, 1.789e15, room=30)
        assert lines[2:] == [
            "min 1787000000000000",
            "p25 1787200000000000",
            "p50 1787500000000000",
            "p75 1787900000000000",
            "max 1789000000000000",
        ]

    def test_a_named_line_wraps_between_items(self):
        lines = _plot(0.0, 0.2, 0.4, 0.7, 250.0, histogram=[40] + [1] * 19, room=20)
        assert lines[3:] == ["p25 0.2 · p50 0.4", "p75 0.7"]

    def test_an_item_longer_than_the_room_overflows_whole(self):
        lines = _plot(1.787e15, 1.7872e15, 1.7875e15, 1.7879e15, 1.789e15, room=10)
        assert lines[:2] == ["██████████", "├█│██────┤"]
        assert lines[2:] == [
            "min 1787000000000000",
            "p25 1787200000000000",
            "p50 1787500000000000",
            "p75 1787900000000000",
            "max 1789000000000000",
        ]

    @pytest.mark.parametrize(("cells", "room", "drawn"), [(40, 76, 40), (160, 76, 76), (160, 200, 160)])
    def test_a_record_is_merged_to_the_room_but_never_stretched(self, cells, room, drawn):
        lines = _plot(0.0, 25.0, 50.0, 75.0, 100.0, histogram=[1] * cells, room=room)
        assert {len(line) for line in lines} == {drawn}

    def test_an_empty_histogram_draws_nothing(self):
        assert _plot(0.0, 1.0, 2.0, 3.0, 4.0, histogram=[0, 0]) is None


class TestRenderConfigSection:
    def test_full_config(self):
        """Full config renders all top-level keys."""
        resolved = {
            "sources": [{"name": "src", "dataset": "ds"}],
            "workflow": {"name": "clean", "type": "data-cleaning"},
            "extractor": {"name": "ext", "model": "onnx"},
        }
        lines = _config(resolved)
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
        lines = _config(resolved)
        text = "\n".join(lines)
        assert "method: kneighbors" in text
        assert "k: 10" in text
        assert "method: domain_classifier" in text

    def test_int_lists_compacted(self):
        """Contiguous int lists are replaced with range shorthand."""
        resolved = {"workflow": {"indices": [0, 1, 2, 3, 4]}}
        lines = _config(resolved)
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
        lines = _config(resolved)
        text = "\n".join(lines)
        assert "view: subset" in text
        assert "type: Limit" in text
        assert "size: 1000" in text


# ---------------------------------------------------------------------------
# flow_repr
# ---------------------------------------------------------------------------


class TestFlowRepr:
    def test_scalar(self):
        """Scalars render as str()."""
        assert flow_repr("hello") == "hello"
        assert flow_repr(3.14) == "3.14"
        assert flow_repr(True) == "True"

    def test_dict(self):
        """Dicts render as {k: v} without quotes."""
        assert flow_repr({"a": 1, "b": 2}) == "{a: 1, b: 2}"

    def test_list(self):
        """Lists render as [v1, v2]."""
        assert flow_repr(["dim", "pixel"]) == "[dim, pixel]"

    def test_nested(self):
        """Nested structures render inline."""
        result = flow_repr({"params": {"size": 100}})
        assert result == "{params: {size: 100}}"

    def test_int_list_compacted(self):
        """Contiguous int lists collapse to range()."""
        assert flow_repr([0, 1, 2, 3, 4]) == "range(0, 5)"

    def test_non_contiguous_int_list(self):
        """Non-contiguous int lists render normally."""
        assert flow_repr([1, 3, 7]) == "[1, 3, 7]"

    @pytest.mark.parametrize("flags", [[True, False], [False, True]])
    def test_a_list_of_flags_is_not_read_as_a_range(self, flags):
        """A bool is an int to Python, but two flags are not a run of indices."""
        assert flow_repr(flags) == str(flags)


class TestRatioLine:
    def test_a_first_part_of_zero_draws_an_empty_track(self):
        """A bar floored at one block would show a share the first part does not have."""
        line = ratio_line([("numeric", 0), ("text", 5), ("bool", 5)])
        assert line.startswith("░" * 20)
        assert "█" not in line


# ---------------------------------------------------------------------------
# format_value
# ---------------------------------------------------------------------------


class TestFormatValue:
    def test_dict_inline(self):
        """Short dict values render inline."""
        lines: list[str] = []
        format_value(lines, {"threshold": 2.5}, indent=4, max_width=80)
        assert lines == ["    threshold: 2.5"]

    def test_dict_expanded(self):
        """Dict value that exceeds width expands to block style."""
        lines: list[str] = []
        long_val = {"a" * 40: "b" * 40}
        format_value(lines, {"params": long_val}, indent=0, max_width=50)
        text = "\n".join(lines)
        assert "params:" in text
        assert "a" * 40 in text

    def test_list_inline(self):
        """Short list items render inline."""
        lines: list[str] = []
        format_value(lines, [{"method": "knn", "k": 5}], indent=4, max_width=80)
        assert lines == ["    - {method: knn, k: 5}"]

    def test_list_expanded(self):
        """Long list items expand to block style."""
        lines: list[str] = []
        format_value(lines, [{"method": "a" * 60}], indent=4, max_width=40)
        text = "\n".join(lines)
        assert "    -" in text
        assert "method:" in text

    def test_scalar(self):
        """Plain scalar renders with indent."""
        lines: list[str] = []
        format_value(lines, "hello", indent=4, max_width=80)
        assert lines == ["    hello"]


# ---------------------------------------------------------------------------
# compact_indices
# ---------------------------------------------------------------------------


class TestCompactIndices:
    def test_empty_list(self):
        """Empty list returns '[]'."""
        assert compact_indices([]) == "[]"

    def test_single_element(self):
        """Single element returns str(list)."""
        assert compact_indices([42]) == "[42]"

    def test_contiguous_range(self):
        """Contiguous range collapses to range()."""
        assert compact_indices([5, 6, 7, 8, 9]) == "range(5, 10)"

    def test_range_with_step(self):
        """Range with step collapses to range(start, stop, step)."""
        assert compact_indices([0, 2, 4, 6]) == "range(0, 7, 2)"

    def test_non_contiguous(self):
        """Non-contiguous list returns str(list)."""
        result = compact_indices([1, 3, 7])
        assert result == "[1, 3, 7]"

    def test_zero_step(self):
        """Repeated elements (step=0) returns str(list) (line 505)."""
        result = compact_indices([5, 5, 5])
        assert result == "[5, 5, 5]"


# ---------------------------------------------------------------------------
# format_value — non-dict list fallback
# ---------------------------------------------------------------------------


class TestFormatValueListFallback:
    def test_non_dict_list_item_exceeds_width(self):
        """List item that is not a dict and exceeds width triggers fallback (lines 477-478)."""
        lines: list[str] = []
        long_item = "a" * 80
        format_value(lines, [long_item], indent=4, max_width=40)
        text = "\n".join(lines)
        assert "a" * 80 in text
        assert "    -" in text
