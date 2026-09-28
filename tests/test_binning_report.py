"""The METADATA FACTORS section and the factor charts, built as blocks and drawn as text."""

from typing import Any

import pytest

from dataeval_flow._binning_report import (
    _MAX_ENUMERATED,
    _factor_detail,
    _factor_table,
    _record_blocks,
    _review_state,
    _split_comparability,
    binning_blocks,
    distribution_blocks,
    proportion_block,
)
from dataeval_flow._blocks import Fields, Section
from dataeval_flow._blocks._draw import BAR_CELLS as _BAR_MAX
from dataeval_flow._blocks._text import DEFAULT_WIDTH as _WIDTH
from dataeval_flow._blocks._text import Frame, render_text

pytestmark = pytest.mark.required


def _factor_lines(name: str, info: dict[str, Any]) -> list[str]:
    """One factor's detail as the Per-factor detail section draws it, four columns in."""
    return render_text([_factor_detail(name, info)], Frame(indent="    ", depth=3))


def _table_lines(record: dict[str, Any]) -> list[str]:
    return render_text(_factor_table(record), Frame(indent="  ", depth=2))


def _record_lines(record: dict[str, Any], *, detailed: bool = False) -> list[str]:
    return render_text(_record_blocks(record, detailed=detailed), Frame(indent="  ", depth=2))


def _section(binning: Any, diagnostics: list[str] | tuple[str, ...] = (), *, detailed: bool = False) -> list[str]:
    return render_text(binning_blocks(binning, diagnostics, detailed=detailed), Frame(indent="  ", depth=1))


def _comparability(per_split: dict[str, Any]) -> list[str]:
    return render_text(_split_comparability(per_split), Frame(indent="  ", depth=2))


def _review(record: dict[str, Any]) -> list[str]:
    items = _review_state(record)
    return render_text([Fields(items=items)], Frame(indent="  ", depth=2)) if items else []


def _distribution(info: dict[str, Any]) -> list[str]:
    """Each chart block on its own lines with no blank between, as triage prints them."""
    return [line for block in distribution_blocks(info) for line in render_text([block])]


def _ratio(counts: dict[str, int]) -> str:
    return "".join(render_text([proportion_block(counts)]))


# ---------------------------------------------------------------------------
# per-factor detail
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


def test_the_section_is_titled_as_a_finding_is():
    """In title case, as every finding is; the text report capitalizes each section's title alike."""
    (section,) = binning_blocks(_record(), ("a diagnostic",), detailed=False)
    assert isinstance(section, Section)
    assert section.title == "Metadata Factors"
    assert _section(_record(), ("a diagnostic",))[1] == "  METADATA FACTORS"


class TestFactorDetailLevels:
    def test_enumerates_at_threshold(self):
        """A factor with exactly _MAX_ENUMERATED levels still lists every one."""
        info = _digitized([(f"v{i}", i + 1) for i in range(_MAX_ENUMERATED)])
        lines = _factor_lines("sensor", info)
        assert lines[0] == f"    sensor [categorical @ unit] — {_MAX_ENUMERATED} levels, derived"
        assert len(lines) == 1 + 2 + _MAX_ENUMERATED  # the title, a header and rule, then one row per level
        assert lines[3].split() == ["v0", "0", "1"]

    def test_small_factor_lists_every_level_with_its_provenance(self):
        lines = _factor_lines("sensor", _digitized([("a", 15), ("b", 22), ("c", 23)]))
        assert lines == [
            "    sensor [categorical @ unit] — 3 levels, derived",
            "      value  code   n",
            "      -----  ----  --",
            "      a         0  15",
            "      b         1  22",
            "      c         2  23",
        ]

    def test_declared_vocabulary_says_so(self):
        """`declared` is the state a reviewer audits for; `derived` is nobody's decision."""
        lines = _factor_lines("sensor", _digitized([("a", 1)], provenance="declared"))
        assert lines[0] == "    sensor [categorical @ unit] — 1 levels, declared"

    def test_levels_sort_by_value_not_by_code(self):
        """A vocabulary grows append-only, so a late level carries an out-of-order code."""
        info = _digitized([("b", 1), ("c", 2)])
        # `a` arrived after the first structuring and took the next free code.
        info["fit"]["levels"].append({"code": 2, "value": "a", "count": 3})  # type: ignore[index]
        lines = _factor_lines("sensor", info)
        assert [line.split()[0] for line in lines[3:]] == ["a", "b", "c"]

    def test_collapses_above_threshold_with_spread(self):
        info = _digitized([(f"c{i}", 3 + (i % 17)) for i in range(40)])
        lines = _factor_lines("klass", info)
        assert lines == ["    klass [categorical @ unit] — 40 levels, derived, n=3–19 per level"]

    def test_uniform_population_reports_single_count(self):
        lines = _factor_lines("klass", _digitized([(f"c{i}", 5) for i in range(20)]))
        assert lines == ["    klass [categorical @ unit] — 20 levels, derived, n=5 per level"]

    def test_identifier_factor_flagged(self):
        """One level per sample is called out instead of dumping every filename."""
        lines = _factor_lines("file_name", _digitized([(f"{i:05d}.jpg", 1) for i in range(250)]))
        assert lines == ["    file_name [categorical @ unit] — 250 levels, derived (one per sample)"]

    def test_identifier_flag_needs_high_cardinality(self):
        """A genuinely tiny all-singleton factor is still enumerated rather than flagged."""
        lines = _factor_lines("pair", _digitized([("a", 1), ("b", 1)]))
        assert lines[0] == "    pair [categorical @ unit] — 2 levels, derived"
        assert [line.split() for line in lines[3:]] == [["a", "0", "1"], ["b", "1", "1"]]

    def test_unreadable_fit_renders_the_policy_alone(self):
        """A companion column renamed upstream costs the counts, not the record."""
        info = _digitized([("a", 1)])
        del info["fit"]
        assert _factor_lines("sensor", info) == ["    sensor [categorical @ unit] — derived"]

    def test_no_encoding_at_all(self):
        lines = _factor_lines("sensor", {"type": "categorical", "level": "unit"})
        assert lines == ["    sensor [categorical @ unit] — not encoded"]


class TestFactorDetailBins:
    def test_names_bins_from_the_record_not_from_their_contents(self):
        """Regression: a declared cutoff never reached its own label.

        `{"temp_c": [-inf, 0.0, inf]}` used to render as `[-40, -0.3]`: a fact about the
        sample, printed where the declared edges belonged.
        """
        info = _binned(["-inf", 0.0, "inf"], {1: (30, -40.0, -0.3), 2: (30, 0.1, 25.0)}, "edges", None)
        lines = _factor_lines("temp_c", info)
        assert lines[0] == "    temp_c [continuous @ unit] — 2 bins, edges declared"
        assert lines[3].split()[:2] == ["<", "0"]
        assert lines[4].split()[:2] == [">=", "0"]
        # The observed span is reported beside the name, never as it.
        assert lines[3].endswith("[-40, -0.3]")

    def test_falls_back_to_the_code_where_the_record_carries_no_name(self):
        """A release without the accessor costs the labels, not the report."""
        info = _binned(["-inf", 0.0, "inf"], {1: (1, -1.0, -0.5)}, "edges", None, names={})
        lines = _factor_lines("temp_c", info)
        assert lines[3].split()[0] == "1"

    def test_reports_a_declared_bin_nothing_reached(self):
        """An empty declared bin is what a locked policy no longer fitting looks like."""
        info = _binned(["-inf", 0.0, 10.0, "inf"], {3: (60, 12.9, 25.07)}, "edges", None)
        lines = _factor_lines("temp_c", info)
        assert lines[0] == "    temp_c [continuous @ unit] — 3 bins, edges declared, 2 empty"
        assert lines[3].endswith("empty")
        assert lines[5].endswith("[12.9, 25.07]")

    def test_derived_cut_names_the_method_that_placed_it(self):
        info = _binned(["-inf", 50.0, "inf"], {1: (30, 1.0, 49.0), 2: (30, 51.0, 99.0)})
        assert _factor_lines("elevation", info)[0] == (
            "    elevation [continuous @ unit] — 2 bins, derived (uniform_width)"
        )

    def test_interior_bins_render_as_half_open_intervals(self):
        info = _binned(["-inf", 0.0, 10.0, "inf"], {2: (5, 1.0, 9.0)}, "edges", None)
        lines = _factor_lines("temp_c", info)
        assert lines[4].split()[:2] == ["[0,", "10)"]

    def test_large_magnitudes_keep_the_digits_that_distinguish_them(self):
        """Four significant figures printed every epoch-millisecond span identically."""
        base = 1787011240000000.0
        info = _binned(["-inf", base, "inf"], {2: (5, base + 1788.0, base + 191686.0)}, "edges", None)
        lines = _factor_lines("capture_ms", info)
        assert "1.787e+15" not in lines[4]
        assert lines[4].endswith("[1787011240001788, 1787011240191686]")

    def test_collapses_above_threshold_with_occupied_span(self):
        """Many bins collapse to the count and the span the bins actually covered."""
        edges: list[object] = ["-inf", *[float(i) for i in range(1, 100)], "inf"]
        info = _binned(edges, {i: (2, float(i), i + 0.5) for i in range(1, 101)})
        lines = _factor_lines("elevation", info)
        heading = "elevation [continuous @ unit] — 100 bins, derived (uniform_width), [1, 100.5] occupied"
        assert " ".join(line.strip() for line in lines) == heading
        assert all(len(line) <= _WIDTH for line in lines)

    def test_unreadable_fit_renders_the_policy_alone(self):
        info = _binned(["-inf", 0.0, "inf"], {1: (1, 0.0, 1.0)}, "edges", None)
        del info["fit"]
        assert _factor_lines("temp_c", info) == ["    temp_c [continuous @ unit] — edges declared"]


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
        rows = _table_lines(record)
        assert rows[0].split() == ["factor", "lvl", "bins", "shape", "(low", "\u2192", "high)", "n/bin", "range"]
        assert set(rows[1]) == {"-", " "}
        assert [line.split()[0] for line in rows[2:]] == ["height", "width"]

    def test_occupancy_range_counts_a_bin_nothing_reached_as_zero(self):
        """A cut with gaps whose range skipped its own empty bins hid the gap."""
        info = _binned(["-inf", 0.0, 10.0, "inf"], {2: (61, 1.0, 9.0)}, "edges", None, hist=[0, 61, 0])
        row = _table_lines(_record(temp_c=info))[2]
        assert "0\u201361" in row

    def test_shape_is_drawn_from_the_record_not_from_the_cut(self):
        """Two bins over a bimodal column still show both modes."""
        hist = [9, *([0] * 38), 9]
        info = _binned(["-inf", 5.0, "inf"], {1: (9, 0.0, 1.0), 2: (9, 9.0, 10.0)}, "count", hist=hist)
        (row,) = _table_lines(_record(temp_c=info))[2:]
        assert "\u2588" + " " * 38 + "\u2588" in row

    def test_a_finer_record_merges_into_the_same_shape_column(self):
        """Recorded at four times the column's forty cells, the shape merges four to one and loses no mode."""
        hist = [9, *([0] * 158), 9]
        info = _binned(["-inf", 5.0, "inf"], {1: (9, 0.0, 1.0), 2: (9, 9.0, 10.0)}, "count", hist=hist)
        (row,) = _table_lines(_record(temp_c=info))[2:]
        assert "\u2588" + " " * 38 + "\u2588" in row

    def test_a_declared_edge_list_is_printed_because_the_row_cannot_imply_it(self):
        """A bin *count* places edges uniformly across the span, so bins and range recover
        them. A verbatim edge list is arbitrary by construction, and nothing else says where
        it fell. The policy exists to make that decision visible."""
        info = _binned(
            ["-inf", 0.0, 10.0, "inf"],
            {1: (5, -3.0, -1.0), 2: (60, 1.0, 9.0), 3: (5, 11.0, 20.0)},
            "edges",
            None,
            hist=[5, 60, 5],
        )
        assert "  temp_c edges: 0, 10" in _table_lines(_record(temp_c=info))

    def test_a_count_declared_cut_carries_no_edge_line(self):
        assert not any("edges:" in line for line in _table_lines(_record(height=self._height())))

    def test_a_long_edge_list_wraps_within_the_report_width(self):
        """Every declared edge stays on the page; the list wraps under itself rather than overrunning."""
        edges: list[object] = ["-inf", *[float(i) for i in range(40)], "inf"]
        info = _binned(edges, {1: (5, -1.0, 0.0)}, "edges", None)
        lines = _table_lines(_record(elevation=info))
        assert all(len(line) <= _WIDTH for line in lines), max(lines, key=len)
        start = next(i for i, line in enumerate(lines) if "elevation edges:" in line)
        listed = " ".join(lines[start:]).split(":", 1)[1]
        assert [edge.strip() for edge in listed.split(",")] == [str(i) for i in range(40)]

    def test_a_digitized_factor_reports_its_levels_and_no_numeric_span(self):
        """A vocabulary has no extremes to report, so the column stays empty rather than
        inventing an ordering for values that have none."""
        row = _table_lines(_record(sensor=_digitized([("a", 15), ("b", 22), ("c", 23)])))[2]
        assert row.startswith("  sensor  unit")
        assert row.endswith("15\u201323")

    def test_a_record_without_a_histogram_falls_back_to_its_buckets(self):
        """A result written before the shape was recorded still draws something true."""
        info = _binned(["-inf", 5.0, "inf"], {1: (1, 0.0, 1.0), 2: (99, 6.0, 9.0)}, "count")
        row = _table_lines(_record(temp=info))[2]
        assert "\u2588" in row
        assert "\u2581" in row

    def test_a_factor_that_was_never_encoded_stays_out_of_the_table(self):
        """Nothing to compare: no cut, no shape, no occupancy."""
        assert _table_lines(_record(sensor={"type": "categorical", "level": "unit"})) == []

    def test_the_shape_column_gives_up_cells_before_the_table_gives_up_its_width(self):
        """A span like `1.15e-06 – 0.07204` is nineteen characters beside a nineteen
        character factor name. The shape is the one column that can shrink without the
        table losing a fact. While it does, it stays one width across every row."""
        record = _record(
            instance_brightness=self._height(),
            unit_zeros=_binned(
                ["-inf", 0.5, "inf"],
                {1: (4, 1.145e-06, 0.05), 2: (123, 0.06, 0.07204)},
                "count",
                hist=[1] * 40,
            ),
        )
        lines = _table_lines(record)
        assert all(len(line) <= _WIDTH for line in lines), max(len(line) for line in lines)
        # The span is still printed in full: the fit came out of the shape, not the facts.
        assert "1.15e-06 \u2013 0.07204" in lines[3]

    def test_row_carries_level_bin_count_and_observed_span(self):
        (row,) = _table_lines(_record(height=self._height()))[2:]
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
        lines = _record_lines(self._record_with_height())
        assert any("n/bin" in line for line in lines)
        assert not any(line.startswith("    height [") for line in lines)

    def test_section_passes_the_detail_request_to_every_split(self):
        binning = {"per_split": {"train": self._record_with_height(), "test": self._record_with_height()}}
        assert sum(1 for line in _section(binning, detailed=True) if "Per-factor detail" in line) == 2
        assert not any("Per-factor detail" in line for line in _section(binning))

    def test_each_split_is_set_off_from_the_one_above_it(self):
        """Two tables running together read as one table with a stray heading in it."""
        binning = {"per_split": {"train": self._record_with_height(), "test": self._record_with_height()}}
        lines = _section(binning)
        assert lines[lines.index("  [test]") - 1] == ""

    def test_every_split_table_shares_one_column_layout(self):
        """Splits drawn at two resolutions cannot be read against each other. That is the
        one thing this section promises about them when they share an encoding. Two tables
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
        headers = [line for line in _section(binning) if "n/bin" in line]
        assert len(headers) == 2
        assert len(set(headers)) == 1

    def test_detailed_keeps_the_full_breakdown_under_the_table(self):
        lines = _record_lines(self._record_with_height(), detailed=True)
        table_at = next(i for i, line in enumerate(lines) if "n/bin" in line)
        detail_at = next(i for i, line in enumerate(lines) if line.startswith("    height ["))
        assert table_at < detail_at
        assert any(line.endswith("[52, 223]") for line in lines)


class TestDiagnostics:
    """Library prose printed into a report that has a width."""

    def test_a_long_diagnostic_wraps_to_the_report_width(self):
        """A diagnostic has no length contract; the report around it does. Left unwrapped,
        one sentence drags the whole block sideways. In the documentation the block scrolls,
        so it takes the factor table with it."""
        message = "dataeval: Declared cuts left bins unused: " + ", ".join(
            f"factor_{i} ({i} of 40 bins hold rows)" for i in range(10)
        )
        lines = _section(None, [message])
        assert all(len(line) <= _WIDTH for line in lines), max(len(line) for line in lines)
        body = [line for line in lines if line.startswith("    ")]
        assert len(body) > 1, "a message this long has to occupy more than one line"
        assert body[1].startswith("      "), "continuations indent under the message they belong to"

    def test_a_short_diagnostic_stays_on_one_line(self):
        lines = _section(None, ["dataeval: nothing to report"])
        assert "    - dataeval: nothing to report" in lines


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
        lines = _comparability({"train": split, "test": split})
        assert any("comparable across them" in line and "NOT" not in line for line in lines)

    def test_a_grown_vocabulary_is_still_comparable(self):
        """Append-only growth leaves every shared code meaning what it meant.

        Reporting this as a disagreement would flag the ordinary case of one split holding
        a level another lacks. That false alarm teaches people to ignore the true ones.
        """
        lines = _comparability(
            {
                "train": self._split(sensor=self._levels("a", "b")),
                "test": self._split(sensor=self._levels("a", "b", "c")),
            }
        )
        assert any("comparable across them" in line and "NOT" not in line for line in lines)

    def test_a_reordered_vocabulary_is_not_comparable(self):
        """Same values, different codes — the one thing append-only ordering prevents."""
        lines = _comparability(
            {
                "train": self._split(sensor=self._levels("a", "b")),
                "test": self._split(sensor=self._levels("b", "a")),
            }
        )
        assert any("NOT comparable" in line for line in lines)

    def test_different_cuts_are_not_comparable(self):
        lines = _comparability(
            {
                "train": self._split(temp_c=self._bins("-inf", 0.0, "inf")),
                "test": self._split(temp_c=self._bins("-inf", 5.0, "inf")),
            }
        )
        assert any("NOT comparable" in line for line in lines)

    def test_names_only_the_factors_that_differ(self):
        lines = _comparability(
            {
                "train": self._split(temp_c=self._bins("-inf", 0.0, "inf"), sensor=self._levels("a")),
                "test": self._split(temp_c=self._bins("-inf", 5.0, "inf"), sensor=self._levels("a")),
            }
        )
        verdict = next(line for line in lines if "NOT comparable" in line)
        assert "temp_c" in verdict
        assert "sensor" not in verdict

    def test_three_splits_that_each_extend_a_third_are_not_comparable(self):
        """Append-only agreement is not transitive. Comparing only to the first misses it.

        `test` and `val` each merely extend `train`, so pairwise-against-the-first says
        comparable. Code 1 means `b` in one and `c` in the other. That is the false
        reassurance this section exists to prevent.
        """
        lines = _comparability(
            {
                "train": self._split(sensor=self._levels("a")),
                "test": self._split(sensor=self._levels("a", "b")),
                "val": self._split(sensor=self._levels("a", "c")),
            }
        )
        assert any("NOT comparable" in line for line in lines)

    def test_a_grown_vocabulary_does_not_claim_one_shared_digest(self):
        """The line must not contradict the `encoding_digest` printed above it.

        Growth appends, so the codes still agree. The digests differ, so the envelope's
        `encoding_digest` is then None. Saying "share one encoding" over that is a report
        disagreeing with its own record.
        """
        train = {**self._split(sensor=self._levels("a", "b")), "encoding_digest": "aaaa"}
        test = {**self._split(sensor=self._levels("a", "b", "c")), "encoding_digest": "bbbb"}
        lines = _comparability({"train": train, "test": test})

        assert any("comparable" in line and "NOT" not in line for line in lines)
        assert not any("share one encoding" in line for line in lines)

    def test_one_split_says_nothing(self):
        assert _comparability({"train": self._split(a=self._levels("x"))}) == []

    def test_no_encodings_says_nothing(self):
        assert _comparability({"train": self._split(), "test": self._split()}) == []

    def test_the_section_carries_the_digest(self):
        section = _section({"encoding_digest": "2b6530cc3015f1fe", "factors": {}, "dropped": {}})
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
        lines = _review(self._record(["b"], ["a", "b"]))
        assert "1 of 2 factors still derived" in lines[0]
        assert "(b)" in lines[1]

    def test_says_so_when_everything_was_decided(self):
        lines = _review(self._record([], ["a", "b"]))
        assert lines == ["  Policy: all 2 factors declared or reviewed"]

    def test_says_nothing_when_no_factor_was_encoded(self):
        assert _review(self._record([], [])) == []

    def test_says_nothing_for_a_record_written_before_this_was_tracked(self):
        record = self._record([], ["a"])
        del record["unreviewed"]
        assert _review(record) == []


# ---------------------------------------------------------------------------
# distribution_blocks / proportion_block
# ---------------------------------------------------------------------------


def _binned_counts(counts: list[int], empty: list[int] | None = None) -> dict:
    """A factor entry whose fit holds the given per-bin counts.

    Named distinctly from the module-level ``_binned`` fixture, which builds a factor
    from explicit edges and per-bucket spans. Both live at module scope; reusing the
    name would silently rebind it out from under the earlier tests.
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
    """ "Empty" against "one sample landed here" is what a bin count is argued from.

    A bar that rounds the second down to the first argues for the wrong answer. This
    isolates the bar cell itself. The row's label alone would read non-blank regardless
    of what the bar drew.
    """
    lines = _distribution(_binned_counts([1000, 1]))
    # Row layout is `{label:>w}  {bar:<_BAR_MAX}  {count:>5}{note}`. The bucket with
    # count=1 carries no "empty" note, so its row's fixed-width tail (bar + gutter +
    # count) can be sliced from the end regardless of the label column's width.
    row = lines[-1]
    bar_cell = row[-(5 + 2 + _BAR_MAX) : -(5 + 2)]
    assert bar_cell.strip()


def test_an_empty_bin_is_marked_and_drawn_blank():
    lines = "\n".join(_distribution(_binned_counts([50, 0, 50], empty=[2])))
    assert "empty" in lines


def test_the_count_is_always_printed():
    lines = "\n".join(_distribution(_binned_counts([412, 7])))
    assert "412" in lines
    assert "7" in lines


def test_a_wide_factor_falls_back_to_a_sparkline():
    lines = _distribution(_binned_counts([10] * 30))
    # One sparkline line, no per-bucket enumeration above the cap.
    assert len(lines) == 1


def test_a_narrow_factor_gets_both():
    lines = _distribution(_binned_counts([10, 20, 30]))
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
    lines = "\n".join(_distribution(info))
    assert "a" in lines
    assert "30" in lines


def test_a_missing_fit_renders_nothing():
    assert _distribution({"type": "continuous", "level": "unit"}) == []


def test_the_ratio_bar_shows_both_kinds():
    line = _ratio({"numeric": 1842, "text": 58})
    assert "1,842 numeric" in line
    assert "58 text" in line


def test_a_long_sparkline_does_not_run_into_its_count():
    """A factor with more buckets than the pad still needs a gap before `n=`."""
    lines = _distribution(_binned_counts([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]))
    assert "n=" in lines[0]
    assert " n=" in lines[0], f"sparkline runs into the count: {lines[0]!r}"


def test_a_single_kind_column_gets_counts_without_an_empty_bar():
    """A bar comparing one kind to nothing renders as an empty track that reads as zero."""
    line = _ratio({"text": 200})
    assert line == "200 text"
    assert "░" not in line


def test_two_kinds_still_get_a_bar():
    line = _ratio({"numeric": 198, "text": 2})
    assert "\u2588" in line
    assert "\u2591" in line
    assert "198 numeric" in line
    assert "2 text" in line


def test_a_thin_minority_still_shows_in_the_ratio_bar():
    """198 against 2 is 19.8 blocks; rounding it up hides the reason the line is printed."""
    line = _ratio({"numeric": 198, "text": 2})
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
    lines = _distribution(_shaped(q, [60, 20, 10, 5, 3, 2, 1, 1]))
    # Eight recorded cells leave the median no room beside -1, so the quartiles are named.
    assert lines == ["█▃▁▁▁▁▁▁", "█│─────┤", "-1   240", "p25 -1 · p50 30 · p75 42"]


def test_a_skewed_box_never_collapses_to_bare_whiskers():
    """p25 and p75 within a fraction of a cell still leave a visible box."""
    q = {"0.0": 110.0, "0.25": 900.0, "0.5": 1806.0, "0.75": 5508.0, "1.0": 680264.0}
    lines = _distribution(_shaped(q, [80, 5, 2, 1, 0, 0, 0, 1]))
    # Where the box collapses onto the median, the median mark is the box. A line of
    # bare whiskers must not remain, which reads as broken.
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
    lines = "\n".join(_distribution(info))
    assert "m210" in lines
    assert "p25" not in lines
