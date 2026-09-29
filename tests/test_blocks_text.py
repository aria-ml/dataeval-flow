"""The text renderer: headings by depth, prose that wraps to the width it inherits, charts drawn from data."""

from typing import get_args

import pytest

from dataeval_flow._blocks import (
    Block,
    BulletList,
    Code,
    Distribution,
    Fields,
    Paragraph,
    Proportion,
    Quantiles,
    Section,
    Summary,
    SummaryItem,
    Tree,
)
from dataeval_flow._blocks._text import DEFAULT_WIDTH, DRAW, MIN_WIDTH, Frame, render_text

pytestmark = pytest.mark.required

_LONG = (
    "The uncovered rate is not health-checked: coverage_method='adaptive' flags a fixed coverage_percent "
    "of observations by construction, so the per-class columns are the data-driven signal to read instead."
)


def test_every_block_type_has_a_text_drawing():
    members = {member.model_fields["type"].default for member in get_args(get_args(Block)[0])}
    assert members == set(DRAW)


class TestSiblings:
    def test_siblings_are_separated_by_one_blank_line(self):
        lines = render_text([Paragraph(text="one"), Paragraph(text="two")])
        assert lines == ["one", "", "two"]

    def test_a_block_that_draws_nothing_leaves_no_gap(self):
        lines = render_text([Paragraph(text="one"), Proportion(parts=[]), Paragraph(text="two")])
        assert lines == ["one", "", "two"]

    def test_no_line_keeps_trailing_whitespace(self):
        lines = render_text([Code(text="a   \n\nb  ")], Frame(indent="  "))
        assert lines == ["  a", "", "  b"]


class TestSections:
    def test_the_root_is_a_banner_around_its_content(self):
        lines = render_text([Section(title="Data Cleaning\nsplit: train", blocks=[Paragraph(text="body")])])
        rule = "=" * DEFAULT_WIDTH
        assert lines == ["", rule, "  DATA CLEANING", "  SPLIT: TRAIN", rule, "  body", "", rule]

    def test_a_top_level_section_has_rules_and_a_right_aligned_brief(self):
        section = Section(title="Duplicates", brief="3 groups", blocks=[Paragraph(text="found some")])
        lines = render_text([section], Frame(indent="  ", depth=1))
        rule = "=" * DEFAULT_WIDTH
        head = "  DUPLICATES" + " " * (DEFAULT_WIDTH - 2 - len("DUPLICATES") - len("3 groups")) + "3 groups"
        assert lines == [rule, head, rule, "  found some"]
        assert len(head) == DEFAULT_WIDTH

    def test_a_brief_too_long_for_the_title_line_moves_under_it(self):
        brief = "b" * (DEFAULT_WIDTH - 5)
        lines = render_text([Section(title="Title", brief=brief)], Frame(indent="  ", depth=1))
        assert lines[1] == "  TITLE"
        assert lines[2] == brief.rjust(DEFAULT_WIDTH)

    def test_a_nested_section_is_a_title_line_with_its_content_indented(self):
        inner = Section(title="latitude", brief="declare 8 bins", blocks=[Paragraph(text="detail")])
        lines = render_text([inner], Frame(indent="  ", depth=2))
        assert lines == ["  latitude \u2014 declare 8 bins", "    detail"]

    def test_width_is_inherited_through_nesting(self):
        deep = Section(
            title="a", blocks=[Section(title="b", blocks=[Section(title="c", blocks=[Paragraph(text=_LONG)])])]
        )
        lines = render_text([deep], Frame(width=60, indent="  ", depth=2))
        prose = [line for line in lines if line.startswith("        ")]
        assert prose
        assert all(len(line) <= 60 for line in lines)


class TestProse:
    def test_a_paragraph_wraps_at_the_frame_width_and_indent(self):
        lines = render_text([Paragraph(text=_LONG)], Frame(width=50, indent="    "))
        assert all(line.startswith("    ") and len(line) <= 50 for line in lines)
        assert " ".join(line.strip() for line in lines) == _LONG

    def test_identifiers_are_never_split(self):
        text = "set coverage_method='adaptive' and data-coverage/label-space-digest together"
        lines = render_text([Paragraph(text=text)], Frame(width=30))
        words = {word for line in lines for word in line.split()}
        assert "coverage_method='adaptive'" in words
        assert "data-coverage/label-space-digest" in words

    def test_a_single_newline_is_a_hard_break(self):
        assert render_text([Paragraph(text="first\nsecond")], Frame(indent="  ")) == ["  first", "  second"]

    def test_a_bullet_wraps_under_its_text(self):
        lines = render_text([BulletList(items=[_LONG])], Frame(width=50, indent="  "))
        assert lines[0].startswith("  - ")
        assert all(line.startswith("    ") for line in lines[1:])
        assert all(len(line) <= 50 for line in lines)


class TestFields:
    def test_values_align_after_the_longest_label(self):
        block = Fields(items=[("Timestamp", "2026-09-25T00:00:00"), ("Preprocessor", "resize")])
        assert render_text([block], Frame(indent="  ")) == [
            "  Timestamp:    2026-09-25T00:00:00",
            "  Preprocessor: resize",
        ]

    def test_a_multi_line_value_hangs_under_the_value_column(self):
        block = Fields(items=[("Source", "train.parquet\ntest.parquet")])
        assert render_text([block], Frame(indent="  ")) == ["  Source: train.parquet", "          test.parquet"]

    def test_a_long_value_wraps_under_the_value_column(self):
        lines = render_text([Fields(items=[("Policy", _LONG)])], Frame(width=50, indent="  "))
        assert all(line.startswith(" " * len("  Policy: ")) for line in lines[1:])
        assert all(len(line) <= 50 for line in lines)

    def test_scalars_render_as_text_and_none_as_empty(self):
        block = Fields(items=[("n", 3), ("ok", True), ("gone", None)])
        assert render_text([block]) == ["n:    3", "ok:   True", "gone:"]


class TestCodeAndTree:
    def test_code_is_verbatim_and_never_wrapped(self):
        long_line = "key: " + "x" * 200
        assert render_text([Code(text=f"a:\n  {long_line}\n")], Frame(indent="  ")) == ["  a:", f"    {long_line}"]

    def test_a_tree_stays_inline_when_it_fits(self):
        lines = render_text([Tree(value={"threshold": 2.5, "indices": [0, 1, 2, 3]})], Frame(indent="  "))
        assert lines == ["  threshold: 2.5", "  indices: range(0, 4)"]

    def test_a_tree_goes_block_style_when_it_does_not_fit(self):
        lines = render_text([Tree(value={"params": {"k" + str(i): i for i in range(20)}})], Frame(width=40))
        assert lines[0] == "params:"
        assert lines[1] == "  k0: 0"


class TestSummary:
    def test_a_dotted_leader_joins_label_to_value_and_marker(self):
        item = SummaryItem(label="Duplicates", value="3 groups", severity="warning")
        line = render_text([Summary(items=[item], warnings=1)], Frame(indent="  "))[0]
        assert line.startswith("  Duplicates ....")
        assert line.endswith(" 3 groups  [!!]")
        assert len(line) == DEFAULT_WIDTH - 2

    @pytest.mark.parametrize(("severity", "marker"), [("ok", "[ok]"), ("info", "[..]"), ("warning", "[!!]")])
    def test_each_severity_has_its_marker(self, severity, marker):
        line = render_text(
            [Summary(items=[SummaryItem(label="x", severity=severity)], warnings=int(severity == "warning"))]
        )[0]
        assert line.endswith(marker)

    def test_a_long_label_wraps_and_its_last_line_carries_the_value(self):
        item = SummaryItem(label=" ".join(["Unpinned categorical vocabularies"] * 4), value="12 factors")
        lines = render_text([Summary(items=[item], warnings=0)], Frame(width=60, indent="  "))
        assert len(lines) > 3
        assert lines[-3].endswith("12 factors  [..]")
        assert all(len(line) <= 60 for line in lines)

    def test_the_lines_end_with_the_health_their_warnings_add_up_to(self):
        items = [SummaryItem(label="a", severity="warning"), SummaryItem(label="b", severity="ok")]
        assert render_text([Summary(items=items, warnings=1)])[-2:] == [
            "",
            "Health: 1 warning(s) [!!] — review flagged findings",
        ]
        assert render_text([Summary(items=items[1:], warnings=0)])[-1] == "Health: All checks passed [ok]"


class TestCharts:
    def test_a_proportion_draws_the_first_parts_share(self):
        (line,) = render_text([Proportion(parts=[("numeric", 198), ("text", 2)])])
        assert line == "\u2588" * 19 + "\u2591" + "  198 numeric, 2 text"

    def test_a_proportion_of_one_kind_lists_counts_without_a_bar(self):
        assert render_text([Proportion(parts=[("text", 1200)])]) == ["1,200 text"]

    def test_a_distribution_without_quantiles_is_a_sparkline_and_its_range(self):
        assert render_text([Distribution(histogram=[3, 0, 12, 7])]) == ["\u2582 \u2588\u2585        n=3\u201312"]

    def test_a_distribution_with_quantiles_is_a_box_plot(self):
        quantiles = Quantiles(low=0.0, q1=25.0, median=50.0, q3=75.0, high=100.0)
        assert render_text([Distribution(histogram=[1] * 40, quantiles=quantiles)]) == [
            "████████████████████████████████████████",
            "├───── 25 ██████████│█████████ 75 ─────┤",
            "0                  50                100",
        ]

    def test_an_empty_histogram_draws_nothing(self):
        assert render_text([Distribution(histogram=[0, 0])]) == []


@pytest.mark.parametrize("width", [40, 80, 120])
def test_long_prose_fits_the_width_in_every_wrapping_block(width):
    blocks = [
        Section(
            title="Findings",
            blocks=[
                Paragraph(text=_LONG),
                BulletList(items=[_LONG]),
                Fields(items=[("Remedy", _LONG)]),
                Section(
                    title="Nested",
                    blocks=[Paragraph(text=_LONG), Summary(items=[SummaryItem(label=_LONG)], warnings=0)],
                ),
            ],
        )
    ]
    lines = render_text(blocks, Frame(width=width, indent="  ", depth=1))
    assert all(len(line) <= width for line in lines), max(lines, key=len)


class TestWidth:
    def test_the_default_is_eighty_columns_and_the_floor_forty(self):
        assert (DEFAULT_WIDTH, MIN_WIDTH) == (80, 40)

    def test_a_box_plot_too_crowded_to_label_names_its_quartiles_within_the_width(self):
        """At triage depth: the box is under a cell wide, so its quartiles are named on their own line."""
        quantiles = Quantiles(low=0.001234, q1=12.5, median=40.25, q3=118.75, high=65432.1)
        block = Distribution(histogram=[(i * 7) % 11 + 1 for i in range(40)], quantiles=quantiles)
        lines = render_text([block], Frame(width=80, indent="      "))
        assert all(len(line) <= 80 for line in lines), max(lines, key=len)
        assert lines[1:] == [
            "      │──────────────────────────────────────┤",
            "      0.001234                           65432",
            "      p25 12.5 · p50 40.25 · p75 118.8",
        ]

    def test_box_plots_share_their_left_edge_whatever_their_labels(self):
        """The range sits beneath the box, so a wide min label no longer pushes the chart right."""
        narrow = Quantiles(low=-1.0, q1=-1.0, median=34.9, q3=46.5, high=90.0)
        wide = Quantiles(low=-11.5, q1=-1.0, median=-0.5, q3=0.0, high=11.1)
        drawn = [
            render_text([Distribution(histogram=[1] * 40, quantiles=q)], Frame(indent="    ")) for q in (narrow, wide)
        ]
        assert drawn[0][0] == drawn[1][0] == "    " + "█" * 40
        assert drawn[0][2].startswith("    -1 ")
        assert drawn[1][2].startswith("    -11.5 ")

    def test_a_box_plot_too_wide_for_its_frame_is_resampled_not_cut(self):
        quantiles = Quantiles(low=0.0, q1=2.0, median=4.0, q3=6.0, high=8.0)
        block = Distribution(histogram=[1] * 40, quantiles=quantiles)
        assert render_text([block], Frame(width=40, indent="      ")) == [
            "      ██████████████████████████████████",
            "      ├──── 2 ████████│█████████ 6 ────┤",
            "      0               4                8",
        ]

    def test_a_finer_record_fills_the_frame(self):
        """A record of 160 cells is merged down to the room: the plot spans the frame, labels and all."""
        quantiles = Quantiles(low=0.0, q1=25.0, median=50.0, q3=75.0, high=100.0)
        lines = render_text([Distribution(histogram=[1] * 160, quantiles=quantiles)], Frame(width=80, indent="    "))
        assert {len(line) for line in lines} == {80}

    @pytest.mark.parametrize("width", [40, 80, 120])
    @pytest.mark.parametrize(
        "quartiles",
        [
            (0.0, 25.0, 50.0, 75.0, 100.0),
            (-1.0, -1.0, 34.9, 46.5, 90.0),
            (0.0, 0.2, 0.4, 0.7, 250.0),
            (0.0, 0.0, 1.0, 30.0, 100.0),
            (1.787e15, 1.7872e15, 1.7875e15, 1.7879e15, 1.789e15),
        ],
    )
    def test_every_box_plot_line_fits_the_width(self, width, quartiles):
        low, q1, median, q3, high = quartiles
        quantiles = Quantiles(low=low, q1=q1, median=median, q3=q3, high=high)
        lines = render_text([Distribution(histogram=[1] * 160, quantiles=quantiles)], Frame(width=width, indent="    "))
        assert all(len(line) <= width for line in lines), max(lines, key=len)

    def test_a_long_sparkline_is_resampled_to_fit(self):
        lines = render_text([Distribution(histogram=list(range(1, 200)))], Frame(width=50, indent="  "))
        assert len(lines) == 1
        assert len(lines[0]) <= 50
        assert lines[0].endswith(" n=1\u2013199")


class TestAwkwardInputs:
    def test_a_token_wider_than_the_line_overflows_whole(self):
        path = "/data/" + "very_long_directory_name/" * 6 + "labels.parquet"
        lines = render_text([Paragraph(text=f"Could not read {path} at all.")], Frame(width=40, indent="  "))
        assert f"  {path}" in lines
        assert " ".join(line.strip() for line in lines) == f"Could not read {path} at all."

    def test_an_unbreakable_summary_label_at_the_narrowest_width_still_renders(self):
        item = SummaryItem(label="x" * 60, value="3 of 3", severity="warning")
        lines = render_text([Summary(items=[item], warnings=1)], Frame(width=40, indent="  "))
        assert lines[lines.index("") - 1].endswith("3 of 3  [!!]")


class TestNarrowWidths:
    """At the narrowest width a caller may ask for, headings and summaries wrap rather than overflow."""

    def test_a_long_banner_title_wraps(self):
        root = Section(title="Data cleaning complete. Dataset: 22 items. Mode: advisory.", blocks=[Paragraph(text="x")])
        lines = render_text([root], Frame(width=40))
        assert all(len(line) <= 40 for line in lines), max(lines, key=len)
        first, second = [i for i, line in enumerate(lines) if line == "=" * 40][:2]
        title = " ".join(line.strip() for line in lines[first + 1 : second])
        assert title == "DATA CLEANING COMPLETE. DATASET: 22 ITEMS. MODE: ADVISORY."

    def test_a_long_top_level_title_and_brief_wrap(self):
        section = Section(
            title="Class distribution across splits (fold 3 of 5)", brief="3 classes, 60 items, imbalance 4.0:1"
        )
        lines = render_text([section], Frame(width=40, indent="  ", depth=1))
        assert all(len(line) <= 40 for line in lines), max(lines, key=len)
        assert "4.0:1" in lines[-2]

    def test_a_long_value_moves_under_its_label(self):
        item = SummaryItem(label="Label Distribution", value="3 classes, 60 items, imbalance 4.0:1", severity="warning")
        lines = render_text([Summary(items=[item], warnings=1)], Frame(width=40, indent="  "))
        assert all(len(line) <= 40 for line in lines), max(lines, key=len)
        assert lines[0] == "  Label Distribution"
        assert lines[lines.index("") - 1].endswith("[!!]")

    @pytest.mark.parametrize("width", [40, 60])
    def test_a_realistic_report_fits(self, width):
        findings = [
            ("Label Distribution", "3 classes, 60 items, imbalance 4.0:1"),
            ("Classwise Outliers Across Every Split In The Dataset", "12 classes over threshold"),
            ("Duplicates", "3 groups (7 images)"),
        ]
        summary = Summary(
            items=[SummaryItem(label=title, value=brief, severity="warning") for title, brief in findings],
            warnings=len(findings),
        )
        root = Section(
            title="Data cleaning complete. Dataset: 22 items. Mode: advisory.",
            blocks=[
                Section(title="SUMMARY", blocks=[summary]),
                *(Section(title=title, brief=brief, blocks=[Paragraph(text=_LONG)]) for title, brief in findings),
            ],
        )
        lines = render_text([root], Frame(width=width))
        assert all(len(line) <= width for line in lines), max(lines, key=len)


class TestNonFiniteDistribution:
    def test_non_finite_quantiles_fall_back_to_the_sparkline(self):
        quantiles = Quantiles(low=float("nan"), q1=1.0, median=2.0, q3=3.0, high=4.0)
        lines = render_text([Distribution(histogram=[1, 3, 2], quantiles=quantiles)])
        assert len(lines) == 1
        assert lines[0].endswith("n=1–3")
