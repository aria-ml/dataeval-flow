"""Report blocks drawn as HTML: one fragment per block, one self-contained page, nothing unescaped."""

import re
from html.parser import HTMLParser
from typing import Any, get_args

import pytest

from dataeval_flow._blocks import (
    Block,
    BulletList,
    Code,
    Column,
    Distribution,
    Fields,
    Flag,
    Paragraph,
    Proportion,
    Quantiles,
    Section,
    Summary,
    SummaryItem,
    Table,
    Tree,
)
from dataeval_flow._blocks._html import DRAW, html_page, render_html
from dataeval_flow._blocks._html_scale import place_labels
from dataeval_flow._blocks._html_style import SCRIPT, STYLE
from tests.test_blocks_models import EVERY_BLOCK

pytestmark = pytest.mark.required

_VOID = {"meta", "br", "rect", "line"}


class _Balance(HTMLParser):
    """Every element opened is closed, in order: what a browser and a PDF engine need to agree on."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if not self.stack or self.stack.pop() != tag:
            self.errors.append(tag)


def _well_formed(text: str) -> bool:
    parser = _Balance()
    parser.feed(text)
    parser.close()
    return not parser.errors and not parser.stack


class TestVocabulary:
    def test_the_draw_table_covers_every_block_type(self):
        union = get_args(get_args(Block)[0])
        assert set(DRAW) == {member.model_fields["type"].default for member in union}

    @pytest.mark.parametrize("block", EVERY_BLOCK, ids=lambda block: block.type)
    def test_every_block_renders_well_formed(self, block):
        fragment = render_html([block])
        assert fragment
        assert _well_formed(fragment)


class TestSections:
    def test_headings_deepen_with_nesting_and_stop_at_h4(self):
        nested = Section(
            title="d0",
            blocks=[
                Section(
                    title="d1", blocks=[Section(title="d2", blocks=[Section(title="d3", blocks=[Section(title="d4")])])]
                )
            ],
        )
        fragment = render_html([nested])
        assert [int(level) for level in re.findall(r"<h(\d)>", fragment)] == [1, 2, 3, 4, 4]

    def test_brief_and_severity_badge(self):
        report = Section(title="Report", blocks=[Section(title="Duplicates", brief="3 groups", severity="warning")])
        fragment = render_html([report])
        assert '<span class="brief">3 groups</span>' in fragment
        assert '<span class="badge warning">warning</span>' in fragment

    def test_a_multi_line_title_breaks_where_it_does_in_text(self):
        assert "<h1>Report<br>Second line</h1>" in render_html([Section(title="Report\nSecond line")])


def _report(*findings: Section, title: str = "Data cleaning") -> Section:
    """A report as a result builds one: its envelope, the summary linking each finding, the findings, the config."""
    items = [SummaryItem(label=f.title, value=f.brief or "", severity=f.severity or "info") for f in findings]
    return Section(
        title=title,
        blocks=[
            Fields(items=[("Dataset", "cifar10-train"), ("Duration", "4.1s")]),
            Section(title="SUMMARY", blocks=[Summary(items=items), Paragraph(text="Health: 1 warning(s)")]),
            *findings,
            Section(title="CONFIGURATION", blocks=[Tree(value={"seed": 1})]),
        ],
    )


_DUPLICATES = Section(title="Duplicates", brief="3 groups", severity="warning", blocks=[Paragraph(text="3 groups.")])
_LABELS = Section(title="Label Distribution", brief="10 classes", severity="ok")


class TestLayout:
    """A report reads as a header, then a card per finding, then its reference sections folded away."""

    def test_the_header_holds_the_title_and_the_health(self):
        fragment = render_html([_report(_DUPLICATES, _LABELS)])
        assert (
            '<header class="report-head"><h1>Data cleaning</h1><span class="badge warning">1 warning</span></header>'
            in fragment
        )

    def test_a_report_with_no_warning_has_passed(self):
        assert '<span class="badge ok">passed</span></header>' in render_html([_report(_LABELS)])

    def test_a_short_report_takes_its_verdict_from_its_summary(self):
        """`to_html(detailed=False)` has the summary but no cards: its lines carry the same verdicts."""
        short = _report(_DUPLICATES, _LABELS)
        short = short.model_copy(update={"blocks": [b for b in short.blocks if b not in (_DUPLICATES, _LABELS)]})
        fragment = render_html([short])
        assert '<span class="badge warning">1 warning</span></header>' in fragment
        assert '<table class="summary">' in fragment, "with no cards, the summary is the report"

    def test_a_report_without_findings_has_no_health_badge(self):
        fragment = render_html([Section(title="Evaluator", blocks=[Paragraph(text="12 rows")])])
        assert '<header class="report-head"><h1>Evaluator</h1></header>' in fragment

    def test_the_envelope_is_the_provenance_line(self):
        fragment = render_html([_report(_LABELS)])
        assert '<dl class="fields provenance"><dt>Dataset</dt><dd>cifar10-train</dd>' in fragment

    def test_each_finding_is_a_card_that_opens_on_its_title(self):
        fragment = render_html([_report(_DUPLICATES)])
        assert (
            '<details class="card warning" id="duplicates" open><summary><h2>Duplicates '
            '<span class="brief">3 groups</span> <span class="badge warning">warning</span></h2></summary>'
            "\n<p>3 groups.</p></details>"
        ) in fragment

    def test_the_cards_stand_for_the_summary(self):
        """Each card's title, brief and badge already say what its summary line says, and the header the health."""
        fragment = render_html([_report(_DUPLICATES, _LABELS)])
        assert "SUMMARY" not in fragment
        assert '<table class="summary">' not in fragment
        assert "Health:" not in fragment

    def test_only_a_warning_starts_open(self):
        """The findings that need a look are open on arrival; the rest are one line each until opened."""
        fragment = render_html([_report(_DUPLICATES, _LABELS)])
        assert '<details class="card warning" id="duplicates" open>' in fragment
        assert '<details class="card ok" id="label-distribution"><summary>' in fragment

    def test_the_metadata_factors_and_the_configuration_fold_away(self):
        """Reference a reader opens when they need it, drawn apart from the findings and closed."""
        report = _report(_LABELS)
        report = report.model_copy(
            update={"blocks": [*report.blocks[:-1], Section(title="METADATA FACTORS"), report.blocks[-1]]}
        )
        fragment = render_html([report])
        assert '<details class="panel"><summary><h2>METADATA FACTORS</h2></summary></details>' in fragment
        assert '<details class="panel"><summary><h2>CONFIGURATION</h2></summary>' in fragment

    def test_findings_that_share_a_title_get_their_own_cards(self):
        second = Section(title="Duplicates", brief="1 group", severity="info")
        fragment = render_html([_report(_DUPLICATES, second)])
        assert 'id="duplicates"' in fragment
        assert 'id="duplicates-2"' in fragment

    def test_a_card_never_takes_an_id_another_already_has(self):
        """ "Outliers 2" names `outliers-2`, so the second "Outliers" moves on to `outliers-3`."""
        titles = ("Outliers", "Outliers 2", "Outliers")
        fragment = render_html([_report(*(Section(title=title, severity="info") for title in titles))])
        assert re.findall(r'<details class="card info" id="([^"]+)"', fragment) == [
            "outliers",
            "outliers-2",
            "outliers-3",
        ]

    def test_a_report_s_own_output_stays_open(self):
        """An evaluator's output and a failed run's errors are the report itself, so they never fold away."""
        for title in ("OUTPUT", "FAILED"):
            report = Section(title="Run", blocks=[Section(title=title, blocks=[Paragraph(text="12 rows")])])
            assert f'<section class="section"><h2>{title}</h2>' in render_html([report])

    def test_a_page_of_several_reports_lists_them_first_and_keeps_every_anchor_its_own(self):
        page = html_page("Results", [_report(_DUPLICATES, title="train"), _report(_DUPLICATES, title="test")])
        assert (
            '<nav class="contents"><h2>Reports</h2><ol><li><a href="#r1">train</a></li>'
            '<li><a href="#r2">test</a></li></ol></nav>'
        ) in page
        assert '<article class="report" id="r1">' in page
        assert 'id="r1-duplicates"' in page
        assert 'id="r2-duplicates"' in page
        assert _well_formed(page)

    def test_a_page_of_one_report_has_no_contents_list(self):
        assert '<nav class="contents">' not in html_page("R", [_report(_LABELS)])


class TestProse:
    def test_backticks_become_code_and_line_breaks_stay(self):
        fragment = render_html([Paragraph(text="Run `dataeval-flow encoding`.\nThen commit it.")])
        assert fragment == "<p>Run <code>dataeval-flow encoding</code>.<br>Then commit it.</p>"

    def test_bullets(self):
        assert render_html([BulletList(items=["a", "b"])]) == "<ul><li>a</li><li>b</li></ul>"

    def test_fields_are_a_definition_list(self):
        fragment = render_html([Fields(items=[("Distance", 0.25), ("p-value", None)])])
        assert fragment == '<dl class="fields"><dt>Distance</dt><dd>0.25</dd><dt>p-value</dt><dd></dd></dl>'

    def test_code_keeps_its_language(self):
        fragment = render_html([Code(text="metadata:\n  - name: a", language="yaml")])
        assert fragment == '<pre><code class="language-yaml">metadata:\n  - name: a</code></pre>'

    def test_a_tree_is_the_text_report_s_tree(self):
        fragment = render_html([Tree(value={"seed": 1, "tasks": ["a", "b"]})])
        assert fragment == '<pre class="tree">seed: 1\ntasks: [a, b]</pre>'


class TestTables:
    _TABLE = Table(
        columns=[
            Column(key="name", header="Class"),
            Column(key="pct", header="%", format="{:.1f}%"),
            Column(key="pct", kind="bar", format="{:.1f}%", markers=[("Threshold", 50.0)]),
        ],
        rows=[{"name": "cat", "pct": 75.0}, {"name": "dog", "pct": 25.0}],
    )

    def test_cells_print_as_the_text_report_does_and_carry_their_raw_values(self):
        fragment = render_html([self._TABLE])
        assert '<td class="left" data-value="cat">cat</td>' in fragment
        assert '<td class="right" data-value="75.0">75.0%</td>' in fragment

    def test_a_bar_is_a_share_of_the_column_s_scale(self):
        fragment = render_html([self._TABLE])
        assert '<span class="bar" style="margin-left:0%;width:100%"></span>' in fragment
        assert '<span class="bar" style="margin-left:0%;width:33.33%"></span>' in fragment

    def test_a_marker_is_drawn_through_every_bar(self):
        fragment = render_html([self._TABLE])
        assert (
            '<span class="track"><span class="bar" style="margin-left:0%;width:100%"></span>'
            '<span class="mark" style="left:66.67%"></span></span>'
        ) in fragment
        assert '<span class="mark" style="left:66.67%"></span>' in fragment.split("</tr>")[2]

    def test_the_scale_row_closes_the_table_with_its_ends_and_markers(self):
        fragment = render_html([self._TABLE])
        assert "<caption>" not in fragment
        (scale,) = re.findall(r'<tfoot><tr class="scale">.*?</tr></tfoot>', fragment)
        assert scale.count("<td></td>") == 2
        assert '<span class="label" style="left:0%;top:0em">0.0%</span>' in scale
        assert ">75.0%</span>" in scale
        assert ">Threshold 50.0%</span>" in scale

    def test_a_table_whose_bars_have_no_markers_has_no_scale_row(self):
        table = Table(columns=[Column(key="k", header="K"), Column(key="n", kind="bar")], rows=[{"k": "a", "n": 1.0}])
        assert "<tfoot>" not in render_html([table])
        assert render_html([table]).startswith("<table>")

    def test_a_table_with_a_scale_keeps_its_charts_as_wide_as_the_labels_are_placed_for(self):
        """Labels are placed for a 12rem column, so under a scale the column never shrinks below it."""
        assert render_html([self._TABLE]).startswith('<table class="scaled">')
        assert "table.scaled td.chart { min-width: 12rem; }" in STYLE

    def test_a_negative_bar_runs_left_of_zero(self):
        table = Table(columns=[Column(key="d", kind="bar")], rows=[{"d": -1.0}, {"d": 3.0}])
        fragment = render_html([table])
        assert '<span class="bar" style="margin-left:0%;width:25%"></span>' in fragment
        assert '<span class="bar" style="margin-left:25%;width:75%"></span>' in fragment

    def test_an_all_negative_column_draws_from_zero_inside_its_cell(self):
        table = Table(columns=[Column(key="d", kind="bar")], rows=[{"d": -1.0}, {"d": -3.0}])
        fragment = render_html([table])
        assert '<span class="bar" style="margin-left:66.67%;width:33.33%"></span>' in fragment
        assert '<span class="bar" style="margin-left:0%;width:100%"></span>' in fragment

    def test_a_number_that_is_not_finite_draws_no_bar(self):
        table = Table(columns=[Column(key="d", kind="bar")], rows=[{"d": float("nan")}, {"d": 1.0}])
        assert '<td class="chart" data-value="nan"></td>' in render_html([table])

    def test_stacked_segments_share_the_row_peak_and_the_header_is_the_legend(self):
        table = Table(
            columns=[Column(key="split", kind="stacked", series=["in-dist", "OOD"])],
            rows=[{"split": [3.0, 1.0]}, {"split": [1.0, 1.0]}],
        )
        fragment = render_html([table])
        assert '<span class="swatch s0"></span>in-dist <span class="swatch s1"></span>OOD' in fragment
        assert (
            '<span class="seg s0" style="width:75%"></span><span class="seg s1" style="width:25%"></span>' in fragment
        )
        assert (
            '<span class="seg s0" style="width:25%"></span><span class="seg s1" style="width:25%"></span>' in fragment
        )

    def test_stacked_segments_stay_on_one_line(self):
        """The segments share one flex row, so a tiny segment's minimum width can't push the last onto a new line."""
        table = Table(
            columns=[Column(key="split", kind="stacked", series=["in-dist", "OOD"])], rows=[{"split": [999.0, 1.0]}]
        )
        fragment = render_html([table])
        assert '<td class="chart" data-value="999.0,1.0"><span class="stack"><span class="seg s0"' in fragment

    def test_a_bar_column_is_scaled_once_per_table_not_once_per_cell(self, monkeypatch):
        import dataeval_flow._blocks._html_tables as html_module

        calls: list[str] = []
        scale = html_module._scale

        def counted(table: Table, column: Column) -> tuple[float, float]:
            calls.append(column.key)
            return scale(table, column)

        monkeypatch.setattr(html_module, "_scale", counted)
        render_html([Table(columns=[Column(key="n", kind="bar")], rows=[{"n": float(i)} for i in range(50)])])
        assert calls == ["n"]

    def test_a_list_cell_s_data_value_holds_its_counts_unrounded(self):
        table = Table(columns=[Column(key="h", kind="sparkline")], rows=[{"h": [12345.0, 0.0]}])
        assert 'data-value="12345.0,0.0"' in render_html([table])

    def test_a_sparkline_is_an_svg_with_one_bar_per_count(self):
        table = Table(columns=[Column(key="shape", kind="sparkline")], rows=[{"shape": [1.0, 0.0, 2.0]}])
        fragment = render_html([table])
        assert '<svg class="spark" viewBox="0 0 3 1" preserveAspectRatio="none">' in fragment
        assert len(re.findall(r"<rect ", fragment)) == 3

    def test_a_table_without_headers_has_no_head_and_one_without_rows_draws_nothing(self):
        assert "<thead>" not in render_html([Table(columns=[Column(key="k")], rows=[{"k": "a"}])])
        assert render_html([Table(columns=[Column(key="k", header="K")], rows=[])]) == ""


class TestCharts:
    def test_a_proportion_draws_each_part_and_lists_the_counts(self):
        fragment = render_html([Proportion(parts=[("numeric", 1842), ("text", 58)])])
        assert fragment.startswith('<div class="proportion"><span class="stack">')
        assert "1,842 numeric, 58 text" in fragment

    def test_a_proportion_of_one_kind_lists_it_without_a_bar(self):
        assert render_html([Proportion(parts=[("numeric", 10), ("text", 0)])]) == (
            '<div class="proportion">10 numeric</div>'
        )

    def test_series_past_the_fourth_reuse_the_colours(self):
        """Four colours are defined, so a fifth series or part wraps round to the first rather than drawing blank."""
        table = Table(columns=[Column(key="s", kind="stacked", series=list("abcde"))], rows=[{"s": [1.0] * 5}])
        parts = render_html([Proportion(parts=[(name, 1) for name in "abcde"])])
        for fragment in (render_html([table]), parts):
            assert "s4" not in fragment
            assert fragment.count('class="seg s0"') == 2
        assert render_html([table]).count('class="swatch s0"') == 2

    def test_a_distribution_with_quantiles_draws_its_box_in_the_same_svg(self):
        quantiles = Quantiles(low=0.0, q1=1.0, median=2.0, q3=3.0, high=4.0)
        fragment = render_html([Distribution(histogram=[1, 3, 2, 1], quantiles=quantiles)])
        assert fragment.count("<svg") == 1
        assert len(re.findall(r'<rect class="hist"', fragment)) == 4
        assert '<rect class="box"' in fragment
        assert "p25 1 · p50 2 · p75 3" in fragment

    def test_a_distributions_lines_keep_their_width_however_fine_its_record(self):
        """The svg stretches one unit per recorded cell to fit, so a stroke measured in those units thins as
        the record grows finer: at 160 cells a median line of 0.05 units draws about 0.16px wide."""
        quantiles = Quantiles(low=0.0, q1=1.0, median=2.0, q3=3.0, high=4.0)
        page = html_page("Report", [Distribution(histogram=[1] * 160, quantiles=quantiles)])
        rule = re.search(r"svg\.dist line \{([^}]*)\}", page)
        assert rule is not None
        assert "vector-effect: non-scaling-stroke" in rule.group(1)
        assert re.search(r"stroke-width: [\d.]+px", rule.group(1))

    def test_a_distribution_without_quantiles_draws_the_counts_alone(self):
        fragment = render_html([Distribution(histogram=[5, 0, 7])])
        assert '<rect class="box"' not in fragment
        assert "n=5–7" in fragment

    def test_a_summary_is_a_table_of_badges(self):
        item = SummaryItem(label="Duplicates", value="3 groups", severity="warning")
        fragment = render_html([Summary(items=[item])])
        assert fragment == (
            '<table class="summary"><tbody><tr><td>Duplicates</td><td>3 groups</td>'
            '<td><span class="badge warning">warning</span></td></tr></tbody></table>'
        )


class TestFlagsCells:
    _FLAGS = [
        Flag(name="contrast", value=0.91, direction="upper", bound=0.77, percentile=99.6, mean=0.5, std=0.1),
        Flag(name="brightness", value=0.99, direction="upper", bound=0.84, percentile=99.95, mean=0.52, std=0.11),
    ]
    _TABLE = Table(
        columns=[Column(key="item", header="Item"), Column(key="flags", header="Flagged by", kind="flags")],
        rows=[{"item": 41, "flags": _FLAGS}],
    )

    def test_a_flags_cell_sorts_by_how_many_flags_it_holds(self):
        assert '<td class="flags" data-sort="2">' in render_html([self._TABLE])

    def test_each_tag_reads_as_its_value_against_its_limit_by_name(self):
        """No shade and no rank: a tag is the comparison that flagged the value."""
        fragment = render_html([self._TABLE])
        assert '<span class="tag" tabindex="0">brightness 0.99 &gt; 0.84<span class="tip">' in fragment
        assert fragment.index("brightness 0.99") < fragment.index("contrast 0.91")

    def test_each_tag_carries_its_card(self):
        fragment = render_html([self._TABLE])
        assert '<span class="tip-title">brightness · above the upper limit</span>' in fragment
        assert '<span class="tip-row"><span>Percentile</span><span>p99.95</span></span>' in fragment
        assert '<span class="tip-row"><span>Population</span><span>mean 0.52 ± 0.11 (std)</span></span>' in fragment

    def test_the_flags_header_sits_with_its_tags(self):
        assert '<th class="flags">Flagged by</th>' in render_html([self._TABLE])

    def test_a_flags_table_is_well_formed(self):
        assert _well_formed(render_html([self._TABLE]))


class TestLabelPlacement:
    """Scale labels are placed without measuring text, on up to three lines, and never overlap."""

    def test_labels_far_apart_share_the_first_line(self):
        placed, also = place_labels([(0.0, "0"), (100.0, "1.0"), (50.0, "warn 0.3")])
        assert [label.line for label in placed] == [0, 0, 0]
        assert also == []

    def test_close_labels_stack_onto_the_next_free_line(self):
        placed, also = place_labels([(0.0, "0"), (100.0, "1"), (50.0, "a 0.301"), (51.0, "b 0.302"), (52.0, "c 0.303")])
        assert [label.line for label in placed[2:]] == [0, 1, 2]
        assert also == []

    def test_a_label_that_clears_no_line_is_listed_instead(self):
        markers = [(50.0 + i / 2, f"m{i} 0.30{i}") for i in range(4)]
        placed, also = place_labels([(0.0, "0"), (100.0, "1"), *markers])
        assert len(placed) == 5
        assert also == ["m3 0.303"]

    def test_the_ends_stay_inside_the_scale(self):
        placed, _ = place_labels([(0.0, "0.0%"), (100.0, "75.0%")])
        assert placed[0].start == 0
        assert placed[1].start + placed[1].width == pytest.approx(100)

    @pytest.mark.parametrize("seed", range(20))
    def test_no_two_labels_on_a_line_overlap(self, seed):
        import random
        from itertools import pairwise

        rng = random.Random(seed)  # noqa: S311 - reproducible test positions, not a secret
        labels = [(rng.uniform(0, 100), "x" * rng.randint(1, 12)) for _ in range(rng.randint(1, 12))]
        placed, also = place_labels(labels)
        assert len(placed) + len(also) == len(labels)
        for line in range(3):
            spans = sorted((p.start, p.start + p.width) for p in placed if p.line == line)
            assert all(end <= start for (_, end), (start, _) in pairwise(spans))


class TestEscaping:
    _EVIL = '<script>alert("1")</script>'

    def test_dataset_strings_are_escaped_in_text_and_in_attributes(self):
        blocks: list[Block] = [
            Section(title=self._EVIL, brief=self._EVIL, blocks=[Paragraph(text=self._EVIL)]),
            Fields(items=[(self._EVIL, self._EVIL)]),
            Table(columns=[Column(key="c", header=self._EVIL)], rows=[{"c": self._EVIL}]),
            BulletList(items=[self._EVIL]),
            Code(text=self._EVIL, language='"><script>'),
            Summary(items=[SummaryItem(label=self._EVIL, value=self._EVIL)]),
            Table(
                columns=[Column(key="f", header=self._EVIL, kind="flags")],
                rows=[
                    {"f": [Flag(name=self._EVIL, value=1, direction="upper", bound=0, percentile=99, mean=0, std=1)]}
                ],
            ),
            Table(
                columns=[Column(key="n", kind="bar", markers=[(self._EVIL, 1.0)])],
                rows=[{"n": 2.0}],
            ),
        ]
        fragment = render_html(blocks)
        assert "<script" not in fragment
        assert 'data-value="&lt;script&gt;alert(&quot;1&quot;)&lt;/script&gt;"' in fragment
        assert _well_formed(fragment)


class TestOverrides:
    def test_an_override_replaces_only_its_block_type(self):
        blocks: list[Block] = [Paragraph(text="kept"), Distribution(histogram=[1, 2])]
        fragment = render_html(blocks, draw={"distribution": lambda _block, _ctx: "<figure>plotted</figure>"})
        assert fragment == "<p>kept</p>\n<figure>plotted</figure>"

    def test_a_container_override_draws_its_children_through_the_context(self):
        def boxed(block: Any, ctx: Any) -> str:
            return f'<div class="box" data-depth="{ctx.depth}">{ctx.render(block.blocks)}</div>'

        fragment = render_html([Section(title="T", blocks=[Paragraph(text="inner")])], draw={"section": boxed})
        assert fragment == '<div class="box" data-depth="0"><p>inner</p></div>'


_BRIGHT = Flag(name="brightness", value=0.99, direction="upper", bound=0.84, percentile=99.95, mean=0.52, std=0.11)


class TestScript:
    """One inline script adds sorting, filtering and expand-all; the page reads the same without it."""

    _TABLE = Table(columns=[Column(key="k", header="K")], rows=[{"k": str(i)} for i in range(12)])

    def test_the_page_s_own_markup_has_no_controls(self):
        """The script adds its buttons and filter boxes, so a page with scripts blocked shows none that do nothing."""
        page = html_page("R", [_report(_DUPLICATES), self._TABLE])
        markup = page.replace(f"<script>{SCRIPT}</script>", "")
        assert "<button" not in markup
        assert "<input" not in markup

    @pytest.mark.parametrize(
        "reach", ["fetch", "XMLHttpRequest", "import(", "eval(", "Function(", "innerHTML", "document.write", "http"]
    )
    def test_the_script_reaches_nothing_outside_the_page(self, reach):
        """It loads nothing and runs no text as code, and it writes result strings only as text."""
        assert reach not in SCRIPT

    def test_the_script_does_what_the_page_needs(self):
        assert 'querySelectorAll("details.card, details.panel")' in SCRIPT, "the buttons open the panels too"
        for need in ("aria-sort", "Expand all", "Collapse all", "beforeprint", "afterprint", "Filter rows"):
            assert need in SCRIPT

    def test_the_script_parses(self, tmp_path):
        import subprocess

        script = tmp_path / "page.js"
        script.write_text(SCRIPT)
        subprocess.run([_node(), "--check", str(script)], check=True, capture_output=True)  # noqa: S603

    def test_a_stray_nan_sorts_last_and_leaves_the_rest_numeric(self, tmp_path):
        """A NaN or infinite cell (`str(float("nan"))` etc.) sorts as blank: last, not first via a text compare."""
        values = [0.3, 0.25, -1.5, -0.2, 10, float("nan")]
        table = Table(columns=[Column(key="v", header="V")], rows=[{"v": v} for v in values])
        _drive(tmp_path, table, 'check("up", sortBy(0), ["-1.5", "-0.2", "0.25", "0.3", "10", "nan"]);')

    def test_a_column_mixing_numbers_and_text_sorts_numbers_as_numbers_and_first(self, tmp_path):
        """Two numbers compare as numbers wherever text sits among them; blanks stay last either way."""
        values = [0.3, 0.25, "varies", -1.5, 10, None]
        table = Table(columns=[Column(key="v", header="V")], rows=[{"v": v} for v in values])
        _drive(
            tmp_path,
            table,
            """
            check("up", sortBy(0), ["-1.5", "0.25", "0.3", "10", "varies", ""]);
            check("down", sortBy(0), ["varies", "10", "0.3", "0.25", "-1.5", ""]);
            """,
        )

    _FLAGGED = Table(
        columns=[
            Column(key="item", header="Item"),
            Column(key="flags", header="Flags"),
            Column(key="by", header="Flagged by", kind="flags"),
        ],
        rows=[
            *({"item": item, "flags": 1, "by": [_BRIGHT]} for item in range(11)),
            {
                "item": 41,
                "flags": 3,
                "by": [
                    _BRIGHT,
                    _BRIGHT.model_copy(update={"name": "contrast"}),
                    _BRIGHT.model_copy(update={"name": "mean"}),
                ],
            },
        ],
    )

    def test_the_filter_reads_what_the_rows_show_cell_by_cell(self, tmp_path):
        """A hover card's words don't match, and a number never runs on into the next cell."""
        _drive(
            tmp_path,
            self._FLAGGED,
            """
            check("metric", filter("mean"), ["41"]);
            for (const hidden of ["upper", "std", "percentile", "population"]) check(hidden, filter(hidden), []);
            check("across cells", filter("413"), []);
            check("item", filter("41"), ["41"]);
            check("tag", filter("brightness 0.99 > 0.84").length, 12);
            """,
        )

    def test_a_filtered_table_prints_whole_and_filters_again_afterwards(self, tmp_path):
        """Print hides the filter's count, so rows it left out would go missing without a word."""
        _drive(
            tmp_path,
            self._FLAGGED,
            """
            check("filtered", filter("mean"), ["41"]);
            fire(window, "beforeprint");
            check("printing", shown().length, 12);
            fire(window, "afterprint");
            check("after", shown(), ["41"]);
            """,
        )


# Just enough DOM for the page's script to run on one real rendered table: elements rebuilt from the
# parsed HTML, the calls the script makes, and `sortBy`, `filter`, `shown` and `check` for a scenario.
_HARNESS = """
"use strict";
const fs = require("fs");
const [source, tree] = [2, 3].map((index) => fs.readFileSync(process.argv[index], "utf8"));

const listen = (target) => Object.assign(target, {
  on: {},
  addEventListener(type, fn) { (this.on[type] ??= []).push(fn); },
});
const fire = (target, type) => (target.on[type] ?? []).forEach((fn) => fn({}));
const text = (data) => ({ nodeType: 3, data });
const element = ({ tag, attrs = {}, children = [] }) => {
  const classes = (attrs.class ?? "").split(" ").filter(Boolean);
  const node = listen({
    nodeType: 1,
    tagName: tag.toUpperCase(),
    hidden: false,
    value: "",
    attrs: { ...attrs },
    dataset: {},
    childNodes: children.map((child) => (typeof child === "string" ? text(child) : element(child))),
    classList: { add: (name) => classes.push(name), contains: (name) => classes.includes(name) },
    setAttribute(name, value) { this.attrs[name] = String(value); },
    getAttribute(name) { return this.attrs[name] ?? null; },
    append(...items) { this.childNodes = this.childNodes.filter((child) => !items.includes(child)).concat(items); },
    before(item) { this.previous = item; },
    get textContent() {
      return this.childNodes.map((child) => (child.nodeType === 3 ? child.data : child.textContent)).join("");
    },
    set textContent(value) { this.childNodes = [text(String(value))]; },
    get cells() { return this.childNodes.filter((child) => child.tagName === "TD" || child.tagName === "TH"); },
    get rows() { return this.childNodes.filter((child) => child.tagName === "TR"); },
    get tHead() { return this.childNodes.find((child) => child.tagName === "THEAD") ?? null; },
    get tBodies() { return this.childNodes.filter((child) => child.tagName === "TBODY"); },
  });
  for (const [name, value] of Object.entries(attrs)) {
    if (name.startsWith("data-")) node.dataset[name.slice(5)] = value ?? "";
  }
  return node;
};

const table = element(JSON.parse(tree));
global.window = listen({});
global.document = {
  querySelectorAll: (selector) => (selector === "main table:not(.summary)" ? [table] : []),
  querySelector: () => null,
  createElement: (tag) => element({ tag }),
};
eval(source);

const body = table.tBodies[0];
const shown = (column = 0) => body.rows.filter((row) => !row.hidden).map((row) => row.cells[column].textContent);
const sortBy = (column) => {
  fire(table.tHead.rows[0].cells[column], "click");
  return shown(column);
};
const filter = (query) => {
  const input = table.previous.childNodes[0];
  input.value = query;
  fire(input, "input");
  return shown();
};
const check = (label, got, want) => {
  if (JSON.stringify(got) !== JSON.stringify(want)) {
    console.error(label, "got", JSON.stringify(got), "want", JSON.stringify(want));
    process.exitCode = 1;
  }
};
"""


class _Tree(HTMLParser):
    """An HTML fragment as the nested ``{tag, attrs, children}`` the harness rebuilds, text as strings."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[dict[str, Any]] = [{"tag": "root", "attrs": {}, "children": []}]

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        node = {"tag": tag, "attrs": dict(attrs), "children": []}
        self.stack[-1]["children"].append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if tag not in _VOID:
            self.stack.pop()

    def handle_data(self, data: str) -> None:
        self.stack[-1]["children"].append(data)


def _node() -> str:
    """Where `node` is; a test that needs it skips where it isn't installed."""
    import shutil

    node = shutil.which("node")
    if node is None:
        raise pytest.skip.Exception("node is not installed")
    return node


def _drive(tmp_path: Any, table: Table, scenario: str) -> None:
    """Run the page's script on *table* as rendered, then *scenario*, whose every `check` must hold."""
    import json
    import subprocess

    node = _node()
    parser = _Tree()
    parser.feed(render_html([table]))
    (element,) = parser.stack[0]["children"]
    for name, content in (("page.js", SCRIPT), ("tree.json", json.dumps(element)), ("harness.js", _HARNESS + scenario)):
        (tmp_path / name).write_text(content)
    paths = [str(tmp_path / name) for name in ("harness.js", "page.js", "tree.json")]
    result = subprocess.run([node, *paths], capture_output=True, text=True)  # noqa: S603
    assert result.returncode == 0, result.stderr


class TestTheme:
    """The stylesheet's promises: a palette for each color scheme, print always light, and nothing to fetch."""

    @staticmethod
    def _print_rules() -> str:
        return STYLE.split("@media print", 1)[1]

    def test_dark_mode_follows_the_system_setting(self):
        dark = STYLE.split("@media (prefers-color-scheme: dark)", 1)[1].split("}", 2)[0]
        for token in ("--bg:", "--ink:", "--card:", "--rule:", "--warning:", "--mark:"):
            assert token in dark

    def test_print_always_uses_the_light_palette(self):
        light = STYLE.split(":root {", 1)[1].split("}", 1)[0]
        printed = self._print_rules()
        for token in ("--bg:", "--ink:", "--card:", "--rule:"):
            assert token in light
            assert f"{token} {light.split(token, 1)[1].split(';', 1)[0].strip()}" in printed

    def test_print_hides_what_only_works_on_screen(self):
        printed = self._print_rules()
        for hidden in (".controls", ".filter", ".tip"):
            assert hidden in printed.split("display: none", 1)[0]

    def test_tags_share_one_style_with_no_shade_of_severity(self):
        """A tag is a fact, the value against its limit, so every tag looks alike."""
        assert ".tag {" in STYLE
        assert ".tag.b" not in STYLE
        dark = STYLE.split("@media (prefers-color-scheme: dark)", 1)[1].split("}", 2)[0]
        assert "--tag:" in STYLE.split(":root {", 1)[1].split("}", 1)[0]
        assert "--tag:" in dark

    def test_a_card_opens_on_hover_and_on_keyboard_focus(self):
        assert ".tag:hover .tip" in STYLE
        assert ".tag:focus .tip" in STYLE

    def test_numbers_line_up(self):
        assert "font-variant-numeric: tabular-nums" in STYLE

    def test_the_stylesheet_fetches_nothing(self):
        assert "url(" not in STYLE
        assert "@import" not in STYLE


class TestPage:
    def test_a_page_is_self_contained_and_prints_cleanly(self):
        page = html_page("Data <cleaning>", [Section(title="Report", blocks=[Paragraph(text="body")])])
        assert page.startswith("<!doctype html>")
        assert '<meta charset="utf-8">' in page
        assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in page
        assert "<title>Data &lt;cleaning&gt;</title>" in page
        assert "@media print" in page
        assert "break-inside: avoid" in page
        assert "break-after: avoid" in page
        assert page.endswith(f"<script>{SCRIPT}</script>\n</body>\n</html>\n")
        assert page.count("<script>") == 1
        assert "http" not in page
        assert _well_formed(page)

    def test_a_printed_page_keeps_its_chart_colours(self):
        """Bars, segments and badges are backgrounds, which a browser leaves out of a print unless told not to."""
        page = html_page("Report", [Section(title="Report")])
        assert "print-color-adjust: exact" in page
        assert "-webkit-print-color-adjust: exact" in page


class TestNumbersWithNoPosition:
    """A NaN or an infinity places nothing: its chart draws blank or falls back, and the page stays whole."""

    def test_stacked_and_sparkline_cells(self):
        table = Table(
            columns=[Column(key="s", kind="stacked", series=["a", "b"]), Column(key="h", kind="sparkline")],
            rows=[{"s": [float("nan"), 1.0], "h": [float("inf"), 1.0]}, {"s": [1.0, 1.0], "h": [1.0, 2.0]}],
        )
        fragment = render_html([table])
        assert '<td class="chart" data-value="nan,1.0"></td>' in fragment
        assert _well_formed(fragment)

    def test_a_distribution_whose_quantiles_are_not_finite_draws_its_counts_alone(self):
        nan = float("nan")
        quantiles = Quantiles(low=nan, q1=nan, median=nan, q3=nan, high=nan)
        fragment = render_html([Distribution(histogram=[2, 4], quantiles=quantiles)])
        assert '<rect class="box"' not in fragment
        assert "n=2–4" in fragment

    def test_a_proportion_of_nothing_draws_nothing(self):
        assert render_html([Proportion(parts=[("numeric", 0), ("text", 0)])]) == ""
