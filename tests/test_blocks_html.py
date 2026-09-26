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
        fragment = render_html([Section(title="Duplicates", brief="3 groups", severity="warning")])
        assert '<section class="section warning">' in fragment
        assert '<span class="brief">3 groups</span>' in fragment
        assert '<span class="badge warning">warning</span>' in fragment

    def test_a_multi_line_title_breaks_where_it_does_in_text(self):
        assert "<h1>Report<br>Second line</h1>" in render_html([Section(title="Report\nSecond line")])


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

    def test_markers_are_listed_in_a_caption(self):
        assert "<caption>Threshold: 50.0%</caption>" in render_html([self._TABLE])

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
        import dataeval_flow._blocks._html as html_module

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
        assert "<script" not in page
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
