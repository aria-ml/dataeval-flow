"""The finding-block lookups workflow tests share, and the ranked table the workflows share."""

import pytest

from dataeval_flow._blocks import BulletList, Code, Column, Fields, Paragraph, Section, Table
from dataeval_flow.workflows import Finding
from dataeval_flow.workflows._tables import ranked_table
from tests.finding_blocks import bullets, codes, column, fields, paragraphs, rendered, sections, tables, walk

pytestmark = pytest.mark.required

_TABLE = Table(columns=[Column(key="name", header="Name")], rows=[{"name": "cat"}, {"name": "dog"}])
_NESTED = Section(
    title="Group",
    brief="2 items",
    blocks=[Paragraph(text="inner"), BulletList(items=["a", "b"]), Fields(items=[("Radius", 0.5)]), _TABLE],
)
_FINDING = Finding(
    title="Evidence",
    brief="3 things",
    description="The lede.",
    blocks=[Paragraph(text="outer"), Fields(items=[("Count", 3), ("Radius", 0.25)]), Code(text="a: 1"), _NESTED],
)


class TestLookups:
    def test_walk_descends_into_sections_in_document_order(self):
        kinds = [block.type for block in walk(_FINDING.blocks)]
        assert kinds == ["paragraph", "fields", "code", "section", "paragraph", "bullet_list", "fields", "table"]

    def test_paragraphs_and_bullets_come_from_every_depth(self):
        assert paragraphs(_FINDING) == ["outer", "inner"]
        assert bullets(_FINDING) == ["a", "b"]

    def test_fields_merge_every_block_and_a_later_label_wins(self):
        assert fields(_FINDING) == {"Count": 3, "Radius": 0.5}

    def test_tables_codes_and_sections(self):
        assert tables(_FINDING) == [_TABLE]
        assert codes(_FINDING) == ["a: 1"]
        assert [section.title for section in sections(_FINDING)] == ["Group"]

    def test_column_reads_one_key_top_to_bottom(self):
        assert column(_TABLE, "name") == ["cat", "dog"]
        assert column(_TABLE, "missing") == [None, None]

    def test_the_description_is_not_a_block(self):
        assert "The lede." not in paragraphs(_FINDING)


class TestRendered:
    def test_draws_the_detail_section_as_the_report_does(self):
        lines = rendered(Finding(title="Pruning", brief="3 items", description="Removed three.")).splitlines()
        assert lines[0] == "=" * 80
        assert lines[1] == "  PRUNING" + "3 items".rjust(71)
        assert lines[3] == "  Removed three."

    def test_honors_the_width(self):
        assert rendered(Finding(title="T"), width=60).splitlines()[0] == "=" * 60


class TestRankedTable:
    def test_rows_run_largest_first_with_a_bar_on_the_value(self):
        table = ranked_table({"dog": 3, "cat": 12, "eel": 7}, headers=("Class", "Count"))
        assert column(table, "name") == ["cat", "eel", "dog"]
        assert column(table, "value") == [12, 7, 3]
        assert [(c.key, c.header, c.kind) for c in table.columns] == [
            ("name", "Class", "text"),
            ("value", "Count", "text"),
            ("value", "", "bar"),
        ]

    def test_names_are_text_whatever_the_key(self):
        assert column(ranked_table({0: 1.5}, headers=("Factor", "MI")), "name") == ["0"]

    def test_ties_keep_the_order_given(self):
        assert column(ranked_table({"b": 1, "a": 1}, headers=("K", "V")), "name") == ["b", "a"]
