"""The finding-block lookups workflow tests share, and the tables the workflows share."""

import pytest

from dataeval_flow._blocks import BulletList, Code, Column, Fields, ItemRef, Paragraph, Section, Table
from dataeval_flow.workflows import Finding
from dataeval_flow.workflows._tables import group_cells, groups_table, ranked_table, unlabelled_blocks
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


def _refs(source: str, *indices: int) -> list[ItemRef]:
    return [ItemRef(source=source, index=index) for index in indices]


class TestGroupCells:
    def test_a_small_group_names_and_pictures_every_item(self):
        assert group_cells(_refs("train", 0, 5)) == ("0, 5", _refs("train", 0, 5))

    def test_a_large_group_shows_eight_and_counts_the_rest(self):
        items, shown = group_cells(_refs("train", *range(20)))
        assert items == "0, 1, 2, 3, 4, 5, 6, 7, … 12 more"
        assert shown == _refs("train", *range(8))

    def test_boxes_are_named_with_their_item(self):
        boxes = [ItemRef(source="train", index=3, target=0), ItemRef(source="train", index=9, target=2)]
        assert group_cells(boxes)[0] == "3 box 0, 9 box 2"


class TestGroupsTable:
    def test_groups_run_largest_first_exact_before_near_at_one_size(self):
        (table,) = groups_table(
            [("exact", 0, _refs("s", 0, 1)), ("near", 0, _refs("s", 2, 3)), ("exact", 1, _refs("s", 4, 5, 6))],
            "images",
        )
        assert isinstance(table, Table)
        assert [(row["kind"], row["group"], row["count"], row["items"]) for row in table.rows] == [
            ("exact", 1, 3, "4, 5, 6"),
            ("exact", 0, 2, "0, 1"),
            ("near", 0, 2, "2, 3"),
        ]
        assert table.rows[0]["image"] == _refs("s", 4, 5, 6)
        assert table.columns[-1] == Column(key="image", kind="image")
        assert table.preview == 10

    def test_no_groups_is_no_table(self):
        assert groups_table([], "images") == []

    def test_past_500_groups_a_paragraph_names_the_rest(self):
        table, note = groups_table([("exact", n, _refs("s", n, n + 1000)) for n in range(503)], "boxes")
        assert isinstance(table, Table)
        assert len(table.rows) == 500
        assert note == Paragraph(
            text="503 groups of boxes; the 500 largest are listed, and every one is in `output.raw`."
        )


class TestUnlabelledBlocks:
    def test_each_source_with_unlabelled_images_counts_names_and_pictures_them(self):
        (section,) = unlabelled_blocks({"train": list(range(10)), "val": [], "test": [4]}, header="Split")
        assert isinstance(section, Section)
        assert section.title == "Images with no labels"
        (table,) = section.blocks
        assert isinstance(table, Table)
        assert [(c.key, c.header, c.kind) for c in table.columns] == [
            ("source", "Split", "text"),
            ("count", "Count", "text"),
            ("items", "Items", "text"),
            ("image", "", "image"),
        ]
        assert [(row["source"], row["count"], row["items"]) for row in table.rows] == [
            ("train", 10, "0, 1, 2, 3, 4, 5, 6, 7, … 2 more"),
            ("test", 1, "4"),
        ]
        assert table.rows[1]["image"] == _refs("test", 4)

    def test_no_unlabelled_images_is_no_section(self):
        assert unlabelled_blocks({"train": []}, header="Source") == []
