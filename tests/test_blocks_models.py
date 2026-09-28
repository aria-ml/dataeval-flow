"""Report blocks as data: every block survives JSON unchanged, and nothing else validates as one."""

from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

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

pytestmark = pytest.mark.required

_BLOCKS = TypeAdapter(list[Block])

EVERY_BLOCK: list[Block] = [
    Section(
        title="Outer",
        brief="2 items",
        severity="warning",
        blocks=[Section(title="Inner", blocks=[Paragraph(text="nested")])],
    ),
    Paragraph(text="Some `code` and\na hard break."),
    BulletList(items=["one", "two"]),
    Fields(items=[("Count", 3), ("Ratio", 0.5), ("Flag", True), ("Missing", None), ("Name", "cat")]),
    Table(
        columns=[
            Column(key="name", header="Name"),
            Column(key="count", header="Count", format="{:,}"),
            Column(key="count", kind="bar", markers=[("Threshold", 5.0)]),
            Column(key="split", kind="stacked", series=["in", "out"]),
            Column(key="shape", kind="sparkline", align="left"),
        ],
        rows=[{"name": "cat", "count": 12, "split": [3.0, 9.0], "shape": [1.0, 4.0, 2.0]}],
    ),
    Proportion(parts=[("numeric", 198), ("text", 2)]),
    Distribution(histogram=[1, 3, 2], quantiles=Quantiles(low=0.0, q1=1.0, median=2.0, q3=3.0, high=4.0)),
    Distribution(histogram=[5, 0, 7]),
    Code(text="metadata:\n  - name: standard\n", language="yaml"),
    Tree(value={"tasks": [{"name": "a", "sources": ["s1", "s2"]}], "seed": None}),
    Summary(items=[SummaryItem(label="Duplicates", value="3 groups", severity="warning")]),
]


@pytest.mark.parametrize("block", EVERY_BLOCK, ids=lambda block: block.type)
def test_every_block_round_trips_through_json(block):
    dumped = _BLOCKS.dump_python([block], mode="json")
    assert _BLOCKS.validate_python(dumped) == [block]


def test_every_member_of_the_union_is_exercised():
    """A block type added to the union without a round-trip case here fails this test."""
    union = get_args(get_args(Block)[0])
    members = {member.model_fields["type"].default for member in union}
    assert members == {block.type for block in EVERY_BLOCK}


def test_fields_serialize_as_pairs_so_their_order_survives_javascript():
    dumped = Fields(items=[("2", 1), ("1", 2)]).model_dump(mode="json")
    assert dumped == {"type": "fields", "items": [["2", 1], ["1", 2]]}


def test_an_unknown_block_type_is_rejected():
    with pytest.raises(ValidationError):
        _BLOCKS.validate_python([{"type": "image", "src": "x.png"}])


def test_a_block_refuses_fields_it_does_not_define():
    with pytest.raises(ValidationError):
        Paragraph.model_validate({"text": "x", "html": "<b>x</b>"})


def test_blocks_are_frozen():
    block = Paragraph(text="x")
    with pytest.raises(ValidationError):
        block.text = "y"


class TestNumpyScalars:
    """DataEval hands back numpy scalars; the blocks keep them as the Python values they print as.

    Built through ``model_validate``, the way untyped data from outside arrives: the constructors
    are typed for the Python values these tests expect to come out.
    """

    def test_a_numpy_integer_cell_stays_an_integer(self):
        import numpy as np

        (row,) = Table.model_validate({"columns": [{"key": "n"}], "rows": [{"n": np.int64(3)}]}).rows
        assert row["n"] == 3
        assert type(row["n"]) is int

    def test_a_numpy_bool_field_stays_a_bool(self):
        import numpy as np

        ((_, value),) = Fields.model_validate({"items": [("flag", np.bool_(True))]}).items
        assert value is True

    def test_a_numpy_float32_keeps_the_digits_it_prints_with(self):
        import numpy as np

        (row,) = Table.model_validate({"columns": [{"key": "x"}], "rows": [{"x": np.float32(0.1)}]}).rows
        assert row["x"] == 0.1

    def test_a_float32_marker_keeps_the_digits_it_prints_with(self):
        """A drift threshold from DataEval: results.json holds 0.2, not its float64 widening."""
        import numpy as np

        column = Column.model_validate({"key": "d", "kind": "bar", "markers": [("Threshold", np.float32(0.2))]})
        assert column.markers == [("Threshold", 0.2)]

    def test_float32_quantiles_keep_the_digits_they_print_with(self):
        import numpy as np

        values = {
            name: np.float32(v)
            for name, v in zip(("low", "q1", "median", "q3", "high"), (0.1, 0.2, 0.3, 0.4, 0.5), strict=True)
        }
        quantiles = Quantiles.model_validate(values)
        assert quantiles == Quantiles(low=0.1, q1=0.2, median=0.3, q3=0.4, high=0.5)

    def test_numpy_values_inside_a_list_cell_become_floats(self):
        import numpy as np

        rows = [{"h": [np.int64(1), np.float32(2.5)]}]
        (row,) = Table.model_validate({"columns": [{"key": "h", "kind": "sparkline"}], "rows": rows}).rows
        assert row["h"] == [1.0, 2.5]

    def test_a_numpy_array_cell_becomes_its_items(self):
        import numpy as np

        rows = [{"h": np.array([1, 2, 3])}]
        (row,) = Table.model_validate({"columns": [{"key": "h", "kind": "sparkline"}], "rows": rows}).rows
        assert row["h"] == [1.0, 2.0, 3.0]


class TestJsonForm:
    """A block's JSON leaves out every field at its default, and reads back to the same block."""

    def test_a_table_writes_only_what_differs_from_the_defaults(self):
        table = Table(
            columns=[Column(key="class", header="Class"), Column(key="count", kind="bar")],
            rows=[{"class": "cat", "count": 40}],
        )
        assert table.model_dump(mode="json") == {
            "type": "table",
            "columns": [{"key": "class", "header": "Class"}, {"key": "count", "kind": "bar"}],
            "rows": [{"class": "cat", "count": 40}],
        }

    def test_the_type_tag_is_always_written(self):
        assert Section(title="T").model_dump(mode="json") == {"type": "section", "title": "T"}

    def test_a_value_given_as_its_default_is_left_out_too(self):
        item = SummaryItem(label="Duplicates", value="", severity="info")
        assert Summary(items=[item]).model_dump(mode="json") == {"type": "summary", "items": [{"label": "Duplicates"}]}

    @pytest.mark.parametrize("block", EVERY_BLOCK, ids=lambda block: block.type)
    def test_the_compact_form_reads_back_to_the_same_block(self, block):
        assert _BLOCKS.validate_json(_BLOCKS.dump_json([block])) == [block]


_FLAG = Flag(name="brightness", value=0.99, direction="upper", bound=0.84, percentile=99.95, mean=0.52, std=0.11)


class TestFlagsAndPreviews:
    """A flags cell holds measurements, not text, and a preview says how many rows a narrow renderer shows."""

    def test_a_flag_is_its_measurements(self):
        assert _FLAG.model_dump(mode="json") == {
            "name": "brightness",
            "value": 0.99,
            "direction": "upper",
            "bound": 0.84,
            "percentile": 99.95,
            "mean": 0.52,
            "std": 0.11,
        }

    def test_a_flags_table_with_a_preview_round_trips(self):
        table = Table(
            columns=[Column(key="item", header="Item"), Column(key="flags", header="Flagged by", kind="flags")],
            rows=[{"item": 41, "flags": [_FLAG]}],
            preview=10,
        )
        assert _BLOCKS.validate_json(_BLOCKS.dump_json([table])) == [table]
        assert table.model_dump(mode="json")["rows"][0]["flags"][0]["name"] == "brightness"

    def test_a_list_of_numbers_is_still_a_chart_cell(self):
        """Adding flags to the cell union leaves a sparkline's counts as numbers."""
        (row,) = Table(columns=[Column(key="h", kind="sparkline")], rows=[{"h": [1, 2.5]}]).rows
        assert row["h"] == [1.0, 2.5]

    def test_a_float32_flag_keeps_the_digits_it_prints_with(self):
        import numpy as np

        flag = Flag.model_validate({**_FLAG.model_dump(), "value": np.float32(0.99), "bound": np.float32(0.84)})
        assert (flag.value, flag.bound) == (0.99, 0.84)

    def test_a_table_without_a_preview_leaves_it_out(self):
        table = Table(columns=[Column(key="k")], rows=[{"k": "a"}])
        assert table.preview is None
        assert "preview" not in table.model_dump(mode="json")

    def test_a_flag_names_its_direction(self):
        with pytest.raises(ValidationError):
            Flag.model_validate({**_FLAG.model_dump(), "direction": "sideways"})
