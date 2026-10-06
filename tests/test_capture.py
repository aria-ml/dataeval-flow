"""Capture: a thumbnail of each item a report names, read once from the run's datasets, and never a failed run."""

import base64
import io
import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from dataeval_flow._blocks import Asset, Block, Column, ItemRef, Section, Table
from dataeval_flow._blocks._items import refs_in
from dataeval_flow._capture import capture, references

pytestmark = pytest.mark.required


def _images(*cells: Any, key: str = "image") -> Table:
    return Table(columns=[Column(key=key, kind="image"), Column(key="n", header="N")], rows=[{key: c} for c in cells])


def _ref(index: int, source: str = "train", target: int | None = None) -> ItemRef:
    return ItemRef(source=source, index=index, target=target)


class _Items:
    """A dataset of 8 × 8 images, recording the order it's read in; *broken* indices raise when read."""

    def __init__(self, count: int = 100, *, image: Any = None, broken: tuple[int, ...] = ()) -> None:
        self.count, self.image, self.broken, self.reads = count, image, broken, []

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        self.reads.append(index)
        if index in self.broken:
            raise OSError(f"cannot open item {index}")
        image = self.image if self.image is not None else np.full((3, 8, 8), index, dtype=np.uint8)
        return image, SimpleNamespace(boxes=np.array([[1.0, 1.0, 5.0, 5.0]])), {}


def _shade(asset: Asset) -> int:
    """A flat thumbnail's grey level, decoded."""
    picture = Image.open(io.BytesIO(base64.b64decode(asset.data)))
    return round(float(np.asarray(picture.convert("L")).mean()))


def _finding(*sizes: int, start: int = 0) -> Section:
    """A finding with a table of each size, naming items from *start* on, each named once."""
    blocks: list[Block] = []
    for size in sizes:
        blocks.append(_images(*(_ref(i) for i in range(start, start + size))))
        start += size
    return Section(title="Finding", blocks=blocks)


def _per_finding(found: list[ItemRef], findings: list[Section]) -> list[int]:
    """How many of each finding's items were taken."""
    named = [
        {ref for table in finding.blocks if isinstance(table, Table) for ref in _column(table)} for finding in findings
    ]
    return [sum(ref in items for ref in found) for items in named]


def _column(table: Table) -> list[ItemRef]:
    return [ref for row in table.rows for ref in refs_in(row.get("image"))]


class TestReferences:
    """Which items get a thumbnail: those the report names first, once each, the limit shared between findings."""

    def test_in_reading_order_table_by_table_row_by_row_then_through_a_group(self):
        blocks: list[Block] = [
            Section(title="Finding", blocks=[_images(_ref(9), [_ref(4), _ref(2, target=1)])]),
            _images(_ref(1, source="test")),
        ]
        assert references(blocks, 200) == [_ref(9), _ref(4), _ref(2, target=1), _ref(1, source="test")]

    def test_an_item_named_twice_is_captured_once(self):
        assert references([_images(_ref(3)), _images([_ref(3), _ref(5)])], 200) == [_ref(3), _ref(5)]

    def test_only_image_columns_name_items(self):
        table = Table(columns=[Column(key="i", header="Item")], rows=[{"i": 3}])
        assert references([table], 200) == []

    def test_the_limit_is_shared_evenly_between_the_findings_that_name_items(self):
        findings = [_finding(100, start=n * 100) for n in range(4)]
        found = references([Section(title="Summary", blocks=[]), *findings], 200)
        assert _per_finding(found, findings) == [50, 50, 50, 50]
        assert found[:2] == [_ref(0), _ref(1)], "each finding's share is its first items, in reading order"

    def test_a_finding_that_needs_fewer_passes_its_spare_to_the_rest(self):
        findings = [_finding(10), _finding(100, start=10), _finding(100, start=110), _finding(100, start=210)]
        assert _per_finding(references(findings, 200), findings) == [10, 63, 63, 64]

    def test_a_finding_s_share_is_split_between_its_tables_the_same_way(self):
        first, second = _finding(100, 3), _finding(100, start=103)
        found = references([first, second], 20)
        assert _per_finding(found, [first, second]) == [10, 10]
        assert [ref.index for ref in found[:10]] == [0, 1, 2, 3, 4, 5, 6, 100, 101, 102]

    def test_a_table_gives_as_many_rows_as_its_share_allows(self):
        assert len(references([_finding(300)], 300)) == 300

    def test_each_item_of_a_group_counts_against_the_share(self):
        assert references([_images([_ref(i) for i in range(8)])], 5) == [_ref(i) for i in range(5)]

    def test_a_limit_of_zero_names_none(self):
        assert references([_finding(10)], 0) == []

    def test_no_limit_names_every_item(self):
        assert len(references([_finding(300), _finding(700, start=300)], None)) == 1000

    def test_findings_naming_the_same_items_still_fill_the_limit(self):
        assert len(references([_finding(300), _finding(300)], 200)) == 200
        assert len(references([_finding(300), _finding(10)], 200)) == 200, "an unusable share goes round again"

    def test_an_item_another_finding_took_costs_nothing(self):
        """A finding listing the same items in another order still gets thumbnails in its first rows."""
        backwards = Section(title="Finding", blocks=[_images(*(_ref(i) for i in reversed(range(300))))])
        assert set(references([_finding(300), backwards], 200)) == {_ref(i) for i in [*range(100), *range(200, 300)]}


class TestCapture:
    def test_each_item_is_read_once_in_ascending_order(self):
        """A whole item and its box are one read: the one pass a streaming dataset will need."""
        dataset = _Items()
        blocks = [_images(_ref(7), _ref(2), _ref(7, target=0), _ref(2))]
        assets = capture(blocks, {"train": dataset}, {}, limit=200)
        assert dataset.reads == [2, 7]
        assert [asset.item for asset in assets] == [_ref(2), _ref(7), _ref(7, target=0)]
        assert (assets[2].width, assets[2].height) == (6, 6), "box 1–5 widened by 10% each side, rounded outward"

    def test_each_source_is_read_from_its_own_dataset(self):
        train, test = _Items(), _Items()
        assets = capture([_images(_ref(1), _ref(2, source="test"))], {"train": train, "test": test}, {}, limit=200)
        assert (train.reads, test.reads) == ([1], [2])
        assert [asset.item.source for asset in assets] == ["train", "test"]

    def test_a_float_image_is_read_by_its_source_s_declared_range(self):
        """0.5 of a declared 0–0.5 is white; undeclared, a constant image has nothing to stretch, so it's grey."""
        dataset = _Items(image=np.full((1, 4, 4), 0.5, dtype=np.float32))
        (declared,) = capture([_images(_ref(0))], {"train": dataset}, {"train": (0.0, 0.5)}, limit=200)
        (undeclared,) = capture([_images(_ref(0))], {"train": dataset}, {"train": None}, limit=200)
        assert (_shade(declared), _shade(undeclared)) == (255, 128)

    def test_an_item_that_cannot_be_read_is_named_and_the_rest_are_captured(self, caplog):
        dataset = _Items(broken=(3,))
        with caplog.at_level(logging.WARNING):
            assets = capture([_images(_ref(3), _ref(4))], {"train": dataset}, {}, limit=200)
        assert [asset.item for asset in assets] == [_ref(4)]
        assert "Could not read item 3 of source 'train' for its thumbnail: cannot open item 3" in caplog.text

    def test_a_box_that_cannot_be_found_is_named_with_its_reason(self, caplog):
        with caplog.at_level(logging.WARNING):
            assets = capture([_images(_ref(3, target=4))], {"train": _Items()}, {}, limit=200)
        assert assets == []
        assert "No thumbnail for item 3 box 4 of source 'train': its annotation has no box 4" in caplog.text

    def test_items_that_are_not_images_are_named_with_one_warning_for_their_source(self, caplog):
        with caplog.at_level(logging.WARNING):
            assets = capture(
                [_images(*(_ref(i) for i in range(5)))], {"train": _Items(image=np.zeros(64))}, {}, limit=200
            )
        assert assets == []
        assert caplog.text.count("holds items that aren't images") == 1

    @pytest.mark.parametrize(
        "datasets", [{}, {"train": None}, {"train": iter([1, 2])}], ids=["none", "unset", "stream"]
    )
    def test_a_source_that_cannot_be_read_by_index_is_skipped_with_one_warning(self, datasets, caplog):
        with caplog.at_level(logging.WARNING):
            assert capture([_images(_ref(1), _ref(2))], datasets, {}, limit=200) == []
        assert caplog.text.count("Source 'train' can't be read by index, so its items have no thumbnails.") == 1


class TestTheRun:
    """Flow captures once a run returns, from the post-view datasets it read, and never fails a run over it."""

    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch, *, success: bool = True, max_images: int | None = None) -> Any:
        """An outliers task over the toy dataset, whose report names item 7, the one image it flags."""
        from dataeval_flow import run_tasks
        from dataeval_flow.config import ResultConfig, TaskConfig
        from dataeval_flow.evaluators.quality import OutliersConfig, OutliersEvaluator
        from tests.evaluator_toys import toy_pipeline

        def boom(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("boom")

        if not success:
            monkeypatch.setattr(OutliersEvaluator, "run", boom)
        config = toy_pipeline(
            evaluators=[OutliersConfig(name="clean", flags=["pixel", "visual"], outlier_threshold="zscore")],
            tasks=[TaskConfig(name="t", workflow="clean", kind="evaluator", sources="src")],
        )
        if max_images is not None:
            config.result = ResultConfig(max_images=max_images)
        return run_tasks(config)["t"]

    def test_a_result_keeps_a_thumbnail_of_each_item_its_report_names(self, monkeypatch):
        result = self._run(monkeypatch)
        (asset,) = result.assets
        assert asset.item == _ref(7, source="src")
        assert _shade(asset) == 255, "the toy dataset's item 7 is solid white"
        assert result.to_dict()["assets"][0]["item"] == {"source": "src", "index": 7}

    def test_the_configured_limit_caps_the_result_s_thumbnails(self, monkeypatch):
        assert self._run(monkeypatch, max_images=0).assets == []

    def test_a_failed_run_is_not_captured(self, monkeypatch):
        assert self._run(monkeypatch, success=False).assets == []

    def test_a_capture_that_fails_costs_only_the_thumbnails(self, monkeypatch, caplog):
        import dataeval_flow._capture as capture_module

        def broken(*_args: Any, **_kwargs: Any) -> list[Asset]:
            raise RuntimeError("capture broke")

        monkeypatch.setattr(capture_module, "capture", broken)
        with caplog.at_level(logging.WARNING):
            result = self._run(monkeypatch)
        assert result.success
        assert result.assets == []
        assert "Could not capture the report's thumbnails, so it names its items instead." in caplog.text
