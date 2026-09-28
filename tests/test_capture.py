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
from dataeval_flow._capture import LIMIT, ROWS, capture, references

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


class TestReferences:
    """Which items get a thumbnail: those the report names first, once each, within the caps."""

    def test_in_reading_order_table_by_table_row_by_row_then_through_a_group(self):
        blocks: list[Block] = [
            Section(title="Finding", blocks=[_images(_ref(9), [_ref(4), _ref(2, target=1)])]),
            _images(_ref(1, source="test")),
        ]
        assert references(blocks) == [_ref(9), _ref(4), _ref(2, target=1), _ref(1, source="test")]

    def test_an_item_named_twice_is_captured_once(self):
        assert references([_images(_ref(3)), _images([_ref(3), _ref(5)])]) == [_ref(3), _ref(5)]

    def test_only_image_columns_name_items(self):
        table = Table(columns=[Column(key="i", header="Item")], rows=[{"i": 3}])
        assert references([table]) == []

    def test_each_table_gives_its_first_50_rows(self):
        found = references([_images(*(_ref(i) for i in range(80))), _images(_ref(500))])
        assert found == [*(_ref(i) for i in range(ROWS)), _ref(500)]

    def test_a_result_gets_at_most_200_thumbnails(self):
        tables = [_images(*(_ref(t * 100 + i) for i in range(ROWS))) for t in range(5)]
        found = references(tables)
        assert len(found) == LIMIT
        assert found[-1] == _ref(349), "four tables of 50 fill it, and the fifth adds nothing"


class TestCapture:
    def test_each_item_is_read_once_in_ascending_order(self):
        """A whole item and its box are one read: the one pass a streaming dataset will need."""
        dataset = _Items()
        blocks = [_images(_ref(7), _ref(2), _ref(7, target=0), _ref(2))]
        assets = capture(blocks, {"train": dataset}, {})
        assert dataset.reads == [2, 7]
        assert [asset.item for asset in assets] == [_ref(2), _ref(7), _ref(7, target=0)]
        assert (assets[2].width, assets[2].height) == (6, 6), "box 1–5 widened by 10% each side, rounded outward"

    def test_each_source_is_read_from_its_own_dataset(self):
        train, test = _Items(), _Items()
        assets = capture([_images(_ref(1), _ref(2, source="test"))], {"train": train, "test": test}, {})
        assert (train.reads, test.reads) == ([1], [2])
        assert [asset.item.source for asset in assets] == ["train", "test"]

    def test_a_float_image_is_read_by_its_source_s_declared_range(self):
        """0.5 of a declared 0–0.5 is white; undeclared, a constant image has nothing to stretch, so it's grey."""
        dataset = _Items(image=np.full((1, 4, 4), 0.5, dtype=np.float32))
        (declared,) = capture([_images(_ref(0))], {"train": dataset}, {"train": (0.0, 0.5)})
        (undeclared,) = capture([_images(_ref(0))], {"train": dataset}, {"train": None})
        assert (_shade(declared), _shade(undeclared)) == (255, 128)

    def test_an_item_that_cannot_be_read_is_named_and_the_rest_are_captured(self, caplog):
        dataset = _Items(broken=(3,))
        with caplog.at_level(logging.WARNING):
            assets = capture([_images(_ref(3), _ref(4))], {"train": dataset}, {})
        assert [asset.item for asset in assets] == [_ref(4)]
        assert "Could not read item 3 of source 'train' for its thumbnail: cannot open item 3" in caplog.text

    def test_a_box_that_cannot_be_found_is_named_with_its_reason(self, caplog):
        with caplog.at_level(logging.WARNING):
            assets = capture([_images(_ref(3, target=4))], {"train": _Items()}, {})
        assert assets == []
        assert "No thumbnail for item 3 box 4 of source 'train': its annotation has no box 4" in caplog.text

    def test_items_that_are_not_images_are_named_with_one_warning_for_their_source(self, caplog):
        with caplog.at_level(logging.WARNING):
            assets = capture([_images(*(_ref(i) for i in range(5)))], {"train": _Items(image=np.zeros(64))}, {})
        assert assets == []
        assert caplog.text.count("holds items that aren't images") == 1

    @pytest.mark.parametrize(
        "datasets", [{}, {"train": None}, {"train": iter([1, 2])}], ids=["none", "unset", "stream"]
    )
    def test_a_source_that_cannot_be_read_by_index_is_skipped_with_one_warning(self, datasets, caplog):
        with caplog.at_level(logging.WARNING):
            assert capture([_images(_ref(1), _ref(2))], datasets, {}) == []
        assert caplog.text.count("Source 'train' can't be read by index, so its items have no thumbnails.") == 1


class TestTheRun:
    """Flow captures once a run returns, from the post-view datasets it read, and never fails a run over it."""

    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch, *, success: bool = True) -> Any:
        import dataeval_flow._orchestrator as orchestrator
        from dataeval_flow import run_tasks
        from dataeval_flow.config import TaskConfig
        from dataeval_flow.workflows import Finding
        from dataeval_flow.workflows.data_cleaning import DataCleaningConfig, DataCleaningResult
        from dataeval_flow.workflows.data_cleaning._outputs import (
            DataCleaningMetadata,
            DataCleaningOutput,
            DataCleaningRawOutput,
            DataCleaningReport,
        )
        from tests.evaluator_toys import toy_pipeline

        def naming_item_7(_runner: Any, _config: Any, _context: Any) -> DataCleaningResult:
            if not success:
                return DataCleaningResult.failed(type="data-cleaning", errors=["boom"])
            finding = Finding(title="Outliers", blocks=[_images(_ref(7, source="src"))])
            output = DataCleaningOutput(
                raw=DataCleaningRawOutput(dataset_size=12), report=DataCleaningReport(summary="s", findings=[finding])
            )
            return DataCleaningResult(
                type="data-cleaning", success=True, output=output, metadata=DataCleaningMetadata()
            )

        monkeypatch.setattr(orchestrator, "_run_target", naming_item_7)
        config = toy_pipeline(
            workflows=[DataCleaningConfig(name="clean", outlier_method="zscore", outlier_flags=["pixel"])],
            tasks=[TaskConfig(name="t", workflow="clean", sources="src")],
        )
        return run_tasks(config)["t"]

    def test_a_result_keeps_a_thumbnail_of_each_item_its_report_names(self, monkeypatch):
        result = self._run(monkeypatch)
        (asset,) = result.assets
        assert asset.item == _ref(7, source="src")
        assert _shade(asset) == 255, "the toy dataset's item 7 is solid white"
        assert result.to_dict()["assets"][0]["item"] == {"source": "src", "index": 7}

    def test_a_failed_run_is_not_captured(self, monkeypatch):
        assert self._run(monkeypatch, success=False).assets == []

    def test_a_capture_that_fails_costs_only_the_thumbnails(self, monkeypatch, caplog):
        import dataeval_flow._capture as capture_module

        def broken(*_args: Any) -> list[Asset]:
            raise RuntimeError("capture broke")

        monkeypatch.setattr(capture_module, "capture", broken)
        with caplog.at_level(logging.WARNING):
            result = self._run(monkeypatch)
        assert result.success
        assert result.assets == []
        assert "Could not capture the report's thumbnails, so it names its items instead." in caplog.text
