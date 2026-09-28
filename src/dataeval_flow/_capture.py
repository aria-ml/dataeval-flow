"""Thumbnails of the items a report names, captured from the run's datasets once the run is done.

A workflow names the items its findings are about, with item references in image columns, and never
touches pixels. Once the run returns, while its datasets are still at hand, Flow walks the report,
reads each named item once, and keeps a thumbnail of it in the result for the HTML report to show.
Capture never fails a run and never changes a finding: an item it can't preview is named instead.
"""

__all__ = ["capture", "references"]

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from dataeval_flow._blocks import Asset, Block, ItemRef
from dataeval_flow._blocks._items import item_name, refs_in
from dataeval_flow._blocks._table import tables
from dataeval_flow._preview import NotAnImageError, preview

_logger = logging.getLogger(__name__)

# Items are taken from at most this many rows of each table, and at most this many per result.
ROWS = 50
LIMIT = 200


def references(blocks: Sequence[Block]) -> list[ItemRef]:
    """Every item the blocks' image columns name, once each, in reading order, within the caps.

    Table by table, row by row, then item by item within a group's cell; from each table's first 50
    rows, and 200 items in all. Rows past a cap keep their references, and just have no thumbnail.
    """
    found: dict[ItemRef, None] = {}
    for table in tables(blocks):
        keys = [column.key for column in table.columns if column.kind == "image"]
        for row in table.rows[:ROWS]:
            for ref in (ref for key in keys for ref in refs_in(row.get(key))):
                found.setdefault(ref)
                if len(found) == LIMIT:
                    return list(found)
    return list(found)


def capture(
    blocks: Sequence[Block],
    datasets: Mapping[str, Any],
    value_ranges: Mapping[str, tuple[float, float] | None],
) -> list[Asset]:
    """A thumbnail of each item the blocks name, read from *datasets* by source name.

    Each source's items are read once each, in ascending order: the one pass a streaming dataset
    will need. *value_ranges* are the ranges the sources declare, which a float image is read by.
    Anything that stops an item's thumbnail, a failed read, a box not found, an item that's no
    image, costs that item its thumbnail and a warning, never the run.
    """
    wanted: dict[str, dict[int, list[ItemRef]]] = {}
    for ref in references(blocks):
        wanted.setdefault(ref.source, {}).setdefault(ref.index, []).append(ref)
    assets: list[Asset] = []
    for source, items in wanted.items():
        dataset = datasets.get(source)
        if dataset is None or not hasattr(dataset, "__getitem__"):
            _logger.warning("Source %r can't be read by index, so its items have no thumbnails.", source)
            continue
        assets.extend(_source_assets(source, dataset, items, value_ranges.get(source)))
    return assets


def _source_assets(
    source: str, dataset: Any, items: Mapping[int, list[ItemRef]], value_range: tuple[float, float] | None
) -> list[Asset]:
    """The thumbnails of one source's items, each item read once; a warning for each that has none."""
    assets: list[Asset] = []
    unimaged = False
    for index in sorted(items):
        try:
            item = dataset[index]
            image, target = item[0], item[1]
        except Exception as error:  # noqa: BLE001 - a dataset may raise anything; the run has already finished
            _logger.warning("Could not read item %d of source %r for its thumbnail: %s", index, source, error)
            continue
        for ref in items[index]:
            try:
                assets.append(preview(ref, image, target, value_range))
            except NotAnImageError:
                unimaged = True
            except Exception as error:  # noqa: BLE001 - one item's thumbnail never costs the run
                _logger.warning("No thumbnail for item %s of source %r: %s", item_name(ref), source, error)
    if unimaged:
        _logger.warning(
            "Source %r holds items that aren't images, which have no thumbnail yet; the report names them instead.",
            source,
        )
    return assets
