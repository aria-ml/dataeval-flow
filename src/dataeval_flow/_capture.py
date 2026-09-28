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

from dataeval_flow._blocks import Asset, Block, ItemRef, Table
from dataeval_flow._blocks._items import item_name, refs_in
from dataeval_flow._blocks._table import fair_shares, tables
from dataeval_flow._preview import NotAnImageError, preview

_logger = logging.getLogger(__name__)


def references(blocks: Sequence[Block], limit: int | None) -> list[ItemRef]:
    """The items the blocks' image columns name that get a thumbnail: at most *limit*, once each, in reading order.

    ``None`` takes every item named.

    The limit is shared evenly between the top-level blocks that name items, a report's findings, and
    each finding's share between its tables in the same way; a share more than its items need goes to
    the rest. A table gives its share row by row, then item by item within a group's cell. Rows past a
    share keep their references, and just have no thumbnail.
    """
    findings = [[_named(table) for table in tables([block])] for block in blocks]
    demands = [sum(map(len, named)) for named in findings]
    found: dict[ItemRef, None] = {}
    for named, share in zip(findings, fair_shares(demands, sum(demands) if limit is None else limit), strict=True):
        for refs, part in zip(named, fair_shares([len(refs) for refs in named], share), strict=True):
            found.update(dict.fromkeys(refs[:part]))
    return list(found)


def _named(table: Table) -> list[ItemRef]:
    """The items a table's image columns name, once each, row by row, then through a group's cell."""
    keys = [column.key for column in table.columns if column.kind == "image"]
    return list(dict.fromkeys(ref for row in table.rows for key in keys for ref in refs_in(row.get(key))))


def capture(
    blocks: Sequence[Block],
    datasets: Mapping[str, Any],
    value_ranges: Mapping[str, tuple[float, float] | None],
    *,
    limit: int | None,
) -> list[Asset]:
    """A thumbnail of each item the blocks name, at most *limit*, read from *datasets* by source name.

    Each source's items are read once each, in ascending order: the one pass a streaming dataset
    will need. *value_ranges* are the ranges the sources declare, which a float image is read by.
    Anything that stops an item's thumbnail, a failed read, a box not found, an item that's no
    image, costs that item its thumbnail and a warning, never the run.
    """
    wanted: dict[str, dict[int, list[ItemRef]]] = {}
    for ref in references(blocks, limit):
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
