"""A run's items, read back from their sources and served only while they match the manifest the run recorded.

A run records a source's manifest when it runs a ``content-digest`` task over it: each item's content and metadata
hashes, its position, and its position beneath the source's view. To serve item ``i``, the service loads the source
from the run's own pipeline snapshot, finds the item the run read at ``i`` by that position beneath the view, so a view
that reorders items differently on each load still finds it, and hashes it again. Only a match is served: anything else
is ``input_changed``, a source that will not load is ``input_unavailable``, and a run with no manifest of the source is
``evidence_unavailable``. Nothing is ever served in place of what the run read.
"""

from __future__ import annotations

__all__ = ["Evidence", "PAGE_SIZE"]

import functools
import io
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dataeval_flow._digest import DatasetManifest, _parts, item_hashes

if TYPE_CHECKING:
    from dataeval_flow._service._store import RunStore

SCHEMA = 1
PAGE_SIZE = 100
_MANIFEST = "content-digest.json"


class UnknownItemError(LookupError):
    """The run's pipeline names no such source, or its manifest holds no such item."""


class Evidence:
    """The items of the runs in `store`, read from datasets under `data_root`."""

    def __init__(self, store: RunStore, data_root: Path) -> None:
        self._store = store
        self._data_root = data_root
        # ponytail: a few loaded sources per process, loaded again when evicted; a reader process if loads stall it.
        self._source = functools.lru_cache(maxsize=4)(self._load_source)
        self._manifest = functools.lru_cache(maxsize=8)(self._load_manifest)
        self._drawn = functools.lru_cache(maxsize=32)(self._drawn_per_task)

    def items(self, run_id: str, source: str, offset: int, limit: int) -> dict[str, Any]:
        """A page of the source's items, in the order the run read them."""
        manifest, reason = self._recorded(run_id, source)
        if manifest is None:
            page = {"status": "evidence_unavailable", "reason": reason, "total": 0, "items": []}
            return {"schema": SCHEMA, "source": source, "offset": offset, "limit": limit, **page}
        total = len(manifest.entries)
        items = [self.item(run_id, source, index) for index in range(offset, min(offset + limit, total))]
        return {"schema": SCHEMA, "source": source, "status": "recorded", "total": total, "offset": offset,
                "limit": limit, "items": items}  # fmt: skip

    def item(self, run_id: str, source: str, index: int) -> dict[str, Any]:
        """One item: its boxes or label, metadata and size where it still matches the run's record, else why not."""
        status, reason, datum, manifest = self._read(run_id, source, index)
        found: dict[str, Any] = {"schema": SCHEMA, "source": source, "index": index, "status": status}
        if manifest is not None:
            entry = manifest.entries[index]
            found |= {"root_index": entry.root, "id": entry.id}
        if datum is None or manifest is None:
            return found | {"reason": reason}
        image, target, meta = datum
        shape = np.asarray(image).shape
        found |= {
            "height": int(shape[-2]),
            "width": int(shape[-1]),
            "channels": int(shape[0]) if len(shape) == 3 else 1,
        }
        found |= _labels(target, manifest.classes)
        return found | {
            "metadata": _jsonable(meta or {}),
            "image_url": f"/v1/runs/{run_id}/items/{source}/{index}/image",
        }

    def image(self, run_id: str, source: str, index: int, target: int | None, max_side: int | None) -> bytes | None:
        """The item's image as a PNG, or the box `target` cropped from it, at most `max_side` pixels across; ``None``
        where the item no longer matches the run's record."""
        from dataeval_flow._preview import render

        status, _, datum, _ = self._read(run_id, source, index)
        if status != "verified" or datum is None:
            return None
        image, annotation, _ = datum
        picture = render(image, annotation, target, self._source(run_id, source)[2])
        if max_side is not None:
            picture.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        return buffer.getvalue()

    def _read(self, run_id: str, source: str, index: int) -> tuple[str, str | None, Any, DatasetManifest | None]:
        """The item's status, why it is not verified, the item where it is, and the manifest it was checked against."""
        manifest, reason = self._recorded(run_id, source)
        if manifest is None:
            return "evidence_unavailable", reason, None, None
        if not 0 <= index < len(manifest.entries):
            raise UnknownItemError(f"Source '{source}' held {len(manifest.entries)} items in this run")
        entry = manifest.entries[index]
        try:
            dataset, positions, _ = self._source(run_id, source)
        except Exception as error:  # noqa: BLE001 - any failure to load is the input being unavailable
            return "input_unavailable", f"The source could not be loaded: {error}", None, manifest
        position = index if entry.root is None else positions.get(entry.root)
        if position is None:
            return "input_changed", "The source no longer holds this item.", None, manifest
        try:
            datum = dataset[position]
        except Exception as error:  # noqa: BLE001 - any failure to read is the item being unavailable
            return "input_unavailable", f"The item could not be read: {error}", None, manifest
        content, metadata = item_hashes(datum)
        if content != entry.content:
            return "input_changed", "The item's image or labels differ from what the run read.", None, manifest
        if entry.metadata is not None and metadata != entry.metadata:
            return "input_changed", "The item's metadata differs from what the run read.", None, manifest
        return "verified", None, _parts(datum), manifest

    def _recorded(self, run_id: str, source: str) -> tuple[DatasetManifest | None, str | None]:
        """The manifest the run recorded of `source`, or ``None`` and why there is none to resolve evidence by; a source
        the pipeline doesn't name is unknown."""
        pipeline = self._store.get(run_id)["pipeline"]
        if source not in {entry.get("name") for entry in pipeline.get("sources") or []}:
            raise UnknownItemError(f"This run's pipeline names no source '{source}'")
        if reason := self._drawn(run_id, source):
            return None, reason
        try:
            return self._manifest(run_id, source), None
        except LookupError:  # not cached, so a manifest the run writes later is found then
            return None, f"This run recorded no manifest of source '{source}': run a content-digest task over it."

    def _drawn_per_task(self, run_id: str, source: str) -> str | None:
        """Why each task of the run drew its own items from `source`, so that no manifest names another task's."""
        from dataeval_flow._sources import drawn_per_task
        from dataeval_flow.config._loader import load_config

        return drawn_per_task(load_config(self._store.directory(run_id) / "pipeline.json"), source)

    def _load_manifest(self, run_id: str, source: str) -> DatasetManifest:
        """The manifest a ``content-digest`` task of the run wrote for `source`; ``LookupError`` where none did."""
        for path in sorted((self._store.directory(run_id) / "results" / "manifests").glob(f"*/{_MANIFEST}")):
            manifest = DatasetManifest.load(path)
            if manifest.source == source:
                return manifest
        raise LookupError(source)

    def _load_source(self, run_id: str, source: str) -> tuple[Any, dict[int, int], tuple[float, float] | None]:
        """The source as the run's pipeline snapshot loads it, each item's position by its position beneath the views,
        and the value range its datasets declare."""
        from dataeval_flow._orchestrator import _value_range_of
        from dataeval_flow._sources import _refuse_unseeded, resolve_source
        from dataeval_flow._view import root_indices
        from dataeval_flow.config._loader import load_config

        config = load_config(self._store.directory(run_id) / "pipeline.json")
        resolved = resolve_source(source, config, self._data_root)
        _refuse_unseeded(resolved)
        dataset = resolved.realized()
        positions = {root: position for position, root in enumerate(root_indices(dataset))}
        return dataset, positions, _value_range_of(resolved)


def _labels(target: Any, classes: Mapping[int, str]) -> dict[str, Any]:
    """A detection target's boxes, as ``x0, y0, x1, y1`` in pixels, or a classification target's label."""
    boxes, labels = getattr(target, "boxes", None), getattr(target, "labels", None)
    if boxes is not None and labels is not None:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(labels).reshape(-1)
        targets = [
            {"target": k, "box": [float(v) for v in box], "label": int(label), "class": classes.get(int(label))}
            for k, (box, label) in enumerate(zip(boxes, labels, strict=True))
        ]
        return {"box_format": "xyxy", "targets": targets}
    scores = np.asarray(target if target is not None else [], dtype=np.float64).reshape(-1)
    if not scores.size or not np.any(scores):
        return {"label": None}
    label = int(np.argmax(scores))
    return {"label": {"label": label, "class": classes.get(label)}}


def _jsonable(value: Any) -> Any:
    """`value` as plain JSON, as a manifest hashes it, with any value that has no JSON form shown as its text."""
    from dataeval_flow._digest import _plain

    try:
        return _plain(value)
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return str(value)
