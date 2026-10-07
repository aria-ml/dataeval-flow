"""A dataset's identity as SHA-256 digests over every item: what a model trains on, and the metadata beside it."""

__all__ = [
    "DatasetDigest",
    "DatasetManifest",
    "ManifestDiff",
    "ManifestEntry",
    "dataset_digest",
    "dataset_manifest",
    "item_hashes",
]

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

# Each digest opens with its scheme, so a later change to what is hashed gives new digests, never colliding ones.
SCHEME = 1
_CONTENT_SCHEME = f"dataeval-flow content digest {SCHEME}".encode()
_METADATA_SCHEME = f"dataeval-flow metadata digest {SCHEME}".encode()


@dataclass(frozen=True)
class DatasetDigest:
    """A dataset's identity: SHA-256 digests over every item, whatever their order.

    ``content`` covers what a model trains on: each item's image and labels, and the dataset's class names.
    ``metadata`` covers each item's metadata as attached to that item's content, so moving metadata from one item to
    another changes it. ``items`` is how many items were read. Two datasets holding the same items, in any order,
    share both digests; adding, removing or editing any item changes at least one of them.
    """

    content: str
    """The content digest, 64 hex characters: every item's image and labels, and the class names."""
    metadata: str
    """The metadata digest, 64 hex characters: every item's metadata as canonical JSON, bound to its item's content."""
    items: int
    """How many items the dataset held."""
    scheme: int
    """The version of the digest scheme, so a change of scheme can't be mistaken for changed data."""


@dataclass(frozen=True)
class ManifestEntry:
    """One item of a manifest: where it sat, its metadata ``id`` where it has one, and its hashes."""

    index: int
    """The item's position in the dataset it was read from."""
    id: str | int | None
    """The item's metadata ``id``, or ``None`` where it has none."""
    content: str
    """The SHA-256 of the item's image and labels, which the content digest sorts and hashes."""
    root: int | None = None
    """The item's position in the dataset beneath the views it was read through; ``None`` in a manifest written
    before manifests recorded it."""
    metadata: str | None = None
    """The SHA-256 of the item's metadata bound to its content, which the metadata digest sorts and hashes; ``None``
    in a manifest written before manifests recorded it."""


@dataclass(frozen=True)
class ManifestDiff:
    """What a dataset changed against a manifest: items by ``id`` where every item on both sides has a unique one,
    else by index. Falsy when nothing differs."""

    changed: tuple[str | int, ...] = ()
    """The ids of items on both sides whose image or labels differ."""
    missing: tuple[str | int, ...] = ()
    """The manifest's items the dataset doesn't hold: by id, or by their index in the manifest."""
    added: tuple[str | int, ...] = ()
    """The dataset's items the manifest doesn't hold: by id, or by their index in the dataset."""
    classes: bool = False
    """Whether the class names differ."""

    def __bool__(self) -> bool:
        """Whether anything differs."""
        return bool(self.changed or self.missing or self.added or self.classes)


@dataclass(frozen=True)
class DatasetManifest:
    """A dataset's digest with each item's content hash and its class names, so a mismatch can name what changed."""

    digest: DatasetDigest
    """The digest :func:`dataset_digest` gives."""
    entries: tuple[ManifestEntry, ...]
    """One entry per item, in the dataset's order."""
    classes: dict[int, str]
    """The class names by index, which the content digest covers."""
    source: str | None = None
    """The source the items were read from, where a run recorded one."""

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the manifest to `path` as JSON, making its directory. It is replaced whole, so a reader never sees one
        half-written."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scheme": self.digest.scheme,
            "content": self.digest.content,
            "metadata": self.digest.metadata,
            "items": self.digest.items,
            "classes": {str(index): name for index, name in self.classes.items()},
            "source": self.source,
            "entries": [
                {
                    "index": entry.index,
                    "id": entry.id,
                    "content": entry.content,
                    "root": entry.root,
                    "metadata": entry.metadata,
                }
                for entry in self.entries
            ],
        }
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "DatasetManifest":
        """Read a manifest :meth:`save` wrote. One written by another digest scheme is refused."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("scheme") != SCHEME:
            raise ValueError(f"{path} is a scheme {data.get('scheme')} manifest; this Flow reads scheme {SCHEME}.")
        digest = DatasetDigest(content=data["content"], metadata=data["metadata"], items=data["items"], scheme=SCHEME)
        entries = tuple(
            ManifestEntry(entry["index"], entry["id"], entry["content"], entry.get("root"), entry.get("metadata"))
            for entry in data["entries"]
        )
        classes = {int(index): name for index, name in data["classes"].items()}
        return cls(digest, entries, classes, data.get("source"))

    def compare(self, other: "DatasetManifest") -> ManifestDiff:
        """What `other`, the data as it is now, changed against this manifest, the data as recorded. Content only:
        the metadata digest covers metadata as stored, paths included, which differ between machines."""
        if self.digest.content == other.digest.content:
            return ManifestDiff()
        classes = self.classes != other.classes
        mine, theirs = _by_id(self.entries), _by_id(other.entries)
        if mine is not None and theirs is not None:
            return ManifestDiff(
                changed=tuple(key for key in mine if key in theirs and mine[key] != theirs[key]),
                missing=tuple(key for key in mine if key not in theirs),
                added=tuple(key for key in theirs if key not in mine),
                classes=classes,
            )
        recorded = Counter(entry.content for entry in self.entries)
        current = Counter(entry.content for entry in other.entries)
        return ManifestDiff(
            missing=_unmatched(self.entries, recorded - current),
            added=_unmatched(other.entries, current - recorded),
            classes=classes,
        )


def dataset_digest(dataset: Any) -> DatasetDigest:
    """Digest every item of `dataset`, as the ``content-digest`` evaluator does.

    A training job calls it before it trains and compares the result with the digests a run recorded, so it can refuse
    data that is not what the run recorded. It reads each item once, never through a cache.

    Each item's image is hashed as its dtype, shape and bytes. Its target is hashed the same way: the array for
    classification, and the ``boxes`` and ``labels`` arrays for detection (``scores`` don't count). Its metadata is
    hashed as JSON with sorted keys, NumPy and tensor values as plain numbers and lists, NaN as ``null``, dates as
    ISO 8601 and bytes as hex; a value with no stable form, such as an arbitrary object, raises ``TypeError``. The
    content digest is the SHA-256 of the sorted item hashes, the item count and the class names. An item's metadata
    hash covers its metadata and its content hash, so the metadata digest, the same over those hashes, also changes
    when metadata moves between items. The content digest never depends on metadata.

    The images are hashed as decoded pixels, which are what a model sees. A different image decoder can change them
    slightly, so compare digests made with the same libraries (a result records them in ``library_versions``).

    Parameters
    ----------
    dataset : AnnotatedDataset
        Any dataset Flow reads: ``len()`` and indexing, each item an ``(image, target, metadata)`` tuple, and its class
        names in ``dataset.metadata["index2label"]`` where it declares them. Pass the dataset as Flow loads it, before
        any training transform: from :func:`~dataeval_flow.load_dataset` with the ``datasets:`` entry's format and
        options, through the same views the recorded source applied, which
        :func:`~dataeval_flow.load_source` applies for you. Data in another shape gives another digest:
        resized or normalized images, ``(image, int)`` tuples, images in height-width-channel order, or a dataset
        with no ``index2label``.

    Returns
    -------
    DatasetDigest
        The content digest, the metadata digest and the item count.

    Examples
    --------
    >>> from dataeval_flow import dataset_digest
    >>> digest = dataset_digest(train)  # doctest: +SKIP
    >>> assert digest.content == recorded["content"], "not the data the run recorded"  # doctest: +SKIP
    """
    return dataset_manifest(dataset).digest


def dataset_manifest(dataset: Any) -> DatasetManifest:
    """Digest every item of `dataset` as :func:`dataset_digest` does, keeping each item's content hash, so a later
    mismatch can name the items that changed (see :meth:`DatasetManifest.compare`).

    Parameters
    ----------
    dataset : AnnotatedDataset
        As :func:`dataset_digest` takes it.

    Returns
    -------
    DatasetManifest
        The digest, one entry per item in the dataset's order, and the class names.
    """
    from dataeval_flow._view import root_indices

    entries: list[ManifestEntry] = []
    for index, root in enumerate(root_indices(dataset)):
        datum = dataset[index]
        content, metadata = item_hashes(datum)
        entries.append(ManifestEntry(index, _item_id(_parts(datum)[2]), content, root, metadata))
    hashes = [str(entry.metadata) for entry in entries]
    count = len(entries).to_bytes(8, "little")
    classes = _index2label(dataset)
    names = _canonical_json(sorted(classes.items()))
    digest = DatasetDigest(
        content=_hash([_CONTENT_SCHEME, count, names, *(item.encode() for item in sorted(e.content for e in entries))]),
        metadata=_hash([_METADATA_SCHEME, count, *(item.encode() for item in sorted(hashes))]),
        items=len(entries),
        scheme=SCHEME,
    )
    return DatasetManifest(digest=digest, entries=tuple(entries), classes=classes)


def item_hashes(datum: Any) -> tuple[str, str]:
    """One item's content hash and metadata hash, as a manifest records them: the image and labels, and the metadata
    bound to that content. `datum` is what indexing a dataset gives, an ``(image, target, metadata)`` tuple."""
    image, target, meta = _parts(datum)
    content = _hash([*_array_parts(image), *_target_parts(target)])
    return content, _hash([content.encode(), _canonical_json(meta)])


def _parts(datum: Any) -> tuple[Any, Any, Any]:
    """An item's image, target and metadata, ``None`` for any it lacks."""
    parts = datum if isinstance(datum, tuple) else (datum,)
    return parts[0], parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None


def _item_id(meta: Any) -> str | int | None:
    """An item's metadata ``id`` as JSON holds it, or ``None`` where it has none."""
    value = meta.get("id") if isinstance(meta, Mapping) else None
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, (int, str)) and not isinstance(value, bool)):
        return value
    return str(value)


def _by_id(entries: tuple[ManifestEntry, ...]) -> dict[str | int, str] | None:
    """Each entry's content by its id, or ``None`` where an entry has no id or two share one."""
    by_id: dict[str | int, str] = {}
    for entry in entries:
        if entry.id is None or entry.id in by_id:
            return None
        by_id[entry.id] = entry.content
    return by_id


def _unmatched(entries: tuple[ManifestEntry, ...], surplus: "Counter[str]") -> tuple[int, ...]:
    """The indices of `entries` whose content `surplus` counts, as many of each as it counts."""
    left = Counter(surplus)
    found: list[int] = []
    for entry in entries:
        if left[entry.content] > 0:
            left[entry.content] -= 1
            found.append(entry.index)
    return tuple(found)


def _hash(parts: Iterable[bytes]) -> str:
    """The SHA-256 of `parts`, each prefixed with its length, so no two sequences of parts hash alike."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "little"))
        hasher.update(part)
    return hasher.hexdigest()


def _array_parts(value: Any) -> list[bytes]:
    """An array's dtype, shape and bytes in C order, as NumPy reads it. An array of objects has no stable bytes."""
    from dataeval.utils import as_numpy

    array = np.ascontiguousarray(as_numpy(value))
    if array.dtype.hasobject:
        raise TypeError(f"Can't digest an array of Python objects ({type(value).__name__}).")
    return [array.dtype.str.encode(), str(array.shape).encode(), array.tobytes()]


def _target_parts(target: Any) -> list[bytes]:
    """A target's labels: boxes and labels for detection, the array for classification, a marker where there is none.

    A detection target's ``scores`` are left out: ground truth carries no confidence a training loader must match.
    """
    if target is None:
        return [b"none"]
    boxes, labels = getattr(target, "boxes", None), getattr(target, "labels", None)
    if boxes is not None and labels is not None:
        return [b"detection", *_array_parts(boxes), *_array_parts(labels)]
    try:
        return [b"array", *_array_parts(target)]
    except (TypeError, ValueError):
        return [b"other", _canonical_json(target)]


def _index2label(dataset: Any) -> dict[int, str]:
    """The dataset's class names by index, or none where it declares none."""
    declared = getattr(dataset, "metadata", None)
    return dict(declared.get("index2label") or {}) if isinstance(declared, Mapping) else {}


def _canonical_json(value: Any) -> bytes:
    """`value` as JSON with sorted keys and no spaces, so equal values give equal bytes (see :func:`_plain`)."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _plain(value: Any) -> Any:
    """`value` in plain JSON types: NumPy scalars, arrays and tensors as numbers and lists, NaN as ``None``, dates as
    ISO 8601, bytes as hex, paths as text and sets sorted. Anything else raises ``TypeError``: its ``str`` could hold a
    memory address or a summary, and so give digests that differ between runs or agree for different values."""
    if isinstance(value, Mapping):
        # JSON keys are text, so {1: x} and {"1": x} still share a key.
        return {_plain_key(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(item) for item in value), key=repr)
    if isinstance(value, np.datetime64):
        unit = "D" if np.datetime_data(value.dtype)[0] in ("Y", "M", "W", "D") else "us"
        return _plain(value.astype(f"datetime64[{unit}]").item())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "M":
            return _plain(value[()]) if value.ndim == 0 else [_plain(item) for item in value]
        return _plain(value.tolist())
    if hasattr(value, "__array__"):
        from dataeval.utils import as_numpy

        return _plain(as_numpy(value).tolist())
    return _plain_leaf(value)


def _plain_key(key: Any) -> str:
    """A mapping key as JSON text, through the same strict path as a value: an opaque key raises ``TypeError``."""
    if isinstance(key, np.generic):
        key = key.item()
    leaf = _plain_leaf(key)
    return leaf if isinstance(leaf, str) else json.dumps(leaf)


def _plain_leaf(value: Any) -> Any:
    """A scalar, date, bytes or path in plain JSON form; anything else raises ``TypeError``."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(
        f"Can't digest a {type(value).__name__} in metadata or a target: it has no stable form. "
        "Convert it to numbers, text, lists or mappings."
    )
