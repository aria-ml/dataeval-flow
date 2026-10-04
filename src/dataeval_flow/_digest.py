"""A dataset's identity as SHA-256 digests over every item: what a model trains on, and the metadata beside it."""

__all__ = ["DatasetDigest", "dataset_digest"]

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
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


def dataset_digest(dataset: Any) -> DatasetDigest:
    """Digest every item of `dataset`, as the ``content-digest`` evaluator does.

    A training job calls it on the data it is about to train on and compares the result with the digests a run
    recorded, so it can refuse data that is not what was audited. It reads each item once, never through a cache.

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
        names in ``dataset.metadata["index2label"]`` where it declares them. Pass the data as it will be trained on:
        read through the same views the audited source applied.

    Returns
    -------
    DatasetDigest
        The content digest, the metadata digest and the item count.

    Examples
    --------
    >>> from dataeval_flow import dataset_digest
    >>> digest = dataset_digest(train)  # doctest: +SKIP
    >>> assert digest.content == recorded["content"], "not the data that was audited"  # doctest: +SKIP
    """
    contents: list[str] = []
    metadata: list[str] = []
    for index in range(len(dataset)):
        datum = dataset[index]
        parts = datum if isinstance(datum, tuple) else (datum,)
        image = parts[0]  # before the len() checks below, which narrow `parts` to include tuple[()]
        target = parts[1] if len(parts) > 1 else None
        content = _hash([*_array_parts(image), *_target_parts(target)])
        contents.append(content)
        metadata.append(_hash([content.encode(), _canonical_json(parts[2] if len(parts) > 2 else None)]))
    count = len(contents).to_bytes(8, "little")
    names = _canonical_json(sorted(_index2label(dataset).items()))
    return DatasetDigest(
        content=_hash([_CONTENT_SCHEME, count, names, *(item.encode() for item in sorted(contents))]),
        metadata=_hash([_METADATA_SCHEME, count, *(item.encode() for item in sorted(metadata))]),
        items=len(contents),
        scheme=SCHEME,
    )


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
