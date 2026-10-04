"""`dataset_digest`: SHA-256 digests over every item of a dataset, whatever their order (audit spec §7.1)."""

import hashlib
from datetime import date, datetime
from typing import Any

import numpy as np
import pytest

from dataeval_flow import DatasetDigest, dataset_digest
from tests.evaluator_toys import Items, ToyImages


class _Target:
    """A detection target: boxes, labels and scores, as an object-detection dataset gives them."""

    def __init__(self, boxes: list[list[float]], labels: list[int], scores: float = 1.0) -> None:
        self.boxes = np.asarray(boxes, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.intp)
        self.scores = np.full(len(labels), scores, dtype=np.float32)


def _toy_items(count: int = 6) -> list[tuple[Any, ...]]:
    toy = ToyImages(count=count)
    return [toy[index] for index in range(count)]


def _framed(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "little"))
        hasher.update(part)
    return hasher.hexdigest()


def test_it_gives_both_digests_and_the_item_count() -> None:
    digest = dataset_digest(ToyImages(count=6))
    assert isinstance(digest, DatasetDigest)
    assert digest.items == 6
    assert len(digest.content) == 64
    assert int(digest.content, 16) >= 0
    assert len(digest.metadata) == 64
    assert int(digest.metadata, 16) >= 0


def test_the_digest_follows_the_documented_scheme() -> None:
    image = np.arange(4, dtype=np.uint8).reshape(1, 2, 2)
    target = np.array([0.0, 1.0], dtype=np.float32)
    digest = dataset_digest(Items([(image, target, {"id": 7})], {0: "a", 1: "b"}))
    item = _framed(b"|u1", b"(1, 2, 2)", image.tobytes(), b"array", b"<f4", b"(2,)", target.tobytes())
    count = (1).to_bytes(8, "little")
    assert digest.content == _framed(b"dataeval-flow content digest 1", count, b'[[0,"a"],[1,"b"]]', item.encode())
    assert digest.metadata == _framed(
        b"dataeval-flow metadata digest 1", count, _framed(item.encode(), b'{"id":7}').encode()
    )


def test_the_same_items_in_another_order_give_the_same_digests() -> None:
    items = _toy_items()
    assert dataset_digest(Items(items)) == dataset_digest(Items(items[::-1]))


def test_editing_one_pixel_changes_both_digests() -> None:
    items = _toy_items()
    image, target, meta = items[3]
    image = image.copy()
    image[0, 0, 0] ^= 1
    before, after = dataset_digest(Items(items)), dataset_digest(Items([*items[:3], (image, target, meta), *items[4:]]))
    assert after.content != before.content
    assert after.metadata != before.metadata  # each item's metadata is bound to its content


def test_editing_a_label_changes_the_content_digest() -> None:
    items = _toy_items()
    image, target, meta = items[2]
    relabelled = [*items[:2], (image, target[::-1].copy(), meta), *items[3:]]
    assert dataset_digest(Items(relabelled)).content != dataset_digest(Items(items)).content


def test_removing_or_repeating_an_item_changes_both_digests() -> None:
    items = _toy_items()
    whole, fewer, repeated = (dataset_digest(Items(i)) for i in (items, items[:-1], [*items, items[0]]))
    assert (fewer.items, repeated.items) == (5, 7)
    assert fewer.content != whole.content
    assert fewer.metadata != whole.metadata
    assert repeated.content != whole.content
    assert repeated.metadata != whole.metadata


def test_renaming_a_class_changes_the_content_digest_only() -> None:
    items = _toy_items()
    named, renamed = dataset_digest(Items(items, {0: "a", 1: "b"})), dataset_digest(Items(items, {0: "a", 1: "c"}))
    assert renamed.content != named.content
    assert renamed.metadata == named.metadata


def test_editing_metadata_changes_the_metadata_digest_only() -> None:
    items = _toy_items()
    image, target, meta = items[2]
    edited = [*items[:2], (image, target, {**meta, "site": "north"}), *items[3:]]
    before, after = dataset_digest(Items(items)), dataset_digest(Items(edited))
    assert after.metadata != before.metadata
    assert after.content == before.content


def test_equal_metadata_in_numpy_and_python_forms_digests_alike() -> None:
    image, target = np.zeros((1, 2, 2), dtype=np.uint8), np.array([1.0, 0.0], dtype=np.float32)
    plain = {"angle": 3, "speed": 1.5, "bands": [1, 2], "when": "2025-06-01", "gap": None, "raw": "0aff"}
    numpy = {
        "raw": b"\x0a\xff",
        "gap": float("nan"),
        "when": date(2025, 6, 1),
        "bands": np.array([1, 2]),
        "speed": np.float64(1.5),
        "angle": np.int64(3),
    }
    assert dataset_digest(Items([(image, target, plain)])) == dataset_digest(Items([(image, target, numpy)]))


def test_detection_targets_hash_their_boxes_and_labels_not_their_scores() -> None:
    image = np.zeros((3, 16, 16), dtype=np.uint8)
    base = dataset_digest(Items([(image, _Target([[1, 1, 6, 9]], [0]), {})]))
    rescored = dataset_digest(Items([(image, _Target([[1, 1, 6, 9]], [0], scores=0.5), {})]))
    moved = dataset_digest(Items([(image, _Target([[2, 1, 7, 9]], [0]), {})]))
    relabelled = dataset_digest(Items([(image, _Target([[1, 1, 6, 9]], [1]), {})]))
    assert rescored.content == base.content
    assert moved.content != base.content
    assert relabelled.content != base.content


def test_items_without_labels_or_metadata_still_digest() -> None:
    image = np.zeros((1, 2, 2), dtype=np.uint8)
    unlabelled = dataset_digest(Items([(image, np.zeros(0, dtype=np.float32), {"id": 0})]))
    bare = dataset_digest(Items([image], metadata=False))  # an image alone, and no `metadata` attribute
    assert unlabelled.items == bare.items == 1
    assert unlabelled.content != bare.content  # an empty label array is not the absence of a target


def test_an_empty_dataset_digests() -> None:
    empty, named = dataset_digest(Items([], {0: "a"})), dataset_digest(Items([], {0: "b"}))
    assert empty.items == 0
    assert empty.content != named.content  # its class names still count
    assert empty.metadata == named.metadata


def test_moving_metadata_between_items_changes_the_metadata_digest_only() -> None:
    image_a, image_b = np.zeros((1, 2, 2), dtype=np.uint8), np.ones((1, 2, 2), dtype=np.uint8)
    target = np.array([1.0, 0.0], dtype=np.float32)
    before = dataset_digest(Items([(image_a, target, {"site": "n"}), (image_b, target, {"site": "s"})]))
    after = dataset_digest(Items([(image_a, target, {"site": "s"}), (image_b, target, {"site": "n"})]))
    assert after.content == before.content
    assert after.metadata != before.metadata


def test_an_opaque_object_is_refused_not_digested_by_its_repr() -> None:
    class Opaque:
        pass

    image, target = np.zeros((1, 2, 2), dtype=np.uint8), np.zeros(2, dtype=np.float32)

    class BoxesOnly:
        boxes = np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(TypeError, match="no stable form"):
        dataset_digest(Items([(image, target, {"o": Opaque()})]))
    with pytest.raises(TypeError, match="no stable form"):
        dataset_digest(Items([(image, BoxesOnly(), {})]))
    with pytest.raises(TypeError):
        dataset_digest(Items([(np.array([object()], dtype=object), target, {})]))


def test_a_long_tensor_in_metadata_digests_by_every_value() -> None:
    torch = pytest.importorskip("torch")
    image, target = np.zeros((1, 2, 2), dtype=np.uint8), np.zeros(2, dtype=np.float32)
    one, other = torch.zeros(2000), torch.zeros(2000)
    other[1000] = 1.0
    first = dataset_digest(Items([(image, target, {"t": one})]))
    second = dataset_digest(Items([(image, target, {"t": other})]))
    assert first.metadata != second.metadata


def test_datetime64_and_infinities_digest_by_value() -> None:
    image, target = np.zeros((1, 2, 2), dtype=np.uint8), np.zeros(2, dtype=np.float32)

    def metadata(value: Any) -> str:
        return dataset_digest(Items([(image, target, {"v": value})])).metadata

    assert metadata(np.datetime64("2025-06-01T12:00:00", "ns")) == metadata(datetime(2025, 6, 1, 12))
    assert metadata(float("inf")) != metadata(float("-inf"))
    assert metadata(float("nan")) == metadata(None)
