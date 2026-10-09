"""TC-22-1 — Dataset digest and manifest: what `dataset_digest` pins, and how a manifest names what changed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataeval_flow import DatasetDigest, DatasetManifest, ManifestDiff, dataset_digest, dataset_manifest
from verification.functional.integrity._data import Items, make_items

pytestmark = pytest.mark.required


class _Detections:
    """A detection target as an object-detection dataset gives it: boxes, labels and confidence scores."""

    def __init__(self, label: int = 0, score: float = 1.0) -> None:
        self.boxes = np.array([[1, 1, 6, 9]], dtype=np.float32)
        self.labels = np.array([label], dtype=np.intp)
        self.scores = np.array([score], dtype=np.float32)


def _with(items: list[Any], index: int, *, image: Any = None, target: Any = None, meta: Any = None) -> list[Any]:
    """`items` with parts of item `index` replaced."""
    old_image, old_target, old_meta = items[index]
    new = (old_image if image is None else image, old_target if target is None else target, meta or old_meta)
    return [*items[:index], new, *items[index + 1 :]]


class TestDatasetDigest:
    def test_digest_holds_both_hashes_the_item_count_and_the_scheme(self) -> None:
        digest = dataset_digest(Items(make_items(6)))
        assert isinstance(digest, DatasetDigest)
        assert (digest.items, digest.scheme) == (6, 1)
        for value in (digest.content, digest.metadata):
            assert len(value) == 64
            int(value, 16)

    def test_digest_is_repeatable_and_ignores_item_order(self) -> None:
        items = make_items()
        assert dataset_digest(Items(items)) == dataset_digest(Items(items))
        assert dataset_digest(Items(items)) == dataset_digest(Items(items[::-1]))

    def test_editing_one_pixel_changes_both_digests(self) -> None:
        items = make_items()
        edited = items[3][0].copy()
        edited[0, 0, 0] ^= 1
        before, after = dataset_digest(Items(items)), dataset_digest(Items(_with(items, 3, image=edited)))
        assert after.content != before.content
        assert after.metadata != before.metadata

    def test_editing_a_label_changes_the_content_digest(self) -> None:
        items = make_items()
        flipped = items[2][1][::-1].copy()
        assert dataset_digest(Items(_with(items, 2, target=flipped))).content != dataset_digest(Items(items)).content

    def test_removing_or_repeating_an_item_changes_both_digests(self) -> None:
        items = make_items()
        whole = dataset_digest(Items(items))
        for changed in (items[:-1], [*items, items[0]]):
            digest = dataset_digest(Items(changed))
            assert digest.items == len(changed)
            assert digest.content != whole.content
            assert digest.metadata != whole.metadata

    def test_renaming_a_class_changes_the_content_digest_only(self) -> None:
        items = make_items()
        named = dataset_digest(Items(items, {0: "boat", 1: "swimmer"}))
        renamed = dataset_digest(Items(items, {0: "boat", 1: "kayak"}))
        assert renamed.content != named.content
        assert renamed.metadata == named.metadata

    def test_editing_metadata_changes_the_metadata_digest_only(self) -> None:
        items = make_items()
        edited = _with(items, 2, meta={"id": 2, "site": "south"})
        before, after = dataset_digest(Items(items)), dataset_digest(Items(edited))
        assert after.metadata != before.metadata
        assert after.content == before.content

    def test_moving_metadata_between_items_changes_the_metadata_digest_only(self) -> None:
        items = make_items(2)
        swapped = [(items[0][0], items[0][1], items[1][2]), (items[1][0], items[1][1], items[0][2])]
        before, after = dataset_digest(Items(items)), dataset_digest(Items(swapped))
        assert after.content == before.content
        assert after.metadata != before.metadata

    def test_equal_metadata_in_numpy_and_python_forms_digests_alike(self) -> None:
        image, target = np.zeros((1, 2, 2), dtype=np.uint8), np.array([1.0, 0.0], dtype=np.float32)
        plain = {"angle": 3, "speed": 1.5, "bands": [1, 2], "gap": None}
        numpy = {"angle": np.int64(3), "speed": np.float64(1.5), "bands": np.array([1, 2]), "gap": float("nan")}
        assert dataset_digest(Items([(image, target, plain)])) == dataset_digest(Items([(image, target, numpy)]))

    def test_detection_digest_covers_boxes_and_labels_but_not_scores(self) -> None:
        image = np.zeros((3, 16, 16), dtype=np.uint8)

        def digest(target: _Detections) -> str:
            return dataset_digest(Items([(image, target, {})])).content

        assert digest(_Detections(score=0.5)) == digest(_Detections())
        assert digest(_Detections(label=1)) != digest(_Detections())

    def test_a_value_with_no_stable_form_is_refused(self) -> None:
        image, target = np.zeros((1, 2, 2), dtype=np.uint8), np.zeros(2, dtype=np.float32)
        with pytest.raises(TypeError, match="no stable form"):
            dataset_digest(Items([(image, target, {"opaque": object()})]))

    def test_an_empty_dataset_digests_and_its_class_names_still_count(self) -> None:
        empty, named = dataset_digest(Items([], {0: "a"})), dataset_digest(Items([], {0: "b"}))
        assert empty.items == 0
        assert empty.content != named.content


class TestDatasetManifest:
    def test_manifest_has_one_entry_per_item_and_the_digest_dataset_digest_gives(self) -> None:
        data = Items(make_items(6))
        manifest = dataset_manifest(data)
        assert manifest.digest == dataset_digest(data)
        assert [(entry.index, entry.id, entry.root) for entry in manifest.entries] == [(i, i, i) for i in range(6)]
        assert manifest.classes == {0: "boat", 1: "swimmer"}

    def test_manifest_reads_back_as_written(self, tmp_path: Path) -> None:
        manifest = dataset_manifest(Items(make_items()))
        manifest.save(tmp_path / "nested" / "train.json")
        assert DatasetManifest.load(tmp_path / "nested" / "train.json") == manifest

    def test_manifest_of_another_digest_scheme_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "train.json"
        dataset_manifest(Items(make_items())).save(path)
        path.write_text(json.dumps({**json.loads(path.read_text()), "scheme": 99}))
        with pytest.raises(ValueError, match="scheme 99 manifest"):
            DatasetManifest.load(path)

    def test_the_same_items_compare_equal(self) -> None:
        assert not dataset_manifest(Items(make_items())).compare(dataset_manifest(Items(make_items())))

    def test_an_edited_item_is_named_as_changed_by_its_id(self) -> None:
        items = make_items()
        edited = _with(items, 3, image=255 - items[3][0])
        diff = dataset_manifest(Items(items)).compare(dataset_manifest(Items(edited)))
        assert diff == ManifestDiff(changed=(3,))
        assert diff

    def test_a_dropped_item_is_named_as_missing_and_an_extra_one_as_added(self) -> None:
        items = make_items()
        recorded = dataset_manifest(Items(items))
        assert recorded.compare(dataset_manifest(Items(items[:-1]))) == ManifestDiff(missing=(5,))
        extra = (items[0][0] ^ 1, items[0][1], {"id": 99, "site": "north"})
        assert recorded.compare(dataset_manifest(Items([*items, extra]))) == ManifestDiff(added=(99,))

    def test_items_without_unique_ids_are_named_by_index(self) -> None:
        bare = [(image, target, {}) for image, target, _ in make_items()]
        edited = _with(bare, 3, image=255 - bare[3][0], meta={})
        diff = dataset_manifest(Items(bare)).compare(dataset_manifest(Items(edited)))
        assert diff == ManifestDiff(missing=(3,), added=(3,))

    def test_renamed_classes_are_reported(self) -> None:
        items = make_items()
        diff = dataset_manifest(Items(items)).compare(dataset_manifest(Items(items, {0: "boat", 1: "kayak"})))
        assert diff == ManifestDiff(classes=True)
