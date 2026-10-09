"""TC-5-2 — dataset views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from dataeval_flow import dataset_digest, load_source, run_tasks
from dataeval_flow.config import PipelineConfig, ViewConfig, ViewOperation
from verification.fixtures import write_coco_dataset, write_image_folder

pytestmark = [pytest.mark.required, pytest.mark.usefixtures("fresh_caches", "unseeded")]

PER_CLASS = 6  # twelve images: class_0 first, then class_1


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    write_image_folder(tmp_path / "imgs", n_per_class=PER_CLASS, n_classes=2)
    write_coco_dataset(tmp_path / "coco", n=4)
    return tmp_path


def _config(operations: list[dict[str, Any]], *, dataset: str = "ds", **sections: Any) -> PipelineConfig:
    data: dict[str, Any] = {
        "datasets": [
            {"name": "ds", "format": "image_folder", "path": "imgs", "infer_labels": True},
            {
                "name": "od",
                "format": "coco",
                "path": "coco",
                "annotations_file": "annotations.json",
                "images_dir": "images",
            },
        ],
        "views": [{"name": "v", "operations": operations}],
        "sources": [{"name": "s", "dataset": dataset, "view": "v"}],
    }
    data.update(sections)
    return PipelineConfig.model_validate(data)


def _classes(dataset: Any) -> list[int]:
    return [int(np.argmax(dataset[i][1])) for i in range(len(dataset))]


def _load(root: Path, operations: list[dict[str, Any]], **kwargs: Any) -> Any:
    return load_source(_config(operations, **kwargs), "s", data_dir=root)


class TestViewConfig:
    def test_view_operation_constructs(self) -> None:
        operation = ViewOperation(type="Limit", params={"size": 10})
        assert (operation.type, operation.params) == ("Limit", {"size": 10})
        assert ViewOperation(type="Reverse").params == {}

    def test_view_config_stacks_operations(self) -> None:
        view = ViewConfig(
            name="sample",
            operations=[
                ViewOperation(type="ClassFilter", params={"classes": [0, 3]}),
                ViewOperation(type="Shuffle", params={"seed": 42}),
                ViewOperation(type="Limit", params={"size": 500}),
            ],
        )
        assert [operation.type for operation in view.operations] == ["ClassFilter", "Shuffle", "Limit"]

    def test_a_view_needs_a_name_and_operations(self) -> None:
        with pytest.raises(ValidationError) as caught:
            ViewConfig()  # type: ignore[call-arg]
        assert {error["loc"][0] for error in caught.value.errors()} == {"name", "operations"}

    def test_an_operation_with_an_unknown_key_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            ViewOperation(type="Limit", size=10)  # type: ignore[call-arg]

    def test_a_pipeline_defines_each_view_name_once(self) -> None:
        view = {"name": "v", "operations": [{"type": "Reverse"}]}
        with pytest.raises(ValidationError, match="Duplicate name 'v' in views"):
            PipelineConfig.model_validate({"views": [view, view]})

    def test_an_index_range_expands_to_the_indices_it_names(self) -> None:
        assert ViewOperation(type="Indices", params={"indices": {"start": 500, "stop": 505}}).params["indices"] == [
            500,
            501,
            502,
            503,
            504,
        ]
        stepped = ViewOperation(type="Indices", params={"indices": {"start": 0, "stop": 10, "step": 3}})
        assert stepped.params["indices"] == [0, 3, 6, 9]
        assert ViewOperation(type="Indices", params={"indices": [4, 2]}).params["indices"] == [4, 2]

    @pytest.mark.parametrize(
        ("indices", "match"),
        [
            ({"start": 0}, "requires both 'start' and 'stop'"),
            ({"stop": 5}, "requires both 'start' and 'stop'"),
            ({"start": 0, "stop": 5, "count": 2}, "Invalid keys in indices range shorthand"),
            ({"start": 0, "stop": 2_000_000}, "expands to 2,000,000 elements"),
        ],
    )
    def test_an_index_range_that_is_incomplete_or_too_large_is_refused(self, indices: dict, match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            ViewOperation(type="Indices", params={"indices": indices})


class TestViewOperations:
    def test_a_limit_operation_reduces_the_dataset_to_the_requested_size(self, data_root: Path) -> None:
        assert len(_load(data_root, [{"type": "Limit", "params": {"size": 5}}])) == 5
        assert len(_load(data_root, [{"type": "Limit", "params": {"size": 500}}])) == 2 * PER_CLASS

    def test_a_source_without_a_view_reads_the_whole_dataset(self, data_root: Path) -> None:
        config = _config([{"type": "Limit", "params": {"size": 5}}])
        config.sources = [config.sources[0].model_copy(update={"view": None})]  # type: ignore[index]
        assert len(load_source(config, "s", data_dir=data_root)) == 2 * PER_CLASS

    def test_operations_apply_in_the_order_listed(self, data_root: Path) -> None:
        keep_class_1 = {"type": "ClassFilter", "params": {"classes": [1]}}
        limit = {"type": "Limit", "params": {"size": 4}}
        filtered_first = _load(data_root, [keep_class_1, limit])
        limited_first = _load(data_root, [limit, keep_class_1])
        assert _classes(filtered_first) == [1, 1, 1, 1]  # four items of the class asked for
        assert len(limited_first) == 0  # the first four items are all class 0, so none are left

    def test_class_filter_keeps_only_the_classes_named(self, data_root: Path) -> None:
        kept = _load(data_root, [{"type": "ClassFilter", "params": {"classes": [1]}}])
        assert _classes(kept) == [1] * PER_CLASS

    def test_reverse_and_indices_select_by_position(self, data_root: Path) -> None:
        reversed_ = _load(data_root, [{"type": "Reverse"}])
        assert _classes(reversed_) == [1] * PER_CLASS + [0] * PER_CLASS
        picked = _load(data_root, [{"type": "Indices", "params": {"indices": {"start": 2, "stop": 10, "step": 3}}}])
        assert _classes(picked) == [0, 0, 1]  # items 2, 5 and 8

    def test_resize_and_crop_take_the_lists_a_config_file_writes(self, data_root: Path) -> None:
        """YAML and JSON have no tuples; `size` and `region` are given as lists and still work."""
        assert _load(data_root, [{"type": "Resize", "params": {"size": [4, 4]}}])[0][0].shape == (3, 4, 4)
        assert _load(data_root, [{"type": "Crop", "params": {"region": [0, 0, 4, 6]}}])[0][0].shape == (3, 6, 4)

    def test_select_channels_keeps_the_channels_named(self, data_root: Path) -> None:
        assert _load(data_root, [{"type": "SelectChannels", "params": {"channels": [0]}}])[0][0].shape == (1, 8, 8)

    def test_relabel_conforms_the_class_names_to_a_target(self, data_root: Path) -> None:
        relabel = {"type": "Relabel", "params": {"class_remap": {"class_0": "x", "class_1": "x"}, "target": ["x"]}}
        dataset = _load(data_root, [relabel])
        assert dataset.metadata["index2label"] == {0: "x"}
        assert set(_classes(dataset)) == {0}

    def test_an_operation_dataeval_does_not_have_is_refused(self, data_root: Path) -> None:
        with pytest.raises(ValueError, match="Unknown view operation type: 'Sharpen'"):
            _load(data_root, [{"type": "Sharpen"}])

    def test_an_operation_works_on_detection_datasets_too(self, data_root: Path) -> None:
        limited = _load(data_root, [{"type": "Limit", "params": {"size": 3}}], dataset="od")
        assert len(limited) == 3
        assert len(limited[0][1].boxes) == 1


class TestReproducibleViews:
    @staticmethod
    def _shuffled(**params: Any) -> list[dict[str, Any]]:
        return [{"type": "Shuffle", "params": params}, {"type": "Limit", "params": {"size": 6}}]

    def test_a_seeded_shuffle_view_is_the_same_each_time(self, data_root: Path) -> None:
        first = _load(data_root, self._shuffled(seed=1))
        second = _load(data_root, self._shuffled(seed=1))
        other = _load(data_root, self._shuffled(seed=2))
        assert dataset_digest(first) == dataset_digest(second)
        assert dataset_digest(first) != dataset_digest(other)

    def test_a_seeded_shuffle_is_a_permutation_of_the_dataset(self, data_root: Path) -> None:
        shuffled = _load(data_root, [{"type": "Shuffle", "params": {"seed": 5}}])
        assert len(shuffled) == 2 * PER_CLASS
        assert sorted(_classes(shuffled)) == [0] * PER_CLASS + [1] * PER_CLASS
        assert _classes(shuffled) != [0] * PER_CLASS + [1] * PER_CLASS

    def test_seeded_shuffle_view_pipeline_is_deterministic(self, data_root: Path) -> None:
        """The same config, run twice, reads the same items and gives the same result."""
        config = _config(
            self._shuffled(seed=7),
            evaluators=[{"name": "digest", "type": "content-digest"}, {"name": "labels", "type": "label-health"}],
            tasks=[
                {"name": "digest", "evaluator": "digest", "sources": "s"},
                {"name": "labels", "evaluator": "labels", "sources": "s"},
            ],
        )
        first = run_tasks(config, data_dir=data_root)
        second = run_tasks(config, data_dir=data_root)
        assert first["digest"].success, first["digest"].errors
        assert first["digest"].output.data()["content"] == second["digest"].output.data()["content"]
        assert first["labels"].output.data() == second["labels"].output.data()

    def test_the_pipeline_seed_fixes_a_shuffle_that_has_none_of_its_own(self, data_root: Path) -> None:
        def digest(seed: int) -> str:
            config = _config(
                self._shuffled(),
                seed=seed,
                evaluators=[{"name": "digest", "type": "content-digest"}],
                tasks=[{"name": "digest", "evaluator": "digest", "sources": "s"}],
            )
            return run_tasks(config, data_dir=data_root)["digest"].output.data()["content"]

        assert digest(3) == digest(3)
        assert digest(3) != digest(4)

    def test_the_view_a_task_read_is_recorded_in_its_result(self, data_root: Path) -> None:
        config = _config(
            [{"type": "ClassFilter", "params": {"classes": [1]}}],
            evaluators=[{"name": "labels", "type": "label-health"}],
            tasks=[{"name": "t", "evaluator": "labels", "sources": "s"}],
        )
        result = run_tasks(config, data_dir=data_root)["t"]
        assert result.success, result.errors
        assert result.output.data()["item_count"] == PER_CLASS
        assert result.output.data()["label_counts_per_class"] == {"class_0": 0, "class_1": PER_CLASS}
        assert result.metadata.selection_id == "v"
        (source,) = result.metadata.resolved_config["sources"]
        assert source["view"] == "v"
        assert source["view_config"]["operations"] == [{"type": "ClassFilter", "params": {"classes": [1]}}]
