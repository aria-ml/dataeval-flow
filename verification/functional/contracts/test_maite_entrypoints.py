"""TC-17-1 — MAITE interoperability: declared entry points and the datasets Flow hands to MAITE tools.

DataEval Flow advertises MAITE image-classification and object-detection ``Dataset`` components and two MAITE
tasks through ``[project.entry-points]`` in ``pyproject.toml``. These tests read the entry points from the
installed metadata, load every target, and check at run time that the datasets Flow loads satisfy the MAITE
dataset protocols.
"""

from __future__ import annotations

from importlib.metadata import distribution
from pathlib import Path

import pytest

from verification.fixtures import write_coco_dataset, write_image_folder, write_yolo_dataset

pytestmark = pytest.mark.required

DIST_NAME = "dataeval-flow"

# group -> {entry point name: target}, exactly as pyproject.toml declares them.
DECLARED = {
    "maite.protocols.image_classification.Dataset": {
        "dataeval_flow_HFImageClassificationDataset": "datamaite:ImageClassificationDataset",
    },
    "maite.protocols.object_detection.Dataset": {
        "dataeval_flow_HFObjectDetectionDataset": "datamaite:ObjectDetectionDataset",
    },
    "maite.tasks": {
        "dataeval_flow_run_tasks": "dataeval_flow:run_tasks",
        "dataeval_flow_load_dataset": "dataeval_flow:load_dataset",
    },
}


def _maite_entrypoints():
    return [ep for ep in distribution(DIST_NAME).entry_points if ep.group.startswith("maite.")]


class TestMaiteEntryPoints:
    def test_declares_maite_entrypoints(self) -> None:
        declared: dict[str, dict[str, str]] = {}
        for ep in _maite_entrypoints():
            declared.setdefault(ep.group, {})[ep.name] = ep.value

        assert declared == DECLARED

    @pytest.mark.parametrize("ep", _maite_entrypoints(), ids=lambda ep: f"{ep.group}:{ep.name}")
    def test_entrypoint_target_importable(self, ep) -> None:
        assert ep.load() is not None

    def test_the_task_entry_points_are_the_public_functions(self) -> None:
        import dataeval_flow

        tasks = {ep.name: ep.load() for ep in _maite_entrypoints() if ep.group == "maite.tasks"}

        assert tasks == {
            "dataeval_flow_run_tasks": dataeval_flow.run_tasks,
            "dataeval_flow_load_dataset": dataeval_flow.load_dataset,
        }

    def test_the_dataset_entry_points_are_the_datamaite_classes_flow_loads_with(self, tmp_path: Path) -> None:
        import dataeval_flow

        classes = {ep.group: ep.load() for ep in _maite_entrypoints() if ep.group.startswith("maite.protocols")}
        write_image_folder(tmp_path / "ic", n_per_class=2, n_classes=2, size=16)
        write_yolo_dataset(tmp_path / "od", n=2)

        image_classification = dataeval_flow.load_dataset(tmp_path / "ic", dataset_format="huggingface")
        object_detection = dataeval_flow.load_dataset(tmp_path / "od", dataset_format="yolo")

        assert type(image_classification) is classes["maite.protocols.image_classification.Dataset"]
        assert type(object_detection) is classes["maite.protocols.object_detection.Dataset"]


class TestDatasetsSatisfyTheMaiteProtocols:
    def test_an_image_classification_dataset_yields_image_label_metadata_triples(self, tmp_path: Path) -> None:
        import maite.protocols.image_classification as ic
        import numpy as np

        from dataeval_flow import load_dataset

        write_image_folder(tmp_path, n_per_class=3, n_classes=2, size=16)

        dataset = load_dataset(tmp_path, dataset_format="huggingface")

        assert isinstance(dataset, ic.Dataset)
        assert len(dataset) == 6
        image, target, datum_metadata = dataset[0]
        assert np.asarray(image).shape == (3, 16, 16)  # channels first
        assert np.asarray(target).tolist() == [1.0, 0.0]  # one-hot over the two classes
        assert datum_metadata["id"] == "class_0/img_0.png"
        assert dataset.metadata["index2label"] == {0: "class_0", 1: "class_1"}

    def test_an_image_folder_with_labels_satisfies_the_image_classification_protocol_too(self, tmp_path: Path) -> None:
        import maite.protocols.image_classification as ic

        from dataeval_flow import load_dataset

        write_image_folder(tmp_path, n_per_class=2, n_classes=2, size=16)

        dataset = load_dataset(tmp_path, dataset_format="image_folder", infer_labels=True)

        assert isinstance(dataset, ic.Dataset)
        assert dataset.metadata["index2label"] == {0: "class_0", 1: "class_1"}

    @pytest.mark.parametrize("fmt", ["coco", "yolo"])
    def test_an_object_detection_dataset_yields_image_boxes_metadata_triples(self, tmp_path: Path, fmt: str) -> None:
        import maite.protocols.object_detection as od
        import numpy as np

        from dataeval_flow import load_dataset

        if fmt == "coco":
            write_coco_dataset(tmp_path, n=3, size=16)
            dataset = load_dataset(tmp_path, dataset_format="coco", images_dir="images")
        else:
            write_yolo_dataset(tmp_path, n=3, size=16)
            dataset = load_dataset(tmp_path, dataset_format="yolo")

        assert isinstance(dataset, od.Dataset)
        assert len(dataset) == 3
        image, target, datum_metadata = dataset[0]
        assert np.asarray(image).shape == (3, 16, 16)
        assert np.asarray(target.boxes).shape == (1, 4)
        assert len(target.labels) == 1
        assert isinstance(datum_metadata, dict)
        assert "index2label" in dataset.metadata

    def test_a_maite_dataset_runs_through_a_flow_evaluator(self, tmp_path: Path) -> None:
        from dataeval_flow import load_dataset, run
        from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesResult

        write_image_folder(tmp_path, n_per_class=4, n_classes=2, size=32)
        dataset = load_dataset(tmp_path, dataset_format="huggingface")

        result = run(DuplicatesConfig(), dataset)

        assert isinstance(result, DuplicatesResult)
        assert result.success, result.errors
