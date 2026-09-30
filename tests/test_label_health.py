"""`quality.label-health`: a Dataset's labels counted by class, over DataEval's `label_stats` (spec §9.4)."""

from typing import Any, cast

from dataeval import Metadata

from dataeval_flow import run
from dataeval_flow.evaluators import EvaluatorInputs
from dataeval_flow.evaluators.quality import LabelHealthConfig, LabelHealthEvaluator, LabelHealthOutput
from tests.chain_toys import ToyDetections
from tests.evaluator_toys import ToyImages, output_json


def test_counts_a_classification_dataset_s_labels_by_class() -> None:
    result = run(LabelHealthConfig(), ToyImages(count=6))
    assert result.success, result.errors
    assert isinstance(result.output, LabelHealthOutput)
    assert result.output.data() == {
        "item_count": 6,
        "class_count": 2,
        "label_count": 6,
        "label_counts_per_class": {"a": 3, "b": 3},
        "image_counts_per_class": {"a": 3, "b": 3},
        "empty_image_count": 0,
        "label_source": "protocol",
    }


def test_counts_a_detection_dataset_s_boxes_and_the_images_that_have_none() -> None:
    boxes = ToyDetections([[0, 1], [], [1], [0]], {0: "car", 1: "van", 2: "bus"})
    assert run(LabelHealthConfig(), boxes).output.data() == {
        "item_count": 4,
        "class_count": 3,
        "label_count": 4,
        "label_counts_per_class": {"car": 2, "van": 2},
        "image_counts_per_class": {"car": 2, "van": 2},
        "empty_image_count": 1,
        "label_source": "protocol",
    }


def test_a_dataset_without_labels_counts_none_and_still_declares_its_classes() -> None:
    assert run(LabelHealthConfig(), ToyImages(count=6, labeled=False)).output.data() == {
        "item_count": 6,
        "class_count": 2,
        "label_count": 0,
        "label_counts_per_class": {},
        "image_counts_per_class": {},
        "empty_image_count": 6,
        "label_source": "protocol",
    }


def test_it_says_where_the_labels_came_from() -> None:
    source = EvaluatorInputs(source="s", metadata=Metadata(cast("Any", ToyImages(count=4))), label_source="filepath")
    output = LabelHealthEvaluator().run(LabelHealthConfig(), [source])
    assert output.data()["label_source"] == "filepath"


def test_its_json_is_the_counts_as_a_mapping_and_it_records_the_core_call() -> None:
    result = run(LabelHealthConfig(), ToyImages(count=6))
    payload = output_json(result)
    assert payload["shape"] == "mapping"
    assert payload["data"]["label_counts_per_class"] == {"a": 3, "b": 3}
    assert result.metadata.dataeval.name == "dataeval.core.label_stats"
