"""`label-health`: a Dataset's labels counted by class, over DataEval's `label_stats` (spec §9.4)."""

from typing import Any, cast

from dataeval import Metadata

from dataeval_flow import run
from dataeval_flow._blocks import Fields
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
        "empty_image_indices": [],
        "label_source": "protocol",
    }


def test_counts_a_detection_dataset_s_boxes_and_the_images_that_have_none() -> None:
    boxes = ToyDetections([[0, 1], [], [1], [0]], {0: "car", 1: "van", 2: "bus"})
    assert run(LabelHealthConfig(), boxes).output.data() == {
        "item_count": 4,
        "class_count": 3,
        "label_count": 4,
        "label_counts_per_class": {"car": 2, "van": 2, "bus": 0},
        "image_counts_per_class": {"car": 2, "van": 2, "bus": 0},
        "empty_image_count": 1,
        "empty_image_indices": [1],
        "label_source": "protocol",
    }


def test_a_dataset_without_labels_counts_none_and_still_declares_its_classes() -> None:
    assert run(LabelHealthConfig(), ToyImages(count=6, labeled=False)).output.data() == {
        "item_count": 6,
        "class_count": 2,
        "label_count": 0,
        "label_counts_per_class": {"a": 0, "b": 0},
        "image_counts_per_class": {"a": 0, "b": 0},
        "empty_image_count": 6,
        "empty_image_indices": [0, 1, 2, 3, 4, 5],
        "label_source": "protocol",
    }


def test_it_says_where_the_labels_came_from() -> None:
    source = EvaluatorInputs(source="s", metadata=Metadata(cast("Any", ToyImages(count=4))), label_source="filepath")
    output = LabelHealthEvaluator().run(LabelHealthConfig(), [source])
    assert output.data()["label_source"] == "filepath"


def test_its_report_shows_fields_and_a_table_of_classes_not_the_raw_output() -> None:
    lines = [line.strip() for line in run(LabelHealthConfig(), ToyImages(count=6)).report().splitlines()]
    assert [
        line for line in lines if line.split(":")[0] in {"Items", "Classes", "Labels", "Empty images", "Label source"}
    ] == ["Items:        6", "Classes:      2", "Labels:       6", "Empty images: 0", "Label source: protocol"]
    assert lines[lines.index("Class  Labels  Images") + 2 :][:2] == ["a           3       3", "b           3       3"]
    assert "label_counts_per_class" not in "\n".join(lines)
    assert "OUTPUT" not in lines


def test_its_report_leaves_out_a_label_source_the_dataset_does_not_say() -> None:
    from dataeval_flow.evaluators.quality._report import label_health_section

    data = {
        "item_count": 2,
        "class_count": 1,
        "label_count": 0,
        "label_counts_per_class": {},
        "image_counts_per_class": {},
        "empty_image_count": 2,
        "label_source": None,
    }
    blocks = label_health_section({"shape": "mapping", "data": data})
    fields = blocks[0]
    assert isinstance(fields, Fields)
    assert len(blocks) == 1  # fields only: no classes occur, so no table
    assert all(key != "Label source" for key, _ in fields.items)


def test_its_json_the_counts_as_a_mapping_and_it_records_the_core_call() -> None:
    result = run(LabelHealthConfig(), ToyImages(count=6))
    payload = output_json(result)
    assert payload["shape"] == "mapping"
    assert payload["data"]["label_counts_per_class"] == {"a": 3, "b": 3}
    assert result.metadata.dataeval.name == "dataeval.core.label_stats"


def test_a_declared_class_with_no_labels_counts_zero() -> None:
    data = run(LabelHealthConfig(), ToyDetections([[0, 1], [], [1], [0]], {0: "car", 1: "van", 2: "bus"})).output.data()
    assert data["label_counts_per_class"] == {"car": 2, "van": 2, "bus": 0}
    assert data["image_counts_per_class"] == {"car": 2, "van": 2, "bus": 0}
    assert data["empty_image_indices"] == [1]


def test_an_unlabelled_dataset_lists_its_declared_classes_at_zero_and_every_image_as_empty() -> None:
    data = run(LabelHealthConfig(), ToyImages(count=6, labeled=False)).output.data()
    assert data["label_counts_per_class"] == {"a": 0, "b": 0}
    assert data["empty_image_indices"] == [0, 1, 2, 3, 4, 5]
