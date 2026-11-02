"""The `sufficiency` evaluator and the trainer file it loads from the data root's `trainers/` folder."""

import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from dataeval.protocols import DatasetMetadata
from pydantic import ValidationError

from dataeval_flow import run, run_tasks
from dataeval_flow.config import PipelineConfig
from dataeval_flow.evaluators.performance import SufficiencyConfig, SufficiencyResult, TrainerConfig
from dataeval_flow.evaluators.performance._evaluator import _log_progress
from dataeval_flow.evaluators.performance._trainer import load_trainer
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import output_json

# A nearest-centroid classifier over the flattened images. Its model is a dataclass under postponed annotations, which
# only loads where the module is in `sys.modules` while it runs; `fail` plants an error in `build()`.
TRAINER = """
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dataeval_flow.evaluators.performance import Trainer, TrainerKit


@dataclass
class Centroids:
    means: dict[int, np.ndarray] = field(default_factory=dict)


class Fit:
    def train(self, model: Centroids, dataset, indices: list[int]) -> None:
        groups: dict[int, list[np.ndarray]] = {}
        for i in indices:
            image, target, _ = dataset[i]
            groups.setdefault(int(np.argmax(target)), []).append(np.ravel(image))
        model.means = {label: np.mean(rows, axis=0) for label, rows in groups.items()}


class Score:
    def evaluate(self, model: Centroids, dataset) -> dict[str, float]:
        labels = list(model.means)
        means = np.stack([model.means[label] for label in labels])
        hits = 0
        for i in range(len(dataset)):
            image, target, _ = dataset[i]
            hits += labels[int(np.argmin(((means - np.ravel(image)) ** 2).sum(axis=1)))] == int(np.argmax(target))
        return {"accuracy": hits / len(dataset)}


class CentroidTrainer(Trainer):
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def build(self) -> TrainerKit:
        if self.fail:
            raise ZeroDivisionError("planted in build")
        return TrainerKit(model=Centroids(), training_strategy=Fit(), evaluation_strategy=Score())


class NotATrainer:
    pass


def helper() -> None:
    pass
"""
DIGEST = hashlib.sha256(TRAINER.encode()).hexdigest()
ENTRY = {"path": "trainers/centroids.py", "class": "CentroidTrainer"}
CENTROIDS = TrainerConfig(path="trainers/centroids.py", class_="CentroidTrainer")


class Blobs:
    """Image classification items in three classes, whose class sets the mean of the image's 16 pixels (0, 60, 120),
    with noise of deviation 40: a centroid learns them from a few items."""

    def __init__(self, count: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._labels = np.arange(count) % 3
        self._images = [rng.normal(60.0 * label, 40.0, (1, 4, 4)).astype(np.float32) for label in self._labels]
        self.metadata: DatasetMetadata = {"id": f"blobs-{seed}-{count}", "index2label": {0: "a", 1: "b", 2: "c"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(3, dtype=np.float32)
        target[self._labels[index]] = 1.0
        return self._images[index], target, {"id": index}


def _data(train: int = 120, test: int = 60) -> dict[str, Blobs]:
    return {"train": Blobs(train, seed=0), "test": Blobs(test, seed=1)}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A data root holding `trainers/centroids.py`."""
    (tmp_path / "trainers").mkdir()
    (tmp_path / "trainers" / "centroids.py").write_text(TRAINER)
    return tmp_path


@pytest.fixture
def mounted(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`root`, as the data root `run()` resolves through `DATAEVAL_DATA`."""
    monkeypatch.setenv("DATAEVAL_DATA", str(root))
    return root


def _run(
    *, data: dict[str, Blobs] | None = None, trainer: dict[str, Any] = ENTRY, **settings: Any
) -> SufficiencyResult:
    config = SufficiencyConfig(trainer=TrainerConfig.model_validate(trainer), **settings)
    return run(config, data or _data())


def _keys(value: Any) -> set[str]:
    """Every mapping key in `value`, at any depth."""
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


@pytest.mark.usefixtures("mounted")
def test_a_run_measures_each_size_and_writes_json_without_the_fits_private_cache() -> None:
    result = _run(schedule=[6, 12, 30, 60, 120])
    assert result.success, result.errors
    assert isinstance(result, SufficiencyResult)
    output = output_json(result)
    assert output["data"]["steps"] == [6, 12, 30, 60, 120]
    assert all(0.0 <= value <= 1.0 for value in output["data"]["averaged_measures"]["accuracy"])
    assert not {key for key in _keys(output) if key.startswith("_")}, "the curve fit's private cache leaked"
    assert set(output["extras"]) == {"trainer", "params"}
    assert len(output["extras"]["params"]["accuracy"]) == 3  # the power law's scale, exponent and asymptote
    json.dumps(result.to_dict())
    assert "_dataeval_flow_trainer_" + DIGEST[:12] not in sys.modules  # unloaded after the run


@pytest.mark.usefixtures("mounted")
def test_the_result_records_the_trainer_and_the_curve_the_output_projects_from() -> None:
    result = _run(schedule=[6, 12, 30, 60, 120])
    extras = output_json(result)["extras"]
    assert extras["trainer"] == {"path": "trainers/centroids.py", "class": "CentroidTrainer", "sha256": DIGEST}
    assert extras["params"]["accuracy"] == result.output.params["accuracy"].tolist()


@pytest.mark.usefixtures("mounted")
def test_the_trainer_entry_writes_class_back() -> None:
    result = _run(schedule=[30, 120])
    assert result.metadata.resolved_config["evaluator"]["trainer"] == {**ENTRY, "params": {}}
    config = SufficiencyConfig(trainer=CENTROIDS)
    assert SufficiencyConfig.model_validate(config.model_dump(mode="json")) == config
    assert config.model_dump()["trainer"]["class"] == "CentroidTrainer"


def test_a_chain_step_records_the_trainer_and_its_check_judges_the_output(root: Path) -> None:
    config = chain_pipeline(
        datasets=_data(),
        evaluators=[SufficiencyConfig(name="suff", trainer=CENTROIDS, schedule=[30, 120])],
        workflows=[
            {
                "name": "w",
                "inputs": ["train", "test"],
                "steps": [
                    {"name": "measured", "evaluator": "suff", "input": ["train", "test"]},
                    {
                        "name": "target",
                        "check": "performance-sufficiency",
                        "input": "measured",
                        "targets": [{"metric": "accuracy", "value": 0.0}],
                    },
                ],
            }
        ],
        tasks=[{"name": "t", "workflow": "w", "sources": ["train", "test"]}],
    )
    result = run_tasks(config, data_dir=root)["t"]  # the root reaches the step without `DATAEVAL_DATA`
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    step = cast("dict[str, Any]", result.to_dict()["steps"])["measured"]
    assert step["output"]["extras"]["trainer"]["sha256"] == DIGEST
    (finding,) = result.findings
    assert (finding.title, finding.severity) == ("Performance Sufficiency", "ok")
    assert finding.brief == "1 of 1 targets reached at 120 items"


@pytest.mark.usefixtures("mounted")
def test_runs_substeps_and_schedule_reach_dataeval() -> None:
    output = _run(runs=2, substeps=3).output
    assert output.measures["accuracy"].shape == (2, 3)  # two runs of DataEval's three-size default schedule
    assert output.steps.tolist() == [1, 11, 120]  # np.geomspace(1.2, 120, 3), 11.99... truncated to 11
    assert _run(schedule=[12, 60]).output.steps.tolist() == [12, 60]


@pytest.mark.usefixtures("mounted")
def test_a_run_logs_a_line_for_each_size_it_measures(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=_log_progress.__module__):
        _run(runs=2, schedule=[12, 60])
    lines = caplog.messages
    assert len(lines) == 4  # two runs of two sizes
    assert lines[0].startswith("Sufficiency 1/4: run 1 of 2, 12 items, accuracy=")
    assert lines[3].startswith("Sufficiency 4/4: run 2 of 2, 60 items, accuracy=")


def test_the_progress_line_leaves_out_metrics_that_are_not_scalars(caplog: pytest.LogCaptureFixture) -> None:
    info = {"run": 0, "runs": 1, "size": 10, "metrics": {"accuracy": 0.5, "per_class": np.array([0.1, 0.9])}}
    with caplog.at_level(logging.INFO, logger=_log_progress.__module__):
        _log_progress(1, total=1, extra_info=info)
    assert caplog.messages == ["Sufficiency 1/1: run 1 of 1, 10 items, accuracy=0.5"]


@pytest.mark.parametrize(
    ("path", "why"),
    [
        ("models/centroids.py", "trainers"),
        ("centroids.py", "trainers"),  # a bare name is not looked up under trainers/
        ("trainers/../centroids.py", "trainers"),
        ("trainers/centroids.txt", "trainers"),
        ("trainers", "trainers"),
        ("/data/trainers/centroids.py", "relative"),
        ("trainers/../../centroids.py", "escapes"),
    ],
)
def test_a_trainer_outside_trainers_is_refused_when_the_config_loads(path: str, why: str) -> None:
    with pytest.raises(ValidationError, match=why):
        TrainerConfig(path=path, class_="CentroidTrainer")


def test_a_symlink_out_of_trainers_is_refused_when_the_run_starts(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (root / "elsewhere.py").write_text(TRAINER)
    (root / "trainers" / "link.py").symlink_to(root / "elsewhere.py")
    monkeypatch.setenv("DATAEVAL_DATA", str(root))
    result = _run(trainer={**ENTRY, "path": "trainers/link.py"}, schedule=[30])
    assert not result.success
    (message,) = result.errors
    assert message == (
        f"ValueError: Trainer 'CentroidTrainer' in trainers/link.py: the file resolves to {root / 'elsewhere.py'}, "
        f"outside the data root's trainers folder {root / 'trainers'}."
    )


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ({"path": "trainers/missing.py"}, "FileNotFoundError: Trainer 'CentroidTrainer' in trainers/missing.py:"),
        ({"class": "Absent"}, "ValueError: Trainer 'Absent' in trainers/centroids.py: the file defines no 'Absent'."),
        (
            {"class": "NotATrainer"},
            "ValueError: Trainer 'NotATrainer' in trainers/centroids.py: 'NotATrainer' is not a Trainer subclass.",
        ),
        ({"class": "helper"}, "ValueError: Trainer 'helper' in trainers/centroids.py: 'helper' is not a Trainer"),
        (
            {"params": {"fail": True}},
            (
                "RuntimeError: Trainer 'CentroidTrainer' in trainers/centroids.py failed to build: "
                "ZeroDivisionError: planted in build"
            ),
        ),
        (
            {"params": {"rate": 2}},
            "RuntimeError: Trainer 'CentroidTrainer' in trainers/centroids.py failed to build: TypeError: ",
        ),
    ],
)
@pytest.mark.usefixtures("mounted")
def test_a_trainer_that_cannot_load_fails_naming_file_class_and_cause(entry: dict[str, Any], error: str) -> None:
    result = _run(trainer={**ENTRY, **entry}, schedule=[30])
    assert not result.success
    (message,) = result.errors
    assert message.startswith(error)


def test_the_trainers_own_error_is_chained(root: Path) -> None:
    (root / "trainers" / "broken.py").write_text("import not_a_module_anywhere\n")
    config = TrainerConfig(path="trainers/broken.py", class_="CentroidTrainer")
    with pytest.raises(RuntimeError, match="failed running the file: ModuleNotFoundError") as raised:
        load_trainer(config, root).__enter__()
    assert isinstance(raised.value.__cause__, ModuleNotFoundError)
    with pytest.raises(RuntimeError, match="failed to build") as raised:
        load_trainer(TrainerConfig.model_validate({**ENTRY, "params": {"fail": True}}), root).__enter__()
    assert isinstance(raised.value.__cause__, ZeroDivisionError)


def test_build_must_return_a_kit(root: Path) -> None:
    (root / "trainers" / "odd.py").write_text(
        "from dataeval_flow.evaluators.performance import Trainer\n\n"
        "class Odd(Trainer):\n    def build(self):\n        return 3\n"
    )
    with pytest.raises(ValueError, match=r"'Odd' in trainers/odd.py: build\(\) returned int, not a TrainerKit"):
        load_trainer(TrainerConfig(path="trainers/odd.py", class_="Odd"), root).__enter__()


def test_a_dataclass_under_postponed_annotations_loads_and_the_module_is_kept_only_for_the_block(root: Path) -> None:
    name = "_dataeval_flow_trainer_" + DIGEST[:12]
    with load_trainer(CENTROIDS, root) as (kit, trainer):
        assert type(kit.model).__module__ == name
        assert sys.modules[name].__file__ == str(root / "trainers" / "centroids.py")
        assert trainer["sha256"] == DIGEST
    assert name not in sys.modules
    assert not (root / "trainers" / "__pycache__").exists()  # nothing is written into the data root


def test_the_how_tos_trainer_loads_and_builds_without_reading_its_weights(tmp_path: Path) -> None:
    (tmp_path / "trainers").mkdir()
    shutil.copy(Path(__file__).parents[1] / "docs" / "trainers" / "backbone.py", tmp_path / "trainers")
    entry = {"path": "trainers/backbone.py", "class": "BackboneTrainer", "params": {"weights": "models/absent.onnx"}}
    with load_trainer(TrainerConfig.model_validate(entry), tmp_path) as (kit, _):
        assert (type(kit.model).__name__, type(kit.training_strategy).__name__) == ("Centroids", "Fit")


@pytest.mark.parametrize("schedule", [[30, 10], [10, 10], [0, 10], []])
def test_a_schedule_must_be_strictly_increasing_positive_integers(schedule: list[int]) -> None:
    with pytest.raises(ValidationError, match="strictly increasing positive integers"):
        SufficiencyConfig(trainer=CENTROIDS, schedule=schedule)


def test_a_schedule_and_substeps_are_refused_together() -> None:
    with pytest.raises(ValidationError, match="`schedule` or `substeps`, not both"):
        SufficiencyConfig(trainer=CENTROIDS, schedule=[10], substeps=3)


@pytest.mark.usefixtures("mounted")
def test_a_size_past_the_training_set_is_refused_when_the_run_starts() -> None:
    result = _run(schedule=[60, 200])
    assert result.errors == [
        (
            "ValueError: `schedule` asks for [200], past the 120 items of the training set, source 'train'; DataEval "
            "would train on all of them and record the larger size."
        )
    ]


@pytest.mark.usefixtures("mounted")
def test_no_schedule_under_100_training_items_is_refused_when_the_run_starts() -> None:
    result = _run(data=_data(train=99))
    assert result.errors == [
        (
            "ValueError: The training set, source 'train', holds 99 items: the default schedule needs at least 100, "
            "or its first size is 0 items. Set `schedule`."
        )
    ]


@pytest.mark.parametrize("count", [1, 3])
def test_one_source_or_three_are_refused_when_the_config_loads(count: int) -> None:
    pipeline = {
        "datasets": [{"name": "ds", "format": "huggingface", "path": "./d", "task": "image_classification"}],
        "sources": [{"name": name, "dataset": "ds"} for name in "abc"],
        "evaluators": [{"name": "e", "type": "sufficiency", "trainer": ENTRY}],
        "tasks": [{"name": "t", "evaluator": "e", "sources": list("abc")[:count]}],
    }
    with pytest.raises(ValidationError, match="takes exactly two sources"):
        PipelineConfig.model_validate(pipeline)
