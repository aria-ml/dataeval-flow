"""`provenance:` on a `datasets:` entry: facts people supply, shown as written, in no cache key (audit spec §7.2)."""

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from dataeval_flow import PipelineConfig, run_tasks
from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._cache import DatasetCache
from dataeval_flow._dataset import _config_key
from dataeval_flow.config import HuggingFaceDatasetConfig, ImageFolderDatasetConfig

_FACTS = {"owner": "Perception team", "license": "CC-BY-4.0", "frames": 1200, "public": False}


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def test_a_dataset_entry_takes_names_and_plain_values() -> None:
    config = HuggingFaceDatasetConfig(
        name="d", format="huggingface", path="./d", task="image_classification", provenance=_FACTS
    )
    assert config.provenance == _FACTS


def test_it_is_optional() -> None:
    assert (
        HuggingFaceDatasetConfig(name="d", format="huggingface", path="./d", task="image_classification").provenance
        is None
    )


def test_a_yaml_date_is_kept_as_iso_text() -> None:
    text = """
datasets:
  - name: d
    format: huggingface
    task: image_classification
    path: ./d
    provenance: {collected: 2025-06-01, owner: Perception team}
"""
    raw = yaml.safe_load(text)
    assert raw["datasets"][0]["provenance"]["collected"] == date(2025, 6, 1)  # YAML reads a date
    config = PipelineConfig.model_validate(raw)
    assert config.datasets
    entry = config.datasets[0]
    assert isinstance(entry, HuggingFaceDatasetConfig)
    assert entry.provenance == {"collected": "2025-06-01", "owner": "Perception team"}


def test_a_nested_value_is_refused() -> None:
    with pytest.raises(ValidationError, match="provenance"):
        HuggingFaceDatasetConfig(
            name="d",
            format="huggingface",
            path="./d",
            task="image_classification",
            provenance={"owner": {"team": "x"}},  # pyright: ignore[reportArgumentType]
        )


def test_it_is_not_part_of_the_dataset_cache_key() -> None:
    plain = HuggingFaceDatasetConfig(name="d", format="huggingface", path="./d", task="image_classification")
    described = HuggingFaceDatasetConfig(
        name="d", format="huggingface", path="./d", task="image_classification", provenance=_FACTS
    )
    assert _config_key(described) == _config_key(plain)
    assert "provenance" not in _config_key(described)


def test_it_reaches_the_results_resolved_config(tmp_path: Path) -> None:
    (tmp_path / "frames").mkdir()
    for index in range(4):
        Image.new("RGB", (8, 8), color=(index * 60, 0, 0)).save(tmp_path / "frames" / f"{index}.png")
    config = PipelineConfig.model_validate(
        {
            "datasets": [
                ImageFolderDatasetConfig(name="frames_data", path="frames", provenance=_FACTS).model_dump(),
            ],
            "sources": [{"name": "frames", "dataset": "frames_data"}],
            "evaluators": [{"name": "labels", "type": "label-health"}],
            "tasks": [{"name": "t", "evaluator": "labels", "sources": "frames"}],
        }
    )
    result = run_tasks(config, data_dir=tmp_path)["t"]
    assert result.success, result.errors
    (source,) = result.metadata.resolved_config["sources"]
    assert source["dataset_config"]["provenance"] == _FACTS


def test_a_config_with_provenance_saves_as_json(tmp_path: Path) -> None:
    state = ConfigState()
    state.load_dict(
        PipelineConfig.model_validate(
            yaml.safe_load(
                "datasets:\n"
                "  - {name: d, format: huggingface, path: ./d, task: image_classification,\n"
                "     provenance: {collected: 2025-06-01}}\n"
            )
        )
    )
    path = tmp_path / "pipeline.json"
    state.save_file(path)
    assert json.loads(path.read_text())["datasets"][0]["provenance"] == {"collected": "2025-06-01"}
