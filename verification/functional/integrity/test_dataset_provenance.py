"""TC-22-4 — Dataset provenance: `provenance:` facts on a `datasets:` entry travel with every result, as written."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow.config import CocoDatasetConfig, PipelineConfig
from verification.functional.integrity._data import digest_pipeline, write_coco, write_pipeline
from verification.helpers import run_cli

pytestmark = pytest.mark.required

_FACTS = {"owner": "Survey team", "license": "CC-BY-4.0", "frames": 6, "public": False}


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The result.json of a run over a dataset that declares `provenance:`."""
    root = tmp_path_factory.mktemp("provenance")
    write_coco(root)
    pipeline = digest_pipeline()
    pipeline["datasets"][0]["provenance"] = _FACTS
    config = write_pipeline(root / "pipeline.yaml", pipeline)
    proc = run_cli("-c", str(config), "-d", str(root), "-o", str(root / "out"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads((root / "out" / "results" / "result.json").read_text())


class TestDatasetProvenance:
    def test_a_dataset_entry_takes_names_mapped_to_text_numbers_and_booleans(self) -> None:
        config = CocoDatasetConfig(name="d", path="d", provenance=_FACTS)
        assert config.provenance == _FACTS

    def test_provenance_is_optional(self) -> None:
        assert CocoDatasetConfig(name="d", path="d").provenance is None

    def test_a_nested_value_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="provenance"):
            CocoDatasetConfig(name="d", path="d", provenance={"owner": {"team": "x"}})  # type: ignore[dict-item]

    def test_a_yaml_date_is_kept_as_iso_text(self, tmp_path: Path) -> None:
        text = (
            "datasets:\n  - {name: d, format: coco, path: d, provenance: {collected: 2025-06-01, owner: Survey team}}\n"
        )
        assert yaml.safe_load(text)["datasets"][0]["provenance"]["collected"] == date(2025, 6, 1)
        config = PipelineConfig.model_validate(yaml.safe_load(text))
        assert config.datasets
        assert config.datasets[0].provenance == {"collected": "2025-06-01", "owner": "Survey team"}

    def test_a_result_records_the_facts_as_written_in_its_resolved_config(self, result: dict) -> None:
        (source,) = result["digest"]["metadata"]["resolved_config"]["sources"]
        assert source["dataset_config"]["provenance"] == _FACTS

    def test_provenance_is_no_part_of_the_dataset_s_cache_key(self) -> None:
        from dataeval_flow._dataset import _config_key

        plain = CocoDatasetConfig(name="d", path="d")
        described = CocoDatasetConfig(name="d", path="d", provenance=_FACTS)
        assert _config_key(described) == _config_key(plain)
