"""TC-2-1 — configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import PipelineConfig, load_config, load_config_folder

MINIMAL_CONFIG = {
    "datasets": [
        {"name": "main", "format": "image_folder", "path": "images"},
    ],
    "sources": [
        {"name": "main_src", "dataset": "main"},
    ],
    "tasks": [],
}


@pytest.mark.test_case("2-1")
class TestConfigLoading:
    def test_load_yaml_single_file(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "params.yaml"
        cfg_path.write_text(yaml.safe_dump(MINIMAL_CONFIG))
        cfg = load_config(cfg_path)
        assert isinstance(cfg, PipelineConfig)

    def test_load_json_single_file(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "params.json"
        cfg_path.write_text(json.dumps(MINIMAL_CONFIG))
        cfg = load_config(cfg_path)
        assert isinstance(cfg, PipelineConfig)

    def test_load_config_folder_merges_files(self, tmp_path: Path) -> None:
        (tmp_path / "datasets.yaml").write_text(yaml.safe_dump({"datasets": MINIMAL_CONFIG["datasets"]}))
        (tmp_path / "sources.yaml").write_text(yaml.safe_dump({"sources": MINIMAL_CONFIG["sources"]}))
        (tmp_path / "tasks.yaml").write_text(yaml.safe_dump({"tasks": []}))
        cfg = load_config_folder(tmp_path)
        assert isinstance(cfg, PipelineConfig)

    def test_invalid_config_raises_validation_error(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml.safe_dump({"sources": "this is not a dict"}))
        with pytest.raises((ValidationError, ValueError, TypeError)):
            load_config(cfg_path)
