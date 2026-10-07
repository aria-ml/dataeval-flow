"""TC-2-1 — configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import (
    ImageFolderDatasetConfig,
    PipelineConfig,
    export_params_schema,
    load_config,
    load_config_folder,
)
from dataeval_flow.config._loader import resolve_path, validate_config_path
from dataeval_flow.dataset import resolve_dataset
from verification.fixtures import write_image_folder

pytestmark = pytest.mark.required

MINIMAL_CONFIG = {
    "datasets": [
        {"name": "main", "format": "image_folder", "path": "images"},
    ],
    "sources": [
        {"name": "main_src", "dataset": "main"},
    ],
    "tasks": [],
}


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

    def test_export_params_schema_describes_pipeline_config(self, tmp_path: Path) -> None:
        out = tmp_path / "schema" / "params.schema.json"
        export_params_schema(out)
        schema = json.loads(out.read_text())
        assert schema["title"] == "PipelineConfig"
        assert {"datasets", "sources", "workflows", "tasks", "seed", "max_processes"} <= set(schema["properties"])

    def test_relative_dataset_paths_resolve_against_data_root(self, tmp_path: Path) -> None:
        write_image_folder(tmp_path / "root" / "imgs", n_per_class=3, n_classes=2)
        data_root = tmp_path / "root"

        assert resolve_path("imgs", data_root) == data_root / "imgs"
        resolved = resolve_dataset(
            ImageFolderDatasetConfig(name="ds", path="imgs", infer_labels=True), data_dir=data_root
        )
        assert len(resolved.dataset) == 6

    def test_dataset_paths_must_stay_relative_to_the_data_root(self) -> None:
        assert validate_config_path("sub/imgs") == "sub/imgs"
        for bad in ("/abs/imgs", "../outside"):
            with pytest.raises(ValueError):
                validate_config_path(bad)
