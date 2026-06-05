"""TC-14-1 — reporting & export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from dataeval_flow import run_tasks

if TYPE_CHECKING:
    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("14-1")
class TestReporting:
    def test_report_returns_nonempty_string(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        text = results[0].report()
        assert isinstance(text, str)
        assert len(text.splitlines()) > 5

    def test_export_json_writes_file(
        self,
        synthetic_pipeline_config: tuple[PipelineConfig, Path],
        tmp_path: Path,
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        out = results[0].export(tmp_path / "result.json", fmt="json")
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert "metadata" in parsed

    def test_export_yaml_writes_file(
        self,
        synthetic_pipeline_config: tuple[PipelineConfig, Path],
        tmp_path: Path,
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        out = results[0].export(tmp_path / "result.yaml", fmt="yaml")
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert "metadata" in parsed

    def test_to_dict_includes_metadata_and_data(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        d = results[0].to_dict()
        assert "metadata" in d
