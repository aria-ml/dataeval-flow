"""TC-14-1 — reporting & export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from dataeval_flow import run_tasks

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from dataeval_flow import PipelineConfig


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

    def test_summary_report_is_shorter_than_detailed(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        detailed = result.report()
        summary = result.report(detailed=False)
        assert summary.strip()
        assert len(summary.splitlines()) > 5
        assert len(summary.splitlines()) < len(detailed.splitlines())

    def test_export_without_a_path_returns_the_serialized_string(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        as_json = result.export()
        as_yaml = result.export(fmt="yaml")
        assert isinstance(as_json, str)
        assert json.loads(as_json) == result.to_dict()
        assert isinstance(as_yaml, str)
        assert yaml.safe_load(as_yaml) == result.to_dict()

    def test_health_warning_count_and_findings_support_gating(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        findings = result.findings
        assert findings
        warnings = [f for f in findings if f.severity == "warning"]
        assert result.warning_count == len(warnings)
        assert result.health == {
            "status": "warning" if warnings else "ok",
            "warnings": len(warnings),
            "findings": len(findings),
        }
        text = result.report()
        if warnings:
            assert f"Health: {len(warnings)} warning(s)" in text
        else:
            assert "Health: All checks passed" in text
