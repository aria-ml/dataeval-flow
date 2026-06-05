"""TC-18-1 — JATIC ResultMetadata envelope (IR-3-H-12)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import dataeval_flow
from dataeval_flow import run_tasks

if TYPE_CHECKING:
    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("18-1")
class TestResultMetadataEnvelope:
    def test_version_field_set(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.version

    def test_timestamp_is_timezone_aware(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        ts = result.metadata.timestamp
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_tool_identifier(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.tool == "dataeval-flow"

    def test_tool_version_matches_package(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.tool_version == dataeval_flow.__version__

    def test_resolved_config_is_dict_and_nonempty(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert isinstance(result.metadata.resolved_config, dict)
        assert result.metadata.resolved_config

    def test_execution_time_nonnegative(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.execution_time_s is not None
        assert result.metadata.execution_time_s >= 0
