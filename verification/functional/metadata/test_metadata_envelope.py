"""TC-18-1 — JATIC ResultMetadata envelope (IR-3-H-12)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import dataeval_flow
from dataeval_flow import run_tasks

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable

    from dataeval_flow import PipelineConfig


class TestResultMetadataEnvelope:
    def test_version_field_set(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.version

    def test_timestamp_is_timezone_aware(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        ts = result.metadata.timestamp
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_tool_identifier(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.tool == "dataeval-flow"

    def test_tool_version_matches_package(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.tool_version == dataeval_flow.__version__

    def test_resolved_config_is_dict_and_nonempty(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert isinstance(result.metadata.resolved_config, dict)
        assert result.metadata.resolved_config

    def test_execution_time_nonnegative(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.execution_time_s is not None
        assert result.metadata.execution_time_s >= 0

    def test_dataset_id_names_the_sources_and_resolved_config_holds_the_seed(
        self, synthetic_pipeline_config: tuple[PipelineConfig, Path]
    ) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg.model_copy(update={"seed": 17}), data_dir=data_dir)[0]
        assert result.metadata.dataset_id == "main_ds"
        resolved = result.metadata.resolved_config
        assert resolved["seed"] == 17
        assert [s["dataset"] for s in resolved["sources"]] == ["main_ds"]

    def test_multi_source_dataset_id_lists_every_source(
        self, image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]]
    ) -> None:
        from dataeval_flow import DriftMonitoringTaskConfig, DriftMonitoringWorkflowConfig
        from dataeval_flow.workflows.drift.params import DriftDetectorKNeighbors

        cfg, data_dir = image_folder_pipeline_builder(
            sources=(("ref", 0), ("test", 1)),
            workflows=[
                DriftMonitoringWorkflowConfig(
                    name="drift",
                    type="drift-monitoring",
                    detectors=[DriftDetectorKNeighbors(method="kneighbors", k=3)],
                )
            ],
            tasks=[
                DriftMonitoringTaskConfig(
                    name="drift_task", workflow="drift", sources=["ref", "test"], extractor="flat"
                )
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.dataset_id == "ref_ds,test_ds"

    def test_envelope_carries_a_health_block(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        health = result.to_dict()["health"]
        assert isinstance(health, dict)
        assert set(health) == {"status", "warnings", "findings"}
        assert health["status"] in {"ok", "warning"}
        assert health["warnings"] == result.warning_count
        assert health["findings"] == len(result.findings)
