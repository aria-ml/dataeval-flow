"""TC-10-1 — audit workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.audit import AuditConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow.config import PipelineConfig


@pytest.mark.test_case("10-1")
class TestAuditWorkflow:
    def test_audit_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            sources=(("train", 0), ("test", 1)),
            workflows=[
                AuditConfig.model_validate(
                    {
                        "name": "audit_main",
                        "type": "audit",
                        "outliers": {"flags": ["dimension", "pixel"], "outlier_threshold": "zscore"},
                    },
                ),
            ],
            tasks=[
                TaskConfig(
                    name="audit_task",
                    workflow="audit_main",
                    sources=["train", "test"],
                    extractor="flat",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)["audit_task"]
        assert isinstance(result, ChainResult)
        assert result.success
        text = result.report()
        assert isinstance(text, str)
        assert text.strip()
        # The preset judges the splits: a verdict, and the five questions' findings
        assert result.verdict is not None
        assert result.verdict.level in {"not-ready", "ready-with-caveats", "ready"}
        assert result.verdict.label in text
        assert result.findings
