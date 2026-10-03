"""TC-20-2 — label-space judges labels against a declared ontology."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.label_space import LabelSpaceConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("20-2")
class TestLabelSpace:
    def test_a_declared_ontology_yields_an_ontology_assessment(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            n_classes=2,
            include_extractor=False,
            workflows=[
                LabelSpaceConfig(name="vocab", ontology={"root": {"class_0": [], "class_1": [], "class_2": []}})
            ],
            tasks=[TaskConfig(name="vocab_task", workflow="vocab", sources="main")],
        )
        result = run_tasks(cfg, data_dir=data_dir)["vocab_task"]
        assert result.success
        assert isinstance(result, ChainResult)
        assert [finding.title for finding in result.findings] == [
            "Label Space Coverage",
            "Label Conformance",
            "Label Alignment",
            "Ontology Structure",
        ]
        assert result.report().strip()
        assert result.metadata.label_space_digest is not None
