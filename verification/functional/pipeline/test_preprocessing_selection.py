"""TC-5-1 — preprocessing & selection pipeline."""

from __future__ import annotations

import pytest

from dataeval_flow import PreprocessorConfig, SelectionConfig, SelectionStep
from dataeval_flow.preprocessing import PreprocessingStep
from verification.fixtures import make_synthetic_dataset


@pytest.mark.test_case("5-1")
class TestPreprocessingSelection:
    def test_preprocessor_config_accepts_transforms(self) -> None:
        cfg = PreprocessorConfig(
            name="pp",
            steps=[PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True})],
        )
        assert len(cfg.steps) == 1

    def test_selection_step_constructs(self) -> None:
        step = SelectionStep(type="Limit", params={"size": 4})
        assert step.type == "Limit"

    def test_selection_config_stacks_steps(self) -> None:
        cfg = SelectionConfig(
            name="sel",
            steps=[
                SelectionStep(type="Limit", params={"size": 4}),
                SelectionStep(type="Shuffle", params={}),
            ],
        )
        assert len(cfg.steps) == 2

    def test_selection_limit_reduces_dataset_length(self) -> None:
        from dataeval.selection import Limit, Select

        ds = make_synthetic_dataset(n=8)
        limited = Select(ds, [Limit(size=3)])
        assert len(limited) == 3
