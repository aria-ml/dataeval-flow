"""TC-5-1 — preprocessing & view pipeline."""

from __future__ import annotations

import pytest

from dataeval_flow import PreprocessorConfig, ViewConfig, ViewOperation
from dataeval_flow.preprocessing import PreprocessingStep
from verification.fixtures import make_synthetic_dataset


@pytest.mark.test_case("5-1")
class TestPreprocessingView:
    def test_preprocessor_config_accepts_transforms(self) -> None:
        cfg = PreprocessorConfig(
            name="pp",
            steps=[PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True})],
        )
        assert len(cfg.steps) == 1

    def test_view_operation_constructs(self) -> None:
        op = ViewOperation(type="Limit", params={"size": 4})
        assert op.type == "Limit"

    def test_view_config_stacks_operations(self) -> None:
        cfg = ViewConfig(
            name="view",
            operations=[
                ViewOperation(type="Limit", params={"size": 4}),
                ViewOperation(type="Shuffle", params={}),
            ],
        )
        assert len(cfg.operations) == 2

    def test_view_limit_reduces_dataset_length(self) -> None:
        from dataeval.data import Limit, View

        ds = make_synthetic_dataset(n=8)
        limited = View(ds, [Limit(size=3)])
        assert len(limited) == 3
