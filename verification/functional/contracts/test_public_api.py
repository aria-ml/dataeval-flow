"""TC-1-1 — public API surface of dataeval_flow."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.required

PUBLIC_API = [
    "load_config",
    "load_dataset",
    "run",
    "run_task",
    "run_tasks",
    "PipelineConfig",
    "Result",
    "ResultMetadata",
    "InputSpec",
    "InputKind",
    "SourceCount",
    "__version__",
]


@pytest.mark.test_case("1-1")
class TestPublicAPI:
    def test_top_level_all_exports(self) -> None:
        import dataeval_flow

        assert hasattr(dataeval_flow, "__all__")
        assert len(dataeval_flow.__all__) > 0

    def test_all_exports_are_importable(self) -> None:
        import dataeval_flow

        for name in dataeval_flow.__all__:
            assert hasattr(dataeval_flow, name), f"missing public symbol: {name}"

    def test_top_level_exports_exactly_the_front_door(self) -> None:
        import dataeval_flow

        assert sorted(dataeval_flow.__all__) == sorted(PUBLIC_API)

    def test_run_tasks_and_load_config_present(self) -> None:
        from dataeval_flow import load_config, run, run_task, run_tasks

        assert callable(load_config)
        assert callable(run)
        assert callable(run_task)
        assert callable(run_tasks)
