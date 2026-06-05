"""TC-1-1 — public API surface of dataeval_flow."""

from __future__ import annotations

import pytest


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

    def test_run_tasks_and_load_config_present(self) -> None:
        from dataeval_flow import load_config, load_config_folder, run_tasks

        assert callable(load_config)
        assert callable(load_config_folder)
        assert callable(run_tasks)
