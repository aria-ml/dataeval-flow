"""TC-10-1 (NFR-3) — type-safety infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.test_case("10-1")
class TestTypeSafety:
    def test_py_typed_marker_present(self) -> None:
        import dataeval_flow

        assert (Path(dataeval_flow.__file__).parent / "py.typed").exists()

    def test_top_level_all_defined(self) -> None:
        import dataeval_flow

        assert hasattr(dataeval_flow, "__all__")
        assert isinstance(dataeval_flow.__all__, list)
        assert len(dataeval_flow.__all__) > 0

    def test_workflow_protocol_runtime_checkable(self) -> None:
        from dataeval_flow.workflow import WorkflowProtocol, get_workflow

        wf = get_workflow("data-cleaning")
        assert isinstance(wf, WorkflowProtocol)
