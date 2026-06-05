"""TC-1-1 / TC-1-2 — installation, import, and optional extras."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.test_case("1-1")
class TestPipInstall:
    def test_import_dataeval_flow(self) -> None:
        import dataeval_flow

        assert dataeval_flow is not None

    def test_version_is_set(self) -> None:
        import dataeval_flow

        assert dataeval_flow.__version__
        assert dataeval_flow.__version__ != "unknown"

    def test_core_modules_importable(self) -> None:
        for name in (
            "dataeval_flow.config",
            "dataeval_flow.dataset",
            "dataeval_flow.workflow",
            "dataeval_flow.workflows",
            "dataeval_flow.runner",
            "dataeval_flow.cache",
            "dataeval_flow.preprocessing",
            "dataeval_flow.embeddings",
            "dataeval_flow.metadata",
            "dataeval_flow.selection",
        ):
            assert importlib.import_module(name) is not None

    def test_py_typed_marker_present(self) -> None:
        import dataeval_flow

        assert (Path(dataeval_flow.__file__).parent / "py.typed").exists()


@pytest.mark.test_case("1-2")
class TestOptionalExtras:
    @pytest.mark.parametrize(
        ("module", "label"),
        [
            ("torch", "cpu/cu*"),
            ("torchvision", "cpu/cu*"),
            ("onnxruntime", "onnx"),
            ("cv2", "opencv"),
            ("textual", "app"),
        ],
    )
    def test_optional_module_importable_if_installed(self, module: str, label: str) -> None:
        mod = pytest.importorskip(module, reason=f"{label} extra not installed")
        assert mod is not None
