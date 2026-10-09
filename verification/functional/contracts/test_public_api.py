"""TC-33-1 — the public API surface of dataeval_flow: what ``__all__`` lists is what imports."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

pytestmark = pytest.mark.required

# The classes a plug-in subclasses, each at the one public path it is documented under.
EXTENSION_POINTS = [
    "dataeval_flow.workflows:Workflow",
    "dataeval_flow.workflows:WorkflowConfig",
    "dataeval_flow.evaluators:Evaluator",
    "dataeval_flow.evaluators:EvaluatorConfig",
    "dataeval_flow.evaluators:EvaluatorResult",
    "dataeval_flow.config.extractors:Extractor",
    "dataeval_flow.config.extractors:ExtractorConfig",
    "dataeval_flow.config.image_transforms:ImageTransform",
    "dataeval_flow.steps:Transform",
    "dataeval_flow.steps:Check",
    "dataeval_flow.steps:Combine",
    "dataeval_flow:Result",
]


def _public_modules() -> list[str]:
    import dataeval_flow

    names = ["dataeval_flow"]
    names.extend(
        info.name
        for info in pkgutil.walk_packages(dataeval_flow.__path__, "dataeval_flow.")
        if not any(part.startswith("_") for part in info.name.split(".")[1:])
    )
    return sorted(names)


class TestPublicAPI:
    def test_top_level_all_exports(self) -> None:
        import dataeval_flow

        assert isinstance(dataeval_flow.__all__, list)
        assert len(dataeval_flow.__all__) > 0
        assert len(set(dataeval_flow.__all__)) == len(dataeval_flow.__all__)

    def test_all_exports_are_importable(self) -> None:
        import dataeval_flow

        for name in dataeval_flow.__all__:
            assert hasattr(dataeval_flow, name), f"missing public symbol: {name}"

    @pytest.mark.parametrize("module_name", _public_modules())
    def test_every_name_a_public_module_lists_in_all_imports_from_it(self, module_name: str) -> None:
        module = importlib.import_module(module_name)
        listed = getattr(module, "__all__", [])

        assert len(set(listed)) == len(listed), f"{module_name}.__all__ lists a name twice"
        assert not [name for name in listed if not hasattr(module, name)], module_name

    @pytest.mark.parametrize("path", EXTENSION_POINTS)
    def test_each_extension_point_is_exported_where_it_is_documented(self, path: str) -> None:
        module_name, name = path.split(":")
        module = importlib.import_module(module_name)

        assert name in module.__all__
        assert isinstance(getattr(module, name), type)

    def test_run_tasks_run_and_load_config_present(self) -> None:
        from dataeval_flow import load_config, run, run_task, run_tasks

        assert callable(load_config)
        assert callable(run)
        assert callable(run_task)
        assert callable(run_tasks)
