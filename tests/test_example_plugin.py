"""A plugin written against the public API alone works everywhere a built-in does."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from dataeval_flow import PipelineConfig, run, run_tasks
from dataeval_flow.config import PreprocessingStep, PreprocessorConfig, TaskConfig
from dataeval_flow.config.extractors import list_extractors
from dataeval_flow.config.transforms import list_transforms
from dataeval_flow.evaluators import list_evaluators
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesResult
from dataeval_flow.workflows import get_workflow, list_workflows
from tests.evaluator_toys import ToyImages, toy_pipeline
from tests.example_plugin import BrightnessConfig, BrightnessResult, CountConfig, CountResult, Invert, MeanConfig

EXAMPLES = {
    "dataeval_flow.workflows": [("example.count", "tests.example_plugin:CountWorkflow")],
    "dataeval_flow.evaluators": [("example.brightness", "tests.example_plugin:BrightnessEvaluator")],
    "dataeval_flow.extractors": [("example.mean", "tests.example_plugin:MeanExtractor")],
    "dataeval_flow.transforms": [("example.Invert", "tests.example_plugin:Invert")],
}


@pytest.fixture
def examples(plugins: dict[str, list[tuple[str, str]]]) -> dict[str, list[tuple[str, str]]]:
    for group, entries in EXAMPLES.items():
        plugins[group] = list(entries)
    return plugins


@pytest.mark.usefixtures("examples")
def test_every_example_is_listed() -> None:
    assert "example.count" in [cls.name for cls in list_workflows()]
    assert "example.brightness" in [cls.name for cls in list_evaluators()]
    assert "example.mean" in [cls.name for cls in list_extractors()]
    assert "example.Invert" in [cls.name for cls in list_transforms()]


@pytest.mark.usefixtures("examples")
def test_the_examples_validate_from_yaml() -> None:
    text = "workflows:\n  - type: example.count\n    minimum: 5\nevaluators:\n  - type: example.brightness\n"
    config = PipelineConfig.model_validate(yaml.safe_load(text))
    assert config.workflows is not None
    assert config.evaluators is not None


@pytest.mark.usefixtures("examples")
def test_the_schema_has_the_examples() -> None:
    definitions = PipelineConfig.model_json_schema()["$defs"]
    assert {"CountConfig", "BrightnessConfig", "MeanConfig"} <= set(definitions)


@pytest.mark.usefixtures("examples")
def test_the_workflow_runs_as_a_task() -> None:
    config = toy_pipeline(
        workflows=[CountConfig(name="count", minimum=100)],
        tasks=[TaskConfig(name="t", workflow="count", sources="src")],
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, CountResult)
    assert result.output.raw.counts == {"src": len(ToyImages())}
    assert result.warning_count == 1


@pytest.mark.usefixtures("examples")
def test_the_evaluator_runs_in_memory() -> None:
    result = run(BrightnessConfig(), ToyImages())
    assert isinstance(result, BrightnessResult)
    assert result.success


@pytest.mark.usefixtures("examples")
def test_the_extractor_and_transform_serve_a_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []
    invert = Invert.__call__

    def recording(self: Invert, data: Any, /) -> Any:
        seen.append(type(data).__name__)
        return invert(self, data)

    monkeypatch.setattr(Invert, "__call__", recording)
    preprocessor = PreprocessorConfig(name="invert", steps=[PreprocessingStep(step="example.Invert")])
    # Flow sets no default batch size, and DataEval needs one before it builds embeddings.
    extractor = MeanConfig(preprocessor="invert", batch_size=4)
    # A fresh disk cache, so the embeddings are computed here rather than served from another test's run.
    result = run(
        DuplicatesConfig(cluster_sensitivity=1.0),
        ToyImages(),
        extractor=extractor,
        definitions=[preprocessor],
        cache_dir=tmp_path,
    )
    assert isinstance(result, DuplicatesResult)
    assert result.success
    assert seen == ["Tensor"] * len(ToyImages())


def test_broken_and_duplicate_entries_leave_the_rest_working(examples: dict[str, list[tuple[str, str]]]) -> None:
    examples["dataeval_flow.workflows"] += [
        ("example.gone", "tests.nowhere:Nope"),
        ("data-cleaning", "tests.example_plugin:CountWorkflow"),
    ]
    names = [cls.name for cls in list_workflows()]
    assert "example.count" in names
    assert "example.gone" not in names
    assert get_workflow("data-cleaning").__name__ == "DataCleaningWorkflow"
    assert run(CountConfig(), ToyImages()).success


@pytest.fixture
def installed_examples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install the examples as a package would: a ``.dist-info`` whose ``entry_points.txt`` names them.

    Nothing is patched: ``importlib.metadata`` finds the distribution on ``sys.path``, as it finds an installed
    plugin, and every registry reads it from there. Nothing is installed into the environment either.
    """
    from dataeval_flow.config.extractors._registry import EXTRACTORS
    from dataeval_flow.config.transforms._registry import TRANSFORMS
    from dataeval_flow.evaluators._registry import EVALUATORS
    from dataeval_flow.workflows._registry import WORKFLOWS

    dist_info = tmp_path / "flow_example_plugin-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: flow-example-plugin\nVersion: 0.1\n")
    groups = "\n".join(
        f"[{group}]\n" + "".join(f"{name} = {target}\n" for name, target in entries)
        for group, entries in EXAMPLES.items()
    )
    (dist_info / "entry_points.txt").write_text(groups)
    monkeypatch.syspath_prepend(str(tmp_path))
    registries = [WORKFLOWS, EVALUATORS, EXTRACTORS, TRANSFORMS]
    for registry in registries:
        registry._reset()
    yield
    for registry in registries:
        registry._reset()


@pytest.mark.usefixtures("installed_examples")
def test_the_examples_are_found_through_real_package_metadata() -> None:
    """The registries read real ``importlib.metadata`` entry points."""
    from importlib.metadata import distribution

    from dataeval_flow.config.transforms import get_transform

    assert distribution("flow-example-plugin").entry_points
    assert "example.count" in [cls.name for cls in list_workflows()]
    assert "example.brightness" in [cls.name for cls in list_evaluators()]
    assert "example.mean" in [cls.name for cls in list_extractors()]
    assert "example.Invert" in [cls.name for cls in list_transforms()]
    assert get_transform("example.Invert") is Invert
    text = (
        "workflows:\n  - type: example.count\n"
        "evaluators:\n  - type: example.brightness\n"
        "extractors:\n  - model: example.mean\n"
    )
    config = PipelineConfig.model_validate(yaml.safe_load(text))
    assert config.workflows is not None
    assert isinstance(config.workflows[0], CountConfig)
    assert config.evaluators is not None
    assert isinstance(config.evaluators[0], BrightnessConfig)
    assert config.extractors is not None
    assert isinstance(config.extractors[0], MeanConfig)
    assert run(CountConfig(), ToyImages()).success
