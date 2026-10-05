"""Every task runs through the engine: rerouted tasks unchanged, custom workflows as chains (spec §3.4, §7.1)."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from dataeval_flow import load_config, run, run_task, run_tasks
from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._cache import DatasetCache
from dataeval_flow._result import LabelSpaceRecord
from dataeval_flow.config import ImageFolderDatasetConfig, PipelineConfig, SourceConfig, TaskConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import (
    ChainResult,
    CustomWorkflowConfig,
    DataType,
    Port,
    Transform,
    TransformConfig,
    TransformContext,
)
from tests.chain_toys import chain_pipeline, register_toys
from tests.evaluator_toys import ToyImages
from tests.golden.rerouting import CASES, approximately, normalized
from tests.workflow_toys import ToyCountConfig, register_count

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_rerouted_task_produces_the_json_it_did_before(name: str) -> None:
    DatasetCache.clear_instances()
    expected = json.loads((GOLDEN / f"rerouting_{name}.json").read_text())
    assert normalized(json.loads(json.dumps(CASES[name]()))) == approximately(expected)


_WORKFLOW = {
    "name": "w",
    "inputs": ["a"],
    "steps": [
        {"name": "few", "transform": "toy-first", "input": "a", "n": 6},
        {"name": "dupes", "evaluator": "dupes", "input": "few"},
    ],
}


def _config(**extra: Any) -> PipelineConfig:
    task = {"name": "t", "workflow": "w", "sources": ["src"]}
    return chain_pipeline(workflows=[_WORKFLOW], evaluators=[DuplicatesConfig(name="dupes")], tasks=[task], **extra)


@pytest.mark.usefixtures("toys")
def test_a_custom_workflow_task_returns_its_chain() -> None:
    result = run_tasks(_config())["t"]
    assert isinstance(result, ChainResult)
    assert result.success
    assert [record.status for record in result.steps.values()] == ["ok", "ok"]
    assert result.metadata.source_descriptions == ["src (src_data)"]
    assert result.metadata.lineage[0].source == "src"
    assert result.metadata.resolved_config["workflow"] == _WORKFLOW


@pytest.mark.usefixtures("toys")
def test_run_takes_a_custom_workflow_with_its_pool_entries() -> None:
    workflow = CustomWorkflowConfig.model_validate(_WORKFLOW)
    result = run(workflow, ToyImages(), definitions=[DuplicatesConfig(name="dupes")])
    assert isinstance(result, ChainResult)
    assert result.steps["dupes"].status == "ok"


@pytest.mark.usefixtures("toys")
def test_run_binds_a_mapping_to_a_workflows_inputs_in_order() -> None:
    workflow = CustomWorkflowConfig.model_validate(
        {
            "name": "pair",
            "inputs": ["ref", "test"],
            "steps": [
                {"name": "r", "transform": "toy-keep", "input": "ref"},
                {"name": "t", "transform": "toy-keep", "input": "test"},
            ],
        }
    )
    result = run(workflow, {"train": ToyImages(count=10), "incoming": ToyImages(count=14, seed=1)})
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    records = [(record.name, record.source, record.items) for record in result.metadata.lineage]
    assert records == [("ref", "train", 10), ("test", "incoming", 14), ("r", None, 10), ("t", None, 14)]


@pytest.mark.usefixtures("toys")
def test_a_task_run_directly_that_binds_too_few_sources_fails_as_a_chain() -> None:
    two = {**_WORKFLOW, "name": "two", "inputs": ["a", "b"]}
    config = chain_pipeline(workflows=[two], evaluators=[DuplicatesConfig(name="dupes")])
    result = run_task(config, TaskConfig(name="t", workflow="two", sources=["src"]))
    assert isinstance(result, ChainResult)
    assert not result.success
    assert "takes one source for 'a' and one source for 'b', but the task names 1" in result.errors[0]


def _per_source(read: str) -> dict[str, Any]:
    """A workflow over a reference and a list of tests, whose second step reads `read`."""
    return {
        "name": "per",
        "inputs": ["ref", {"name": "tests", "list": True}],
        "steps": [
            {"name": "kept", "transform": "toy-keep", "input": "tests"},
            {"name": "v", "transform": "toy-keep", "input": read},
        ],
    }


_THREE = {"src": ToyImages(), "a": ToyImages(seed=1), "b": ToyImages(seed=2)}


@pytest.mark.usefixtures("toys")
@pytest.mark.parametrize("read", ["tests[nope]", "kept[nope]"])
def test_a_task_whose_sources_leave_a_list_key_unbound_fails_the_load(read: str) -> None:
    task = {"name": "t", "workflow": "per", "sources": ["src", "a", "b"]}
    base = read.split("[")[0]
    message = (
        rf"Task 't' binds sources a, b to `tests`\. Step 'v' reads `{base}\[nope\]`, but `{base}` has elements "
        r"a, b, not `nope`\."
    )
    with pytest.raises(ValidationError, match=message):
        chain_pipeline(workflows=[_per_source(read)], tasks=[task], datasets=_THREE)


@pytest.mark.usefixtures("toys")
def test_a_task_whose_sources_bind_each_list_key_runs() -> None:
    task = {"name": "t", "workflow": "per", "sources": ["src", "a", "b"]}
    result = run_tasks(chain_pipeline(workflows=[_per_source("tests[b]")], tasks=[task], datasets=_THREE))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert result.steps["v"].inputs == ["tests[b]"]


@pytest.mark.usefixtures("toys")
def test_a_task_run_directly_whose_sources_leave_a_list_key_unbound_fails_as_a_chain() -> None:
    config = chain_pipeline(workflows=[_per_source("tests[nope]")], datasets=_THREE)
    result = run_task(config, TaskConfig(name="t", workflow="per", sources=["src", "a", "b"]))
    assert isinstance(result, ChainResult)
    assert not result.success
    assert result.errors == [
        (
            "Task 't' binds sources a, b to `tests`. Step 'v' reads `tests[nope]`, but `tests` has elements a, b, "
            "not `nope`."
        )
    ]
    assert result.steps == {}


@pytest.mark.usefixtures("toys")
def test_run_refuses_a_list_key_its_datasets_do_not_bind() -> None:
    workflow = CustomWorkflowConfig.model_validate(_per_source("tests[nope]"))
    with pytest.raises(ValidationError, match=r"binds sources a, b to `tests`"):
        run(workflow, {"ref": ToyImages(), "a": ToyImages(seed=1), "b": ToyImages(seed=2)})


class WhereConfig(TransformConfig):
    input: str


class Where(Transform[WhereConfig]):
    """Hands its input on, noting where the run writes, and adds one label-space record."""

    name: ClassVar[str] = "toy-where"
    description: ClassVar[str] = "Notes the output directory."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)
    seen: ClassVar[list[Path | None]] = []

    def run(self, config: WhereConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        Where.seen.append(context.output_dir)
        return {"output": inputs["input"].value}

    def label_space(
        self, config: WhereConfig, inputs: Mapping[str, Any], outputs: Mapping[str, Any], *, address: str
    ) -> list[LabelSpaceRecord]:
        return [LabelSpaceRecord(source=address, target=["a", "b"], digest="relabelled")]


@pytest.mark.usefixtures("toys")
def test_a_chain_step_gets_the_output_dir_and_its_label_space_is_kept_once(plugins, tmp_path: Path) -> None:
    plugins["dataeval_flow.transforms"].append(("toy-where", "tests.test_chain_tasks:Where"))
    workflow = {"name": "w", "inputs": ["a"], "steps": [{"name": "where", "transform": "toy-where", "input": "a"}]}
    config = chain_pipeline(workflows=[workflow], tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}])
    Where.seen.clear()
    result = run_tasks(config, output_dir=tmp_path)["t"]
    assert Where.seen == [tmp_path]
    assert [record.source for record in result.metadata.label_space] == ["where"]
    assert result.metadata.label_space_digest == "relabelled"


@pytest.mark.usefixtures("toys")
def test_a_chain_whose_label_spaces_disagree_names_none_of_them(plugins) -> None:
    plugins["dataeval_flow.transforms"].append(("toy-where", "tests.test_chain_tasks:Where"))
    relabel = {"type": "Relabel", "params": {"class_remap": {"a": "x"}, "target": ["x", "b"]}}
    workflow = {"name": "w", "inputs": ["a"], "steps": [{"name": "where", "transform": "toy-where", "input": "a"}]}
    config = chain_pipeline(
        workflows=[workflow],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        extra={"views": [{"name": "renamed", "operations": [relabel]}]},
    )
    source = config.sources[0].model_copy(update={"view": "renamed"})  # type: ignore[index]
    result = run_tasks(config.model_copy(update={"sources": [source]}))["t"]
    assert [record.source for record in result.metadata.label_space] == ["src", "where"]
    assert len({record.digest for record in result.metadata.label_space}) == 2
    assert result.metadata.label_space_digest is None


@pytest.mark.usefixtures("toys")
def test_the_tui_state_saves_a_custom_workflow_unchanged(tmp_path: Path) -> None:
    # A file-backed dataset: YAML cannot hold an in-memory one. Validation does not read the folder.
    config = PipelineConfig.model_validate(
        {
            "datasets": [ImageFolderDatasetConfig(name="src_data", path="images")],
            "sources": [SourceConfig(name="src", dataset="src_data")],
            "evaluators": [DuplicatesConfig(name="dupes")],
            "workflows": [_WORKFLOW],
            "tasks": [{"name": "t", "workflow": "w", "sources": ["src"]}],
        }
    )
    state = ConfigState()
    state.load_dict(config)
    state.save_file(tmp_path / "pipeline.yaml")
    reloaded = load_config(tmp_path / "pipeline.yaml")
    assert [w.model_dump(mode="json") for w in reloaded.workflows or ()] == [_WORKFLOW]
    assert state.validate_item("workflows", _WORKFLOW) == []


def test_the_tui_snippet_names_a_custom_workflow_by_its_steps() -> None:
    from dataeval_flow._app._viewmodel._rendering import _snippet_workflow

    text = _snippet_workflow(_WORKFLOW)
    assert "2 steps" in text
    assert "few: transform toy-first" in text
    assert "dupes: evaluator dupes" in text


def test_a_one_step_task_makes_its_evaluator_once() -> None:
    from unittest.mock import patch

    from dataeval_flow import _orchestrator

    task = TaskConfig(name="t", workflow="dupes", kind="evaluator", sources="src")
    config = chain_pipeline(evaluators=[DuplicatesConfig(name="dupes")], tasks=[task.model_dump()])
    with patch.object(_orchestrator, "_implementation", wraps=_orchestrator._implementation) as made:
        result = run_task(config, task)
    assert result.success
    assert made.call_count == 1


@pytest.mark.parametrize(
    ("kind", "entry"),
    [
        ("evaluator", DuplicatesConfig(name="target")),
        ("workflow", ToyCountConfig(name="target")),
    ],
)
def test_a_one_step_task_hands_its_runner_the_contexts_it_resolved_and_records_no_lineage(
    kind: str, entry, plugins
) -> None:
    from unittest.mock import patch

    register_count(plugins)

    from dataeval_flow import _orchestrator
    from dataeval_flow._chain import _run as engine

    resolve, execute, chain = _orchestrator._source_contexts, _orchestrator._run_target, engine.run_chain
    resolved: list[Any] = []
    handed: list[Any] = []
    runs: list[Any] = []

    def resolving(*args: Any, **kwargs: Any) -> Any:
        resolved.append(resolve(*args, **kwargs))
        return resolved[-1]

    def executing(target: Any, config: Any, context: Any) -> Any:
        handed.append(context)
        return execute(target, config, context)

    def chaining(*args: Any, **kwargs: Any) -> Any:
        runs.append(chain(*args, **kwargs))
        return runs[-1]

    task = TaskConfig(name="t", workflow="target", kind=kind, sources="src")  # type: ignore[arg-type]
    evaluators, workflows = ([entry], []) if kind == "evaluator" else ([], [entry])
    config = chain_pipeline(evaluators=evaluators, workflows=workflows, tasks=[task.model_dump()])
    with (
        patch.object(_orchestrator, "_source_contexts", side_effect=resolving),
        patch.object(_orchestrator, "_run_target", side_effect=executing),
        patch.object(engine, "run_chain", side_effect=chaining),
    ):
        result = run_task(config, task)
    assert result.success, result.errors
    ((contexts, _),) = resolved
    (context,) = handed
    (run,) = runs
    assert list(context.dataset_contexts) == ["src"]
    assert context.dataset_contexts["src"] == contexts["src"]
    assert (run.lineage, run.label_space) == ([], [])
    assert "lineage" not in result.to_dict()["metadata"]  # type: ignore[operator]


def test_a_one_step_task_whose_step_fails_before_its_evaluator_runs_returns_a_failed_result_of_its_class() -> None:
    from unittest.mock import patch

    from dataeval_flow import _orchestrator
    from dataeval_flow.evaluators.quality import DuplicatesResult

    task = TaskConfig(name="t", workflow="dupes", kind="evaluator", sources="src")
    config = chain_pipeline(evaluators=[DuplicatesConfig(name="dupes")], tasks=[task.model_dump()])
    with patch.object(_orchestrator, "_run_target", side_effect=RuntimeError("no context")):
        result = run_task(config, task)
    assert isinstance(result, DuplicatesResult)
    assert (result.success, result.type, result.errors) == (False, "duplicates", ["RuntimeError: no context"])


def test_the_golden_normalizer_drops_a_thumbnail_s_encoded_bytes_and_keeps_its_item() -> None:
    payload = {"assets": [{"data": "UklGR", "item": {"index": 0, "source": "src"}, "width": 16}]}
    assert normalized(payload) == {"assets": [{"item": {"index": 0, "source": "src"}, "width": 16}]}


def test_the_golden_normalizer_drops_volatile_keys_at_every_depth() -> None:
    payload = {
        "timestamp": "2026-09-29",
        "metadata": {"execution_time_s": 1.5, "tool_version": "0.2", "kept": 1},
        "steps": [
            {"execution_time": 2.0, "execution_duration": "2s", "dataeval": {"version": "1.2", "commit": "abc"}},
            [{"timestamp": "nested", "items": 3}],
        ],
        "dataeval": {"version": "1.2", "flags": ["x"]},
    }
    assert normalized(payload) == {
        "metadata": {"kept": 1},
        "steps": [{"dataeval": {"commit": "abc"}}, [{"items": 3}]],
        "dataeval": {"flags": ["x"]},
    }


def test_the_golden_comparison_lets_floats_differ_only_in_their_last_digits() -> None:
    # torch's float32 arithmetic differs from one CPU to another: CI read MMD's distance as 0.3987167477607727
    # where this machine recorded 0.3987165689468384.
    expected = approximately({"distance": 0.3987165689468384, "items": [24, 0.0152357221], "method": "mmd"})
    assert expected == {"distance": 0.3987167477607727, "items": [24, 0.01523590087890625], "method": "mmd"}
    assert expected != {"distance": 0.3988, "items": [24, 0.0152357221], "method": "mmd"}
    assert expected != {"distance": 0.3987165689468384, "items": [25, 0.0152357221], "method": "mmd"}
    assert expected != {"distance": 0.3987165689468384, "items": [24, 0.0152357221], "method": "ks"}


@pytest.mark.usefixtures("toys")
def test_a_pooled_extractor_is_one_a_step_can_name_not_the_tasks() -> None:
    from dataeval_flow.evaluators.scope import CoverageConfig
    from tests.evaluator_toys import FLAT

    def workflow(**step: Any) -> CustomWorkflowConfig:
        steps = [{"name": "c", "evaluator": "cov", "input": "a", **step}]
        return CustomWorkflowConfig.model_validate({"name": "w", "inputs": ["a"], "steps": steps})

    pools = [CoverageConfig(name="cov"), FLAT]
    message = "Task 'w' runs workflow 'w', whose step 'c' needs an extractor to produce embeddings"
    with pytest.raises(ValidationError, match=message):
        run(workflow(), ToyImages(count=30), definitions=pools)
    result = run(workflow(extractor="flat"), ToyImages(count=30), definitions=pools)
    assert isinstance(result, ChainResult)
    assert result.steps["c"].status == "ok", result.steps["c"].errors
