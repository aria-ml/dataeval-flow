"""TC-13-2 — a result that failed keeps the envelope, carries its errors, and refuses to be read as an output."""

from __future__ import annotations

import json

import pytest
import yaml

import dataeval_flow
from dataeval_flow import ResultMetadata, load_config, run_task, set_device
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.quality import DuplicatesResult
from dataeval_flow.steps import ChainResult
from verification.functional.reporting._project import FAILING_QUALITY, pipeline, run_project, task, write_images

pytestmark = pytest.mark.required

STEP_ERROR = "outliers: ValueError: n_expected_clusters=500 should be less than dataset size (8)"


@pytest.fixture(scope="module")
def step_failed(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    """A `quality` run whose `outliers` step raised, so the steps after it were skipped."""
    results = run_project(
        tmp_path_factory.mktemp("step_failed"), workflows=[FAILING_QUALITY], tasks=[task("bad_task", "q_fail")]
    )
    return results["bad_task"]


@pytest.fixture(scope="module")
def succeeded(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    return run_project(tmp_path_factory.mktemp("succeeded"))["clean_task"]


@pytest.fixture(scope="module")
def refused(tmp_path_factory: pytest.TempPathFactory) -> DuplicatesResult:
    """A task refused before it ran: it asks for cluster-mode duplicates over two sources."""
    root = tmp_path_factory.mktemp("refused")
    write_images(root, name="imgs")
    write_images(root, name="imgs_b", seed=1)
    config = pipeline(tasks=[])
    config["datasets"] = [
        {"name": "a_ds", "format": "image_folder", "path": "imgs", "infer_labels": True},
        {"name": "b_ds", "format": "image_folder", "path": "imgs_b", "infer_labels": True},
    ]
    config["sources"] = [{"name": "a", "dataset": "a_ds"}, {"name": "b", "dataset": "b_ds"}]
    config["evaluators"] = [{"name": "clustered", "type": "duplicates", "cluster_sensitivity": 1.0}]
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    set_device("cpu")
    # Run directly, not out of `config.tasks`: a task listed there is refused when the config loads.
    return run_task(
        load_config(root / "config.yaml"),
        TaskConfig(name="refused_task", evaluator="clustered", sources=["a", "b"]),
        data_dir=root,
    )


class TestFailedWorkflowResult:
    def test_the_result_is_unsuccessful_and_lists_the_step_error(self, step_failed: ChainResult) -> None:
        assert isinstance(step_failed, ChainResult)
        assert step_failed.kind == "workflow"
        assert not step_failed.success
        assert step_failed.errors == [STEP_ERROR]

    def test_its_output_refuses_to_be_read(self, step_failed: ChainResult) -> None:
        with pytest.raises(RuntimeError, match="n_expected_clusters=500"):
            _ = step_failed.output

    def test_the_steps_that_ran_stay_readable_and_the_dependents_are_skipped(self, step_failed: ChainResult) -> None:
        assert step_failed.failed_steps == ["outliers"]
        assert step_failed.steps["outliers"].status == "failed"
        assert step_failed.steps["label-health"].status == "ok"
        assert step_failed.steps["clean"].status == "skipped"
        assert "outliers" in (step_failed.steps["clean"].reason or "")

    def test_the_dict_carries_kind_envelope_health_steps_and_errors(self, step_failed: ChainResult) -> None:
        payload = step_failed.to_dict()
        assert payload["kind"] == "workflow"
        assert payload["errors"] == [STEP_ERROR]
        assert payload["health"]["status"] == "failed"  # type: ignore[index]
        assert payload["health"]["failed_steps"] == ["outliers"]  # type: ignore[index]
        outliers = payload["steps"]["outliers"]  # type: ignore[index]
        assert outliers["errors"] == ["ValueError: n_expected_clusters=500 should be less than dataset size (8)"]
        assert "output" not in outliers
        assert json.loads(step_failed.export()) == payload

    def test_the_envelope_is_filled_as_for_a_successful_run(self, step_failed: ChainResult) -> None:
        meta = step_failed.metadata
        assert meta.tool == "dataeval-flow"
        assert meta.tool_version == dataeval_flow.__version__
        assert meta.dataset_id == "ds"
        assert meta.model_id == "flat (flatten)"
        assert meta.device == "cpu"
        assert meta.execution_time_s is not None
        assert meta.library_versions["dataeval"]
        assert meta.resolved_config["workflow"]["name"] == "q_fail"
        assert meta.resolved_config["workflow"]["outliers"]["n_clusters"] == 500


class TestRefusedTaskResult:
    def test_a_task_refused_before_it_ran_comes_back_as_a_failed_result_of_the_evaluators_class(
        self, refused: DuplicatesResult
    ) -> None:
        assert isinstance(refused, DuplicatesResult)
        assert refused.kind == "evaluator"
        assert not refused.success
        assert len(refused.errors) == 1
        assert "reads exactly one source in cluster mode" in refused.errors[0]
        with pytest.raises(RuntimeError, match="did not complete, so it has no output"):
            _ = refused.output

    def test_it_carries_the_envelope_and_its_dict_holds_kind_metadata_and_errors_only(
        self, refused: DuplicatesResult
    ) -> None:
        payload = refused.to_dict()
        assert list(payload) == ["kind", "metadata", "errors"]
        assert payload["errors"] == refused.errors
        meta = refused.metadata
        assert meta.tool_version == dataeval_flow.__version__
        assert meta.dataset_id == "a_ds,b_ds"
        assert meta.execution_time_s == 0.0  # nothing ran, so no time was spent running
        assert [source["name"] for source in meta.resolved_config["sources"]] == ["a", "b"]
        assert meta.resolved_config["evaluator"]["name"] == "clustered"

    def test_its_report_says_it_failed_and_why(self, refused: DuplicatesResult) -> None:
        text = refused.report()
        assert "  FAILED\n" in text
        assert "reads exactly one source in cluster mode" in " ".join(text.split())  # the report wraps its lines


class TestResultContract:
    def test_a_failed_result_can_be_made_without_a_run(self) -> None:
        result = ChainResult.failed(type="quality", errors=["boom"])
        assert not result.success
        assert result.errors == ["boom"]
        assert isinstance(result.metadata, ResultMetadata)
        assert result.metadata.tool == "dataeval-flow"
        assert result.to_dict()["errors"] == ["boom"]
        assert result.health["status"] == "failed"

    def test_a_successful_result_must_carry_an_output_and_a_failed_one_must_not(self) -> None:
        from dataeval_flow.steps import ChainOutput

        template = ChainResult.failed(type="quality", errors=["boom"])
        with pytest.raises(ValueError, match="carries its output"):
            ChainResult(type="quality", success=True, metadata=template.metadata)
        with pytest.raises(ValueError, match="carries no output"):
            ChainResult(type="quality", success=False, metadata=template.metadata, output=ChainOutput({}))

    def test_a_successful_result_reads_its_output(self, succeeded: ChainResult) -> None:
        assert succeeded.success
        assert succeeded.errors == []
        assert list(succeeded.output.steps) == list(succeeded.steps)
        assert "errors" not in succeeded.to_dict()
