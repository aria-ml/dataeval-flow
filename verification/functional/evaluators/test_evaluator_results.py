"""TC-20-3 and TC-20-4 — what an evaluator result holds, and how a task that runs an evaluator is validated."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import dataset_digest, run
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators import EvaluatorResult, get_evaluator
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesResult, OutliersConfig
from dataeval_flow.evaluators.scope import CoverageConfig
from dataeval_flow.evaluators.shift import DriftMMDConfig
from verification.fixtures import plant_duplicate_and_outlier, write_image_folder
from verification.functional.chains._toys import EVALUATOR_DATA, EVALUATOR_SETTINGS, FLAT, Images, pipeline
from verification.helpers import run_cli

pytestmark = pytest.mark.required


def _go(name: str, data: Any = None, **settings: Any) -> EvaluatorResult[Any]:
    default_data, needs_extractor = EVALUATOR_DATA[name]
    config = get_evaluator(name).config_type(**{**EVALUATOR_SETTINGS.get(name, {}), **settings})
    result = run(config, default_data if data is None else data, extractor=FLAT if needs_extractor else None)
    assert result.success, result.errors
    return result


def _output(proc) -> str:
    return proc.stdout + proc.stderr


class TestEvaluatorDeterminations:
    def test_duplicates_finds_the_planted_exact_group(self) -> None:
        result = _go("duplicates")
        assert isinstance(result, DuplicatesResult)
        rows = result.to_dict()["output"]["rows"]
        assert [(row["dup_type"], row["item_indices"]) for row in rows] == [("exact", [0, 5])]

    def test_outliers_flags_the_planted_white_image(self) -> None:
        result = _go("outliers", flags=["pixel"], outlier_threshold="zscore")
        flagged = {row["item_index"] for row in result.to_dict()["output"]["rows"]}
        assert flagged == {7}

    def test_label_health_counts_each_class(self) -> None:
        data = _go("label-health").to_dict()["output"]["data"]
        assert (data["item_count"], data["class_count"], data["label_counts_per_class"]) == (40, 2, {"a": 20, "b": 20})

    def test_content_digest_equals_the_digest_of_the_dataset(self) -> None:
        dataset = EVALUATOR_DATA["content-digest"][0]
        data = _go("content-digest").to_dict()["output"]["data"]
        assert data["content"] == dataset_digest(dataset).content
        assert data["items"] == len(dataset)

    def test_a_drift_detector_tells_a_shifted_source_from_an_unshifted_one(self) -> None:
        shifted = _go("drift-mmd").to_dict()["output"]["data"]
        same = _go("drift-mmd", {"reference": Images(40), "test": Images(40, seed=1)}).to_dict()["output"]["data"]
        assert shifted["drifted"] is True
        assert same["drifted"] is False

    def test_factor_leakage_lists_the_values_each_source_holds(self) -> None:
        data = _go("factor-leakage").to_dict()["output"]["data"]
        assert data["sources"] == ["a", "b"]
        assert data["factors"]["site"]["north"] == [20, 20]

    def test_label_alignment_reports_a_lossless_remap_onto_a_matching_ontology(self) -> None:
        data = _go("label-alignment").to_dict()["output"]["data"]
        assert data["mergeability"] == "lossless"
        assert data["class_remap"] == {"a": "a", "b": "b"}

    def test_an_empty_source_gives_a_result_with_no_rows(self) -> None:
        class Nothing(Images):
            def __init__(self) -> None:
                super().__init__(0)

        result = _go("duplicates", Nothing())
        assert result.to_dict()["output"]["rows"] == []
        assert result.report().strip()

    def test_the_result_holds_dataeval_s_own_output(self) -> None:
        result = _go("duplicates")
        native = result.output
        assert type(native).__name__ == "DuplicatesOutput"
        assert len(native.data()) == 1


class TestEvaluatorEnvelope:
    @pytest.mark.parametrize("name", ["duplicates", "label-health", "balance", "drift-mmd", "coverage"])
    def test_an_evaluator_result_carries_no_health_and_no_findings(self, name: str) -> None:
        payload = _go(name).to_dict()
        assert payload["kind"] == "evaluator"
        assert set(payload) - {"assets"} == {"kind", "metadata", "output"}  # assets: thumbnails of items named
        assert "health" not in payload
        assert "findings" not in payload
        assert "Health" not in _go(name).report()

    def test_the_envelope_names_the_evaluator_dataeval_and_the_resolved_config(self) -> None:
        result = _go("duplicates")
        meta = result.metadata
        assert meta.evaluator == "duplicates"
        assert meta.tool == "dataeval-flow"
        assert meta.dataeval.version
        assert meta.dataeval.name
        assert meta.execution_time_s is not None
        assert meta.resolved_config["evaluator"]["type"] == "duplicates"
        assert "workflow" not in meta.resolved_config
        dumped = result.to_dict()["metadata"]
        assert dumped["evaluator"] == "duplicates"
        assert dumped["dataeval"]["version"] == meta.dataeval.version

    def test_export_writes_the_result_as_json_with_kind_evaluator(self, tmp_path: Path) -> None:
        target = tmp_path / "dupes.json"
        _go("duplicates").export(target)
        written = json.loads(target.read_text())
        assert written["kind"] == "evaluator"
        assert written["output"]["shape"] == "table"
        assert written["metadata"]["evaluator"] == "duplicates"

    def test_output_extras_are_written_beside_data_where_the_evaluator_names_them(self) -> None:
        coverage = _go("coverage").to_dict()["output"]
        assert set(coverage["extras"]) == set(get_evaluator("coverage").output_extras)
        assert {"uncovered_indices", "coverage_radius", "critical_value_radii", "uncovered_classes"} <= set(
            coverage["extras"]
        )
        assert isinstance(coverage["extras"]["coverage_radius"], float)
        prioritization = _go("prioritization").to_dict()["output"]
        assert prioritization["shape"] == "array"
        assert len(prioritization["extras"]["scores"]) == len(prioritization["data"]) == 40

    def test_an_evaluator_without_extras_writes_no_extras_key(self) -> None:
        assert "extras" not in _go("label-health").to_dict()["output"]
        assert get_evaluator("label-health").output_extras == ()


class TestEvaluatorOnTheCommandLine:
    @staticmethod
    def _project(root: Path) -> Path:
        write_image_folder(root / "imgs", n_per_class=10, n_classes=2)
        plant_duplicate_and_outlier(root / "imgs")
        config = {
            "datasets": [{"name": "ds", "format": "image_folder", "path": "imgs", "infer_labels": True}],
            "sources": [{"name": "main", "dataset": "ds"}],
            "evaluators": [{"name": "dupes", "type": "duplicates"}],
            "tasks": [{"name": "find_dupes", "evaluator": "dupes", "sources": "main"}],
        }
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def test_an_evaluator_task_writes_a_result_with_no_health_and_never_fails_on_warning(self, tmp_path: Path) -> None:
        config = self._project(tmp_path)
        out = tmp_path / "out"
        proc = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(out), "--fail-on-warning")
        assert proc.returncode == 0, _output(proc)
        entry = json.loads((out / "results" / "result.json").read_text())["find_dupes"]
        assert entry["kind"] == "evaluator"
        assert "health" not in entry
        assert entry["metadata"]["evaluator"] == "duplicates"
        groups = [row for row in entry["output"]["rows"] if row["dup_type"] == "exact"]
        assert [row["item_indices"] for row in groups] == [[0, 10]]


class TestEvaluatorTaskValidation:
    def test_a_task_naming_both_a_workflow_and_an_evaluator_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="names both a workflow and an evaluator"):
            TaskConfig.model_validate({"name": "t", "workflow": "w", "evaluator": "e", "sources": ["a"]})

    def test_a_task_naming_neither_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="names neither a workflow nor an evaluator"):
            TaskConfig.model_validate({"name": "t", "sources": ["a"]})

    def test_a_task_naming_an_evaluator_the_config_does_not_define_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="names evaluator 'zzz', which `evaluators:` does not define"):
            pipeline({"a": Images()}, tasks=[{"name": "t", "evaluator": "zzz", "sources": ["a"]}])

    def test_a_task_with_the_wrong_number_of_sources_is_refused_at_load(self) -> None:
        with pytest.raises(ValidationError, match=r"takes exactly two sources, but the task names 1"):
            pipeline(
                {"a": Images()},
                evaluators=[DriftMMDConfig(name="m")],
                tasks=[{"name": "t", "evaluator": "m", "sources": ["a"], "extractor": "flat"}],
                extractor=True,
            )

    @pytest.mark.parametrize("config", [DriftMMDConfig(name="m"), CoverageConfig(name="m")], ids=["drift", "coverage"])
    def test_a_task_without_the_extractor_its_evaluator_needs_is_refused_at_load(self, config: Any) -> None:
        sources = ["a", "b"] if isinstance(config, DriftMMDConfig) else ["a"]
        with pytest.raises(ValidationError, match="needs an extractor to produce embeddings"):
            pipeline(
                {name: Images() for name in sources},
                evaluators=[config],
                tasks=[{"name": "t", "evaluator": "m", "sources": sources}],
            )

    def test_an_unknown_parameter_is_refused_not_ignored(self) -> None:
        with pytest.raises(ValidationError, match="flagz"):
            pipeline(
                {"a": Images()},
                evaluators=[{"name": "d", "type": "duplicates", "flagz": ["hash_basic"]}],
                tasks=[{"name": "t", "evaluator": "d", "sources": ["a"]}],
            )

    def test_a_value_dataeval_refuses_is_refused_when_the_config_loads(self) -> None:
        with pytest.raises(ValidationError, match="DataEval rejected these parameters"):
            OutliersConfig(flags=["pixel"], outlier_threshold="nope")  # type: ignore[arg-type]

    def test_a_valid_parameter_is_kept_and_recorded(self) -> None:
        config = DuplicatesConfig(flags=["hash_basic"])  # type: ignore[list-item]
        result = run(config, Images())
        assert result.success, result.errors
        assert result.metadata.resolved_config["evaluator"]["flags"] == ["hash_basic"]
