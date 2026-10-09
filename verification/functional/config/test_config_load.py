"""TC-2-1 — configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import load_config, load_source
from dataeval_flow.config import ImageFolderDatasetConfig, PipelineConfig
from dataeval_flow.config._json_schema import registry_twin
from dataeval_flow.config._loader import get_data_dir, resolve_path
from verification.fixtures import write_image_folder
from verification.functional.orchestration.support import (
    EXAMPLE_ENTRY_POINTS,
    QUALITY,
    pipeline_dict,
)
from verification.helpers import REPO_ROOT

pytestmark = pytest.mark.required

SECTIONS = {
    "datasets",
    "deterministic",
    "evaluators",
    "exports",
    "extractors",
    "logging",
    "max_processes",
    "metadata",
    "ontologies",
    "preprocessors",
    "result",
    "seed",
    "sources",
    "stats",
    "tasks",
    "views",
    "workflows",
}


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data) if path.suffix == ".json" else yaml.safe_dump(data))
    return path


class TestLoadFile:
    def test_load_yaml_single_file(self, tmp_path: Path) -> None:
        config = load_config(_write(tmp_path / "params.yaml", pipeline_dict(seed=7)))
        assert isinstance(config, PipelineConfig)
        assert config.seed == 7
        assert config.datasets is not None
        assert config.datasets[0].name == "ds"
        assert [source.name for source in config.sources or ()] == ["main"]

    def test_load_json_single_file(self, tmp_path: Path) -> None:
        config = load_config(_write(tmp_path / "params.json", pipeline_dict(seed=7)))
        assert isinstance(config, PipelineConfig)
        assert config.seed == 7
        assert [extractor.name for extractor in config.extractors or ()] == ["flat"]

    def test_yaml_and_json_of_one_config_load_equal(self, tmp_path: Path) -> None:
        data = pipeline_dict(workflows=[QUALITY], tasks=[{"name": "t", "workflow": "q", "sources": "main"}])
        assert load_config(_write(tmp_path / "a.yaml", data)) == load_config(_write(tmp_path / "a.json", data))

    def test_the_path_may_be_a_string(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "params.yml", pipeline_dict())
        assert load_config(str(path)) == load_config(path)

    def test_an_empty_file_is_an_empty_pipeline(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert load_config(path) == PipelineConfig()

    def test_a_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nope.yaml"):
            load_config(tmp_path / "nope.yaml")

    def test_a_config_survives_a_dump_and_reload(self, tmp_path: Path) -> None:
        """Every entry, including a workflow type's own fields, is written back as it was read."""
        config = load_config(
            _write(
                tmp_path / "params.yaml",
                pipeline_dict(
                    seed=3,
                    workflows=[QUALITY],
                    evaluators=[{"name": "dup", "type": "duplicates"}],
                    tasks=[
                        {"name": "w", "workflow": "q", "sources": "main"},
                        {"name": "e", "evaluator": "dup", "sources": "main"},
                    ],
                ),
            )
        )
        reloaded = PipelineConfig.model_validate_json(config.model_dump_json())
        assert reloaded == config
        assert json.loads(config.model_dump_json())["workflows"][0]["outliers"]["flags"] == ["pixel"]


class TestLoadFolder:
    def test_load_config_folder_merges_files(self, tmp_path: Path) -> None:
        """Mappings merge, lists extend and a later file's scalar replaces an earlier one's."""
        _write(
            tmp_path / "00-base.yaml",
            {
                "seed": 1,
                "logging": {"app_level": "INFO"},
                "datasets": [{"name": "a", "format": "image_folder", "path": "a"}],
            },
        )
        _write(
            tmp_path / "10-more.json",
            {
                "seed": 2,
                "logging": {"lib_level": "ERROR"},
                "datasets": [{"name": "b", "format": "image_folder", "path": "b"}],
            },
        )
        config = load_config(tmp_path)
        assert config.seed == 2
        assert config.logging is not None
        assert (config.logging.app_level, config.logging.lib_level) == ("INFO", "ERROR")
        assert [dataset.name for dataset in config.datasets or ()] == ["a", "b"]

    def test_files_merge_in_name_order(self, tmp_path: Path) -> None:
        _write(
            tmp_path / "b.yaml", {"seed": 2, "datasets": [{"name": "second", "format": "image_folder", "path": "x"}]}
        )
        _write(tmp_path / "a.yml", {"seed": 1, "datasets": [{"name": "first", "format": "image_folder", "path": "x"}]})
        config = load_config(tmp_path)
        assert [dataset.name for dataset in config.datasets or ()] == ["first", "second"]
        assert config.seed == 2

    def test_a_config_split_across_files_validates_once_merged(self, tmp_path: Path) -> None:
        """A task may name a workflow, source or extractor that another file defines."""
        full = pipeline_dict(workflows=[QUALITY], tasks=[{"name": "t", "workflow": "q", "sources": "main"}])
        for index, (key, value) in enumerate(full.items()):
            _write(tmp_path / f"{index:02d}-{key}.yaml", {key: value})
        assert load_config(tmp_path) == load_config(_write(tmp_path.parent / f"{tmp_path.name}.yaml", full))

    def test_files_that_are_not_pipeline_configs_are_skipped(self, tmp_path: Path) -> None:
        _write(tmp_path / "pipeline.yaml", {"seed": 5})
        (tmp_path / "compose.yaml").write_text("services:\n  web: {image: x}\n")
        (tmp_path / "broken.yaml").write_text("a: [unclosed\n")
        (tmp_path / "notes.md").write_text("not a config")
        assert load_config(tmp_path).seed == 5

    def test_a_folder_without_a_pipeline_file_raises_file_not_found(self, tmp_path: Path) -> None:
        (tmp_path / "compose.yaml").write_text("services: {}\n")
        with pytest.raises(FileNotFoundError, match="No valid pipeline config"):
            load_config(tmp_path)

    def test_a_misspelled_section_in_one_file_names_the_file(self, tmp_path: Path) -> None:
        _write(tmp_path / "00-ok.yaml", {"seed": 1})
        _write(tmp_path / "20-bad.yaml", {"datasets": [], "sourcez": []})
        with pytest.raises(ValueError, match=r"20-bad\.yaml.*'sourcez'.*did you mean 'sources'"):
            load_config(tmp_path)


class TestInvalidConfig:
    def test_invalid_config_raises_validation_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "bad.yaml", {"sources": "this is not a list"})
        with pytest.raises(ValidationError) as caught:
            load_config(path)
        assert caught.value.errors()[0]["loc"] == ("sources",)

    def test_the_error_points_to_the_offending_field(self, tmp_path: Path) -> None:
        data = pipeline_dict(datasets=[{"name": "ds", "format": "image_folder", "path": "imgs", "bogus": 1}])
        with pytest.raises(ValidationError) as caught:
            load_config(_write(tmp_path / "bad.yaml", data))
        located = [error for error in caught.value.errors() if error["loc"][-1] == "bogus"]
        assert located
        assert located[0]["loc"][:2] == ("datasets", 0)
        assert located[0]["type"] == "extra_forbidden"
        assert "bogus" in str(caught.value)

    def test_an_error_in_a_workflow_entry_is_located_by_its_index(self, tmp_path: Path) -> None:
        workflows = [
            QUALITY,
            {"name": "q2", "type": "quality", "outliers": {"flags": ["nonsense"], "outlier_threshold": "zscore"}},
        ]
        with pytest.raises(ValidationError) as caught:
            load_config(_write(tmp_path / "bad.yaml", pipeline_dict(workflows=workflows)))
        assert caught.value.errors()[0]["loc"] == ("workflows", 1, "outliers", "flags", 0)

    def test_a_misspelled_top_level_key_names_the_section_it_resembles(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"Unknown top-level key 'dataset'.*did you mean 'datasets'"):
            load_config(_write(tmp_path / "bad.yaml", {"dataset": []}))

    def test_a_workflow_type_that_is_not_installed_lists_the_installed_ones(self, tmp_path: Path) -> None:
        data = pipeline_dict(workflows=[{"name": "x", "type": "no-such-workflow"}])
        with pytest.raises(ValidationError, match=r"no-such-workflow.*quality"):
            load_config(_write(tmp_path / "bad.yaml", data))

    @pytest.mark.parametrize("section", ["datasets", "sources", "extractors", "workflows"])
    def test_a_name_used_twice_in_one_section_is_refused(self, tmp_path: Path, section: str) -> None:
        data = pipeline_dict(workflows=[QUALITY])
        data[section] = [*data[section], data[section][0]]
        with pytest.raises(ValidationError, match=f"Duplicate name '.+' in {section}"):
            load_config(_write(tmp_path / "bad.yaml", data))


class TestJsonSchema:
    def test_the_schema_describes_pipeline_config(self) -> None:
        schema = PipelineConfig.model_json_schema()
        assert schema["title"] == "PipelineConfig"
        assert set(schema["properties"]) == SECTIONS
        assert schema["additionalProperties"] is False

    def test_the_schema_has_a_branch_per_built_in_workflow_evaluator_and_extractor(self) -> None:
        from dataeval_flow.config.extractors import list_extractors
        from dataeval_flow.evaluators import list_evaluators
        from dataeval_flow.workflows import list_workflows

        definitions = PipelineConfig.model_json_schema()["$defs"]
        for kind, classes, key in (
            ("workflow", list_workflows(), "type"),
            ("evaluator", list_evaluators(), "type"),
            ("extractor", list_extractors(), "model"),
        ):
            consts = {
                definition["properties"][key]["const"]
                for definition in definitions.values()
                if key in definition.get("properties", {}) and "const" in definition["properties"][key]
            }
            assert {cls.name for cls in classes} <= consts, kind

    def test_an_installed_plugin_has_a_branch_in_the_schema(self, example_plugin: dict[str, Any]) -> None:
        definitions = PipelineConfig.model_json_schema()["$defs"]
        assert definitions["CountConfig"]["properties"]["type"]["const"] == "example.count"
        assert definitions["MeanConfig"]["properties"]["model"]["const"] == "example.mean"
        assert set(EXAMPLE_ENTRY_POINTS) >= {"dataeval_flow.workflows", "dataeval_flow.extractors"}

    def test_the_checked_in_schema_matches_the_built_in_types(self) -> None:
        """`config/params.schema.json`, which editors validate a config against, is current."""
        checked_in = json.loads((REPO_ROOT / "config" / "params.schema.json").read_text(encoding="utf-8"))
        assert checked_in == registry_twin(plugins=False).model_json_schema()


class TestDataRoot:
    def test_relative_dataset_paths_resolve_against_data_root(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_image_folder(root / "imgs", n_per_class=3, n_classes=2)
        config = PipelineConfig.model_validate(pipeline_dict())
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        assert len(load_source(config, "main", data_dir=root)) == 6
        with pytest.raises(FileNotFoundError):
            load_source(config, "main", data_dir=elsewhere)

    def test_a_dataset_under_the_data_folder_of_the_root_is_found(self, tmp_path: Path) -> None:
        """A relative path not found directly under the root is looked up in the root's `data` folder."""
        write_image_folder(tmp_path / "data" / "imgs", n_per_class=2, n_classes=2)
        config = PipelineConfig.model_validate(pipeline_dict())
        assert len(load_source(config, "main", data_dir=tmp_path)) == 4

    def test_the_data_root_defaults_to_the_environment_then_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("DATAEVAL_DATA", raising=False)
        assert get_data_dir() == Path(".")
        monkeypatch.setenv("DATAEVAL_DATA", str(tmp_path / "mounted"))
        assert get_data_dir() == tmp_path / "mounted"
        assert get_data_dir(tmp_path / "explicit") == tmp_path / "explicit"
        assert resolve_path("imgs") == tmp_path / "mounted" / "imgs"

    def test_dataset_paths_must_stay_relative_to_the_data_root(self) -> None:
        for bad in ("/abs/imgs", "../outside", "a/../../outside"):
            with pytest.raises(ValidationError, match="path"):
                ImageFolderDatasetConfig(name="ds", path=bad)
        assert ImageFolderDatasetConfig(name="ds", path="sub/../imgs").path == "sub/../imgs"

    def test_a_config_file_with_an_absolute_dataset_path_is_refused(self, tmp_path: Path) -> None:
        data = pipeline_dict(datasets=[{"name": "ds", "format": "image_folder", "path": "/abs/imgs"}])
        with pytest.raises(ValidationError) as caught:
            load_config(_write(tmp_path / "bad.yaml", data))
        assert any(error["loc"][-1] == "path" for error in caught.value.errors())

    def test_a_model_path_must_stay_relative_too(self) -> None:
        from dataeval_flow.config.extractors import OnnxExtractorConfig

        with pytest.raises(ValidationError, match="model_path"):
            OnnxExtractorConfig(model_path="/abs/model.onnx")
        with pytest.raises(ValidationError, match="model_path"):
            OnnxExtractorConfig(model_path="../model.onnx")
