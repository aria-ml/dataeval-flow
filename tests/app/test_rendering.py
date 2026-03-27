"""Unit tests for _app._rendering — pure snippet renderers (no Textual dependency)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from dataeval_flow._app._model._execution import TaskExecution
from dataeval_flow._app._viewmodel._rendering import (
    _SNIPPET_RENDERERS,
    _item_to_yaml_snippet,
    _snippet_dataset,
    _snippet_extractor,
    _snippet_source,
    _snippet_steps,
    _snippet_task,
    _snippet_workflow,
    snippet_config_item,
    snippet_task_with_execution,
)

# ---------------------------------------------------------------------------
# _snippet_steps
# ---------------------------------------------------------------------------


class TestSnippetSteps:
    def test_with_params(self) -> None:
        item: dict[str, Any] = {
            "name": "pre1",
            "steps": [{"step": "Resize", "params": {"size": 256}}],
        }
        result = _snippet_steps(item, "step")
        assert "[bold]pre1[/bold]" in result
        assert "Resize(size=256)" in result

    def test_without_params(self) -> None:
        item: dict[str, Any] = {
            "name": "pre1",
            "steps": [{"step": "ToTensor"}],
        }
        result = _snippet_steps(item, "step")
        assert "ToTensor" in result
        assert "(" not in result

    def test_missing_name(self) -> None:
        item: dict[str, Any] = {"steps": [{"step": "Resize"}]}
        result = _snippet_steps(item, "step")
        assert "[bold]?[/bold]" in result

    def test_empty_steps(self) -> None:
        item: dict[str, Any] = {"name": "pre1", "steps": []}
        result = _snippet_steps(item, "step")
        assert result == "[bold]pre1[/bold]"

    def test_missing_steps_key(self) -> None:
        item: dict[str, Any] = {"name": "pre1"}
        result = _snippet_steps(item, "step")
        assert result == "[bold]pre1[/bold]"

    def test_type_key(self) -> None:
        item: dict[str, Any] = {
            "name": "sel1",
            "steps": [{"type": "Limit", "params": {"size": 100}}],
        }
        result = _snippet_steps(item, "type")
        assert "Limit(size=100)" in result

    def test_multiple_steps(self) -> None:
        item: dict[str, Any] = {
            "name": "pre1",
            "steps": [
                {"step": "Resize", "params": {"size": 256}},
                {"step": "ToTensor"},
            ],
        }
        result = _snippet_steps(item, "step")
        lines = result.split("\n")
        assert len(lines) == 3

    def test_multiple_params(self) -> None:
        item: dict[str, Any] = {
            "name": "pre1",
            "steps": [{"step": "Resize", "params": {"width": 256, "height": 128}}],
        }
        result = _snippet_steps(item, "step")
        assert "width=256" in result
        assert "height=128" in result

    def test_empty_params_dict(self) -> None:
        item: dict[str, Any] = {
            "name": "pre1",
            "steps": [{"step": "Resize", "params": {}}],
        }
        result = _snippet_steps(item, "step")
        assert "(" not in result

    def test_missing_step_key(self) -> None:
        item: dict[str, Any] = {
            "name": "pre1",
            "steps": [{"params": {"size": 256}}],
        }
        result = _snippet_steps(item, "step")
        assert "?" in result


# ---------------------------------------------------------------------------
# _snippet_dataset
# ---------------------------------------------------------------------------


class TestSnippetDataset:
    def test_basic(self) -> None:
        item: dict[str, Any] = {"name": "ds1", "format": "huggingface", "path": "data"}
        result = _snippet_dataset(item)
        assert "[bold]ds1[/bold]" in result
        assert "format: huggingface" in result
        assert "path: data" in result

    def test_with_split(self) -> None:
        item: dict[str, Any] = {"name": "ds1", "format": "coco", "path": "data", "split": "train"}
        result = _snippet_dataset(item)
        assert "split: train" in result

    def test_without_split(self) -> None:
        item: dict[str, Any] = {"name": "ds1", "format": "coco", "path": "data"}
        result = _snippet_dataset(item)
        assert "split" not in result

    def test_missing_fields(self) -> None:
        item: dict[str, Any] = {}
        result = _snippet_dataset(item)
        assert "[bold]?[/bold]" in result
        assert "format: ?" in result
        assert "path: ?" in result

    def test_empty_split_not_shown(self) -> None:
        item: dict[str, Any] = {"name": "ds1", "format": "hf", "path": "data", "split": ""}
        result = _snippet_dataset(item)
        assert "split" not in result


# ---------------------------------------------------------------------------
# _snippet_source
# ---------------------------------------------------------------------------


class TestSnippetSource:
    def test_basic(self) -> None:
        item: dict[str, Any] = {"name": "src1", "dataset": "ds1"}
        result = _snippet_source(item)
        assert "[bold]src1[/bold]" in result
        assert "dataset: ds1" in result

    def test_with_selection(self) -> None:
        item: dict[str, Any] = {"name": "src1", "dataset": "ds1", "selection": "sel1"}
        result = _snippet_source(item)
        assert "selection: sel1" in result

    def test_without_selection(self) -> None:
        item: dict[str, Any] = {"name": "src1", "dataset": "ds1"}
        result = _snippet_source(item)
        assert "selection" not in result

    def test_missing_fields(self) -> None:
        item: dict[str, Any] = {}
        result = _snippet_source(item)
        assert "[bold]?[/bold]" in result
        assert "dataset: ?" in result

    def test_empty_selection_not_shown(self) -> None:
        item: dict[str, Any] = {"name": "src1", "dataset": "ds1", "selection": ""}
        result = _snippet_source(item)
        assert "selection" not in result


# ---------------------------------------------------------------------------
# _snippet_extractor
# ---------------------------------------------------------------------------


class TestSnippetExtractor:
    def test_basic(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "onnx"}
        result = _snippet_extractor(item)
        assert "[bold]ext1[/bold]" in result
        assert "[dim]onnx[/dim]" in result

    def test_with_model_path(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "onnx", "model_path": "models/model.onnx"}
        result = _snippet_extractor(item)
        assert "path: models/model.onnx" in result

    def test_bovw_with_vocab_size(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "bovw", "vocab_size": 500}
        result = _snippet_extractor(item)
        assert "vocab_size: 500" in result

    def test_non_bovw_ignores_vocab_size(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "onnx", "vocab_size": 500}
        result = _snippet_extractor(item)
        assert "vocab_size" not in result

    def test_with_preprocessor(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "onnx", "preprocessor": "pre1"}
        result = _snippet_extractor(item)
        assert "preprocessor: pre1" in result

    def test_with_batch_size(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "onnx", "batch_size": 32}
        result = _snippet_extractor(item)
        assert "batch_size: 32" in result

    def test_missing_fields(self) -> None:
        item: dict[str, Any] = {}
        result = _snippet_extractor(item)
        assert "[bold]?[/bold]" in result
        assert "[dim]?[/dim]" in result

    def test_empty_optional_fields_not_shown(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "onnx", "preprocessor": "", "batch_size": ""}
        result = _snippet_extractor(item)
        assert "preprocessor" not in result
        assert "batch_size" not in result

    def test_no_model_path(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "onnx"}
        result = _snippet_extractor(item)
        assert "path:" not in result

    def test_bovw_without_vocab_size(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "bovw"}
        result = _snippet_extractor(item)
        assert "vocab_size" not in result


# ---------------------------------------------------------------------------
# _snippet_workflow
# ---------------------------------------------------------------------------


class TestSnippetWorkflow:
    def test_basic(self) -> None:
        item: dict[str, Any] = {"name": "wf1", "type": "data-cleaning"}
        result = _snippet_workflow(item)
        assert "[bold]wf1[/bold]" in result
        assert "[dim]data-cleaning[/dim]" in result

    def test_with_extras(self) -> None:
        item: dict[str, Any] = {
            "name": "wf1",
            "type": "data-cleaning",
            "outlier_method": "adaptive",
        }
        result = _snippet_workflow(item)
        assert "outlier_method: adaptive" in result

    def test_empty_extras_not_shown(self) -> None:
        item: dict[str, Any] = {"name": "wf1", "type": "data-cleaning", "extra": ""}
        result = _snippet_workflow(item)
        lines = result.split("\n")
        assert len(lines) == 1

    def test_missing_fields(self) -> None:
        item: dict[str, Any] = {}
        result = _snippet_workflow(item)
        assert "[bold]?[/bold]" in result


# ---------------------------------------------------------------------------
# _snippet_task
# ---------------------------------------------------------------------------


class TestSnippetTask:
    def test_enabled(self) -> None:
        item: dict[str, Any] = {
            "name": "t1",
            "workflow": "wf1",
            "sources": "src1",
            "enabled": True,
        }
        result = _snippet_task(item)
        assert "\u2713" in result
        assert "[bold]t1[/bold]" in result
        assert "workflow: wf1" in result
        assert "sources: src1" in result

    def test_disabled(self) -> None:
        item: dict[str, Any] = {
            "name": "t1",
            "workflow": "wf1",
            "sources": "src1",
            "enabled": False,
        }
        result = _snippet_task(item)
        assert "\u2717" in result
        assert "strikethrough" in result

    def test_default_enabled(self) -> None:
        item: dict[str, Any] = {"name": "t1", "workflow": "wf1", "sources": "src1"}
        result = _snippet_task(item)
        assert "\u2713" in result

    def test_sources_list(self) -> None:
        item: dict[str, Any] = {
            "name": "t1",
            "workflow": "wf1",
            "sources": ["src1", "src2"],
            "enabled": True,
        }
        result = _snippet_task(item)
        assert "src1, src2" in result

    def test_with_extractor(self) -> None:
        item: dict[str, Any] = {
            "name": "t1",
            "workflow": "wf1",
            "sources": "src1",
            "extractor": "ext1",
            "enabled": True,
        }
        result = _snippet_task(item)
        assert "extractor: ext1" in result

    def test_without_extractor(self) -> None:
        item: dict[str, Any] = {
            "name": "t1",
            "workflow": "wf1",
            "sources": "src1",
            "enabled": True,
        }
        result = _snippet_task(item)
        assert "extractor" not in result

    def test_disabled_with_extractor(self) -> None:
        item: dict[str, Any] = {
            "name": "t1",
            "workflow": "wf1",
            "sources": "src1",
            "extractor": "ext1",
            "enabled": False,
        }
        result = _snippet_task(item)
        assert "extractor: ext1" in result
        assert "strikethrough" in result

    def test_missing_fields(self) -> None:
        item: dict[str, Any] = {"enabled": True}
        result = _snippet_task(item)
        assert "[bold]?[/bold]" in result
        assert "workflow: ?" in result


# ---------------------------------------------------------------------------
# _item_to_yaml_snippet + _SNIPPET_RENDERERS
# ---------------------------------------------------------------------------


class TestItemToYamlSnippet:
    def test_known_category_datasets(self) -> None:
        item: dict[str, Any] = {"name": "ds1", "format": "hf", "path": "data"}
        result = _item_to_yaml_snippet("datasets", item)
        assert "ds1" in result

    def test_known_category_tasks(self) -> None:
        item: dict[str, Any] = {"name": "t1", "workflow": "wf1", "sources": "src1", "enabled": True}
        result = _item_to_yaml_snippet("tasks", item)
        assert "t1" in result

    def test_known_category_preprocessors(self) -> None:
        item: dict[str, Any] = {"name": "pre1", "steps": [{"step": "Resize", "params": {"size": 256}}]}
        result = _item_to_yaml_snippet("preprocessors", item)
        assert "pre1" in result
        assert "Resize" in result

    def test_known_category_selections(self) -> None:
        item: dict[str, Any] = {"name": "sel1", "steps": [{"type": "Limit"}]}
        result = _item_to_yaml_snippet("selections", item)
        assert "sel1" in result

    def test_known_category_sources(self) -> None:
        item: dict[str, Any] = {"name": "src1", "dataset": "ds1"}
        result = _item_to_yaml_snippet("sources", item)
        assert "src1" in result

    def test_known_category_extractors(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "onnx"}
        result = _item_to_yaml_snippet("extractors", item)
        assert "ext1" in result

    def test_known_category_workflows(self) -> None:
        item: dict[str, Any] = {"name": "wf1", "type": "data-cleaning"}
        result = _item_to_yaml_snippet("workflows", item)
        assert "wf1" in result

    def test_unknown_category_falls_back_to_yaml(self) -> None:
        item: dict[str, Any] = {"key": "value"}
        result = _item_to_yaml_snippet("unknown_section", item)
        assert "key" in result
        assert "value" in result

    def test_snippet_renderers_keys(self) -> None:
        expected_keys = {"datasets", "preprocessors", "selections", "sources", "extractors", "workflows", "tasks"}
        assert set(_SNIPPET_RENDERERS.keys()) == expected_keys


# ---------------------------------------------------------------------------
# snippet_task_with_execution
# ---------------------------------------------------------------------------


class TestSnippetTaskWithExecution:
    """Tests for snippet_task_with_execution."""

    def _make_task(
        self,
        name: str = "t1",
        workflow: str = "wf1",
        sources: str | list[str] = "src1",
        extractor: str = "",
        enabled: bool = True,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {"name": name, "workflow": workflow, "sources": sources, "enabled": enabled}
        if extractor:
            item["extractor"] = extractor
        return item

    # -- execution=None (defaults to idle indicator) --

    def test_no_execution_shows_idle(self) -> None:
        result = snippet_task_with_execution(self._make_task(), execution=None)
        # idle indicator is the dim bullet ●
        assert "\u25cf" in result

    def test_no_execution_contains_name_and_workflow(self) -> None:
        result = snippet_task_with_execution(self._make_task(name="my_task", workflow="cleaning"))
        assert "[bold]my_task[/bold]" in result
        assert "cleaning" in result

    # -- idle status --

    def test_idle_execution(self) -> None:
        exe = TaskExecution(task_name="t1", status="idle")
        result = snippet_task_with_execution(self._make_task(), execution=exe)
        assert "\u25cf" in result

    # -- running status --

    def test_running_execution(self) -> None:
        exe = TaskExecution(
            task_name="t1",
            status="running",
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        result = snippet_task_with_execution(self._make_task(), execution=exe)
        assert "running..." in result
        assert "\u25d0" in result

    # -- completed status with elapsed time --

    def test_completed_execution_with_elapsed(self) -> None:
        exe = TaskExecution(
            task_name="t1",
            status="completed",
            started_at=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            finished_at=datetime(2025, 1, 1, 0, 0, 12, 500000, tzinfo=timezone.utc),
        )
        result = snippet_task_with_execution(self._make_task(), execution=exe)
        assert "\u2713" in result
        assert "12.5s" in result

    def test_completed_execution_without_elapsed(self) -> None:
        """Completed but started_at is None so elapsed_s is None."""
        exe = TaskExecution(
            task_name="t1",
            status="completed",
            started_at=None,
            finished_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        result = snippet_task_with_execution(self._make_task(), execution=exe)
        assert "\u2713" in result
        # No elapsed time suffix when elapsed_s is None
        assert "s[/dim]" not in result

    # -- failed status --

    def test_failed_execution(self) -> None:
        exe = TaskExecution(
            task_name="t1",
            status="failed",
            error="something broke",
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            finished_at=datetime(2025, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
        )
        result = snippet_task_with_execution(self._make_task(), execution=exe)
        assert "\u2717" in result
        assert "failed" in result

    # -- disabled task --

    def test_disabled_task_with_execution(self) -> None:
        exe = TaskExecution(task_name="t1", status="idle")
        result = snippet_task_with_execution(self._make_task(enabled=False), execution=exe)
        assert "strikethrough" in result
        assert "[dim]" in result

    def test_disabled_task_no_execution(self) -> None:
        result = snippet_task_with_execution(self._make_task(enabled=False))
        assert "strikethrough" in result

    # -- sources as list --

    def test_sources_list_joined(self) -> None:
        result = snippet_task_with_execution(self._make_task(sources=["src1", "src2"]))
        assert "src1, src2" in result

    # -- extractor shown in parentheses --

    def test_extractor_shown(self) -> None:
        result = snippet_task_with_execution(self._make_task(extractor="ext1"))
        assert "(ext1)" in result

    def test_no_extractor_not_shown(self) -> None:
        result = snippet_task_with_execution(self._make_task())
        assert "extractor" not in result.lower() or "()" not in result


# ---------------------------------------------------------------------------
# snippet_config_item
# ---------------------------------------------------------------------------


class TestSnippetConfigItem:
    """Tests for snippet_config_item."""

    # -- datasets --

    def test_datasets(self) -> None:
        item: dict[str, Any] = {"name": "ds1", "format": "huggingface"}
        result = snippet_config_item("datasets", item)
        assert "[bold]ds1[/bold]" in result
        assert "[dim]huggingface[/dim]" in result

    def test_datasets_missing_format(self) -> None:
        item: dict[str, Any] = {"name": "ds1"}
        result = snippet_config_item("datasets", item)
        assert "[bold]ds1[/bold]" in result
        assert "[dim][/dim]" in result

    # -- extractors --

    def test_extractors(self) -> None:
        item: dict[str, Any] = {"name": "ext1", "model": "onnx"}
        result = snippet_config_item("extractors", item)
        assert "[bold]ext1[/bold]" in result
        assert "[dim]onnx[/dim]" in result

    def test_extractors_missing_model(self) -> None:
        item: dict[str, Any] = {"name": "ext1"}
        result = snippet_config_item("extractors", item)
        assert "[bold]ext1[/bold]" in result

    # -- workflows --

    def test_workflows(self) -> None:
        item: dict[str, Any] = {"name": "wf1", "type": "data-cleaning"}
        result = snippet_config_item("workflows", item)
        assert "[bold]wf1[/bold]" in result
        assert "[dim]data-cleaning[/dim]" in result

    def test_workflows_missing_type(self) -> None:
        item: dict[str, Any] = {"name": "wf1"}
        result = snippet_config_item("workflows", item)
        assert "[bold]wf1[/bold]" in result

    # -- sources --

    def test_sources(self) -> None:
        item: dict[str, Any] = {"name": "src1", "dataset": "ds1"}
        result = snippet_config_item("sources", item)
        assert "[bold]src1[/bold]" in result
        assert "[dim](ds1)[/dim]" in result

    def test_sources_missing_dataset(self) -> None:
        item: dict[str, Any] = {"name": "src1"}
        result = snippet_config_item("sources", item)
        assert "[bold]src1[/bold]" in result
        assert "[dim]()[/dim]" in result

    # -- preprocessors --

    def test_preprocessors(self) -> None:
        item: dict[str, Any] = {"name": "pre1", "steps": [{"step": "Resize"}, {"step": "ToTensor"}]}
        result = snippet_config_item("preprocessors", item)
        assert "[bold]pre1[/bold]" in result
        assert "2 steps" in result

    def test_preprocessors_single_step(self) -> None:
        item: dict[str, Any] = {"name": "pre1", "steps": [{"step": "Resize"}]}
        result = snippet_config_item("preprocessors", item)
        assert "1 step" in result
        assert "1 steps" not in result

    def test_preprocessors_no_steps(self) -> None:
        item: dict[str, Any] = {"name": "pre1", "steps": []}
        result = snippet_config_item("preprocessors", item)
        assert "0 steps" in result

    # -- selections --

    def test_selections(self) -> None:
        item: dict[str, Any] = {"name": "sel1", "steps": [{"type": "Limit"}, {"type": "Shuffle"}]}
        result = snippet_config_item("selections", item)
        assert "[bold]sel1[/bold]" in result
        assert "2 steps" in result

    def test_selections_single_step(self) -> None:
        item: dict[str, Any] = {"name": "sel1", "steps": [{"type": "Limit"}]}
        result = snippet_config_item("selections", item)
        assert "1 step" in result
        assert "1 steps" not in result

    # -- unknown category --

    def test_unknown_category(self) -> None:
        item: dict[str, Any] = {"name": "something"}
        result = snippet_config_item("foobar", item)
        assert "[bold]something[/bold]" in result
        # Should only have the bold name, no extra dim annotation
        assert "[dim]" not in result

    def test_unknown_category_missing_name(self) -> None:
        item: dict[str, Any] = {"other": "data"}
        result = snippet_config_item("foobar", item)
        assert "[bold]?[/bold]" in result
