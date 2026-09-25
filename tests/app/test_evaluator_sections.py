"""The TUI builds evaluators and evaluator tasks, and keeps them through a load and save."""

from typing import Any

from textual.widgets import Input, Select

from dataeval_flow._app._model._introspect import FieldKind
from dataeval_flow._app._model._item import finalize_item
from dataeval_flow._app._model._registry import SECTION_KEYS, get_fields, get_variant_choices
from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._app._screens import SectionModal
from dataeval_flow._app._screens._base import _select_value
from dataeval_flow._app._viewmodel._rendering import _item_to_yaml_snippet, snippet_task_with_execution

from .conftest import _MinimalApp


def _config() -> dict[str, Any]:
    return {
        "datasets": [{"name": "ds", "format": "huggingface", "path": "./d", "task": "image_classification"}],
        "sources": [{"name": "a", "dataset": "ds"}],
        "evaluators": [{"name": "dupes", "type": "quality.duplicates", "flags": ["hash_basic"]}],
        "tasks": [{"name": "t", "evaluator": "dupes", "sources": "a", "enabled": True}],
    }


def test_evaluators_is_a_section():
    assert "evaluators" in SECTION_KEYS
    assert set(get_variant_choices("evaluators") or []) >= {"quality.duplicates", "quality.outliers"}


def test_a_load_and_save_keeps_evaluators_and_their_tasks():
    """Review Focus 5: a TUI round trip must not drop the evaluators."""
    state = ConfigState()
    state.load_dict(_config())
    saved = state.to_dict()
    assert saved["evaluators"] == _config()["evaluators"]
    assert saved["tasks"] == _config()["tasks"]
    assert state.validate_all() == []


def test_the_task_form_offers_the_defined_evaluators():
    state = ConfigState()
    state.load_dict(_config())
    fields = {field.name: field for field in get_fields("tasks", None, state)}
    assert fields["evaluator"].kind == FieldKind.SELECT
    assert fields["evaluator"].choices == ["dupes"]


def test_removing_an_evaluator_removes_its_tasks():
    state = ConfigState()
    state.load_dict(_config())
    _, warnings = state.remove("evaluators", 0)
    assert state.items("tasks") == []
    assert warnings == ["Auto-removed task 't'"]


def test_an_empty_target_is_dropped_from_the_saved_task():
    item = finalize_item("tasks", {"name": "t", "workflow": None, "evaluator": "dupes", "sources": "a"})
    assert "workflow" not in item
    assert item["evaluator"] == "dupes"


def test_snippets_name_the_evaluator():
    task = _config()["tasks"][0]
    assert "evaluator: dupes" in _item_to_yaml_snippet("tasks", task)
    assert "dupes" in snippet_task_with_execution(task)
    assert "quality.duplicates" in _item_to_yaml_snippet("evaluators", _config()["evaluators"][0])


class TestTriStateBoolFields:
    """Item 1: a `bool | None = None` evaluator field renders as a three-state selection: unset, true, false."""

    async def test_an_explicit_false_survives_editing_another_field(self) -> None:
        app = _MinimalApp()
        state = ConfigState()
        state.load_dict(_config())
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(
                section="evaluators",
                existing={
                    "name": "dupes",
                    "type": "quality.duplicates",
                    "flags": ["hash_basic"],
                    "merge_near_duplicates": False,
                },
                state=state,
            )
            await app.push_screen(modal)
            await pilot.pause()
            # on_mount sets #md-disc, which queues a Select.Changed that rebuilds the
            # fields once the message pump runs. Repopulate so the assertions see the
            # settled form.
            modal._populate_fields()

            # The loaded `false` renders as the explicit "false" selection.
            select = modal.query_one(f"#{modal._wid('merge_near_duplicates')}", Select)
            assert select.value == "false"

            # Edit an unrelated field and save without touching this one.
            modal.query_one("#md-name", Input).value = "dupes_renamed"
            result = modal._collect_raw()

            assert result is not None
            assert result["merge_near_duplicates"] is False

    async def test_setting_false_from_the_unset_state(self) -> None:
        app = _MinimalApp()
        state = ConfigState()
        state.load_dict(_config())
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(section="evaluators", state=state)
            await app.push_screen(modal)
            await pilot.pause()

            modal.query_one("#md-name", Input).value = "dupes2"
            modal.query_one("#md-disc", Select).value = "quality.duplicates"
            await pilot.pause()

            select = modal.query_one(f"#{modal._wid('merge_near_duplicates')}", Select)
            assert _select_value(select) == ""  # starts unset

            select.value = "false"
            result = modal._collect_raw()

            assert result is not None
            assert result["merge_near_duplicates"] is False

    async def test_an_explicit_null_stays_unset(self) -> None:
        app = _MinimalApp()
        state = ConfigState()
        state.load_dict(_config())
        async with app.run_test(size=(100, 40)) as pilot:
            modal = SectionModal(
                section="evaluators",
                existing={
                    "name": "dupes",
                    "type": "quality.duplicates",
                    "flags": ["hash_basic"],
                    "merge_near_duplicates": None,
                },
                state=state,
            )
            await app.push_screen(modal)
            await pilot.pause()
            modal._populate_fields()  # see the explicit-false test for why

            # A loaded null is DataEval's default, so it must not save as `false`.
            select = modal.query_one(f"#{modal._wid('merge_near_duplicates')}", Select)
            assert _select_value(select) == ""
            result = modal._collect_raw()

            assert result is not None
            assert "merge_near_duplicates" not in result
