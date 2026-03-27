"""Shared mutable config state for the builder.

Both the Textual TUI and Click CLI operate on the same ``ConfigState``
object.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from dataeval_flow._app._model._item import DELETE_SENTINEL, finalize_item
from dataeval_flow._app._model._registry import (
    SECTION_KEYS,
    SECTION_MODELS,
    VARIANT_REGISTRY,
)
from dataeval_flow.config._models import PipelineConfig

__all__ = ["ConfigState"]

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_empty_params(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove empty ``params`` dicts from step lists."""
    cleaned: list[dict[str, Any]] = []
    for item in items:
        item = dict(item)
        if "steps" in item:
            item["steps"] = [{k: v for k, v in s.items() if k != "params" or v} for s in item["steps"]]
        cleaned.append(item)
    return cleaned


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert a Pydantic model or dict to a plain dict."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    return dict(obj)


# ---------------------------------------------------------------------------
# ConfigState
# ---------------------------------------------------------------------------


class ConfigState:
    """Shared mutable config state for the builder.

    Both the Textual TUI and Click CLI operate on this same object.
    The internal representation is a dict of lists (one per section),
    where each item is a plain dict suitable for YAML serialization.
    """

    def __init__(self) -> None:
        self._data: dict[str, list[dict[str, Any]]] = {s: [] for s in SECTION_KEYS}

    # -- Queries -----------------------------------------------------------

    def items(self, section: str) -> list[dict[str, Any]]:
        """Return a shallow copy of items in *section*."""
        return list(self._data[section])

    def names(self, section: str) -> list[str]:
        """Return the names of all items in *section*."""
        return [item.get("name", "") for item in self._data[section]]

    def get(self, section: str, index: int) -> dict[str, Any] | None:
        """Return item at *index*, or ``None``."""
        items = self._data[section]
        if 0 <= index < len(items):
            return items[index]
        return None

    def count(self, section: str) -> int:
        """Return the number of items in *section*."""
        return len(self._data[section])

    def is_empty(self) -> bool:
        """Return ``True`` if all sections are empty."""
        return all(len(v) == 0 for v in self._data.values())

    # -- Mutations ---------------------------------------------------------

    def add(self, section: str, item: dict[str, Any]) -> None:
        """Append *item* to *section*."""
        self._data[section].append(item)

    def update(self, section: str, index: int, item: dict[str, Any]) -> None:
        """Replace the item at *index* in *section*."""
        items = self._data[section]
        if 0 <= index < len(items):
            items[index] = item

    def remove(self, section: str, index: int) -> tuple[str, list[str]]:
        """Remove item at *index*. Returns ``(removed_name, warnings)``."""
        items = self._data[section]
        if not (0 <= index < len(items)):
            return ("", [])
        removed = items.pop(index)
        name = removed.get("name", "")
        warnings = self._scrub_references(section, name) if name else []
        return name, warnings

    # -- Load / Save -------------------------------------------------------

    def load_dict(self, data: dict[str, Any] | BaseModel) -> None:
        """Populate state from a raw dict or a ``PipelineConfig``."""
        if isinstance(data, BaseModel):
            data = data.model_dump(exclude_none=True)
        self._data = {s: [] for s in SECTION_KEYS}
        for section in SECTION_KEYS:
            for item in data.get(section) or []:
                self._data[section].append(_to_dict(item))
        for task in self._data["tasks"]:
            task.setdefault("enabled", True)

    def to_dict(self) -> dict[str, Any]:
        """Export state as a plain dict suitable for YAML serialization."""
        result: dict[str, Any] = {}
        for section in SECTION_KEYS:
            items = self._data[section]
            if items:
                if section in ("preprocessors", "selections"):
                    items = _strip_empty_params(items)
                result[section] = items
        return result

    def load_file(self, path: Path) -> str | None:
        """Load config from *path*. Returns a warning string or ``None``."""
        fallback = False
        if path.is_dir():
            from dataeval_flow.config import load_config_folder

            config = load_config_folder(path)
            self.load_dict(config)
        else:
            try:
                from dataeval_flow.config import load_config

                config = load_config(path)
                self.load_dict(config)
            except (ValueError, TypeError, KeyError, OSError):
                fallback = True
                with open(path, encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                self.load_dict(raw)
        if fallback:
            return "Loaded as raw YAML — some fields may not validate"
        return None

    def save_file(self, path: Path) -> None:
        """Save config to *path* (YAML or JSON based on suffix)."""
        config = self.to_dict()
        if not config:
            return
        if path.suffix not in (".yaml", ".yml", ".json"):
            path = path.with_suffix(".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        else:
            path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8")

    # -- Snapshot / Restore ------------------------------------------------

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """Return a deep copy of internal state."""
        return copy.deepcopy(self._data)

    def restore(self, snap: dict[str, list[dict[str, Any]]]) -> None:
        """Restore from a previous snapshot."""
        self._data = copy.deepcopy(snap)

    # -- Validation --------------------------------------------------------

    def validate_item(self, section: str, data: dict[str, Any]) -> list[str]:
        """Validate a single item dict. Returns a list of error strings."""
        if section in VARIANT_REGISTRY:
            disc_field, variants = VARIANT_REGISTRY[section]
            disc_value = data.get(disc_field)
            model = variants.get(str(disc_value)) if disc_value else None
        else:
            model = SECTION_MODELS.get(section)

        if model is None:
            return [f"Unknown {section} type"]
        try:
            model.model_validate(data)
            return []
        except (ValueError, TypeError) as e:
            return [str(e)]

    def validate_all(self) -> list[str]:
        """Validate the full config via ``PipelineConfig``. Returns errors."""
        try:
            PipelineConfig.model_validate(self.to_dict())
            return []
        except (ValueError, TypeError) as e:
            return [str(e)]

    def to_pipeline_config(self) -> PipelineConfig:
        """Return a validated ``PipelineConfig`` from current state.

        Raises
        ------
        ValueError
            If the current state does not pass validation.
        """
        return PipelineConfig.model_validate(self.to_dict())

    # -- Reference scrubbing -----------------------------------------------

    _TASK_FIELD_MAP: Mapping[str, str] = {
        "sources": "sources",
        "extractors": "extractor",
        "workflows": "workflow",
    }
    _REQUIRED_TASK_KEYS = frozenset({"sources", "workflow"})

    def _scrub_references(self, section: str, removed_name: str) -> list[str]:
        """Remove stale references after a deletion. Returns warnings."""
        warnings: list[str] = []

        task_key = self._TASK_FIELD_MAP.get(section)
        if task_key:
            to_remove: list[int] = []
            for i, task in enumerate(self._data["tasks"]):
                if not self._scrub_task_field(task, task_key, removed_name):
                    to_remove.append(i)
            for i in reversed(to_remove):
                removed_task = self._data["tasks"].pop(i)
                warnings.append(f"Auto-removed task '{removed_task.get('name', '')}'")

        if section == "datasets":
            warnings.extend(self._scrub_comp("sources", "dataset", removed_name, required=True))
        elif section == "selections":
            warnings.extend(self._scrub_comp("sources", "selection", removed_name, required=False))
        elif section == "preprocessors":
            warnings.extend(self._scrub_comp("extractors", "preprocessor", removed_name, required=False))

        return warnings

    def _scrub_comp(self, list_key: str, field_key: str, name: str, *, required: bool) -> list[str]:
        """Scrub composition-layer references."""
        warnings: list[str] = []
        items = self._data[list_key]
        to_remove: list[int] = []
        for i, item in enumerate(items):
            if item.get(field_key) == name:
                if required:
                    to_remove.append(i)
                else:
                    del item[field_key]
        for i in reversed(to_remove):
            removed_item = items.pop(i)
            removed_name = removed_item.get("name", "")
            warnings.append(f"Auto-removed {list_key[:-1]} '{removed_name}'")
            if removed_name:
                warnings.extend(self._scrub_references(list_key, removed_name))
        return warnings

    def _scrub_task_field(self, task: dict[str, Any], key: str, name: str) -> bool:
        """Scrub *name* from a single task field. Returns ``False`` if the task should be removed."""
        val = task.get(key)
        if val is None:
            return True
        if isinstance(val, list):
            filtered = [v for v in val if v != name]
            if not filtered:
                return False
            if len(filtered) != len(val):
                task[key] = filtered[0] if len(filtered) == 1 else filtered
            return True
        if val != name:
            return True
        if key in self._REQUIRED_TASK_KEYS:
            return False
        del task[key]
        return True

    # -- Modal result handling (shared by TUI and CLI) ---------------------

    def apply_modal_result(self, section: str, index: int, result: dict | str | None) -> tuple[str, list[str]] | None:
        """Apply the result of a create/edit/delete operation.

        Returns ``(description, warnings)`` for mutations, or ``None`` for
        cancel (result is ``None``).
        """
        if result is None:
            return None

        if result == DELETE_SENTINEL:
            existing = self.get(section, index)
            item_name = existing.get("name", "?") if existing else "?"
            _, warnings = self.remove(section, index)
            return (f"Delete {section[:-1]} '{item_name}'", warnings)

        if not isinstance(result, dict):
            return None

        result = finalize_item(section, result)
        name = result.get("name", "?")
        if index >= 0:
            self.update(section, index, result)
            return (f"Update {section[:-1]} '{name}'", [])

        self.add(section, result)
        return (f"Add {section[:-1]} '{name}'", [])
