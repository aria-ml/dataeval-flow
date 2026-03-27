"""ViewModel for the SectionModal.

Manages field descriptors, step-builder state, union-list state,
and item collection/validation.  No UI dependencies.
"""

from __future__ import annotations

from typing import Any

from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind, introspect_model
from dataeval_flow._app._model._item import (
    SKIP,
    build_item_dict,
    collect_bool_value,
    collect_field_value,
    collect_json_value,
    collect_multi_select_value,
    diagnose_collect_failure,
)
from dataeval_flow._app._model._registry import (
    STEP_BUILDER_SECTIONS,
    get_discriminator_field,
    get_fields,
    get_variant_choices,
)
from dataeval_flow._app._model._state import ConfigState

__all__ = ["SectionViewModel"]


class SectionViewModel:
    """ViewModel for creating/editing items in any config section."""

    def __init__(
        self,
        section: str,
        existing: dict[str, Any] | None = None,
        state: ConfigState | None = None,
    ) -> None:
        self.section = section
        self.state = state or ConfigState()
        self.existing = existing
        self.original: dict[str, Any] | None = dict(existing) if existing else None
        self.descriptors: list[FieldDescriptor] = []
        self.steps: list[dict[str, Any]] = []
        self.list_items: dict[str, list[dict[str, Any]]] = {}
        self._step_choices: list[str] = []
        self._get_step_params: Any = None

        if existing and section in STEP_BUILDER_SECTIONS:
            self.steps = [dict(s) for s in existing.get("steps", [])]

    # -- Properties --------------------------------------------------------

    @property
    def is_edit_mode(self) -> bool:
        return self.existing is not None

    @property
    def is_step_builder(self) -> bool:
        return self.section in STEP_BUILDER_SECTIONS

    @property
    def variant_choices(self) -> list[str] | None:
        return get_variant_choices(self.section)

    @property
    def disc_field(self) -> str | None:
        return get_discriminator_field(self.section)

    @property
    def step_choices(self) -> list[str]:
        return self._step_choices

    # -- Field descriptors -------------------------------------------------

    def load_fields(self, variant_value: str | None) -> list[FieldDescriptor]:
        """Load field descriptors for the given variant. Stores them internally."""
        self.descriptors = get_fields(self.section, variant_value, self.state)
        return self.descriptors

    # -- Step builder ------------------------------------------------------

    def init_step_builder(self) -> list[str]:
        """Initialize step builder and return available step choices."""
        spec = STEP_BUILDER_SECTIONS[self.section]
        from dataeval_flow._app._model import _discover

        self._step_choices = getattr(_discover, spec["list_fn"])()
        self._get_step_params = getattr(_discover, spec["params_fn"])
        return self._step_choices

    def get_step_params(self, step_name: str) -> list[Any]:
        """Return parameter info for a step type."""
        if self._get_step_params:
            return self._get_step_params(step_name)
        return []

    def add_step(self, step_name: str, params: dict[str, Any]) -> str:
        """Add a step with collected params. Returns notification message."""
        spec = STEP_BUILDER_SECTIONS[self.section]
        step: dict[str, Any] = {spec["step_key"]: step_name}
        if params:
            step["params"] = params
        self.steps.append(step)
        return f"Added step '{step_name}'."

    def remove_step(self, index: int) -> bool:
        """Remove step at index. Returns True if removed."""
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
            return True
        return False

    def step_display_lines(self) -> list[str]:
        """Return formatted display strings for each step."""
        spec = STEP_BUILDER_SECTIONS.get(self.section, {})
        step_key = spec.get("step_key", "step")
        lines: list[str] = []
        for idx, step in enumerate(self.steps):
            name = step.get(step_key, "?")
            params = step.get("params", {})
            if params:
                p_str = ", ".join(f"{k}={v}" for k, v in params.items())
                lines.append(f"{idx + 1}. {name}({p_str})")
            else:
                lines.append(f"{idx + 1}. {name}")
        return lines

    # -- Union list builder ------------------------------------------------

    def add_list_item(self, field_name: str, variant_key: str, field_values: dict[str, Any]) -> str:
        """Add a union-list item. Returns notification message."""
        desc = next((d for d in self.descriptors if d.name == field_name), None)
        if not desc or not desc.discriminator:
            return ""
        item: dict[str, Any] = {desc.discriminator: variant_key}
        item.update(field_values)
        self.list_items.setdefault(field_name, []).append(item)
        return f"Added {variant_key}."

    def remove_list_item(self, field_name: str, index: int) -> bool:
        """Remove a union-list item. Returns True if removed."""
        items = self.list_items.get(field_name, [])
        if 0 <= index < len(items):
            items.pop(index)
            return True
        return False

    def get_variant_descriptors(self, field_name: str, variant_key: str) -> list[FieldDescriptor]:
        """Return field descriptors for a union variant, excluding the discriminator and complex fields."""
        desc = next((d for d in self.descriptors if d.name == field_name), None)
        if not desc or not desc.union_variants or not desc.discriminator:
            return []
        variant_model = desc.union_variants.get(variant_key)
        if not variant_model:
            return []
        return [
            vd
            for vd in introspect_model(variant_model)
            if vd.name != desc.discriminator and vd.kind.value not in ("nested", "list")
        ]

    # -- Collection (pure logic, takes raw values from view) ---------------

    def collect_field(self, desc: FieldDescriptor, raw_value: Any) -> Any:
        """Coerce a raw widget value for a single field. Returns SKIP if empty."""
        if desc.kind == FieldKind.SELECT or (desc.kind == FieldKind.NESTED and desc.union_variants):
            return raw_value if raw_value else SKIP
        if desc.kind == FieldKind.MULTI_SELECT:
            return collect_multi_select_value(raw_value or [], self.section, desc.name)
        if desc.kind == FieldKind.BOOL:
            return collect_bool_value(raw_value, desc.default)
        if desc.kind in (FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING):
            return collect_field_value(desc, raw_value or "")
        if desc.kind == FieldKind.LIST and desc.union_variants:
            items = self.list_items.get(desc.name, [])
            return list(items) if items else SKIP
        if desc.kind == FieldKind.NESTED and desc.item_descriptors and isinstance(raw_value, dict):
            coerced: dict[str, Any] = {}
            for sub_desc in desc.item_descriptors:
                sub_raw = raw_value.get(sub_desc.name)
                sub_val = self.collect_field(sub_desc, sub_raw)
                if sub_val is not SKIP:
                    coerced[sub_desc.name] = sub_val
            return coerced if coerced else SKIP
        if desc.kind in (FieldKind.LIST, FieldKind.NESTED):
            return collect_json_value(raw_value or "")
        return SKIP

    def build_result(
        self,
        name: str,
        variant_value: str | None,
        field_values: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Assemble the final item dict from collected values. Returns None if invalid."""
        if not name:
            return None
        if self.disc_field and not variant_value:
            return None
        if self.is_step_builder:
            if not self.steps:
                return None
            return build_item_dict(self.section, name, variant_value, {"steps": list(self.steps)})

        # Preserve existing enabled state for tasks
        if self.section == "tasks" and self.existing:
            field_values.setdefault("enabled", self.existing.get("enabled", True))

        return build_item_dict(self.section, name, variant_value, field_values)

    def check_dirty(self, collected: dict[str, Any] | None) -> bool:
        """Return True if collected data differs from original."""
        if not collected:
            return False
        if not self.original:
            return True
        return collected != self.original

    def diagnose_failure(self, name: str, result: dict[str, Any] | None = None) -> str:
        """Return a human-readable error message when build_result returns None."""
        return diagnose_collect_failure(self.section, name, self.steps, self.descriptors, result)
