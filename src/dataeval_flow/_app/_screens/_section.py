"""SectionModal — generic modal for any config section."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind
from dataeval_flow._app._model._item import SKIP
from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._app._screens._base import ComponentModal, _select_value
from dataeval_flow._app._screens._params import build_param_form, collect_param_form, validate_param_form
from dataeval_flow._app._viewmodel._section_vm import SectionViewModel

_log = logging.getLogger(__name__)


class SectionModal(ComponentModal):
    """Generic modal for creating/editing items in any config section.

    Delegates all business logic to :class:`SectionViewModel`.
    This class handles only widget creation, event routing, and
    reading/writing widget values.
    """

    def __init__(
        self,
        section: str,
        existing: dict[str, Any] | None = None,
        state: ConfigState | None = None,
        section_vm: SectionViewModel | None = None,
        data_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(existing, data_dir=data_dir, **kwargs)
        self._vm = section_vm if section_vm is not None else SectionViewModel(section, existing, state)
        self._section = section  # convenience alias
        self._gen: int = 0
        self._variant_param_gen: int = 0

    def _wid(self, name: str) -> str:
        return f"md-{self._gen}-{name}"

    def compose(self) -> ComposeResult:
        singular = self._section[:-1].replace("_", " ").title()
        title = f"Edit {singular}" if self.is_edit_mode else f"New {singular}"

        with VerticalScroll(classes="modal-dialog"):
            yield Static(f"[bold]{title}[/bold]")
            yield Label("name:")
            yield Input(id="md-name", placeholder=f"e.g. my_{self._section[:-1]}")

            if self._vm.variant_choices:
                disc_field = self._vm.disc_field or "type"
                yield Label(f"{disc_field.replace('_', ' ')}:")
                yield Select(
                    [(c, c) for c in self._vm.variant_choices],
                    id="md-disc",
                    prompt=f"select {disc_field}",
                )

            if self._vm.is_step_builder:
                yield Static("[bold]steps[/bold]", classes="step-section-title")
                yield Vertical(id="modal-step-list")
                yield Label("add step:")
                yield Select[str]([], id="md-step-select", allow_blank=True, prompt="select step")
                yield Vertical(id="modal-params-form")
                yield Button("Add Step", id="btn-modal-add-step", variant="success")
            else:
                yield Vertical(id="md-fields")

            yield from self.compose_buttons()

    def on_mount(self) -> None:
        if self._vm.is_step_builder:
            choices = self._vm.init_step_builder()
            with contextlib.suppress(NoMatches):
                self.query_one("#md-step-select", Select).set_options([(c, c) for c in choices])

        if self._existing:
            with contextlib.suppress(NoMatches):
                self.query_one("#md-name", Input).value = self._existing.get("name", "")

            if self._vm.variant_choices and self._vm.disc_field:
                val = self._existing.get(self._vm.disc_field, "")
                if val:
                    with contextlib.suppress(NoMatches):
                        self.query_one("#md-disc", Select).value = val
                    self._rebuild_fields()
                    self._populate_fields()
            elif not self._vm.is_step_builder:
                self._rebuild_fields()
                self._populate_fields()

            if self._vm.is_step_builder:
                self._refresh_step_list()
        elif not self._vm.variant_choices and not self._vm.is_step_builder:
            self._rebuild_fields()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "md-disc":
            self._rebuild_fields()
        elif event.select.id == "md-step-select":
            self._rebuild_step_params()
        else:
            # Check for variant picker changes in LIST fields
            select_id = event.select.id or ""
            if select_id.endswith("-picker"):
                field_name = self._parse_picker_field(select_id)
                if field_name:
                    self._rebuild_variant_params(field_name)
        super().on_select_changed(event)

    def _handle_remove_step(self, btn: str) -> None:
        """Handle a step-removal button press."""
        try:
            idx = int(btn.split("-")[-1])
            if self._vm.remove_step(idx):
                self._refresh_step_list()
                self._update_ok_state()
        except (ValueError, IndexError):
            pass

    def _handle_list_remove(self, btn: str) -> None:
        """Handle a list-item removal button press."""
        parsed = self._parse_list_rem(btn)
        if parsed:
            field_name, idx = parsed
            if self._vm.remove_list_item(field_name, idx):
                self._refresh_list_items(field_name)
                self._update_ok_state()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id or ""
        if btn == "btn-modal-add-step":
            self._add_step()
            return
        if btn.startswith("btn-remove-step-"):
            self._handle_remove_step(btn)
            return
        if btn.endswith("-add"):
            field_name = self._parse_add_field(btn)
            if field_name:
                self._add_list_item(field_name)
                return
        if btn.startswith("md-listrem-"):
            self._handle_list_remove(btn)
            return
        if btn.startswith("btn-browse-"):
            input_id = btn[len("btn-browse-") :]
            self._browse_for_input(input_id)
            return
        super().on_button_pressed(event)

    # -- Field rendering ---------------------------------------------------

    @staticmethod
    def _mount_scalar_field(container: Vertical, desc: FieldDescriptor, wid: str, *, browse: bool = False) -> None:
        """Mount a single scalar field widget (SELECT, BOOL, INT, FLOAT, STRING)."""
        suffix = "" if desc.required else " (optional)"
        label_text = f"{desc.name}{suffix}"

        if desc.kind == FieldKind.SELECT:
            container.mount(Label(f"{label_text}:"))
            options = [(c, c) for c in desc.choices]
            has_default = desc.default is not None and desc.default in desc.choices
            kw: dict[str, Any] = {"id": wid, "allow_blank": not desc.required or not options}
            if has_default:
                kw["value"] = desc.default
            elif not desc.required or not options:
                kw["prompt"] = "(default)" if not desc.required else "(no options)"
            container.mount(Select(options, **kw))
        elif desc.kind == FieldKind.BOOL:
            default_val = desc.default if isinstance(desc.default, bool) else False
            container.mount(Checkbox(label_text, value=default_val, id=wid))
        elif desc.kind in (FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING):
            container.mount(Label(f"{label_text}:"))
            placeholder = str(desc.default) if desc.default is not None else ""
            if desc.constraints:
                parts = [f"{k}={v}" for k, v in desc.constraints.items()]
                placeholder += f"  ({', '.join(parts)})" if placeholder else ", ".join(parts)
            if browse and "path" in desc.name.lower():
                row = Horizontal(classes="browse-row")
                container.mount(row)
                row.mount(Input(placeholder=placeholder, id=wid))
                row.mount(Button("Browse", id=f"btn-browse-{wid}"))
            else:
                container.mount(Input(placeholder=placeholder, id=wid))

    def _mount_multi_select_field(self, container: Vertical, desc: FieldDescriptor) -> None:
        suffix = "" if desc.required else " (optional)"
        container.mount(Label(f"{desc.name}{suffix} (check all that apply):"))
        for choice in desc.choices:
            checked = isinstance(desc.default, list) and choice in desc.default
            container.mount(Checkbox(choice, value=checked, id=self._wid(f"{desc.name}-{choice}")))

    def _mount_union_list_field(self, container: Vertical, desc: FieldDescriptor) -> None:
        suffix = "" if desc.required else " (optional)"
        label_text = f"{desc.name}{suffix}"
        if desc.name not in self._vm.list_items:
            self._vm.list_items[desc.name] = []
        container.mount(Static(f"[bold]{label_text}[/bold]", classes="step-section-title"))
        container.mount(Vertical(id=self._wid(f"{desc.name}-list")))
        container.mount(Label("add:"))
        variant_names = list(desc.union_variants)  # type: ignore[arg-type]
        container.mount(
            Select(
                [(v, v) for v in variant_names],
                id=self._wid(f"{desc.name}-picker"),
                prompt=f"select {desc.discriminator}",
            )
        )
        container.mount(Vertical(id=self._wid(f"{desc.name}-params")))
        singular = desc.name[:-1] if desc.name.endswith("s") else desc.name
        container.mount(Button(f"Add {singular}", id=self._wid(f"{desc.name}-add"), variant="success"))
        self._refresh_list_items(desc.name)

    def _mount_nested_union_field(self, container: Vertical, desc: FieldDescriptor, wid: str) -> None:
        suffix = "" if desc.required else " (optional)"
        container.mount(Label(f"{desc.name}{suffix}:"))
        variant_names = list(desc.union_variants)  # type: ignore[arg-type]
        container.mount(
            Select(
                [(v, v) for v in variant_names],
                id=wid,
                prompt=f"select {desc.name}",
                allow_blank=not desc.required,
            )
        )

    def _mount_json_field(self, container: Vertical, desc: FieldDescriptor, wid: str) -> None:
        suffix = "" if desc.required else " (optional)"
        container.mount(Label(f"{desc.name}{suffix} (JSON):"))
        placeholder = ""
        if desc.default is not None:
            try:
                placeholder = json.dumps(desc.default)
            except (TypeError, ValueError):
                placeholder = str(desc.default)
        container.mount(Input(placeholder=placeholder, id=wid))

    def _mount_nested_model_field(self, container: Vertical, desc: FieldDescriptor) -> None:
        """Mount a nested BaseModel as expanded scalar controls with defaults pre-filled."""
        suffix = "" if desc.required else " (optional)"
        container.mount(Static(f"[bold]{desc.name}{suffix}[/bold]"))
        group = Vertical(id=self._wid(f"{desc.name}-group"), classes="nested-group")
        container.mount(group)
        for sub_desc in desc.item_descriptors:  # type: ignore[union-attr]
            sub_wid = self._wid(f"{desc.name}__{sub_desc.name}")
            self._mount_scalar_field(group, sub_desc, sub_wid)
            # Pre-fill with defaults so the form isn't empty
            if sub_desc.default is not None and sub_desc.kind in (FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING):
                group.query_one(f"#{sub_wid}", Input).value = str(sub_desc.default)

    def _mount_field(self, container: Vertical, desc: FieldDescriptor) -> None:
        """Mount a single field descriptor as the appropriate widget(s)."""
        wid = self._wid(desc.name)

        if desc.kind in (FieldKind.SELECT, FieldKind.BOOL, FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING):
            self._mount_scalar_field(container, desc, wid, browse=True)
        elif desc.kind == FieldKind.MULTI_SELECT:
            self._mount_multi_select_field(container, desc)
        elif desc.kind == FieldKind.LIST and desc.union_variants and desc.discriminator:
            self._mount_union_list_field(container, desc)
        elif desc.kind == FieldKind.NESTED and desc.union_variants:
            self._mount_nested_union_field(container, desc, wid)
        elif desc.kind == FieldKind.NESTED and desc.item_descriptors:
            self._mount_nested_model_field(container, desc)
        elif desc.kind in (FieldKind.LIST, FieldKind.NESTED):
            self._mount_json_field(container, desc, wid)

    @staticmethod
    def _force_scroll_recalc(container: Vertical) -> None:
        try:
            parent = container.parent
            while parent is not None:
                if isinstance(parent, VerticalScroll):
                    parent.refresh(layout=True)
                    break
                parent = parent.parent
        except NoMatches:
            pass

    def _rebuild_fields(self) -> None:
        """Rebuild the dynamic fields form."""
        if self._vm.is_step_builder:
            return

        try:
            container = self.query_one("#md-fields", Vertical)
        except NoMatches:
            return

        container.remove_children()
        self._gen += 1
        self._vm.list_items.clear()

        variant = None
        if self._vm.variant_choices:
            try:
                variant = _select_value(self.query_one("#md-disc", Select))
            except NoMatches:
                return
            if not variant:
                return

        self._vm.load_fields(variant)
        for desc in self._vm.descriptors:
            self._mount_field(container, desc)
        self._force_scroll_recalc(container)

    def _populate_nested_subs(self, desc: FieldDescriptor, val: dict[str, Any]) -> None:
        """Populate nested sub-field widgets from a dict."""
        for sub_desc in desc.item_descriptors:  # type: ignore[union-attr]
            if sub_desc.name not in val:
                continue
            sub_wid = self._wid(f"{desc.name}__{sub_desc.name}")
            sub_val = val[sub_desc.name]
            if sub_desc.kind == FieldKind.BOOL:
                self.query_one(f"#{sub_wid}", Checkbox).value = bool(sub_val)
            elif sub_desc.kind == FieldKind.SELECT:
                self.query_one(f"#{sub_wid}", Select).value = str(sub_val)
            elif sub_desc.kind in (FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING):
                self.query_one(f"#{sub_wid}", Input).value = str(sub_val)

    def _populate_one_field(self, desc: FieldDescriptor, val: Any) -> None:
        """Populate a single field widget from existing data."""
        wid = self._wid(desc.name)
        if desc.kind == FieldKind.SELECT or (desc.kind == FieldKind.NESTED and desc.union_variants):
            self.query_one(f"#{wid}", Select).value = str(val)
        elif desc.kind == FieldKind.MULTI_SELECT:
            selected = val if isinstance(val, list) else [val]
            for choice in desc.choices:
                with contextlib.suppress(NoMatches):
                    self.query_one(f"#{self._wid(f'{desc.name}-{choice}')}", Checkbox).value = choice in selected
        elif desc.kind == FieldKind.BOOL:
            self.query_one(f"#{wid}", Checkbox).value = bool(val)
        elif desc.kind in (FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING):
            self.query_one(f"#{wid}", Input).value = str(val)
        elif desc.kind == FieldKind.LIST and desc.union_variants and isinstance(val, list):
            self._vm.list_items[desc.name] = [dict(v) if isinstance(v, dict) else v for v in val]
            self._refresh_list_items(desc.name)
        elif desc.kind == FieldKind.NESTED and desc.item_descriptors and isinstance(val, dict):
            self._populate_nested_subs(desc, val)
        elif desc.kind in (FieldKind.LIST, FieldKind.NESTED):
            try:
                self.query_one(f"#{wid}", Input).value = json.dumps(val)
            except (TypeError, ValueError):
                self.query_one(f"#{wid}", Input).value = str(val)

    def _populate_fields(self) -> None:
        """Populate field widgets from existing data."""
        if not self._existing:
            return
        for desc in self._vm.descriptors:
            if desc.name not in self._existing:
                continue
            try:
                self._populate_one_field(desc, self._existing[desc.name])
            except NoMatches:
                _log.debug("Widget %s not found while populating field '%s'", self._wid(desc.name), desc.name)

    # -- Collection --------------------------------------------------------

    def _read_name_and_variant(self) -> tuple[str, str | None] | None:
        """Read name and discriminator from widgets. Returns None if invalid."""
        try:
            name = self.query_one("#md-name", Input).value.strip()
        except NoMatches:
            return None
        if not name:
            return None

        variant_value: str | None = None
        if self._vm.disc_field:
            try:
                variant_value = _select_value(self.query_one("#md-disc", Select))
            except NoMatches:
                return None
            if not variant_value:
                return None

        return name, variant_value

    def _read_nested_subs(self, desc: FieldDescriptor) -> dict[str, Any]:
        """Read nested sub-field widget values into a dict."""
        result: dict[str, Any] = {}
        for sub_desc in desc.item_descriptors:  # type: ignore[union-attr]
            sub_wid = self._wid(f"{desc.name}__{sub_desc.name}")
            if sub_desc.kind == FieldKind.BOOL:
                result[sub_desc.name] = self.query_one(f"#{sub_wid}", Checkbox).value
            elif sub_desc.kind == FieldKind.SELECT:
                result[sub_desc.name] = _select_value(self.query_one(f"#{sub_wid}", Select))
            elif sub_desc.kind in (FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING):
                result[sub_desc.name] = self.query_one(f"#{sub_wid}", Input).value.strip()
        return result

    def _read_raw_field(self, desc: FieldDescriptor) -> Any:
        """Read a single raw widget value for a field descriptor."""
        wid = self._wid(desc.name)
        if desc.kind == FieldKind.SELECT or (desc.kind == FieldKind.NESTED and desc.union_variants):
            return _select_value(self.query_one(f"#{wid}", Select))
        if desc.kind == FieldKind.MULTI_SELECT:
            return [c for c in desc.choices if self.query_one(f"#{self._wid(f'{desc.name}-{c}')}", Checkbox).value]
        if desc.kind == FieldKind.BOOL:
            return self.query_one(f"#{wid}", Checkbox).value
        if desc.kind in (FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING):
            return self.query_one(f"#{wid}", Input).value.strip()
        if desc.kind == FieldKind.LIST and desc.union_variants:
            return None  # data comes from vm.list_items, not a widget
        if desc.kind == FieldKind.NESTED and desc.item_descriptors:
            return self._read_nested_subs(desc)
        if desc.kind in (FieldKind.LIST, FieldKind.NESTED):
            return self.query_one(f"#{wid}", Input).value.strip()
        return None

    def _collect_all_fields(self) -> dict[str, Any]:
        """Read all widget values and delegate coercion to ViewModel."""
        field_values: dict[str, Any] = {}
        for desc in self._vm.descriptors:
            val = self._try_collect_one(desc)
            if val is not SKIP:
                field_values[desc.name] = val

        if self._section == "tasks" and self._existing:
            field_values.setdefault("enabled", self._existing.get("enabled", True))

        return field_values

    def _try_collect_one(self, desc: FieldDescriptor) -> Any:
        """Read + coerce one field, returning SKIP on missing widget."""
        try:
            raw = self._read_raw_field(desc)
            return self._vm.collect_field(desc, raw)
        except NoMatches:
            return SKIP

    def _collect_raw(self) -> dict[str, Any] | None:
        parsed = self._read_name_and_variant()
        if parsed is None:
            return None
        name, variant_value = parsed
        return self._vm.build_result(name, variant_value, self._collect_all_fields())

    def _check_dirty(self) -> bool:
        collected = self._collect_raw()
        return self._vm.check_dirty(collected)

    def _collect(self) -> dict[str, Any] | None:
        result = self._collect_raw()
        if result is None:
            name = ""
            with contextlib.suppress(NoMatches):
                name = self.query_one("#md-name", Input).value.strip()
            msg = self._vm.diagnose_failure(name, result)
            self.app.notify(msg, severity="error")
        return result

    # -- Step builder (preprocessors / selections) -------------------------

    def _rebuild_step_params(self) -> None:
        step_name = _select_value(self.query_one("#md-step-select", Select))
        container = self.query_one("#modal-params-form", Vertical)
        if step_name:
            params = self._vm.get_step_params(step_name)
            build_param_form(container, params, "mdsp")
        else:
            container.remove_children()

    def _add_step(self) -> None:
        step_name = _select_value(self.query_one("#md-step-select", Select))
        if not step_name:
            self.app.notify("Select a step first.", severity="error")
            return
        param_infos = self._vm.get_step_params(step_name)
        container = self.query_one("#modal-params-form", Vertical)
        errors = validate_param_form(container, param_infos, "mdsp")
        if errors:
            for e in errors:
                self.app.notify(e, severity="error")
            return
        params = collect_param_form(container, param_infos, "mdsp")
        msg = self._vm.add_step(step_name, params)
        self._refresh_step_list()
        self._update_ok_state()
        self.app.notify(msg)

    def _refresh_step_list(self) -> None:
        try:
            container = self.query_one("#modal-step-list", Vertical)
        except NoMatches:
            return
        container.remove_children()
        lines = self._vm.step_display_lines()
        if not lines:
            container.mount(Static("[dim]  No steps yet.[/dim]"))
            return
        for idx, text in enumerate(lines):
            row = Horizontal(classes="step-row")
            container.mount(row)
            row.mount(Static(text))
            row.mount(Button("X", id=f"btn-remove-step-{idx}", variant="error"))

    # -- Union variant list builder (e.g. workflow detectors) ---------------

    def _parse_picker_field(self, select_id: str) -> str | None:
        prefix = f"md-{self._gen}-"
        if not select_id.startswith(prefix) or not select_id.endswith("-picker"):
            return None
        return select_id[len(prefix) :].rsplit("-picker", 1)[0]

    def _parse_add_field(self, btn_id: str) -> str | None:
        prefix = f"md-{self._gen}-"
        if not btn_id.startswith(prefix) or not btn_id.endswith("-add"):
            return None
        return btn_id[len(prefix) :].rsplit("-add", 1)[0]

    def _parse_list_rem(self, btn_id: str) -> tuple[str, int] | None:
        if not btn_id.startswith("md-listrem-"):
            return None
        rest = btn_id[len("md-listrem-") :]
        parts = rest.rsplit("-", 1)
        if len(parts) != 2:
            return None
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return None

    def _rebuild_variant_params(self, field_name: str) -> None:
        picker_id = self._wid(f"{field_name}-picker")
        params_id = self._wid(f"{field_name}-params")
        self._variant_param_gen += 1
        vpg = self._variant_param_gen

        try:
            variant_key = _select_value(self.query_one(f"#{picker_id}", Select))
            container = self.query_one(f"#{params_id}", Vertical)
            container.remove_children()
            if not variant_key:
                return
            variant_descs = self._vm.get_variant_descriptors(field_name, variant_key)
            for vd in variant_descs:
                vid = f"md-vp-{vpg}-{field_name}-{vd.name}"
                self._mount_scalar_field(container, vd, vid)
        except NoMatches:
            _log.debug("Widget not found while rebuilding variant params for '%s'", field_name)

    def _add_list_item(self, field_name: str) -> None:
        desc = next((d for d in self._vm.descriptors if d.name == field_name), None)
        if not desc or not desc.discriminator:
            return

        picker_id = self._wid(f"{field_name}-picker")
        variant_key = _select_value(self.query_one(f"#{picker_id}", Select))
        if not variant_key:
            self.app.notify(f"Select a {desc.discriminator}.", severity="error")
            return

        variant_descs = self._vm.get_variant_descriptors(field_name, variant_key)
        field_values: dict[str, Any] = {}

        params_id = self._wid(f"{field_name}-params")
        try:
            container = self.query_one(f"#{params_id}", Vertical)
        except NoMatches:
            return

        vpg = self._variant_param_gen
        for vd in variant_descs:
            wid = f"md-vp-{vpg}-{field_name}-{vd.name}"
            try:
                raw = self._read_variant_widget(container, vd, wid)
                val = self._vm.collect_field(vd, raw)
                if val is not SKIP:
                    field_values[vd.name] = val
            except NoMatches:
                pass

        msg = self._vm.add_list_item(field_name, variant_key, field_values)
        self._refresh_list_items(field_name)
        self._update_ok_state()
        if msg:
            self.app.notify(msg)

    def _read_variant_widget(self, container: Vertical, vd: FieldDescriptor, wid: str) -> Any:
        """Read a raw value from a variant param widget."""
        if vd.kind == FieldKind.SELECT:
            return _select_value(container.query_one(f"#{wid}", Select))
        if vd.kind == FieldKind.BOOL:
            return container.query_one(f"#{wid}", Checkbox).value
        if vd.kind in (FieldKind.INT, FieldKind.FLOAT, FieldKind.STRING):
            return container.query_one(f"#{wid}", Input).value.strip()
        return None

    def _refresh_list_items(self, field_name: str) -> None:
        list_id = self._wid(f"{field_name}-list")
        try:
            container = self.query_one(f"#{list_id}", Vertical)
        except NoMatches:
            return
        container.remove_children()
        items = self._vm.list_items.get(field_name, [])
        if not items:
            container.mount(Static("[dim]  (none added)[/dim]"))
            return
        for idx, item in enumerate(items):
            parts = [f"{k}={v}" for k, v in item.items()]
            text = f"{idx + 1}. {', '.join(parts)}"
            row = Horizontal(classes="step-row")
            container.mount(row)
            row.mount(Static(text))
            row.mount(Button("X", id=f"md-listrem-{field_name}-{idx}", variant="error"))
