"""Lightweight click-based configuration builder/editor.

Uses ``click.prompt`` / ``click.confirm`` / ``click.Choice`` for interactive
input so it works without the ``textual`` dependency.  Consumes the
:class:`~dataeval_flow._app._viewmodel._builder_vm.BuilderViewModel` for
state management and :class:`SectionViewModel` for field introspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import yaml

from dataeval_flow._app._model._introspect import FieldDescriptor, FieldKind
from dataeval_flow._app._model._registry import SECTIONS, get_discriminator_field
from dataeval_flow._app._viewmodel._builder_vm import BuilderViewModel

__all__ = ["run_cli_builder"]


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _prompt_select(desc: FieldDescriptor, hint: str, existing_value: Any) -> Any:
    display = f"{desc.name}{hint}"
    if not desc.required:
        choices = list(desc.choices) + [""]
        display += " [Enter to skip]"
    else:
        choices = list(desc.choices)
    default = str(existing_value) if existing_value is not None and str(existing_value) in choices else None
    if not desc.required and default is None:
        default = ""
    val = click.prompt(
        display,
        type=click.Choice(choices, case_sensitive=False),
        default=default,
        show_choices=True,
    )
    return val if val else None


def _prompt_multi_select(desc: FieldDescriptor, hint: str, existing_value: Any) -> list[str] | None:
    click.echo(f"\n  {desc.name}{hint} (check all that apply):")
    existing_list = existing_value if isinstance(existing_value, list) else []
    selected: list[str] = []
    for choice in desc.choices:
        default_yn = choice in existing_list
        if click.confirm(f"    Include '{choice}'?", default=default_yn):
            selected.append(choice)
    return selected if selected else None


def _prompt_bool(desc: FieldDescriptor, hint: str, existing_value: Any) -> bool:
    default = (
        existing_value
        if isinstance(existing_value, bool)
        else (desc.default if isinstance(desc.default, bool) else False)
    )
    return click.confirm(f"{desc.name}{hint}", default=default)


def _prompt_numeric(desc: FieldDescriptor, suffix: str, hint: str, existing_value: Any) -> Any:
    prompt_text = f"{desc.name}{suffix}{hint}"
    constraint_parts = [f"{k}={v}" for k, v in desc.constraints.items()]
    if constraint_parts:
        prompt_text += f" [{', '.join(constraint_parts)}]"
    default = existing_value if existing_value is not None else desc.default
    typ = float if desc.kind == FieldKind.FLOAT else int
    if default is not None:
        return click.prompt(prompt_text, default=default, type=typ)
    if desc.required:
        return click.prompt(prompt_text, type=typ)
    raw = click.prompt(prompt_text, default="", show_default=False)
    if raw == "":
        return None
    return typ(raw)


def _prompt_json(desc: FieldDescriptor, suffix: str, hint: str, existing_value: Any) -> Any:
    default = (
        json.dumps(existing_value)
        if existing_value is not None
        else (json.dumps(desc.default) if desc.default is not None else "")
    )
    prompt_text = f"{desc.name} (JSON){suffix}{hint}"
    raw = click.prompt(prompt_text, default=default, show_default=bool(default))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        click.echo("  Warning: could not parse as JSON, storing as string.")
        return raw


def _prompt_field(desc: FieldDescriptor, existing_value: Any = None) -> Any:
    """Prompt the user for a single field value based on its descriptor."""
    suffix = "" if desc.required else " (optional, Enter to skip)"
    hint = f" ({desc.description})" if desc.description else ""

    if desc.kind == FieldKind.SELECT and desc.choices:
        return _prompt_select(desc, hint, existing_value)

    if desc.kind == FieldKind.MULTI_SELECT and desc.choices:
        return _prompt_multi_select(desc, hint, existing_value)

    if desc.kind == FieldKind.BOOL:
        return _prompt_bool(desc, hint, existing_value)

    if desc.kind in (FieldKind.INT, FieldKind.FLOAT):
        return _prompt_numeric(desc, suffix, hint, existing_value)

    if desc.kind in (FieldKind.LIST, FieldKind.NESTED):
        if desc.union_variants and desc.discriminator:
            return _prompt_union_list(desc, existing_value)
        if desc.nested_model:
            return _prompt_nested_model(desc, existing_value)
        return _prompt_json(desc, suffix, hint, existing_value)

    # STRING (default)
    default = (
        str(existing_value) if existing_value is not None else (str(desc.default) if desc.default is not None else "")
    )
    prompt_text = f"{desc.name}{suffix}{hint}"
    val = click.prompt(prompt_text, default=default, show_default=bool(default))
    return val if val else None


def _union_list_add(desc: FieldDescriptor, items: list[dict[str, Any]], singular: str) -> None:
    variant_choices = list(desc.union_variants)  # type: ignore[arg-type]
    variant_key = click.prompt(
        f"  {desc.discriminator}",
        type=click.Choice(variant_choices, case_sensitive=False),
        show_choices=True,
    )
    variant_model = desc.union_variants[variant_key]  # type: ignore[index]
    from dataeval_flow._app._model._introspect import introspect_model as _introspect

    variant_descs = _introspect(variant_model)
    item: dict[str, Any] = {desc.discriminator: variant_key}  # type: ignore[dict-item]
    for vd in variant_descs:
        if vd.name == desc.discriminator:
            continue
        val = _prompt_field(vd)
        if val is not None:
            item[vd.name] = val
    items.append(item)
    click.echo(f"  Added {variant_key} {singular}.")


def _union_list_remove(items: list[dict[str, Any]]) -> None:
    if not items:
        click.echo("  Nothing to remove.")
        return
    idx = click.prompt("  Remove index", type=int) - 1
    if 0 <= idx < len(items):
        items.pop(idx)
    else:
        click.echo("  Invalid index.")


def _prompt_union_list(desc: FieldDescriptor, existing: Any) -> list[dict[str, Any]]:
    """Prompt for a list of discriminated-union items (e.g. detectors)."""
    if not desc.union_variants or not desc.discriminator:
        msg = "union_variants and discriminator are required"
        raise ValueError(msg)
    items: list[dict[str, Any]] = []
    if isinstance(existing, list):
        items = [dict(x) for x in existing]

    singular = desc.name[:-1] if desc.name.endswith("s") else desc.name

    while True:
        if items:
            click.echo(f"\n  Current {desc.name}:")
            for i, item in enumerate(items):
                parts = [f"{k}={v}" for k, v in item.items()]
                click.echo(f"    {i + 1}. {', '.join(parts)}")

        click.echo(f"\n  Options: [a]dd {singular}, [r]emove, [d]one")
        action = click.prompt("  Action", type=click.Choice(["a", "r", "d"]), default="d")

        if action == "d":
            break
        if action == "r":
            _union_list_remove(items)
        elif action == "a":
            _union_list_add(desc, items, singular)

    return items


def _prompt_nested_model(desc: FieldDescriptor, existing: Any) -> dict[str, Any] | None:
    """Prompt for a nested Pydantic model."""
    if desc.nested_model is None:
        return None
    from dataeval_flow._app._model._introspect import introspect_model as _introspect

    click.echo(f"\n  {desc.name}:")
    nested_descs = _introspect(desc.nested_model)
    result: dict[str, Any] = {}
    existing_dict = existing if isinstance(existing, dict) else {}
    for nested_desc in nested_descs:
        val = _prompt_field(nested_desc, existing_dict.get(nested_desc.name))
        if val is not None:
            result[nested_desc.name] = val
    return result if result else None


# ---------------------------------------------------------------------------
# Step builder (preprocessors / selections)
# ---------------------------------------------------------------------------


def _steps_remove(steps: list[dict[str, Any]]) -> None:
    if not steps:
        click.echo("  No steps to remove.")
        return
    idx = click.prompt("  Remove index", type=int) - 1
    if 0 <= idx < len(steps):
        steps.pop(idx)
    else:
        click.echo("  Invalid index.")


def _steps_prompt_param(p: Any) -> tuple[str, Any] | None:
    suffix = "" if p.required else " (optional, Enter to skip)"
    if p.choices:
        return _steps_prompt_choice_param(p, suffix)
    if p.type_hint == "bool":
        return _steps_prompt_bool_param(p)
    return _steps_prompt_text_param(p, suffix)


def _steps_prompt_choice_param(p: Any, suffix: str) -> tuple[str, Any] | None:
    choices = list(p.choices) + [""] if not p.required else list(p.choices)
    val = click.prompt(
        f"    {p.name} [{p.type_hint}]{suffix}",
        type=click.Choice(choices, case_sensitive=False),
        default=str(p.default)
        if p.default is not None and str(p.default) in choices
        else ("" if not p.required else None),
        show_choices=True,
    )
    return (p.name, val) if val else None


def _steps_prompt_bool_param(p: Any) -> tuple[str, Any] | None:
    default = p.default if isinstance(p.default, bool) else False
    val = click.confirm(f"    {p.name}", default=default)
    return (p.name, val) if val != default else None


def _steps_prompt_text_param(p: Any, suffix: str) -> tuple[str, Any] | None:
    default_str = str(p.default) if p.default is not None else ""
    raw = click.prompt(f"    {p.name} [{p.type_hint}]{suffix}", default=default_str, show_default=bool(default_str))
    if raw and raw != default_str or raw and p.required:
        from dataeval_flow._app._model._coerce import coerce_value

        return (p.name, coerce_value(raw, p.type_hint))
    return None


def _steps_add(steps: list[dict[str, Any]], step_key: str, sec_vm: Any) -> None:
    available_steps = sec_vm.step_choices
    step_name = click.prompt(
        f"  {step_key}",
        type=click.Choice(available_steps, case_sensitive=True),
        show_choices=False,
    )
    param_infos = sec_vm.get_step_params(step_name)
    params: dict[str, Any] = {}
    for p in param_infos:
        result = _steps_prompt_param(p)
        if result is not None:
            params[result[0]] = result[1]

    step: dict[str, Any] = {step_key: step_name}
    if params:
        step["params"] = params
    steps.append(step)
    click.echo(f"  Added step '{step_name}'.")


def _prompt_steps(sec_vm: Any, existing_steps: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Interactive step builder for preprocessor/selection pipelines."""
    from dataeval_flow._app._model._registry import STEP_BUILDER_SECTIONS

    spec = STEP_BUILDER_SECTIONS[sec_vm.section]
    step_key = spec["step_key"]

    sec_vm.init_step_builder()

    steps: list[dict[str, Any]] = list(existing_steps) if existing_steps else []

    while True:
        if steps:
            click.echo("\n  Steps:")
            for i, s in enumerate(steps):
                name = s.get(step_key, "?")
                params = s.get("params", {})
                if params:
                    p_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    click.echo(f"    {i + 1}. {name}({p_str})")
                else:
                    click.echo(f"    {i + 1}. {name}")

        click.echo("\n  Options: [a]dd step, [r]emove, [d]one")
        action = click.prompt("  Action", type=click.Choice(["a", "r", "d"]), default="d")

        if action == "d":
            break
        if action == "r":
            _steps_remove(steps)
        elif action == "a":
            _steps_add(steps, step_key, sec_vm)

    return steps


# ---------------------------------------------------------------------------
# Section editing
# ---------------------------------------------------------------------------


def _prompt_item(
    section: str,
    vm: BuilderViewModel,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Prompt for all fields of a section item. Returns the item dict or None."""
    sec_vm = vm.create_section_vm(section, existing)

    # Name
    default_name = existing.get("name", "") if existing else ""
    name = click.prompt("name", default=default_name, show_default=bool(default_name))
    if not name:
        click.echo("name is required.")
        return None

    # Discriminator (if applicable)
    variant_choices = sec_vm.variant_choices
    variant_value: str | None = None

    if variant_choices and sec_vm.disc_field:
        default_variant = existing.get(sec_vm.disc_field, "") if existing else ""
        variant_value = click.prompt(
            sec_vm.disc_field.replace("_", " "),
            type=click.Choice(variant_choices, case_sensitive=False),
            default=default_variant if default_variant in variant_choices else None,
            show_choices=True,
        )

    # Step-builder sections
    if sec_vm.is_step_builder:
        existing_steps = existing.get("steps", []) if existing else []
        steps = _prompt_steps(sec_vm, existing_steps)
        if not steps:
            click.echo("At least one step is required.")
            return None
        sec_vm.steps = steps
        return sec_vm.build_result(name, variant_value, {})

    # Regular fields from introspection
    sec_vm.load_fields(variant_value)
    field_values: dict[str, Any] = {}
    for desc in sec_vm.descriptors:
        existing_val = existing.get(desc.name) if existing else None
        val = _prompt_field(desc, existing_val)
        if val is not None:
            field_values[desc.name] = val

    return sec_vm.build_result(name, variant_value, field_values)


def _show_items(section: str, vm: BuilderViewModel) -> None:
    """Display items in a section."""
    items = vm.items(section)
    if not items:
        click.echo("  (empty)")
        return
    for i, item in enumerate(items):
        name = item.get("name", "?")
        # Build a brief summary
        parts: list[str] = []
        disc_field = get_discriminator_field(section)
        if disc_field and disc_field in item:
            parts.append(f"{disc_field}={item[disc_field]}")
        # Add a few key fields
        for key, val in item.items():
            if key in ("name",) or key == disc_field:
                continue
            if isinstance(val, (list, dict)):
                continue
            parts.append(f"{key}={val}")
            if len(parts) >= 3:
                break
        summary = "  ".join(parts) if parts else ""
        enabled_marker = ""
        if section == "tasks":
            enabled = item.get("enabled", True)
            enabled_marker = "[on] " if enabled else "[off] "
        click.echo(f"  {i + 1}. {enabled_marker}{name}  {summary}")


def _edit_action_add(section: str, vm: BuilderViewModel) -> None:
    item = _prompt_item(section, vm)
    if not item:
        return
    errors = vm.validate_item(section, item)
    if errors:
        click.echo("  Validation warnings:")
        for e in errors:
            click.echo(f"    - {e}")
        if not click.confirm("  Add anyway?", default=True):
            return
    outcome = vm.apply_result(section, -1, item)
    if outcome:
        click.echo(f"  Added '{item.get('name', '?')}'.")


def _edit_action_edit(section: str, vm: BuilderViewModel) -> None:
    idx = click.prompt("  Edit index", type=int) - 1
    existing = vm.get_item(section, idx)
    if existing is None:
        click.echo("  Invalid index.")
        return
    item = _prompt_item(section, vm, existing=existing)
    if not item:
        return
    errors = vm.validate_item(section, item)
    if errors:
        click.echo("  Validation warnings:")
        for e in errors:
            click.echo(f"    - {e}")
        if not click.confirm("  Save anyway?", default=True):
            return
    outcome = vm.apply_result(section, idx, item)
    if outcome:
        click.echo(f"  Updated '{item.get('name', '?')}'.")


def _edit_action_delete(section: str, vm: BuilderViewModel) -> None:
    idx = click.prompt("  Delete index", type=int) - 1
    outcome = vm.delete_item(section, idx)
    if outcome:
        description, warnings = outcome
        # description is like "Delete dataset 'ds1'" — extract the name
        removed_name = description.split("'")[1] if "'" in description else "?"
        click.echo(f"  Removed '{removed_name}'.")
        for w in warnings:
            click.echo(f"  Warning: {w}")
    else:
        click.echo("  Invalid index.")


def _edit_action_toggle(vm: BuilderViewModel) -> None:
    idx = click.prompt("  Toggle index", type=int) - 1
    desc = vm.toggle_task(idx)
    if desc:
        click.echo(f"  {desc}.")
    else:
        click.echo("  Invalid index.")


def _edit_section(section: str, section_title: str, vm: BuilderViewModel) -> None:
    """Interactive loop for editing a single section."""
    while True:
        click.echo(f"\n--- {section_title} ---")
        _show_items(section, vm)

        options = ["a", "b"]
        option_desc = "[a]dd, [b]ack"
        if vm.count(section) > 0:
            options.extend(["e", "d"])
            option_desc = "[a]dd, [e]dit, [d]elete, [b]ack"
            if section == "tasks":
                options.append("t")
                option_desc += ", [t]oggle"

        click.echo(f"\n  {option_desc}")
        action = click.prompt("  Action", type=click.Choice(options), default="b", show_choices=False)

        if action == "b":
            break
        if action == "a":
            _edit_action_add(section, vm)
        elif action == "e":
            _edit_action_edit(section, vm)
        elif action == "d":
            _edit_action_delete(section, vm)
        elif action == "t" and section == "tasks":
            _edit_action_toggle(vm)


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------


def _menu_save(vm: BuilderViewModel) -> None:
    default_path = vm.config_file_path or "config.yaml"
    out = click.prompt("Save to", default=default_path)
    success, msg = vm.save_file(Path(out))
    click.echo(msg)


def _menu_load(vm: BuilderViewModel) -> None:
    load_path = click.prompt("Load from", default=vm.config_file_path or "")
    if load_path:
        p = Path(load_path)
        if p.exists():
            success, msg = vm.load_file(p)
            click.echo(msg)
        else:
            click.echo(f"File not found: {p}")


def _menu_view(vm: BuilderViewModel) -> None:
    config = vm.to_dict()
    if config:
        click.echo("\n" + yaml.dump(config, default_flow_style=False, sort_keys=False))
    else:
        click.echo("(empty config)")


def _menu_display(vm: BuilderViewModel) -> None:
    click.echo("\n=== DataEval Flow Config Builder ===")
    if vm.config_file_path:
        click.echo(f"  File: {vm.config_file_path}")
    for key, title in SECTIONS:
        count = vm.count(key)
        click.echo(f"  {title}: {count} item{'s' if count != 1 else ''}")
    click.echo("\nSections:")
    for i, (_, title) in enumerate(SECTIONS):
        click.echo(f"  {i + 1}. {title}")
    click.echo("  s. Save")
    click.echo("  l. Load")
    click.echo("  v. View YAML")
    click.echo("  q. Quit")


def _main_menu(vm: BuilderViewModel) -> None:
    """Main interactive loop."""
    while True:
        _menu_display(vm)
        choice = click.prompt("\nChoice", default="q")

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(SECTIONS):
                key, title = SECTIONS[idx]
                _edit_section(key, title, vm)
            else:
                click.echo("Invalid choice.")
        elif choice == "s":
            _menu_save(vm)
        elif choice == "l":
            _menu_load(vm)
        elif choice == "v":
            _menu_view(vm)
        elif choice == "q":
            if not vm.is_empty() and not click.confirm("Unsaved changes may be lost. Quit?", default=False):
                continue
            break
        else:
            click.echo("Invalid choice.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_cli_builder(config_path: str | Path | None = None) -> None:
    """Launch the lightweight click-based configuration builder."""
    vm = BuilderViewModel(config_path)

    if config_path:
        p = Path(config_path)
        if p.exists():
            success, msg = vm.load_file(p)
            click.echo(msg)

    _main_menu(vm)
