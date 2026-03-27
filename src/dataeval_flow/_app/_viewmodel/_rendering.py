"""Item snippet rendering for the configuration builder.

Presentation-layer functions that produce Rich-markup snippets for display
in the TUI cards.  These live in the ViewModel layer because they transform
Model data into view-ready strings.  No Textual widget dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from dataeval_flow._app._model._execution import TaskExecution

# ---------------------------------------------------------------------------
# Individual snippet renderers
# ---------------------------------------------------------------------------


def _snippet_steps(item: dict[str, Any], key: str) -> str:
    lines = [f"[bold]{item.get('name', '?')}[/bold]"]
    for s in item.get("steps", []):
        label = s.get(key, "?")
        params = s.get("params", {})
        if params:
            p_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"  {label}({p_str})")
        else:
            lines.append(f"  {label}")
    return "\n".join(lines)


def _snippet_dataset(item: dict[str, Any]) -> str:
    name = item.get("name", "?")
    fmt = item.get("format", "?")
    path = item.get("path", "?")
    split = item.get("split", "")
    lines = [f"[bold]{name}[/bold]", f"  format: {fmt}  path: {path}"]
    if split:
        lines.append(f"  split: {split}")
    return "\n".join(lines)


def _snippet_source(item: dict[str, Any]) -> str:
    name = item.get("name", "?")
    dataset = item.get("dataset", "?")
    selection = item.get("selection", "")
    lines = [f"[bold]{name}[/bold]", f"  dataset: {dataset}"]
    if selection:
        lines.append(f"  selection: {selection}")
    return "\n".join(lines)


def _snippet_extractor(item: dict[str, Any]) -> str:
    name = item.get("name", "?")
    model_type = item.get("model", "?")
    lines = [f"[bold]{name}[/bold]  [dim]{model_type}[/dim]"]
    if item.get("model_path"):
        lines.append(f"  path: {item['model_path']}")
    if model_type == "bovw" and item.get("vocab_size"):
        lines.append(f"  vocab_size: {item['vocab_size']}")
    pre = item.get("preprocessor", "")
    if pre:
        lines.append(f"  preprocessor: {pre}")
    batch = item.get("batch_size", "")
    if batch:
        lines.append(f"  batch_size: {batch}")
    return "\n".join(lines)


def _snippet_workflow(item: dict[str, Any]) -> str:
    name = item.get("name", "?")
    wf_type = item.get("type", "?")
    lines = [f"[bold]{name}[/bold]  [dim]{wf_type}[/dim]"]
    extras = [f"  {k}: {v}" for k, v in item.items() if k not in ("name", "type") and v]
    lines.extend(extras)
    return "\n".join(lines)


def _snippet_task(item: dict[str, Any]) -> str:
    enabled = item.get("enabled", True)
    check = "[bold green]\u2713[/bold green]" if enabled else "[dim]\u2717[/dim]"
    name = item.get("name", "?")
    wf = item.get("workflow", "?")
    srcs = item.get("sources", "")
    if isinstance(srcs, list):
        srcs = ", ".join(srcs)
    lines = [f"{check} [bold]{name}[/bold]"]
    lines.append(f"    workflow: {wf}  sources: {srcs}")
    extractor = item.get("extractor", "")
    if extractor:
        lines.append(f"    extractor: {extractor}")
    text = "\n".join(lines)
    if not enabled:
        text = f"{check} [dim strikethrough][bold]{name}[/bold]\n    workflow: {wf}  sources: {srcs}"
        if extractor:
            text += f"\n    extractor: {extractor}"
        text += "[/dim strikethrough]"
    return text


# ---------------------------------------------------------------------------
# Dispatch table and entry point
# ---------------------------------------------------------------------------

_SNIPPET_RENDERERS: dict[str, Any] = {
    "datasets": _snippet_dataset,
    "preprocessors": lambda item: _snippet_steps(item, "step"),
    "selections": lambda item: _snippet_steps(item, "type"),
    "sources": _snippet_source,
    "extractors": _snippet_extractor,
    "workflows": _snippet_workflow,
    "tasks": _snippet_task,
}


def _item_to_yaml_snippet(category: str, item: dict[str, Any]) -> str:
    renderer = _SNIPPET_RENDERERS.get(category)
    if renderer:
        return renderer(item)
    return yaml.dump(item, default_flow_style=True).strip()


# ---------------------------------------------------------------------------
# Execution-aware task snippet (for dashboard task pane)
# ---------------------------------------------------------------------------

_STATUS_INDICATORS: dict[str, str] = {
    "idle": "[dim]\u25cf[/dim]",
    "running": "[bold yellow]\u25d0[/bold yellow] [yellow]running...[/yellow]",
    "completed": "[bold green]\u2713[/bold green]",
    "failed": "[bold red]\u2717[/bold red] [red]failed[/red]",
}


def snippet_task_with_execution(task: dict[str, Any], execution: TaskExecution | None = None) -> str:
    """Render a task card snippet with execution status indicator.

    Used in the dashboard task pane where status is shown inline.
    """
    enabled = task.get("enabled", True)
    check = "[bold green]\u2713[/bold green]" if enabled else "[dim]\u2717[/dim]"
    name = task.get("name", "?")
    wf = task.get("workflow", "?")
    srcs = task.get("sources", "")
    if isinstance(srcs, list):
        srcs = ", ".join(srcs)

    # Status indicator
    if execution is not None:
        status = _STATUS_INDICATORS.get(execution.status, _STATUS_INDICATORS["idle"])
        if execution.status == "completed" and execution.elapsed_s is not None:
            status += f" [dim]{execution.elapsed_s:.1f}s[/dim]"
    else:
        status = _STATUS_INDICATORS["idle"]

    # Build the snippet
    line1 = f"{check} [bold]{name}[/bold]  {status}"
    line2 = f"    {wf} > {srcs}"
    extractor = task.get("extractor", "")
    if extractor:
        line2 += f"  [dim]({extractor})[/dim]"

    if not enabled:
        return f"{check} [dim strikethrough]{name}  {wf} > {srcs}[/dim strikethrough]  {status}"
    return f"{line1}\n{line2}"


def snippet_config_item(category: str, item: dict[str, Any]) -> str:
    """Compact one-line snippet for config sidebar items."""
    name = item.get("name", "?")
    if category == "datasets":
        fmt = item.get("format", "")
        return f"[bold]{name}[/bold] [dim]{fmt}[/dim]"
    if category == "extractors":
        model = item.get("model", "")
        return f"[bold]{name}[/bold] [dim]{model}[/dim]"
    if category == "workflows":
        wf_type = item.get("type", "")
        return f"[bold]{name}[/bold] [dim]{wf_type}[/dim]"
    if category == "sources":
        dataset = item.get("dataset", "")
        return f"[bold]{name}[/bold] [dim]({dataset})[/dim]"
    if category in ("preprocessors", "selections"):
        n_steps = len(item.get("steps", []))
        return f"[bold]{name}[/bold] [dim]{n_steps} step{'s' if n_steps != 1 else ''}[/dim]"
    return f"[bold]{name}[/bold]"
