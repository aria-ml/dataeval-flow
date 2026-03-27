"""Config sidebar pane — rebuild, section update, and focus management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static

from dataeval_flow._app._panes._widgets import (
    _CONFIG_SECTIONS,
    SECTION_TITLES,
    CfgItem,
    CfgSectionHeader,
    PaneWidget,
    uid,
)
from dataeval_flow._app._viewmodel._rendering import snippet_config_item

if TYPE_CHECKING:
    from textual.app import App
    from textual.widget import Widget

    from dataeval_flow._app._viewmodel._builder_vm import BuilderViewModel
else:
    App = object


class ConfigPaneMixin(App):
    """Mixin providing config sidebar rebuild logic for FlowApp.

    Expects the host class to also inherit from ``App`` and to provide
    ``_vm`` and ``_get_pane_widgets`` (both supplied by ``FlowApp``).
    """

    if TYPE_CHECKING:
        _vm: BuilderViewModel

        def _get_pane_widgets(self, pane_id: str) -> list[PaneWidget]: ...

    def _rebuild_config_pane(self) -> None:
        """Full rebuild of the config sidebar."""
        try:
            container = cast(VerticalScroll, self.query_one("#config-pane", VerticalScroll))
        except NoMatches:
            return

        focused = self.focused
        restore = self._save_config_focus(focused)

        if restore[0] is not None:
            self.set_focus(None)

        container.remove_children()
        for category, _ in _CONFIG_SECTIONS:
            self._mount_config_section(container, category)

        if restore[0] is not None:
            self.call_after_refresh(self._restore_config_focus, *restore)

    def _rebuild_config_section(self, category: str) -> None:
        """Rebuild a single config section in place."""
        try:
            container = cast(VerticalScroll, self.query_one("#config-pane", VerticalScroll))
        except NoMatches:
            return

        focused = self.focused
        restore = self._save_config_focus(focused)

        if restore[0] == category:
            self.set_focus(None)

        insert_before = self._remove_section_widgets(container, category)
        self._mount_config_section(container, category, before=insert_before)

        if restore[0] == category:
            self.call_after_refresh(self._restore_config_focus, *restore)

    def _remove_section_widgets(self, container: VerticalScroll, category: str) -> PaneWidget | None:
        """Remove all widgets for *category* and return the next section's header (or None)."""
        in_section = False
        insert_before: PaneWidget | None = None
        to_remove: list[Widget] = []
        for child in list(container.children):
            if isinstance(child, CfgSectionHeader) and child.fc_category == category:
                in_section = True
                to_remove.append(child)
            elif in_section:
                if isinstance(child, CfgSectionHeader):
                    insert_before = child
                    break
                to_remove.append(child)
        for w in to_remove:
            w.remove()
        return insert_before

    def _mount_config_section(self, container: VerticalScroll, category: str, before: PaneWidget | None = None) -> None:
        """Mount header + items for one config section."""
        title = SECTION_TITLES.get(category, category)
        items = self._vm.items(category)

        widgets: list[Static] = []
        widgets.append(
            CfgSectionHeader(
                f"[bold]{title}[/bold] [dim]({len(items)})[/dim]",
                category=category,
                classes="cfg-section-header",
                id=uid("ch"),
                markup=True,
            )
        )
        if not items:
            widgets.append(Static("[dim]  (empty)[/dim]"))
        else:
            for idx, item in enumerate(items):
                snippet = snippet_config_item(category, item)
                widgets.append(
                    CfgItem(
                        snippet,
                        category=category,
                        index=idx,
                        classes="cfg-item",
                        id=uid("ci"),
                        markup=True,
                    )
                )

        if before is not None:
            for w in widgets:
                container.mount(w, before=before)
        else:
            for w in widgets:
                container.mount(w)

    # -- Config focus save/restore -----------------------------------------

    def _save_config_focus(self, focused: Any) -> tuple[str | None, int | None, bool]:
        """Extract focus context from the currently focused widget."""
        if isinstance(focused, CfgItem) and focused.pane == "config-pane":
            return focused.fc_category, focused.fc_index, False
        if isinstance(focused, CfgSectionHeader) and focused.pane == "config-pane":
            return focused.fc_category, None, True
        return None, None, False

    def _restore_config_focus(self, category: str | None, index: int | None, is_header: bool) -> None:
        """Restore focus by finding the widget with matching attributes."""
        if category is None:
            return
        widgets = self._get_pane_widgets("config-pane")
        if is_header or index is None:
            target = next((w for w in widgets if isinstance(w, CfgSectionHeader) and w.fc_category == category), None)
        else:
            target = next(
                (w for w in widgets if isinstance(w, CfgItem) and w.fc_category == category and w.fc_index == index),
                None,
            )
            if target is None and index > 0:
                target = next(
                    (
                        w
                        for w in reversed(widgets)
                        if isinstance(w, CfgItem) and w.fc_category == category and w.fc_index < index
                    ),
                    None,
                )
            if target is None:
                target = next(
                    (w for w in widgets if isinstance(w, CfgSectionHeader) and w.fc_category == category), None
                )
        if target is not None:
            self.set_focus(target, scroll_visible=False)
