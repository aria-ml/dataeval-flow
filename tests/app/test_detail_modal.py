"""Tests for _screens._detail: _colorize_marker, ResultDetailModal, ErrorDetailModal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from textual.widgets import Button, Static

from dataeval_flow._app._screens._detail import (
    ErrorDetailModal,
    ResultDetailModal,
    _colorize_marker,
    _FindingHeader,
)

from .conftest import _MinimalApp, _wait_for_result

# ---------------------------------------------------------------------------
# _colorize_marker  (lines 35-40)
# ---------------------------------------------------------------------------


class TestColorizeMarker:
    """Pure-function tests for _colorize_marker."""

    def test_ok_marker(self) -> None:
        line = "All checks passed  [ok]"
        result = _colorize_marker(line)
        assert "[green]" in result
        assert result.startswith("All checks passed")

    def test_warning_marker(self) -> None:
        line = "Duplicates found  [!!]"
        result = _colorize_marker(line)
        assert "[bold red]" in result
        assert result.startswith("Duplicates found")

    def test_info_marker(self) -> None:
        line = "Statistics  [..]"
        result = _colorize_marker(line)
        assert "[blue]" in result
        assert result.startswith("Statistics")

    def test_no_marker_returns_unchanged(self) -> None:
        line = "plain text"
        assert _colorize_marker(line) == line

    def test_empty_string(self) -> None:
        assert _colorize_marker("") == ""

    def test_marker_must_be_at_end(self) -> None:
        line = "[ok] at the beginning"
        assert _colorize_marker(line) == line


# ---------------------------------------------------------------------------
# _FindingHeader  (lines 114-121)
# ---------------------------------------------------------------------------


class TestFindingHeader:
    def test_attributes(self) -> None:
        fh = _FindingHeader("header text", finding_idx=3, id="fh-test")
        assert fh.finding_idx == 3
        assert fh.can_focus is True


# ---------------------------------------------------------------------------
# ErrorDetailModal  (lines 306-333)
# ---------------------------------------------------------------------------


class TestErrorDetailModal:
    def test_init_stores_fields(self) -> None:
        modal = ErrorDetailModal("task_x", "boom")
        assert modal._task_name == "task_x"
        assert modal._error == "boom"

    async def test_compose_renders_elements(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ErrorDetailModal("task1", "Something went wrong")
            app.push_screen(modal)
            await pilot.pause()
            # Title, error body, and close button are present
            assert modal.query_one("#ed-title", Static) is not None
            assert modal.query_one("#ed-error", Static) is not None
            assert modal.query_one("#btn-ed-close", Button) is not None

    async def test_close_button_dismisses(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ErrorDetailModal("task1", "error msg")
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            await pilot.click("#btn-ed-close")
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_action_close_dismisses(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ErrorDetailModal("task1", "error msg")
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            modal.action_close()
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_on_button_pressed_non_close_ignored(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ErrorDetailModal("task1", "error msg")
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            # Fire a button press with unrelated id -- should not dismiss
            event = Button.Pressed(Button("Other", id="btn-other"))
            modal.on_button_pressed(event)
            await pilot.pause()
            assert results == []


# ---------------------------------------------------------------------------
# ResultDetailModal  (lines 124-263)
# ---------------------------------------------------------------------------


@dataclass
class _FakeFinding:
    """Minimal finding stub for ResultViewModel."""

    title: str
    severity: str
    report_type: str
    data: Any = None
    description: str = ""

    @property
    def summary(self) -> str:
        return f"{self.title} summary"


def _make_fake_result(
    findings: list[_FakeFinding] | None = None,
    execution_time_s: float = 1.5,
    timestamp: str | None = None,
    model_id: str | None = None,
    preprocessor_id: str | None = None,
) -> MagicMock:
    """Build a mock WorkflowResult that satisfies ResultViewModel."""
    result = MagicMock()

    # metadata
    meta = MagicMock()
    meta.execution_time_s = execution_time_s
    meta.timestamp = timestamp
    meta.model_id = model_id
    meta.preprocessor_id = preprocessor_id
    meta.source_descriptions = []
    result.metadata = meta

    # data.report.findings
    report = MagicMock()
    report.findings = findings or []
    report.summary = "Test report summary"
    result.data = MagicMock()
    result.data.report = report

    return result


class TestResultDetailModal:
    def _mock_result(self, findings: list[_FakeFinding] | None = None) -> MagicMock:
        if findings is None:
            findings = [
                _FakeFinding(title="Dup Check", severity="ok", report_type="scalar"),
                _FakeFinding(title="Coverage", severity="warning", report_type="scalar"),
            ]
        return _make_fake_result(findings=findings)

    def test_init_stores_fields(self) -> None:
        mock_result = self._mock_result()
        modal = ResultDetailModal("task_a", mock_result)
        assert modal._task_name == "task_a"
        assert modal._result is mock_result
        assert modal._expanded_findings == set()
        assert modal._gen == 0

    async def test_compose_renders_title_and_close(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # Title and close button are present
            assert modal.query_one("#rd-title", Static) is not None
            assert modal.query_one("#btn-rd-close", Button) is not None

    async def test_compose_shows_finding_headers(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            headers = modal.query(_FindingHeader)
            assert len(headers) == 2

    async def test_close_button_dismisses(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            await pilot.click("#btn-rd-close")
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_action_close_dismisses(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            modal.action_close()
            await _wait_for_result(pilot, results)
            assert results == [None]

    async def test_on_button_pressed_non_close_ignored(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            results: list[Any] = []
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal, callback=results.append)
            await pilot.pause()
            event = Button.Pressed(Button("Other", id="btn-other"))
            modal.on_button_pressed(event)
            await pilot.pause()
            assert results == []

    async def test_toggle_finding_expands_and_collapses(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # Initially collapsed
            assert 0 not in modal._expanded_findings
            # Expand finding 0
            modal._toggle_finding(0)
            await pilot.pause()
            assert 0 in modal._expanded_findings
            assert modal._gen == 1
            # Collapse finding 0
            modal._toggle_finding(0)
            await pilot.pause()
            assert 0 not in modal._expanded_findings
            assert modal._gen == 2

    async def test_rebuild_content_increments_gen(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            old_gen = modal._gen
            modal._rebuild_content()
            await pilot.pause()
            assert modal._gen == old_gen + 1

    async def test_compose_with_no_findings(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result(findings=[])
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_empty", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # No finding headers
            headers = modal.query(_FindingHeader)
            assert len(headers) == 0

    async def test_compose_warning_health(self) -> None:
        """Findings with warnings should produce a warning health line."""
        app = _MinimalApp()
        findings = [_FakeFinding(title="Issue", severity="warning", report_type="scalar")]
        mock_result = self._mock_result(findings=findings)
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_warn", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # The health line exists -- we just check the modal composed without error
            assert modal.query_one("#btn-rd-close", Button)

    async def test_click_finding_header_toggles(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # Click a finding header to expand — use _toggle_finding directly
            header = modal.query(_FindingHeader).first()
            assert header is not None
            idx = header.finding_idx
            modal._toggle_finding(idx)
            await pilot.pause()
            assert idx in modal._expanded_findings

    async def test_expand_finding_with_table_data(self) -> None:
        """Expanding a finding whose report_type yields table data."""
        app = _MinimalApp()
        finding = _FakeFinding(
            title="Counts",
            severity="ok",
            report_type="table",
            data={"table_data": {"a": 10, "b": 5}, "table_headers": ("Label", "Count")},
        )
        mock_result = self._mock_result(findings=[finding])
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_tbl", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # Expand finding 0 to trigger DataTable path
            modal._toggle_finding(0)
            await pilot.pause()
            assert 0 in modal._expanded_findings

    async def test_populate_tables_no_matches_safe(self) -> None:
        """_populate_tables should not crash when DataTable widget is missing."""
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # Manually add an index to expanded but do not rebuild content
            # This means no DataTable widget exists for that index
            modal._expanded_findings.add(99)
            modal._populate_tables()  # should not raise
            await pilot.pause()

    async def test_rebuild_content_no_scroll_safe(self) -> None:
        """_rebuild_content should not crash if #rd-scroll is missing."""
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # Remove the scroll container
            scroll = modal.query_one("#rd-scroll")
            scroll.remove()
            await pilot.pause()
            # Should not raise
            modal._rebuild_content()
            await pilot.pause()

    async def test_key_enter_on_finding_header(self) -> None:
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # Focus on a finding header and press enter
            header = modal.query(_FindingHeader).first()
            assert header is not None
            header.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert header.finding_idx in modal._expanded_findings

    async def test_key_enter_no_finding_header_focused(self) -> None:
        """Enter key when a non-FindingHeader widget is focused should not crash."""
        app = _MinimalApp()
        mock_result = self._mock_result()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_a", mock_result)
            app.push_screen(modal)
            await pilot.pause()
            # Focus the close button
            btn = modal.query_one("#btn-rd-close", Button)
            btn.focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
