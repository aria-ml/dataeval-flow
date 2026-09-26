"""Tests for _screens._detail: _colorize_marker, ResultDetailModal, ErrorDetailModal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest
from rich.markup import render
from textual.widgets import Button, DataTable, Static

from dataeval_flow._app._screens._detail import (
    ErrorDetailModal,
    ResultDetailModal,
    _BlockTable,
    _BlockText,
    _colorize_marker,
    _FindingHeader,
)
from dataeval_flow._blocks import Column, Paragraph, Section, Table
from dataeval_flow._blocks._text import MIN_WIDTH

from .conftest import _MinimalApp, _wait_for_result

pytestmark = pytest.mark.optional

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
        assert render(_colorize_marker(line)).plain == line

    def test_brackets_in_the_text_show_as_written(self) -> None:
        """A class name is the user's own: a bracket in it is text, never markup."""
        line = "worst: [person], [/x]  [!!]"
        result = _colorize_marker(line)
        assert "[bold red]" in result
        assert render(result).plain == line


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

    async def test_a_bracketed_task_name_and_error_show_as_written(self) -> None:
        app = _MinimalApp()
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ErrorDetailModal("[/t] task", "KeyError: [/x] [person]")
            app.push_screen(modal)
            await pilot.pause()
            assert str(modal.query_one("#ed-title", Static).render()) == "FAILED: [/t] task"
            assert str(modal.query_one("#ed-error", Static).render()) == "KeyError: [/x] [person]"

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
    description: str = ""
    brief: str | None = None
    blocks: list[Any] = field(default_factory=list)

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

    # output.report.findings
    report = MagicMock()
    report.findings = findings or []
    report.summary = "Test report summary"
    result.output.report = report

    return result


class TestResultDetailModal:
    def _mock_result(self, findings: list[_FakeFinding] | None = None) -> MagicMock:
        if findings is None:
            findings = [
                _FakeFinding(title="Dup Check", severity="ok"),
                _FakeFinding(title="Coverage", severity="warning"),
            ]
        return _make_fake_result(findings=findings)

    def test_init_stores_fields(self) -> None:
        mock_result = self._mock_result()
        modal = ResultDetailModal("task_a", mock_result)
        assert modal._task_name == "task_a"
        assert modal._result is mock_result
        assert modal._expanded_findings == set()
        assert modal._gen == 0

    async def test_bracketed_user_strings_show_as_written(self) -> None:
        """Task names, sources, finding titles and briefs are the user's own: brackets in them are text."""
        app = _MinimalApp()
        finding = _FakeFinding(title="[/y] Balance", severity="warning", brief="worst: [person]")
        result = _make_fake_result(findings=[finding])
        result.metadata.source_descriptions = ["[/s] source"]
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("[/t] task", result)
            app.push_screen(modal)
            await pilot.pause()
            drawn = [str(widget.render()) for widget in modal.query(Static)]
            assert "Result: [/t] task" in drawn
            assert any("[/s] source" in text for text in drawn)
            assert any("[/y] Balance" in text and "worst: [person]" in text for text in drawn)
            assert any(text.startswith("\u25b6 DETAIL: [/y] Balance") for text in drawn)

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
        findings = [_FakeFinding(title="Issue", severity="warning")]
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

    async def _expanded(self, pilot: Any, app: Any, finding: _FakeFinding) -> ResultDetailModal:
        modal = ResultDetailModal("task", self._mock_result(findings=[finding]))
        app.push_screen(modal)
        await pilot.pause()
        modal._toggle_finding(0)
        await pilot.pause()
        return modal

    async def test_expand_finding_with_table_data(self) -> None:
        """A data table shows as a native DataTable, every cell printed as the report prints it."""
        app = _MinimalApp()
        counts = Table(
            columns=[Column(key="label", header="Label"), Column(key="pct", header="%", format="{:.1f}%")],
            rows=[{"label": "a", "pct": 62.5}, {"label": "b", "pct": 37.5}],
        )
        async with app.run_test(size=(120, 40)) as pilot:
            modal = await self._expanded(pilot, app, _FakeFinding(title="Counts", severity="ok", blocks=[counts]))
            (table,) = modal.query(DataTable)
            assert [str(column.label) for column in table.columns.values()] == ["Label", "%"]
            assert [str(cell) for cell in table.get_row_at(0)] == ["a", "62.5%"]
            assert table.row_count == 2

    async def test_bracketed_names_in_a_data_table_show_as_written(self) -> None:
        """Class and split names are the user's own: a bracket in one is text, never markup."""
        app = _MinimalApp()
        names = Table(
            columns=[Column(key="class", header="Class"), Column(key="split:[train]", header="[train]")],
            rows=[{"class": "[person]", "split:[train]": "[/x] 3"}],
        )
        async with app.run_test(size=(120, 40)) as pilot:
            modal = await self._expanded(pilot, app, _FakeFinding(title="Names", severity="info", blocks=[names]))
            table = modal.query_one(DataTable)
            assert [str(column.label) for column in table.columns.values()] == ["Class", "[train]"]
            assert [str(cell) for cell in table.get_row_at(0)] == ["[person]", "[/x] 3"]

    async def test_a_multi_line_cell_shows_every_line(self) -> None:
        """A cell holding one factor per line gets a row tall enough for all of them."""
        app = _MinimalApp()
        factors = Table(
            columns=[Column(key="split", header="Split"), Column(key="mi", header="Top High MI Factors")],
            rows=[{"split": "train", "mi": "altitude\nweather\ntime"}],
        )
        finding = _FakeFinding(title="Bias", severity="info", blocks=[factors])
        async with app.run_test(size=(120, 40)) as pilot:
            modal = ResultDetailModal("task_bias", self._mock_result(findings=[finding]))
            app.push_screen(modal)
            await pilot.pause()
            modal._toggle_finding(0)
            await pilot.pause()
            table = modal.query_one(DataTable)
            (row,) = table.ordered_rows
            assert row.height == 3

    async def test_a_description_shows_above_a_data_table(self) -> None:
        app = _MinimalApp()
        counts = Table(columns=[Column(key="k", header="K")], rows=[{"k": "a"}])
        finding = _FakeFinding(title="Counts", severity="ok", description="Two labels.", blocks=[counts])
        async with app.run_test(size=(120, 40)) as pilot:
            modal = await self._expanded(pilot, app, finding)
            detail = [widget for widget in modal.query("#rd-scroll > *") if isinstance(widget, (_BlockText, DataTable))]
            assert [type(widget) for widget in detail] == [_BlockText, _BlockTable]
            assert isinstance(detail[0], _BlockText)
            assert detail[0].text == "Two labels."

    async def test_expanded_text_fits_the_modal_and_follows_a_resize(self) -> None:
        app = _MinimalApp()
        prose = " ".join(["calibrated"] * 40)
        finding = _FakeFinding(title="Prose", severity="info", blocks=[Paragraph(text=prose)])
        async with app.run_test(size=(120, 40)) as pilot:
            modal = await self._expanded(pilot, app, finding)
            (text,) = modal.query(_BlockText)
            wide = text.text.splitlines()
            assert max(len(line) for line in wide) <= text.content_size.width
            await pilot.resize_terminal(80, 40)
            await pilot.pause()
            narrow = text.text.splitlines()
            assert len(narrow) > len(wide)
            assert max(len(line) for line in narrow) <= text.content_size.width

    async def test_tables_with_the_same_columns_line_up_across_a_finding(self) -> None:
        """Tables read against each other, such as one per factor, share column widths as the text report's do."""

        def buckets(names: list[str]) -> Table:
            columns = [Column(key="bin", header="Bin"), Column(key="n", header="Count"), Column(key="n", kind="bar")]
            return Table(columns=columns, rows=[{"bin": name, "n": float(i + 1)} for i, name in enumerate(names)])

        app = _MinimalApp()
        factors = [
            Section(title="altitude", blocks=[buckets(["low", "high"])]),
            Section(title="weather", blocks=[buckets(["partly cloudy", "rain"])]),
        ]
        async with app.run_test(size=(120, 40)) as pilot:
            modal = await self._expanded(pilot, app, _FakeFinding(title="Buckets", severity="info", blocks=factors))
            (text,) = modal.query(_BlockText)
            headers = [line for line in text.text.splitlines() if "Bin" in line and "Count" in line]
            assert len(headers) == 2
            assert len({line.index("Count") for line in headers}) == 1

    async def test_a_modal_narrower_than_the_report_s_minimum_still_draws_its_text(self) -> None:
        """Below 40 columns the text draws at 40, one row per line, and the modal scrolls it sideways."""
        app = _MinimalApp()
        finding = _FakeFinding(title="Prose", severity="info", blocks=[Paragraph(text=" ".join(["word"] * 30))])
        async with app.run_test(size=(40, 30)) as pilot:
            modal = await self._expanded(pilot, app, finding)
            (text,) = modal.query(_BlockText)
            lines = text.text.splitlines()
            assert max(len(line) for line in lines) <= MIN_WIDTH
            assert text.content_size.width == MIN_WIDTH
            assert text.size.height == len(lines)
            assert modal.query_one("#rd-scroll").max_scroll_x > 0

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
