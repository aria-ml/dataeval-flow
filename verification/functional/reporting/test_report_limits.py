"""TC-12-3 — table row limits and thumbnails in the text, HTML and JSON reports."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from dataeval_flow.steps import ChainResult
from verification.functional.reporting._project import Invocation, run_project, write_project

pytestmark = pytest.mark.required

GROUPS = 6  # exact-duplicate groups planted in every dataset below, so the duplicates table has six rows


def _result(root: Path, **result: int) -> ChainResult:
    return run_project(root, duplicates=GROUPS, per_class=10, **result)["clean_task"]


@pytest.fixture(scope="module")
def defaults(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    return _result(tmp_path_factory.mktemp("defaults"))


@pytest.fixture(scope="module")
def limited(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    return _result(tmp_path_factory.mktemp("limited"), max_rows=3, preview_rows=2)


@pytest.fixture(scope="module")
def unlimited(tmp_path_factory: pytest.TempPathFactory) -> ChainResult:
    return _result(tmp_path_factory.mktemp("unlimited"), max_rows=-1, preview_rows=-1)


def _group_rows(text: str) -> list[str]:
    """The rows of the duplicates table in a text report: ``<group> exact <count> <items>``."""
    return re.findall(r"^\s+(\d+)\s+exact\s+2\s+\d+, \d+$", text, flags=re.MULTILINE)


def _json_rows(result: ChainResult) -> list[dict]:
    output = result.to_dict()["steps"]["duplicates"]["output"]  # type: ignore[index]
    return output["rows"]


class TestRowLimits:
    def test_defaults_list_every_row_of_a_short_table(self, defaults: ChainResult) -> None:
        text = defaults.report()
        assert _group_rows(text) == [str(i) for i in range(GROUPS)]
        assert "more row" not in text

    def test_preview_rows_cuts_the_text_report_and_counts_the_rest(self, limited: ChainResult) -> None:
        text = limited.report()
        assert _group_rows(text) == ["0", "1"]  # two rows previewed
        assert "… 1 more row, in the HTML and JSON reports" in text  # of the three max_rows lists

    def test_max_rows_cuts_the_table_and_a_paragraph_says_where_the_rest_is(self, limited: ChainResult) -> None:
        text = " ".join(limited.report().split())
        assert "6 groups of images; the 3 largest are listed, and every one is in `output.rows`." in text

    def test_json_keeps_every_row_whatever_the_limits(self, limited: ChainResult, unlimited: ChainResult) -> None:
        assert len(_json_rows(limited)) == len(_json_rows(unlimited)) == GROUPS

    def test_minus_one_lists_every_row(self, unlimited: ChainResult) -> None:
        text = unlimited.report()
        assert _group_rows(text) == [str(i) for i in range(GROUPS)]
        assert "more row" not in text
        assert "largest are listed" not in " ".join(text.split())

    def test_html_shows_every_listed_row_whatever_preview_rows_says(
        self, limited: ChainResult, unlimited: ChainResult
    ) -> None:
        # Each duplicate group row carries a thumbnail per item: three rows of two, then six.
        assert len(limited.assets) == 3 * 2
        assert limited.to_html().count("<img") == 3 * 2
        assert len(unlimited.assets) == GROUPS * 2
        assert unlimited.to_html().count("<img") == GROUPS * 2


class TestThumbnails:
    def test_the_report_names_items_and_their_thumbnails_are_kept_in_the_result(self, defaults: ChainResult) -> None:
        assert len(defaults.assets) == GROUPS * 2
        assets = defaults.to_dict()["assets"]
        assert len(assets) == GROUPS * 2  # type: ignore[arg-type]
        first = assets[0]  # type: ignore[index]
        assert {"item", "media_type", "width", "height", "data"} <= set(first)
        assert first["media_type"] == "image/webp"
        assert base64.b64decode(first["data"])[:4] == b"RIFF"
        named = {asset.item.index for asset in defaults.assets}
        assert named == {
            int(i) for pair in re.findall(r"(\d+), (\d+)$", defaults.report(), flags=re.MULTILINE) for i in pair
        }

    def test_html_is_one_self_contained_page(self, defaults: ChainResult) -> None:
        page = defaults.to_html()
        assert page.lstrip().lower().startswith("<!doctype html")
        assert page.count("data:image/webp;base64,") == len(defaults.assets)
        assert "http://" not in page
        assert "https://" not in page
        assert "<link" not in page
        assert not re.search(r"<script[^>]+src=", page)
        assert not re.search(r"""(?:src|href)=["'](?!data:|#)""", page)

    def test_summary_html_is_shorter_than_detailed_html(self, defaults: ChainResult) -> None:
        assert len(defaults.to_html(detailed=False)) < len(defaults.to_html())

    def test_max_images_caps_the_thumbnails(self, tmp_path: Path) -> None:
        capped = _result(tmp_path, max_images=5)
        assert len(capped.assets) == 5
        assert capped.to_html().count("<img") == 5
        assert len(capped.to_dict()["assets"]) == 5  # type: ignore[arg-type]

    def test_zero_max_images_keeps_none(self, tmp_path: Path) -> None:
        bare = _result(tmp_path, max_images=0)
        assert bare.assets == []
        assert "assets" not in bare.to_dict()
        assert "<img" not in bare.to_html()
        assert len(_group_rows(bare.report())) == GROUPS  # the report still names the items

    def test_minus_one_max_images_keeps_a_thumbnail_for_every_named_item(self, tmp_path: Path) -> None:
        every = _result(tmp_path, max_images=-1, max_rows=-1)
        assert len(every.assets) == GROUPS * 2

    def test_no_report_images_flag_and_variable_turn_thumbnails_off(
        self, tmp_path: Path, cli: Callable[..., Invocation]
    ) -> None:
        config = write_project(tmp_path, duplicates=GROUPS, per_class=10)
        out = tmp_path / "out"
        results = out / "results"

        assert cli("-c", config, "-d", tmp_path, "-o", out).code == 0
        assert "assets" in json.loads((results / "result.json").read_text())["clean_task"]
        assert (results / "result.html").read_text().count("<img") == GROUPS * 2

        assert cli("-c", config, "-d", tmp_path, "-o", out, "--no-report-images").code == 0
        assert "assets" not in json.loads((results / "result.json").read_text())["clean_task"]
        assert "<img" not in (results / "result.html").read_text()

        assert cli("-c", config, "-d", tmp_path, "-o", out, env={"DATAEVAL_REPORT_IMAGES": "false"}).code == 0
        assert "<img" not in (results / "result.html").read_text()

        flagged = cli("-c", config, "-d", tmp_path, "-o", out, "--report-images", env={"DATAEVAL_REPORT_IMAGES": "0"})
        assert flagged.code == 0
        assert (results / "result.html").read_text().count("<img") == GROUPS * 2
