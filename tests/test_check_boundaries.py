"""A check's bound is the last value that does not warn: a finding warns only past it (naming spec §4.2)."""

from types import SimpleNamespace
from typing import Any

import polars as pl

from dataeval_flow.steps import CheckContext
from dataeval_flow.steps.checks import DriftCheck, DriftConfig, FactorCoverageGapsCheck, FactorCoverageGapsConfig
from dataeval_flow.steps.checks._ood import OODThresholds, ood_severity
from dataeval_flow.steps.combines import FactorGap, FactorGapsOutput

_GAP = FactorGap(
    class_name="dog", factor_name="site", factor_value="site-0", class_count=1, expected_count=13.5, deficit=0.926
)


def test_a_flagged_share_equal_to_warning_informs_and_one_past_it_warns() -> None:
    bands = OODThresholds(warning=10.0, info=1.0)
    assert ood_severity(10.0, bands) == "info"
    assert ood_severity(10.1, bands) == "warning"


def test_a_flagged_share_equal_to_info_is_ok_and_one_past_it_informs() -> None:
    bands = OODThresholds(warning=10.0, info=1.0)
    assert ood_severity(1.0, bands) == "ok"
    assert ood_severity(1.1, bands) == "info"


def _drift(flags: list[bool], **settings: Any) -> str:
    output = SimpleNamespace(details=pl.DataFrame({"drifted": flags}), drifted=any(flags))
    node = SimpleNamespace(value=output, config=None)
    (finding,) = DriftCheck().run(DriftConfig(input="knn", **settings), {"input": node}, None)  # type: ignore[arg-type]
    return finding.severity


def test_a_drifted_share_equal_to_chunk_percent_does_not_warn() -> None:
    # 1 of 10 chunks is 10.0%, the bound itself; 2 of 10 is past it
    assert _drift([True] + [False] * 9, chunk_percent=10.0, consecutive_chunks=None) == "info"
    assert _drift([True, True] + [False] * 8, chunk_percent=10.0, consecutive_chunks=None) == "warning"


def test_a_drifted_run_equal_to_consecutive_chunks_does_not_warn() -> None:
    assert _drift([True, True, False, False], chunk_percent=None, consecutive_chunks=2) == "info"
    assert _drift([True, True, True, False], chunk_percent=None, consecutive_chunks=2) == "warning"


def test_consecutive_chunks_defaults_to_two_so_three_in_a_row_still_warns() -> None:
    assert DriftConfig(input="knn").consecutive_chunks == 2
    assert _drift([True, True, True, False], chunk_percent=None) == "warning"


def _gaps(n: int, **settings: Any) -> str:
    node = SimpleNamespace(value=FactorGapsOutput(mutual_information={"site": 0.5}, gaps=[_GAP] * n))
    config = FactorCoverageGapsConfig(input="gaps", **settings)
    (finding,) = FactorCoverageGapsCheck().run(config, {"input": node}, CheckContext("t", "s"))
    return finding.severity


def test_gaps_equal_to_warning_inform_and_one_more_warns() -> None:
    assert _gaps(2, warning=2) == "info"
    assert _gaps(3, warning=2) == "warning"


def test_factor_coverage_gaps_defaults_to_two_so_three_gaps_still_warn() -> None:
    assert FactorCoverageGapsConfig(input="gaps").warning == 2
    assert _gaps(3) == "warning"
