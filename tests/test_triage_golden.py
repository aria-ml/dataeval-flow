"""metadata-triage agrees with what it produced before its port: its findings' severity, title and brief, in order;
its suggested policy stanza; and the binning record on its envelope (spec §10.10).

Deliberate differences from its legacy run (spec §10.3 item 3), each with its reason:

- **The envelope's `blocking` and `verified` counts are gone.** The findings, the chain's `warning_count`, and the
  `factor-triage` Output's `counts` and `verification` say the same.
- **The result no longer carries the `dataset` it read.** A chain's steps hold the Datasets they read and made.
- **The `metadata_*` fields are gone.** A config names a policy under `metadata:` instead, as data-cleaning's does.
- **It names its items by the chain's input, `data`, not by the source** (spec §7.4).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from tests.golden.triage import CASES, pipeline

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "triage_findings.json").read_text())


def _canonical(value: Any) -> str:
    """`value` as sorted JSON: a binning record can hold NaN, which never equals itself as a float."""
    return json.dumps(value, sort_keys=True)


def _produced(result: ChainResult) -> dict[str, Any]:
    """The preset's findings, its `triage` step's stanza, and its envelope's binning record."""
    return {
        "findings": [[f.severity, f.title, f.brief] for f in result.findings],
        "suggested_policy_yaml": result.steps["triage"].output.data()["suggested_policy_yaml"],
        "metadata_binning": result.metadata.metadata_binning,
    }


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_metadata_triage_gives_what_it_gave_before_its_port(name: str) -> None:
    result = run_tasks(pipeline(name))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert _canonical(_produced(result)) == _canonical(_GOLDEN[name])
