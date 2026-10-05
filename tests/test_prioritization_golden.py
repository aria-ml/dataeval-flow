"""data-prioritization agrees with what it produced before its port: each pool's ranking, as indices into the pool,
and how many items cleaning removed from each source (spec §10.9).

Deliberate differences from its legacy run (spec §10.3 item 3), each with its reason:

- **It makes no findings.** Legacy made a "Pruning" finding, a warning when cleaning removed more than
  `cleaning_removed_pct_warning` of all items, and an info finding per pool. Ranking judges nothing, and the chain has
  no checks. A removal-rate check can bring Pruning back once `remove` exposes its counts to checks. The `rank` step's
  section lists what the info findings showed.
- **Its ranked tables number items within the Dataset ranked.** With cleaning, that is the cleaned pool, not the
  pool: every step's section names items by the node it read (spec §7.4).
- **`prioritization` reads the pool's labels.** Legacy passed Prioritize no `class_labels`; the evaluator passes the
  pool's, so `policy: class_balanced`, which legacy always refused ("class_labels not provided"), ranks by them
  wherever a pool has labels.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from tests.golden.prioritization import CASES, pipeline

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "prioritization_rankings.json").read_text())


def _untied(produced: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """`produced` with each pool's item 5 read as its item 0.

    A toy pool's item 5 is a copy of its item 0 (`ToyImages` plants it), so the two score the same and their order
    between each other is a tie. DataEval breaks it with numpy's sort, which breaks ties differently across numpy
    versions: the lowest-dependency run's numpy 1.24 puts 5 first. Cleaning removes item 5, so only the uncleaned
    cases hold both.
    """
    rankings = {
        pool: [0 if index == 5 else index for index in ranking] for pool, ranking in produced["rankings"].items()
    }
    return {**produced, "rankings": rankings}


def _produced(result: ChainResult) -> dict[str, dict[str, Any]]:
    """Each pool's selection, as indices into the pool, and how many items cleaning removed from each source.

    `View.resolve_indices()` counts within the view's own parent, so with cleaning, a selection's indices are mapped
    through `pool-clean`'s to reach the pool.
    """
    steps = result.steps
    selected = steps["selected"].elements or {}
    if "pool-clean" not in steps:
        return {
            "rankings": {
                pool: [int(i) for i in element.output.resolve_indices()] for pool, element in selected.items()
            },
            "removed": {"ref": 0, **dict.fromkeys(selected, 0)},
        }
    cleaned = steps["pool-clean"].elements or {}
    kept = {pool: element.output.resolve_indices() for pool, element in cleaned.items()}
    reference = steps["reference-clean"].details or {}
    return {
        "rankings": {
            pool: [int(kept[pool][i]) for i in element.output.resolve_indices()] for pool, element in selected.items()
        },
        "removed": {
            "ref": reference["removed"]["items"],
            **{pool: (element.details or {})["removed"]["items"] for pool, element in cleaned.items()},
        },
    }


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_data_prioritization_gives_the_rankings_it_gave_before_its_port(name: str) -> None:
    result = run_tasks(pipeline(name))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert _untied(_produced(result)) == _untied(_GOLDEN[name])
