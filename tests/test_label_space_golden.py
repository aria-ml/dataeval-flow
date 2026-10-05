"""`label-space` agrees with what legacy data-coverage made with `ontology:` set: each finding's severity, title,
brief and description, in order, and what they were computed from (coverage spec §8.1).

Deliberate differences from its legacy run (step-chaining spec §10.3 item 3), each with its reason:

- **An ontology that fails to load fails the steps**, and so the task, with the loader's message, where legacy made an
  info "Ontology Analysis: skipped" finding; so does a dataset with no labels. The ontology is this preset's whole
  input, so a skip would hide a broken config.
- **Class names are the `index2label` values in index order,** where legacy took the class distribution's keys,
  observed then unseen. The correspondences, the unmatched list, the ambiguous mapping and "Dropped:" can come in
  another order; the stanza is sorted, and the digest does not depend on the order. An observed label missing from
  `index2label` is judged under the loader's placeholder `UNDEFINED_CLASS_<index>`, where legacy used `str(index)`.
  The golden's datasets declare every label, in index order, so they agree.
- **The digest's precedence is reversed** (coverage spec §3.5): on a source whose view applies a `Relabel`, the
  source's record wins over the alignment's, where legacy's own stamp won.
- **No "Class Shortfall" without an ontology**: that finding is data-coverage's.
- **The ignored-entries note says `expected`,** the preset's field, where legacy said `ontology_expected`.
- **Names follow the naming pass** (naming spec §3.2): recorded titles are read through `tests/golden/_renames.py`.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from tests.golden._renames import title
from tests.golden.label_space import CASES, pipeline
from tests.golden.rerouting import approximately

_GOLDEN: dict[str, Any] = json.loads((Path(__file__).parent / "golden" / "label_space.json").read_text("utf-8"))


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


def _sorted_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
    """The alignment with its name-ordered lists sorted, the order being a deliberate difference."""
    return alignment | {
        "correspondences": sorted(alignment["correspondences"], key=lambda c: (c["source"], c["target_label"])),
        "unaligned_source": sorted(alignment["unaligned_source"]),
        "unaligned_target": sorted(alignment["unaligned_target"]),
    }


@pytest.mark.parametrize("name", sorted(CASES))
def test_label_space_gives_what_legacy_data_coverage_gave(name: str) -> None:
    result = run_tasks(pipeline(name, legacy=False))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    golden = _GOLDEN[name]
    assert [[f.severity, f.title, f.brief, f.description] for f in result.findings] == [
        [s, title(t), *rest] for s, t, *rest in golden["findings"]
    ]
    representation = result.steps["representation"].output
    assert float(representation.leaf_coverage) == pytest.approx(golden["leaf_coverage"])
    assert int(representation.total_deficit) == golden["total_deficit"]
    assert representation.data().to_dicts() == approximately(golden["worklist"])
    assert representation.dark_branches.to_dicts() == approximately(golden["dark_branches"])
    assert representation.violations.to_dicts() == approximately(golden["violations"])
    assert representation.ignored_expected == golden["ignored_expected"]
    assert result.steps["label-reconciliation"].output.data() == golden["conformance"]
    alignment = result.steps["label-alignment"].output.alignment.model_dump(mode="json")
    assert _sorted_alignment(alignment) == approximately(_sorted_alignment(golden["alignment"]))
    assert result.steps["ontology-validation"].output.data() == golden["structure"]
