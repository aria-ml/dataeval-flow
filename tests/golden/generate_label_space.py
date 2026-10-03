"""Record `label_space.json` from legacy data-coverage with `ontology:` set, before `label-space` exists.

Run once, on the legacy workflow: `.venv/bin/python -m tests.golden.generate_label_space`. It records each case's
ontology findings and what they were computed from: the representation, the conformance, the alignment and the
structure (coverage spec §8.1).
"""

import json
from pathlib import Path
from typing import Any

from dataeval_flow import run_tasks
from tests.golden.label_space import CASES, TITLES, pipeline


def _record(name: str) -> dict[str, Any]:
    result = run_tasks(pipeline(name, legacy=True))["t"]
    assert result.success, result.errors
    raw = result.output.raw
    onto = raw.ontology
    assert onto is not None, raw.ontology_skipped_reason
    assert not onto.synthesized
    assert onto.conformance is not None
    assert onto.alignment is not None
    assert onto.structure is not None
    rep = onto.representation
    return {
        "findings": [
            [f.severity, f.title, f.brief, f.description] for f in result.output.report.findings if f.title in TITLES
        ],
        "leaf_coverage": rep.leaf_coverage,
        "total_deficit": rep.total_deficit,
        "worklist": [row.model_dump() for row in rep.worklist],
        "dark_branches": [row.model_dump() for row in rep.dark_branches],
        "violations": [row.model_dump() for row in rep.violations],
        "ignored_expected": rep.ignored_expected,
        "conformance": {
            "conforms": onto.conformance.conforms,
            "matched": onto.conformance.matched,
            "unmatched": onto.conformance.unmatched,
            "ambiguous": onto.conformance.ambiguous,
        },
        "alignment": onto.alignment.model_dump(mode="json"),
        "structure": onto.structure.model_dump(),
    }


if __name__ == "__main__":
    golden = {name: _record(name) for name in CASES}
    path = Path(__file__).parent / "label_space.json"
    path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
