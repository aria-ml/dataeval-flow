"""Recorded `audit.json` from the data-analysis workflow, before audit replaced it. It now refuses to run.

Commit 6fe9f57 ran it once, on data-analysis, to record each case's per-split outlier, duplicate, class and
empty-image counts, the cross-split duplicate groups and divergence, and train's factor summary. audit must agree with
them (audit spec §12.2).

audit's commit deleted data-analysis, so a `type: data-analysis` entry is now refused. Run now, this could record
nothing; recording audit's own output instead would make the agreement test compare audit with itself.
`tests/test_audit_golden.py` lists audit's deliberate differences from data-analysis.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_audit records from the data-analysis workflow, which audit replaced and deleted: run now, it could "
        "record nothing, and recording audit's own output would make the agreement test compare audit with itself."
    )
