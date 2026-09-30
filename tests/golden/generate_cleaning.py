"""Recorded `cleaning_findings.json` from the legacy data-cleaning workflow, before its port to a preset. It now refuses
to run.

Commit 8cc349f ran it once, on the legacy workflow, to write each agreement case's findings as severity, title and
brief, in order. The preset must agree with them (spec §10.3).

The port deleted the legacy workflow, so `data-cleaning` now names the preset. Run now, this would record the preset's
own findings as the golden the preset is tested against, and the agreement test would only compare the preset with
itself. `tests/test_cleaning_golden.py` lists the preset's deliberate differences from the legacy run.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_cleaning records from the legacy data-cleaning workflow, which the port to a preset deleted: run "
        "now, it would record the preset's own findings, and the agreement test would compare the preset with itself."
    )
