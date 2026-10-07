"""Recorded `triage_findings.json` from the legacy metadata-triage workflow, before its port to a preset. It now refuses
to run.

Commit 7a1f238 ran it once, on the legacy workflow, to record each case's findings, suggested stanza and binning record.
The preset must agree with them (spec §10.10).

The port deleted the legacy workflow, so `triage` now names the preset. Run now, this would record the
preset's own output as the golden the preset is tested against, and the agreement test would only compare the preset
with itself. `tests/test_triage_golden.py` lists the preset's deliberate differences from the legacy run.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_triage records from the legacy metadata-triage workflow, which the port to a preset deleted: run "
        "now, it would record the preset's own output, and the agreement test would compare the preset with itself."
    )
