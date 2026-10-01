"""Recorded `drift.json` from the legacy drift-monitoring workflow, before its port to a preset. It now refuses to run.

Commit ec17a61 ran it once, on the legacy workflow, to record each case's finding severities, each detector's verdict,
threshold and chunks, each class row stored under its detector's key, and which detectors were classwise or chunked.
The preset must agree with all of it (spec §10.11).

The port deleted the legacy workflow, so `drift-monitoring` now names the preset. Run now, this would record the
preset's own output as the golden the preset is tested against, and the agreement test would only compare the preset
with itself. `tests/test_drift_golden.py` lists the preset's deliberate differences from the legacy run.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_drift records from the legacy drift-monitoring workflow, which the port to a preset deleted: run "
        "now, it would record the preset's own output, and the agreement test would compare the preset with itself."
    )
