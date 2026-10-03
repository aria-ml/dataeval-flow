"""Recorded `coverage.json` from the legacy data-coverage workflow, before its port to a preset. It now refuses to run.

Commit f25fd52 ran it once, on the legacy workflow, to record each case's findings and what they were computed from:
coverage, completeness, the label distribution, the metadata summary, the gaps and the class worklist. The preset
must agree with them (coverage spec §8.2).

The port deleted the legacy workflow, so `data-coverage` now names the preset. Run now, this would record the preset's
own output as the golden the preset is tested against, and the agreement test would only compare the preset with
itself. `tests/test_coverage_golden.py` lists the preset's deliberate differences from the legacy run.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_coverage records from the legacy data-coverage workflow, which the port to a preset deleted: run "
        "now, it would record the preset's own output, and the agreement test would compare the preset with itself."
    )
