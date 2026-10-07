"""Recorded `prioritization_rankings.json` from the legacy data-prioritization workflow, before its port to a preset.
It now refuses to run.

Commit e603f6f ran it once, on the legacy workflow, to record each case's pool rankings and per-source removals. The
preset must agree with them (spec §10.9).

The port deleted the legacy workflow, so `prioritization` now names the preset. Run now, this would record the
preset's own output as the golden the preset is tested against, and the agreement test would only compare the preset
with itself. `tests/test_prioritization_golden.py` lists the preset's deliberate differences from the legacy run.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_prioritization records from the legacy data-prioritization workflow, which the port to a preset "
        "deleted: run now, it would record the preset's own output, and the agreement test would compare the preset "
        "with itself."
    )
