"""Recorded `splitting.json` from the legacy data-splitting workflow, before its port to a preset. It now refuses to
run.

Commit 5742b41 ran it once, on the legacy workflow, to record each case's findings, its indices, each part's label
counts, the whole's balance and diversity rows and each part's uncovered indices. The preset must agree with them
(data-splitting spec §9).

The port deleted the legacy workflow, so `data-splitting` now names the preset. Run now, this would record the
preset's own output as the golden the preset is tested against, and the agreement test would only compare the preset
with itself. `tests/test_splitting_golden.py` lists the preset's deliberate differences from the legacy run.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_splitting records from the legacy data-splitting workflow, which the port to a preset deleted: run "
        "now, it would record the preset's own output, and the agreement test would compare the preset with itself."
    )
