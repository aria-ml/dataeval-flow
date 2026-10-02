"""Recorded `ood.json` from the legacy ood-detection workflow, before its port to a preset. It now refuses to run.

Commit 526bace ran it once, on the legacy workflow, to record each case's findings, each detector's
flags, scores and threshold, the union, mutual and unique sets, the normalized scores, and the factor predictors and
deviations. The preset must agree with all of it (ood-detection spec §10).

The port deleted the legacy workflow, so `ood-detection` now names the preset. Run now, this would record the
preset's own output as the golden the preset is tested against, and the agreement test would only compare the preset
with itself. `tests/test_ood_golden.py` lists the preset's deliberate differences from the legacy run.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_ood records from the legacy ood-detection workflow, which the port to a preset deleted: run now, it "
        "would record the preset's own output, and the agreement test would compare the preset with itself."
    )
