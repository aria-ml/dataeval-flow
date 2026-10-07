"""Recorded `label_space.json` from legacy data-coverage with `ontology:` set, before `taxonomy` existed. It now
refuses to run.

Commits d83eb0a and 671ad6f ran it on legacy data-coverage, to record each case's ontology findings and what they
were computed from: the representation, the conformance, the alignment and the structure. The preset must agree with
them (coverage spec §8.1).

scope's port to a preset deleted the legacy workflow, and scope now refuses `ontology:`, so legacy's
settings no longer build. `tests/test_label_space_golden.py` lists the preset's deliberate differences from the legacy
run.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_label_space records from legacy data-coverage with `ontology:` set, which the port to a preset "
        "deleted: scope now refuses `ontology:`, and `taxonomy` must not be tested against its own output."
    )
