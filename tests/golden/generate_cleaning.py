"""Record what the data-cleaning workflow finds on each agreement case, before its port to a preset.

Run it from the repository root, on the code before the port:

    .venv/bin/python -m tests.golden.generate_cleaning

It writes `cleaning_findings.json`: each case's findings as severity, title and brief, in order. The preset must
agree with it (spec §10.3). Rerun it only while the legacy workflow still exists, and say so in that commit.
"""

import json
from pathlib import Path

from tests.golden.cleaning import CASES

HERE = Path(__file__).parent

if __name__ == "__main__":
    recorded = {name: [[f.severity, f.title, f.brief] for f in run()] for name, run in CASES.items()}
    (HERE / "cleaning_findings.json").write_text(json.dumps(recorded, indent=2) + "\n")
    print("wrote", ", ".join(recorded))
