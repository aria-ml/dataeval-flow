"""Record what the rerouting cases' results hold today, to show that running every task through the engine changes
nothing.

Run it from the repository root, with the code as it stands before the change it guards:

    .venv/bin/python -m tests.golden.generate_rerouting

Rerun it only when a change is meant to alter these results, and say so in that commit.
"""

import json
from pathlib import Path

from tests.golden.rerouting import CASES, normalized

HERE = Path(__file__).parent

if __name__ == "__main__":
    for name, run in CASES.items():
        (HERE / f"rerouting_{name}.json").write_text(json.dumps(normalized(run()), indent=2, sort_keys=True) + "\n")
        print("wrote", name)
