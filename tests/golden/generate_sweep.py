"""Record `sweep.json` from the legacy parameter-sweep workflow, before its removal (task-matrix spec §11.3).

Run once: `.venv/bin/python -m tests.golden.generate_sweep`.
"""

import json
from pathlib import Path

from tests.golden.sweep import legacy_counts

if __name__ == "__main__":
    path = Path(__file__).with_name("sweep.json")
    path.write_text(json.dumps(legacy_counts(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
