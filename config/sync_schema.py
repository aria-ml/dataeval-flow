#!/usr/bin/env python
"""Regenerate or verify config/params.schema.json against PipelineConfig.

Usage:
  python config/sync_schema.py          # Check only (CI-friendly)
  python config/sync_schema.py --fix    # Regenerate the file
"""

import json
import sys
from pathlib import Path

from dataeval_flow.config.models import PipelineConfig

SCHEMA_PATH = Path(__file__).parent / "params.schema.json"


def main() -> None:
    """Entry point for JSON schema sync script."""
    new_schema = json.dumps(PipelineConfig.model_json_schema(), indent=2) + "\n"

    if "--fix" in sys.argv:
        SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCHEMA_PATH.write_text(new_schema, encoding="utf-8")
        print(f"Updated {SCHEMA_PATH}")
    else:
        if not SCHEMA_PATH.exists():
            print(f"FAIL: {SCHEMA_PATH} does not exist. Run: python config/sync_schema.py --fix")
            sys.exit(1)
        existing = SCHEMA_PATH.read_text(encoding="utf-8")
        if existing == new_schema:
            print(f"OK: {SCHEMA_PATH} is up to date")
        else:
            print(f"FAIL: {SCHEMA_PATH} is out of date. Run: python config/sync_schema.py --fix")
            sys.exit(1)


if __name__ == "__main__":
    main()
