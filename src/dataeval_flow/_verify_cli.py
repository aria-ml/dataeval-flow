"""``dataeval-flow verify``: whether a configured source still holds the items a manifest records (follow-ons §5.2)."""

__all__ = ["verify"]

import sys
from pathlib import Path

# How many items of each kind a mismatch names before it counts the rest.
_SHOWN = 20


def verify(manifest: Path, config: Path, source: str, data_dir: Path | None = None) -> int:
    """Compare source `source` of the pipeline at `config`, loaded as a run reads it, with `manifest`. Returns 0 when
    it holds the recorded items, and 1 when it doesn't, naming them, or when either can't be read."""
    from dataeval_flow._digest import DatasetManifest, dataset_manifest
    from dataeval_flow._sources import load_source
    from dataeval_flow.config._loader import get_data_dir, load_config

    try:
        recorded = DatasetManifest.load(manifest)
        current = dataset_manifest(load_source(load_config(config), source, data_dir=get_data_dir(data_dir)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    diff = recorded.compare(current)
    if not diff:
        print(f"OK: source '{source}' holds the {current.digest.items:,} items {manifest} records.")
        return 0
    print(f"MISMATCH: source '{source}' doesn't hold the items {manifest} records.")
    for label, items in (("Changed", diff.changed), ("Missing", diff.missing), ("Added", diff.added)):
        if items:
            shown = ", ".join(str(item) for item in items[:_SHOWN])
            more = f", and {len(items) - _SHOWN:,} more" if len(items) > _SHOWN else ""
            print(f"  {label}: {len(items):,} ({shown}{more})")
    if diff.classes:
        print("  The class names differ.")
    return 1
