"""Fields naming another pool's entry, marked so an editor finds each reference to rename (builder spec §4.3)."""

from typing import Any

__all__ = ["pool_ref"]


def pool_ref(pool: str, *, or_path: bool = False) -> dict[str, Any]:
    """`json_schema_extra` for a field holding the name of an entry of `pool`; with `or_path`, else a file path."""
    ref: dict[str, Any] = {"pool": pool}
    if or_path:
        ref["or_path"] = True
    return {"x-flow-ref": ref}
