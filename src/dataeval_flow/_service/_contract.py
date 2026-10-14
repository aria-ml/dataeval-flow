"""What the builder contracts share: the extension error and the service's limits (builder spec §4)."""

import functools
import hashlib
import json
from typing import Any

CONTRACT_SCHEMA = 1
MAX_DOCUMENT_BYTES = 1_048_576
MAX_RESOLVED_NODES = 1000
ACTIVITY_PAGE_SIZE = 100


class ContractError(Exception):
    """A refusal a client branches on by `code`, answered as ``{schema, code, message, retryable, details}``."""

    def __init__(self, status: int, code: str, message: str, *, retryable: bool = False, **details: Any) -> None:
        super().__init__(message)
        self.status = status
        self.body = {
            "schema": CONTRACT_SCHEMA,
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": details,
        }


def fingerprint_of(document: Any) -> str:
    """`document`'s fingerprint: SHA-256 over its canonical JSON, keys sorted, arrays in order (builder spec §4.4)."""
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


@functools.cache
def schema_fingerprint() -> str:
    """The fingerprint of this process's pipeline schema and step catalog, plugins and versions included.

    Computed once: a plugin installed while the service runs is seen by new runs, not by this; restart the service.
    """
    from dataeval_flow.config import PipelineConfig
    from dataeval_flow.steps import list_steps

    return fingerprint_of(
        {"schema": PipelineConfig.model_json_schema(), "catalog": list_steps().model_dump(mode="json")}
    )
