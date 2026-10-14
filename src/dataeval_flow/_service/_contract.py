"""What the builder contracts share: the extension error and the service's limits (builder spec §4)."""

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
