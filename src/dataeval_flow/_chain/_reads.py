"""The Metadata a chain's steps read, noted as they read it, and the binning record the reads make (spec §10.10)."""

__all__ = ["MetadataRead", "ReadingContext", "attach_reads", "note_read", "noting_reads"]

import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dataeval_flow.workflows._context import WorkflowContext

if TYPE_CHECKING:
    from dataeval import Metadata

    from dataeval_flow._policy import ResolvedPolicy
    from dataeval_flow._result import ResultMetadata

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetadataRead:
    """One Metadata a step read: the address of the Dataset it describes, the policy it was built under, and the name
    the step's entry gave that policy, ``None`` where it named none."""

    address: str
    policy_name: str | None
    policy: "ResolvedPolicy | None"
    metadata: "Metadata"


_READS: ContextVar[list[MetadataRead] | None] = ContextVar("chain_metadata_reads", default=None)


@contextmanager
def noting_reads() -> Iterator[list[MetadataRead]]:
    """Collect every Metadata a step reads inside the block, in the order read."""
    reads: list[MetadataRead] = []
    token = _READS.set(reads)
    try:
        yield reads
    finally:
        _READS.reset(token)


def note_read(address: str, policy_name: str | None, policy: "ResolvedPolicy | None", metadata: "Metadata") -> None:
    """Note that a step read `address`'s Metadata under `policy`. Outside :func:`noting_reads`, nothing."""
    reads = _READS.get()
    if reads is not None:
        reads.append(MetadataRead(address, policy_name, policy, metadata))


@dataclass
class ReadingContext(WorkflowContext):
    """An evaluator or workflow step's context: a :class:`WorkflowContext` whose :meth:`metadata` notes each read."""

    policy_name: str | None = None
    """The name the step's entry gave its metadata policy, ``None`` where it named none."""

    def metadata(self, source: str) -> "Metadata":
        """The source's Metadata, as :meth:`WorkflowContext.metadata` builds it, noted for the binning record."""
        metadata = super().metadata(source)
        note_read(source, self.policy_name, self.metadata_policy, metadata)
        return metadata


def attach_reads(result_metadata: "ResultMetadata", reads: Sequence[MetadataRead]) -> None:
    """Record on a chain's envelope the encodings `reads` read: ``metadata_binning`` and ``encoding_digest``.

    Reads of one Dataset that describe identically are one record, whatever policy object each step held. One record
    is written as it stands, as an unported workflow writes one. Several are written as ``{"per_split": {key:
    record}}``, keyed by the Dataset's address; a Dataset read a second way is keyed ``address (policy)``, by the name
    the step's entry gave its policy, and a further collision is numbered. ``encoding_digest`` is set only where every
    record agrees. Never raises: a read that cannot be described costs the record, not the run.
    """
    from dataeval_flow._binning import _common_digest, describe_under

    if not reads:
        return
    try:
        records: dict[str, dict[str, Any]] = {}
        texts: dict[str, str] = {}  # each record as sorted JSON: a record's NaNs never equal themselves as floats
        described: set[tuple[str, int]] = set()
        for read in reads:
            if (read.address, id(read.metadata)) in described:
                continue  # one Metadata read again describes the same
            described.add((read.address, id(read.metadata)))
            record = describe_under(read.metadata, read.policy)
            text = json.dumps(record, sort_keys=True, default=str)
            same = [key for key in records if key == read.address or key.startswith(f"{read.address} (")]
            if any(texts[key] == text for key in same):
                continue
            key = read.address
            if same:
                key = f"{read.address} ({read.policy_name or 'no policy'})"
                if key in records:
                    key = f"{read.address} ({len(same) + 1})"
            records[key], texts[key] = record, text
        if len(records) == 1:
            (record,) = records.values()
            result_metadata.metadata_binning = record
            result_metadata.encoding_digest = record.get("encoding_digest")
        else:
            result_metadata.metadata_binning = {"per_split": records}
            result_metadata.encoding_digest = _common_digest(records.values())
    except Exception:
        _logger.warning("Binning record unavailable", exc_info=True)
