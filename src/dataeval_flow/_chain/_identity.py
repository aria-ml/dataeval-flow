"""Node identity: the keys that name a Dataset node's cache, and the digests lineage shows (spec §5.7)."""

__all__ = [
    "element_key",
    "indices_digest",
    "output_key",
    "plan_digest",
    "settings_of",
    "short_digest",
    "step_key",
]

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from dataeval_flow.steps._port import Port


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def step_key(type_id: str, settings: Mapping[str, Any], input_keys: Sequence[str], digest: str | None) -> str:
    """The key of a transform's output: its type, settings, input node keys in port order, and content digest."""
    payload = json.dumps(
        {"type": type_id, "settings": settings, "inputs": list(input_keys), "digest": digest},
        sort_keys=True,
        default=repr,
    )
    return "step:" + _sha(payload)


def output_key(step_key: str, output: str) -> str:
    """The key of one named output of a step with several."""
    return "out:" + _sha(f"{step_key}|{output}")


def element_key(parent_key: str, key: str) -> str:
    """The key of one element of a list output."""
    return "el:" + _sha(f"{parent_key}|{key}")


def short_digest(key: str) -> str:
    """What lineage shows of a key: 12 hex characters, equal for equal keys."""
    return _sha(key)[:12]


def indices_digest(indices: Iterable[int]) -> str:
    """A digest of item indices, in order."""
    return _sha(",".join(str(int(index)) for index in indices))[:16]


def plan_digest(plan: Iterable[Any]) -> str:
    """A digest of a removal plan's addresses. A ``RemovalPlan`` iterates sorted, so equal plans digest alike."""
    return _sha("\n".join(repr(address) for address in plan))[:16]


def settings_of(config: BaseModel, ports: Iterable[Port]) -> dict[str, Any]:
    """`config` as JSON without the fields its input ports read: names are local, input keys say what was read."""
    names = {port.name for port in ports}
    return json.loads(config.model_dump_json(exclude=names, fallback=repr))
