"""IR-1-H-3 / IR-1-R-2 — advertised MAITE component & task entrypoints.

DataEval-Flow declares ``maite_interop_scope=provider``: it advertises MAITE
image-classification / object-detection ``Dataset`` components and MAITE tasks
via ``[project.entry-points]`` in ``pyproject.toml`` (see IR-1-H-3).

This test guards those advertisements two ways:

* Always — every declared ``maite.*`` entrypoint must import (``ep.load()``),
  which is exactly what the program's ``validate_maite_entrypoints.py`` does for
  ``maite.tasks`` and a prerequisite for the ``maite.protocols.*`` static check.
* When Pyright and a recent MAITE are available — the advertised
  ``maite.protocols.*`` components are statically verified against their
  protocols using MAITE's ``statically_verify_exposed_component_entrypoints``
  (the IR-1-R-2 recommendation).
"""

from __future__ import annotations

import shutil
from importlib.metadata import distribution

import pytest

DIST_NAME = "dataeval-flow"
PROTOCOL_PREFIX = "maite.protocols"
TASKS_PREFIX = "maite.tasks"


def _maite_entrypoints():
    return [ep for ep in distribution(DIST_NAME).entry_points if ep.group.startswith("maite.")]


def test_declares_maite_entrypoints() -> None:
    """provider scope requires at least one advertised MAITE entrypoint."""
    eps = _maite_entrypoints()
    assert eps, "no maite.* entrypoints advertised in installed metadata (IR-1-H-3)"
    # We advertise dataset components for both supported CV tasks plus tasks.
    groups = {ep.group for ep in eps}
    assert any(g.startswith(PROTOCOL_PREFIX) for g in groups), groups
    assert TASKS_PREFIX in groups, groups


@pytest.mark.parametrize("ep", _maite_entrypoints(), ids=lambda ep: f"{ep.group}:{ep.name}")
def test_entrypoint_target_importable(ep) -> None:
    """Every advertised entrypoint target must import (IR-1-H-3)."""
    obj = ep.load()
    assert obj is not None


def test_protocol_entrypoints_statically_verify() -> None:
    """maite.protocols.* components are valid protocol implementers (IR-1-R-2)."""
    if shutil.which("pyright") is None:
        pytest.skip("pyright not installed; static entrypoint verification skipped")
    try:
        from maite._internals.testing.project import (
            statically_verify_exposed_component_entrypoints,
        )
    except ImportError:
        pytest.skip("installed maite predates statically_verify_exposed_component_entrypoints")

    results = statically_verify_exposed_component_entrypoints(
        entrypoint_group_prefix=PROTOCOL_PREFIX,
        package_name=DIST_NAME,
    )
    assert results, "no maite.protocols.* entrypoints were verified"
    failed = {name: ok for name, ok in results.items() if not ok}
    assert not failed, f"entrypoints failed static protocol verification: {sorted(failed)}"
