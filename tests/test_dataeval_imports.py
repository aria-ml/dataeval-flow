"""Pin every ``dataeval`` symbol ``src/`` imports, so a rename fails here.

Flow reaches into dataeval from module scope *and* from inside functions. A
function-local import is invisible until the function runs, which is how
``dataeval.utils._internal`` moving to ``dataeval.utils._array`` took out 46
tests at call time — including ``dataset_fingerprint``, on the path of every run
that resolves a dataset — rather than at collection.

These tests resolve each import statically swept from the source, so an upstream
rename surfaces as one legible failure naming the symbol.
"""

import ast
import importlib
from pathlib import Path
from typing import get_args

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "dataeval_flow"

# Private dataeval modules flow depends on. `__all__` is empty in each, so
# nothing upstream promises they will keep their names -- `utils._internal` has
# already moved once. Adding a row is a deliberate act: prefer a public API, and
# where there is none, say so upstream. Tracked in the 2026-09-03 metadata plan
# under "Cross-cutting: private coupling".
PRIVATE_MODULES = {
    "dataeval.core._clusterer",
    "dataeval.utils._array",
}


def _imports() -> list[tuple[str, str, str]]:
    """Sweep ``src/`` for ``dataeval`` imports as ``(module, name, origin)``."""
    found: list[tuple[str, str, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(SRC.parent.parent)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # `level` is non-zero for relative imports, which are flow's own.
                if node.level or not node.module or not _is_dataeval(node.module):
                    continue
                found.extend((node.module, alias.name, f"{rel}:{node.lineno}") for alias in node.names)
            elif isinstance(node, ast.Import):
                found.extend(
                    (alias.name, "", f"{rel}:{node.lineno}") for alias in node.names if _is_dataeval(alias.name)
                )
    return found


def _is_dataeval(module: str) -> bool:
    """True for ``dataeval`` and its submodules, excluding ``dataeval_flow``."""
    return module == "dataeval" or module.startswith("dataeval.")


IMPORTS = _imports()


def test_the_sweep_found_something():
    # A refactor that moves the package would otherwise make every test below
    # pass vacuously.
    assert len(IMPORTS) > 20, f"expected a substantial sweep of {SRC}, got {len(IMPORTS)}"


@pytest.mark.parametrize(("module", "name", "origin"), IMPORTS, ids=lambda v: str(v))
def test_import_resolves(module: str, name: str, origin: str):
    try:
        mod = importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - only on an upstream rename
        pytest.fail(f"{origin} imports from `{module}`, which no longer exists: {e}")
    if name and not hasattr(mod, name):
        pytest.fail(f"{origin} imports `{name}` from `{module}`, which no longer provides it")


def test_private_coupling_is_the_recorded_set():
    private = {
        module
        for module, _, _ in IMPORTS
        if any(part.startswith("_") for part in module.split(".")[1:])  # `dataeval` itself is public
    }
    assert private == PRIVATE_MODULES, (
        "the set of private dataeval modules flow imports changed. Update PRIVATE_MODULES "
        "only after checking there is no public API for what the new one provides."
    )


class TestPinnedVocabulariesStayInStep:
    """Config Literals hardcode value sets that live in private dataeval modules.

    Pinning them is what makes a misspelled period a config error rather than a failure
    after the dataset walk. The cost is that an upstream addition would otherwise appear as
    a value flow silently refuses, with nothing to say why — so it fails here instead,
    naming what moved.

    These imports are the reason `dataeval.types` exporting the three vocabularies is asked
    for in the API proposal; when it lands, the private import here is what drops.
    """

    def test_the_datetime_periods_match_the_registry(self):
        from dataeval.types._factors import DATETIME_GRANULARITIES

        from dataeval_flow.config.schemas._metadata import DateTimeGranularity

        assert set(get_args(DateTimeGranularity)) == set(DATETIME_GRANULARITIES)

    def test_the_epoch_units_match_the_registry(self):
        from dataeval.types._factors import EPOCH_UNITS

        from dataeval_flow.config.schemas._metadata import EpochUnit

        assert set(get_args(EpochUnit)) == set(EPOCH_UNITS)

    def test_every_pinned_period_is_one_parse_datetime_accepts(self):
        """Set equality would still pass if both sides drifted together onto a value the
        type itself refuses, which is the failure a user would actually hit."""
        from dataeval.types import ParseDateTime

        from dataeval_flow.config.schemas._metadata import DateTimeGranularity

        for period in get_args(DateTimeGranularity):
            ParseDateTime("f", every=period)

    def test_every_pinned_epoch_unit_is_one_parse_datetime_accepts(self):
        from dataeval.types import ParseDateTime

        from dataeval_flow.config.schemas._metadata import EpochUnit

        for unit in get_args(EpochUnit):
            ParseDateTime("f", epoch=unit)
