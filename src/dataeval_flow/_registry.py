"""The plug-in registry: built-ins from a table, plugins from entry points, one set of checks."""

import builtins
import importlib
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Generic, TypeVar

__all__ = ["Registry"]

_logger: logging.Logger = logging.getLogger(__name__)

K = TypeVar("K")
_BUILTIN_ORIGIN = "dataeval-flow"

# One re-entrant lock for every registry. A plugin imported while one registry loads may load another in the same
# thread; with a lock per registry, two threads loading two registries could each wait on the other's lock.
_LOAD_LOCK = threading.RLock()


@dataclass(frozen=True)
class _Entry:
    name: str
    target: str  # "module:attribute", the attribute possibly dotted
    origin: str  # the distribution that declared it


class Registry(Generic[K]):
    """One kind's implementations, by type id.

    Built-ins come from ``builtins`` (Flow runs uninstalled in its Docker images, so its own entry-point
    metadata cannot be relied on); plugins come from the entry-point ``group``. Both pass the same checks. A
    plugin that fails them, raises while being loaded or looks this registry up while it loads is logged once and
    left out; resolving its name raises with the reason. A built-in that fails them is a bug and raises at once. A
    plugin can never take a built-in's name, and a name two plugins claim is refused to both.

    Annotations in the class body spell ``builtins.list``: past ``def list``, a bare ``list`` names the method.
    """

    def __init__(
        self,
        *,
        kind: str,
        group: str,
        base: Callable[[], type[K]],
        builtins: Mapping[str, str],
        check: Callable[[str, type[K]], str | None] | None = None,
    ) -> None:
        self.kind = kind
        self.group = group
        self._base = base
        self._builtins = dict(builtins)
        self._check = check
        self._loading = False
        self._loaded: dict[str, type[K]] | None = None
        self._broken: dict[str, str] = {}

    def get(self, name: str) -> type[K]:
        """The implementation registered as `name`."""
        loaded = self._load()
        if name in loaded:
            return loaded[name]
        if name in self._broken:
            raise ValueError(self._broken[name])
        raise ValueError(
            f"Unknown {self.kind}: {name!r}. Installed: {sorted(loaded)}. A plugin {self.kind} is found only through "
            f"an entry point in the {self.group!r} group of an installed package; defining its class is not enough."
        )

    def list(self, *, plugins: bool = True) -> builtins.list[type[K]]:
        """Every working implementation, sorted by name; without `plugins`, the built-ins alone."""
        loaded = self._load()
        return [loaded[name] for name in sorted(loaded) if plugins or name in self._builtins]

    def names(self) -> builtins.list[str]:
        """Every working implementation's name, sorted."""
        return sorted(self._load())

    def problem(self, name: str) -> str | None:
        """Why the plugin claiming `name` was refused, or ``None`` when no plugin claiming it was."""
        self._load()
        return self._broken.get(name)

    def _reset(self) -> None:
        with _LOAD_LOCK:
            self._loaded = None
            self._broken = {}

    def _entries(self) -> builtins.list[_Entry]:
        builtin = [_Entry(name, target, _BUILTIN_ORIGIN) for name, target in self._builtins.items()]
        plugins = [
            _Entry(ep.name, ep.value, ep.dist.name if ep.dist is not None else "an unknown package")
            for ep in entry_points(group=self.group)
        ]
        return builtin + plugins

    def _load(self) -> dict[str, type[K]]:
        if self._loaded is not None:
            return self._loaded
        with _LOAD_LOCK:
            if self._loaded is None:
                if self._loading:
                    # The lock is re-entrant, so this is a plugin's import, in this thread, asking for the
                    # table it is being loaded into. Raised into that import, where `_admit` records it.
                    raise RuntimeError(
                        f"a plugin looked up the {self.kind} registry while it was loading; "
                        "import the class directly instead of looking it up by name"
                    )
                self._loading = True
                try:
                    loaded, broken = self._build()
                finally:
                    self._loading = False
                self._broken = broken
                self._loaded = loaded  # built locally and assigned once, so no thread sees a half-filled dict
        return self._loaded

    def _build(self) -> tuple[dict[str, type[K]], dict[str, str]]:
        """Admit every entry, recording why each refused plugin was refused."""
        loaded: dict[str, type[K]] = {}
        broken: dict[str, str] = {}
        claims: dict[str, builtins.list[_Entry]] = {}
        for entry in self._entries():  # built-ins first, so a built-in is always a name's first claim
            claims.setdefault(entry.name, []).append(entry)
        for name, (first, *rivals) in claims.items():
            if rivals and first.origin != _BUILTIN_ORIGIN:
                sides = " and ".join(f"{entry.target} from {entry.origin}" for entry in (first, *rivals))
                problem = f"{self.kind} {name!r} is claimed by {sides}; with no way to choose, none is registered."
                self._refuse(broken, first, problem)
                continue
            for rival in rivals:
                problem = (
                    f"{self.kind} {name!r} from {rival.origin} clashes with the one from {first.origin}; "
                    "a plugin cannot take a built-in's name."
                )
                self._refuse(broken, rival, problem)
            admitted = self._admit(first)
            if not isinstance(admitted, str):
                loaded[name] = admitted
            elif first.origin == _BUILTIN_ORIGIN:
                raise RuntimeError(admitted)
            else:
                self._refuse(broken, first, admitted)
        return loaded, broken

    def _refuse(self, broken: dict[str, str], entry: _Entry, problem: str) -> None:
        broken[entry.name] = problem
        _logger.warning("Skipping %s plugin %r from %s: %s", self.kind, entry.name, entry.origin, problem)

    def _admit(self, entry: _Entry) -> type[K] | str:
        """The class `entry` names, or why it cannot be registered. Nothing a plugin raises escapes."""
        module_name, _, attribute = entry.target.partition(":")
        try:
            target: Any = importlib.import_module(module_name)
            for part in filter(None, attribute.split(".")):
                target = getattr(target, part)
            base = self._base()
            if not (isinstance(target, type) and issubclass(target, base)):
                return f"{entry.target} is not a subclass of {base.__name__}."
            if (own := getattr(target, "name", None)) != entry.name:
                return f"{entry.target} is registered as {entry.name!r} but names itself {own!r}."
            if self._check is not None and (problem := self._check(entry.name, target)) is not None:
                return f"{entry.target}: {problem}"
        except Exception as error:  # noqa: BLE001 - a broken plugin must not break unrelated runs
            return f"{self.kind} {entry.name!r} from {entry.origin} failed to load {entry.target}: {error}"
        return target
