"""Turn workflow configuration into a :class:`dataeval.Ontology`.

Private to :mod:`dataeval_flow.workflows`. Four sources are supported: an inline nested
mapping, a path to a serialized RDF artifact, a set of concepts declared directly in
config, and — when a workflow is run without an ontology at all — a flat vocabulary
synthesized from the dataset's ``index2label``. Declared concepts also merge onto an
inline mapping or an RDF artifact, replacing any concept that shares their id.

Kept out of any single workflow package because its concerns are configuration
concerns (path resolution, an optional dependency, format inference) rather than
the concerns of whichever workflow happens to consume the ontology first.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval import Ontology
    from dataeval.types import OntologyConcept

__all__ = ["OntologyLoadError", "load_ontology", "resolve_ontology", "synthesize_ontology"]

_logger = logging.getLogger(__name__)

#: File suffix to the rdflib format hint. ``None`` lets rdflib guess.
_RDF_FORMATS: dict[str, str] = {
    ".ttl": "turtle",
    ".rdf": "xml",
    ".owl": "xml",
    ".xml": "xml",
    ".nt": "nt",
    ".jsonld": "json-ld",
    ".json": "json-ld",
}


class OntologyLoadError(Exception):
    """An ontology could not be built from the configuration.

    The message is surfaced verbatim to the user as a skip reason, so it must say
    what went wrong and — where there is one — what to do about it.
    """


def _one_declared_concept(entry: "Mapping[str, Any] | Any") -> "OntologyConcept":
    """Convert one declared-concept entry — a mapping or an ``OntologyConceptConfig`` — to DataEval's type."""
    from dataeval.types import OntologyConcept

    fields = dict(entry if isinstance(entry, Mapping) else entry.model_dump())
    return OntologyConcept(**fields)


def _declared_concepts(concepts: "Sequence[Mapping[str, Any] | Any]") -> "list[OntologyConcept]":
    """Config-declared concepts as DataEval's own type.

    Accepts either the pydantic config model or a plain mapping of the same shape, so the
    loader is usable from a config and from a hand-written call alike.
    """
    built: list[OntologyConcept] = []
    for entry in concepts:
        try:
            built.append(_one_declared_concept(entry))
        except Exception as exc:  # noqa: PERF203 - config-time, over at most a handful of entries
            name = entry.get("id", "<no id>") if isinstance(entry, Mapping) else getattr(entry, "id", "<no id>")
            raise OntologyLoadError(f"declared concept {name!r} is not valid: {exc}") from exc
    return built


def _build(concepts: "list[OntologyConcept]") -> "Ontology":
    """Build an :class:`Ontology` from a flat list of concepts.

    Wraps any failure — most commonly two concepts sharing an id — as an
    :class:`OntologyLoadError` naming the problem, so a config typo reads as a skip reason
    rather than an unhandled `dataeval` exception.
    """
    from dataeval import Ontology

    try:
        return Ontology(concepts)
    except Exception as exc:
        raise OntologyLoadError(f"declared concepts do not form a valid ontology: {exc}") from exc


def _extended(base: "Ontology", declared: "list[OntologyConcept]") -> "Ontology":
    """*base* with *declared* merged in, or *base* itself when nothing was declared.

    A declared concept replaces one the artifact already defines under the same id: the
    config is the more local statement, and silently keeping both would leave the id
    ambiguous.
    """
    if not declared:
        return base
    replaced = {concept.id for concept in declared}
    return _build([*(c for c in base if c.id not in replaced), *declared])


def load_ontology(
    spec: "Mapping[str, Any] | str | None",
    *,
    concepts: "Sequence[Mapping[str, Any] | Any]" = (),
    data_dir: "Path | None" = None,
) -> "tuple[Ontology, str]":
    """Build an ontology from an inline hierarchy, an RDF artifact, declared concepts, or a mix.

    Parameters
    ----------
    spec : Mapping or str or None
        A nested mapping of concept to children, or a path to a serialized RDF file.
        Relative paths resolve against *data_dir*. ``None`` builds the space from
        *concepts* alone.
    concepts : Sequence, optional
        Concepts to add on top of *spec*, each an ``OntologyConceptConfig`` or a mapping of
        the same shape. A declared concept whose id the artifact already defines replaces it.
    data_dir : Path or None, optional
        Data root that a relative *spec* path resolves against. ``None`` falls back to the
        process-wide root.

    Returns
    -------
    tuple[Ontology, str]
        The ontology and a source label — ``"inline"``, ``"concepts"``, or the resolved path.

    Raises
    ------
    OntologyLoadError
        If nothing is given, the mapping is malformed, a declared concept is malformed, the
        file is unreadable, the RDF does not parse, or ``rdflib`` is not installed.
    """
    from dataeval import Ontology

    declared = _declared_concepts(concepts)

    if spec is None:
        if not declared:
            raise OntologyLoadError("no ontology source and no concepts declared")
        return _build(declared), "concepts"

    if isinstance(spec, Mapping):
        try:
            base = Ontology.from_hierarchy(dict(spec))
        except Exception as exc:
            raise OntologyLoadError(f"inline ontology is not a valid hierarchy: {exc}") from exc
        return _extended(base, declared), "inline"

    from dataeval_flow.config._loader import resolve_path

    path = resolve_path(spec, data_dir, default_subdir="config")
    try:
        content = Path(path).read_text()
    except OSError as exc:
        raise OntologyLoadError(f"could not read ontology file '{path}': {exc}") from exc
    except UnicodeDecodeError as exc:
        raise OntologyLoadError(f"ontology file '{path}' is not valid UTF-8: {exc}") from exc

    fmt = _RDF_FORMATS.get(Path(path).suffix.lower())
    try:
        base = Ontology.from_rdf(content, format=fmt)
    except ImportError as exc:
        raise OntologyLoadError(
            "reading an ontology from a file needs rdflib, which is not installed. "
            'Install it with: pip install "dataeval[ontology]"'
        ) from exc
    except Exception as exc:
        raise OntologyLoadError(
            f"could not parse ontology file '{path}' as {fmt or 'an auto-detected format'}: {exc}"
        ) from exc

    _logger.debug("Loaded ontology from %s (%d concepts)", path, len(base.ids))
    return _extended(base, declared), str(path)


def resolve_ontology(
    spec: "Mapping[str, Any] | str | None",
    pool: "Sequence[Any] | None",
    *,
    data_dir: "Path | None" = None,
) -> "tuple[Ontology, str]":
    """Build the ontology a workflow's ``ontology`` field names.

    A string is a name in *pool* first and a path second, which is what lets a definition
    move into ``ontologies:`` without breaking a config that named a file. A mapping is
    always an inline hierarchy and never consults the pool.

    Parameters
    ----------
    spec : Mapping or str or None
        The workflow's ``ontology`` value.
    pool : Sequence of OntologyConfig or None
        The pipeline's ``ontologies`` definitions.
    data_dir : Path or None, optional
        Data root a relative path resolves against.

    Returns
    -------
    tuple[Ontology, str]
        The ontology and a source label — the pool entry's name, ``"inline"``,
        ``"concepts"``, or the resolved path.

    Raises
    ------
    OntologyLoadError
        For anything :func:`load_ontology` refuses, and when *spec* names both a pool entry
        and a readable file.
    """
    if not isinstance(spec, str) or not pool:
        return load_ontology(spec, data_dir=data_dir)

    entry = next((item for item in pool if item.name == spec), None)
    if entry is None:
        return load_ontology(spec, data_dir=data_dir)

    _refuse_if_also_a_file(spec, data_dir)
    ontology, _ = load_ontology(entry.source, concepts=entry.concepts, data_dir=data_dir)
    return ontology, entry.name


def _refuse_if_also_a_file(name: str, data_dir: "Path | None") -> None:
    """Refuse a string that names a pool entry and a readable file at once.

    Picking one by precedence would resolve a genuine ambiguity silently, and the wrong
    choice is a whole run measured against a label space nobody asked for.
    """
    from dataeval_flow.config._loader import resolve_path

    try:
        candidate = resolve_path(name, data_dir, default_subdir="config")
    except Exception:  # noqa: BLE001 - an unresolvable path simply is not a collision
        return
    if Path(candidate).is_file():
        raise OntologyLoadError(
            f"'{name}' names an entry under `ontologies:` and also the file '{candidate}'. "
            "Rename one of them — there is no sound way to choose between them.",
        )


def synthesize_ontology(index2label: Mapping[int, str]) -> "tuple[Ontology, str]":
    """Build a flat ontology from a dataset's ``index2label``.

    Every declared class becomes a concept with no parents and no children, so the
    result is simultaneously all roots and all leaves. This is deliberately *not* a
    taxonomy: it can only name classes the dataset already declares, so it supports
    a balance measurement rather than a coverage one. See the design spec.

    Raises
    ------
    OntologyLoadError
        If ``index2label`` is empty — there is nothing to build from.
    """
    from dataeval import Ontology

    names = [str(name) for name in index2label.values()]
    if not names:
        raise OntologyLoadError("no ontology configured and no index2label to synthesize one from")
    return Ontology.from_hierarchy(names), "index2label"
