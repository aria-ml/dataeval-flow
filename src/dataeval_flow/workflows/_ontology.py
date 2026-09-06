"""Turn workflow configuration into a :class:`dataeval.Ontology`.

Private to :mod:`dataeval_flow.workflows`. Build an ontology from any of four sources: an
inline nested mapping, a path to a serialized RDF artifact, concepts declared in config, or
a flat vocabulary synthesized from the dataset's ``index2label`` when a workflow configures
no ontology. Declared concepts also merge onto an inline mapping or an RDF artifact,
replacing any concept with the same id.

Lives here rather than in a workflow package. It handles configuration: path resolution, an
optional dependency, and format inference.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval import Ontology
    from dataeval.types import OntologyConcept

    from dataeval_flow.config.schemas import OntologyConfig

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

    The message reaches the user verbatim as a skip reason. Write it to say what went wrong
    and what to do about it.
    """


def _one_declared_concept(entry: "Mapping[str, Any] | Any") -> "OntologyConcept":
    """Convert one declared-concept entry to DataEval's type.

    Accept either a mapping or an ``OntologyConceptConfig``.
    """
    from dataeval.types import OntologyConcept

    fields = dict(entry if isinstance(entry, Mapping) else entry.model_dump())
    return OntologyConcept(**fields)


def _declared_concepts(concepts: "Sequence[Mapping[str, Any] | Any]") -> "list[OntologyConcept]":
    """Convert config-declared concepts to DataEval's own type.

    Pass either pydantic config models or plain mappings of the same shape. Both work, so
    you can call the loader from a config or by hand.
    """
    built: list[OntologyConcept] = []
    for entry in concepts:
        try:
            built.append(_one_declared_concept(entry))
        except Exception as exc:  # noqa: PERF203 - runs at config time over a handful of entries
            name = entry.get("id", "<no id>") if isinstance(entry, Mapping) else getattr(entry, "id", "<no id>")
            raise OntologyLoadError(f"declared concept {name!r} is not valid: {exc}") from exc
    return built


def _build(concepts: "list[OntologyConcept]") -> "Ontology":
    """Build an :class:`Ontology` from a flat list of concepts.

    Raise :class:`OntologyLoadError` on any failure, naming the problem. Two concepts
    sharing an id is the common case.
    """
    from dataeval import Ontology

    try:
        return Ontology(concepts)
    except Exception as exc:
        raise OntologyLoadError(f"declared concepts do not form a valid ontology: {exc}") from exc


def _extended(base: "Ontology", declared: "list[OntologyConcept]") -> "Ontology":
    """Merge *declared* into *base*, or return *base* unchanged when nothing was declared.

    A declared concept replaces one the artifact defines under the same id. Keeping both
    would leave that id ambiguous. Replacement is total, so a replaced concept's `parents`
    are dropped too. Log the replaced ids at WARNING, since otherwise the hierarchy is
    re-rooted with nothing to show for it.
    """
    if not declared:
        return base
    replaced = {concept.id for concept in declared}
    try:
        kept: list[OntologyConcept] = []
        overwritten: list[str] = []
        for concept in base:
            if concept.id in replaced:
                overwritten.append(concept.id)
            else:
                kept.append(concept)
    except Exception as exc:
        raise OntologyLoadError(f"could not read the base ontology's concepts: {exc}") from exc
    if overwritten:
        _logger.warning(
            "Declared concept(s) %s replace an artifact concept of the same id. Replacement is "
            "total: restate `parents` on the declared concept, or it becomes a root.",
            sorted(overwritten),
        )
    return _build([*kept, *declared])


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
        The ontology, and a source label: ``"inline"``, ``"concepts"``, or the resolved path.

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
    pool: "Sequence[OntologyConfig] | None",
    *,
    data_dir: "Path | None" = None,
) -> "tuple[Ontology, str]":
    """Build the ontology a workflow's ``ontology`` field names.

    Read a string as a name in *pool* first and as a path second. Moving a definition into
    ``ontologies:`` does not break a config that named a file. Read a mapping as an inline
    hierarchy and never consult the pool.

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
        The ontology, and a source label: the pool entry's name, ``"inline"``,
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
        return _load_unmatched(spec, pool, data_dir)

    _refuse_if_also_a_file(spec, data_dir)
    ontology, _ = load_ontology(entry.source, concepts=entry.concepts, data_dir=data_dir)
    return ontology, entry.name


def _load_unmatched(
    spec: str,
    pool: "Sequence[OntologyConfig]",
    data_dir: "Path | None",
) -> "tuple[Ontology, str]":
    """Load *spec* as a path, naming the declared ontologies if that fails.

    A name-first lookup sends a typo down the path branch, where it fails as a missing
    file and never mentions that ``ontologies:`` holds candidates. List them so a reader
    can spot the typo.
    """
    try:
        return load_ontology(spec, data_dir=data_dir)
    except OntologyLoadError as exc:
        names = ", ".join(sorted(item.name for item in pool))
        raise OntologyLoadError(
            f"{exc} No ontology named {spec!r} is declared under `ontologies:` either. Declared: {names}.",
        ) from exc


def _refuse_if_also_a_file(name: str, data_dir: "Path | None") -> None:
    """Refuse a string that names both a pool entry and a readable file.

    Do not pick one by precedence. A silent choice would run the analysis against the wrong
    label space.
    """
    from dataeval_flow.config._loader import resolve_path

    try:
        candidate = resolve_path(name, data_dir, default_subdir="config")
        is_collision = Path(candidate).is_file()
    except OSError:  # an unresolvable path, or an unsearchable directory, is not a collision
        return
    if is_collision:
        raise OntologyLoadError(
            f"'{name}' names an entry under `ontologies:` and also the file '{candidate}'. Rename one of them.",
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
