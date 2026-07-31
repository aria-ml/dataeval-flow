"""Turn workflow configuration into a :class:`dataeval.Ontology`.

Private to :mod:`dataeval_flow.workflows`. Three sources are supported: an inline
nested mapping, a path to a serialized RDF artifact, and — when a workflow is run
without an ontology at all — a flat vocabulary synthesized from the dataset's
``index2label``.

Kept out of any single workflow package because its concerns are configuration
concerns (path resolution, an optional dependency, format inference) rather than
the concerns of whichever workflow happens to consume the ontology first.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataeval import Ontology

__all__ = ["OntologyLoadError", "load_ontology", "synthesize_ontology"]

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


def load_ontology(spec: "Mapping[str, Any] | str") -> "tuple[Ontology, str]":
    """Build an ontology from an inline hierarchy or a path to an RDF artifact.

    Parameters
    ----------
    spec : Mapping or str
        A nested mapping of concept to children, or a path to a serialized RDF
        file. Relative paths resolve against the data root.

    Returns
    -------
    tuple[Ontology, str]
        The ontology and a source label — ``"inline"`` or the resolved path.

    Raises
    ------
    OntologyLoadError
        If the mapping is malformed, the file is unreadable, the RDF does not
        parse, or ``rdflib`` is not installed.
    """
    from dataeval import Ontology

    if isinstance(spec, Mapping):
        try:
            return Ontology.from_hierarchy(dict(spec)), "inline"
        except Exception as exc:
            raise OntologyLoadError(f"inline ontology is not a valid hierarchy: {exc}") from exc

    from dataeval_flow.config._loader import resolve_path

    path = resolve_path(spec, None, default_subdir="config")
    try:
        content = Path(path).read_text()
    except OSError as exc:
        raise OntologyLoadError(f"could not read ontology file '{path}': {exc}") from exc
    except UnicodeDecodeError as exc:
        raise OntologyLoadError(f"ontology file '{path}' is not valid UTF-8: {exc}") from exc

    fmt = _RDF_FORMATS.get(Path(path).suffix.lower())
    try:
        ontology = Ontology.from_rdf(content, format=fmt)
    except ImportError as exc:
        raise OntologyLoadError(
            "reading an ontology from a file needs rdflib, which is not installed. "
            'Install it with: pip install "dataeval[ontology]"'
        ) from exc
    except Exception as exc:
        raise OntologyLoadError(
            f"could not parse ontology file '{path}' as {fmt or 'an auto-detected format'}: {exc}"
        ) from exc

    _logger.debug("Loaded ontology from %s (%d concepts)", path, len(ontology.ids))
    return ontology, str(path)


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
