"""Identity of the label space a result's labels were read under.

A conformed dataset's numbers are comparable with another's only if both can say which
vocabulary produced them: a bias score over a collapsed label space and one over an
uncollapsed space are different measurements wearing one name.  This is the same argument
``encoding_digest`` already answers for binning, applied to the vocabulary.

Deliberately free of any ``dataeval`` import.  The inputs are a set of ids, a mapping and a
sequence — all plain data — so the digest can be computed from a config's ``Relabel``
parameters as readily as from an alignment result.  That is the mechanism: an audit and a
downstream workflow reach the same value from different starting points, and matching values
are what lets a reader find the audit that justified a vocabulary.
"""

__all__ = ["label_space_digest", "ontology_digest"]

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

#: Digest length.  Twelve rather than the eight `_config_hash` uses: this one is compared
#: across archived artifacts rather than within one cache directory, so the collision budget
#: is larger.
_DIGEST_CHARS = 12


def ontology_digest(concept_ids: Iterable[str]) -> str:
    """Content digest of an ontology's concept set.

    Parameters
    ----------
    concept_ids : Iterable[str]
        Every concept id the ontology defines.

    Returns
    -------
    str
        A 12-character hex digest.

    Notes
    -----
    Sorted before hashing, so the digest names the concepts an ontology defines rather than
    the order a parser happened to yield them in.  Two artifacts defining the same concepts
    are one label space for this purpose; one that gained a concept is not.
    """
    ids = sorted(str(cid) for cid in concept_ids)
    payload = json.dumps(ids, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:_DIGEST_CHARS]


def label_space_digest(
    *,
    ontology: str,
    class_remap: Mapping[str, str],
    target: Sequence[str],
) -> str:
    """Digest of the vocabulary a result's labels were read under.

    Parameters
    ----------
    ontology : str
        Digest of the ontology the vocabulary came from, from :func:`ontology_digest`.
    class_remap : Mapping[str, str]
        Source class name to target concept, as :class:`dataeval.data.Relabel` applies it.
        Order-independent: a mapping is a set of pairs, and two configs listing them
        differently describe one rewrite.
    target : Sequence[str]
        The target vocabulary **in index order**.  Order participates, because the order is
        the integer indexing — two datasets conformed against differently-ordered targets
        carry labels that do not mean the same thing.

    Returns
    -------
    str
        A 12-character hex digest.
    """
    payload = json.dumps(
        {
            "class_remap": dict(sorted((str(k), str(v)) for k, v in class_remap.items())),
            "ontology": ontology,
            "target": [str(name) for name in target],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:_DIGEST_CHARS]
