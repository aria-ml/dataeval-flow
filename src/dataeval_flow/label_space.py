"""Identity of the label space a result's labels were read under.

Two results are only comparable if both record which vocabulary produced them. A bias score
computed over a collapsed label space differs from one computed over an uncollapsed space,
and nothing else in the envelope distinguishes them. This is the vocabulary equivalent of
``encoding_digest``, which serves the same purpose for binning.

The functions take plain data and import nothing from ``dataeval``, so the digest can be
computed from a config's ``Relabel`` parameters as well as from an alignment result. An
audit and a downstream workflow therefore produce the same value from different inputs,
which is what lets a result be matched to its audit.
"""

__all__ = ["label_space_digest", "ontology_digest"]

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence

#: Digest length. Twelve rather than the eight `_config_hash` uses, because this digest is
#: compared across archived artifacts rather than within a single cache directory.
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
    Ids are sorted before hashing, so the digest covers the set of concepts an ontology
    defines and not the order a parser returned them in. Two artifacts defining the same
    concepts produce the same digest; adding or removing a concept changes it.

    Labels, synonyms and hierarchy are not covered. A label rename still changes the
    composite digest through :func:`label_space_digest`'s ``target``.
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
        Order does not affect the digest: two configs listing the same pairs in a different
        order describe the same rewrite.
    target : Sequence[str]
        The target vocabulary in index order. Order does affect the digest, because the
        order is the integer indexing. Datasets conformed against differently ordered
        targets carry labels that mean different things.

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
