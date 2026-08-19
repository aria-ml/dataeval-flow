"""Metadata convenience builder wrapping DataEval."""

__all__ = ["build_metadata"]

from typing import TYPE_CHECKING, Any

from dataeval import Metadata
from dataeval.protocols import AnnotatedDataset

if TYPE_CHECKING:
    from dataeval_flow.policy import ResolvedPolicy


def build_metadata(dataset: AnnotatedDataset[Any], policy: "ResolvedPolicy | None" = None) -> Metadata:
    """Build Metadata from a dataset under a resolved metadata policy.

    Parameters
    ----------
    dataset : AnnotatedDataset
        Input dataset.
    policy : ResolvedPolicy | None
        How factors become codes — the cuts, the vocabularies, and which of them somebody
        chose.  None takes DataEval's defaults, which derive everything from this draw.

    Returns
    -------
    Metadata
        DataEval Metadata instance.
    """
    from dataeval_flow.policy import ResolvedPolicy

    return Metadata(dataset, **(policy or ResolvedPolicy()).metadata_kwargs())
