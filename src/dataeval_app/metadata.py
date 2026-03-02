"""Metadata convenience builder wrapping DataEval."""

__all__ = ["build_metadata"]

from typing import TYPE_CHECKING, Any

from dataeval import Metadata
from dataeval.protocols import AnnotatedDataset

if TYPE_CHECKING:
    from dataeval_app.config.schemas.task import AutoBinMethod


def build_metadata(
    dataset: AnnotatedDataset[Any],
    auto_bin_method: "AutoBinMethod | None" = None,
    exclude: list[str] | None = None,
    continuous_factor_bins: dict[str, int | list[float]] | None = None,
) -> Metadata:
    """Build Metadata from dataset and config.

    Parameters
    ----------
    dataset : AnnotatedDataset
        Input dataset.
    auto_bin_method : AutoBinMethod | None
        Method for automatic binning of continuous values.
    exclude : list[str] | None
        Metadata columns to exclude.
    continuous_factor_bins : dict[str, int | list[float]] | None
        Number of uniform bins (int) or explicit bin edges (list[float])
        for specific continuous factors.

    Returns
    -------
    Metadata
        DataEval Metadata instance.
    """
    kwargs = {}
    if auto_bin_method is not None:
        kwargs["auto_bin_method"] = auto_bin_method
    if exclude is not None:
        kwargs["exclude"] = exclude
    if continuous_factor_bins is not None:
        kwargs["continuous_factor_bins"] = continuous_factor_bins

    return Metadata(dataset, **kwargs)
