"""Metadata convenience builder wrapping DataEval."""

from typing import TYPE_CHECKING

from dataeval import Metadata

if TYPE_CHECKING:
    from dataeval_app.config.schemas.task import AutoBinMethod
    from dataeval_app.dataset import MaiteDataset

__all__ = [
    "build_metadata",
]


def build_metadata(
    dataset: "MaiteDataset",
    auto_bin_method: "AutoBinMethod | None" = None,
    exclude: list[str] | None = None,
    continuous_factor_bins: dict[str, list[float]] | None = None,
) -> Metadata:
    """Build Metadata from dataset and config.

    Parameters
    ----------
    dataset : MaiteDataset
        Input dataset.
    auto_bin_method : AutoBinMethod | None
        Method for automatic binning of continuous values.
    exclude : list[str] | None
        Metadata columns to exclude.
    continuous_factor_bins : dict[str, list[float]] | None
        Manual bin edges for specific continuous factors.

    Returns
    -------
    Metadata
        DataEval Metadata instance.
    """
    # Build kwargs dict, omitting None values so DataEval uses its own defaults.
    # MaiteDataset conforms to DataEval's dataset protocol at runtime but
    # pyright can't verify cross-library structural conformance.
    kwargs: dict[str, object] = {}
    if auto_bin_method is not None:
        kwargs["auto_bin_method"] = auto_bin_method
    if exclude is not None:
        kwargs["exclude"] = exclude
    if continuous_factor_bins is not None:
        kwargs["continuous_factor_bins"] = continuous_factor_bins

    return Metadata(dataset, **kwargs)  # type: ignore[arg-type]
