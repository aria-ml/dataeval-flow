"""Selection convenience builder wrapping DataEval."""

from typing import TYPE_CHECKING

import dataeval.selection as sel
from dataeval.selection import Select

if TYPE_CHECKING:
    from dataeval_app.config.schemas.selection import SelectionStep
    from dataeval_app.dataset import MaiteDataset

__all__ = [
    "build_selection",
]


def build_selection(dataset: "MaiteDataset", steps: list["SelectionStep"]) -> Select:
    """Build selection pipeline from config.

    Pass-through to dataeval.selection - no custom logic.

    Parameters
    ----------
    dataset : MaiteDataset
        Input dataset to wrap with selections.
    steps : list[SelectionStep]
        Selection steps from config.

    Returns
    -------
    Select
        Dataset wrapped with selection criteria.

    Example
    -------
    >>> from dataeval_app.config.schemas.selection import SelectionStep
    >>> steps = [
    ...     SelectionStep(type="Limit", params={"size": 10000}),
    ...     SelectionStep(type="ClassFilter", params={"classes": [0, 1, 2]}),
    ... ]
    >>> filtered = build_selection(dataset, steps)
    """
    selections = []
    for step in steps:
        selection_cls = getattr(sel, step.type, None)
        if selection_cls is None:
            raise ValueError(f"Unknown selection type: '{step.type}'. Check dataeval.selection docs.")
        selections.append(selection_cls(**step.params))

    # MaiteDataset conforms to DataEval's dataset protocol at runtime (duck typing);
    # pyright can't verify cross-library structural conformance.
    return Select(dataset, selections=selections)  # type: ignore[arg-type]
