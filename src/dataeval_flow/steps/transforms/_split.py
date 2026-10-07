"""`split` and `kfold`: DataEval's split_dataset, each part a view of the input (spec §6.1)."""

__all__ = ["KFoldConfig", "KFoldTransform", "SplitConfig", "SplitTransform"]

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import Field, model_validator

from dataeval_flow._chain._identity import indices_digest
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext
from dataeval_flow.steps.transforms._view import root_indices


class _SplitSettings(TransformConfig, MetadataConfigMixin):
    input: str = Field(description="The Dataset to split.")
    test_frac: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description="The share held out as `test`; 0.2 unless set. DataEval's own default, 0, holds nothing out.",
    )
    stratify: bool = Field(
        default=True,
        description=(
            "Whether each part keeps the input's class proportions; true unless set. DataEval's own default is false."
        ),
    )
    split_on: list[str] | None = Field(
        default=None, description="Metadata factors whose values never straddle parts, such as a scene or site."
    )


class SplitConfig(_SplitSettings):
    """A `split` step's settings: one train, val and test."""

    val_frac: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="The share held out as `val`; 0.1 unless set. DataEval's own default, 0, holds out none.",
    )

    @model_validator(mode="after")
    def _fractions_hold_out_and_leave_a_train(self) -> "SplitConfig":
        if self.test_frac == 0 and self.val_frac == 0:
            raise ValueError("`split` holds nothing out: set `test_frac`, `val_frac` or both.")
        if self.test_frac + self.val_frac >= 1.0:
            raise ValueError("`test_frac` and `val_frac` together must leave something to train on.")
        return self


class KFoldConfig(_SplitSettings):
    """A `kfold` step's settings: `folds` train and val pairs, and one test."""

    folds: int = Field(ge=2, description="How many train and val pairs.")


def _parts(
    config: _SplitSettings,
    inputs: Mapping[str, Any],
    context: TransformContext,
    *,
    folds: int,
    test_frac: float,
    val_frac: float,
) -> Any:
    from dataeval.data import split_dataset

    if context.derive_metadata is None:
        raise RuntimeError("split needs the node's metadata")
    metadata = context.derive_metadata(inputs["input"])
    return split_dataset(
        metadata,
        num_folds=folds,
        stratify=config.stratify,
        split_on=config.split_on,
        test_frac=test_frac,
        val_frac=val_frac,
    )


def _spread(sizes: list[int]) -> int | str:
    """The one size every fold shares, or the spread of sizes across the folds."""
    low, high = min(sizes), max(sizes)
    return low if low == high else f"{low}-{high} (range {high - low})"


def _view(dataset: Any, indices: Any) -> Any:
    from dataeval.data import Indices, View

    return View(dataset, Indices([int(index) for index in indices]))


class SplitTransform(Transform[SplitConfig]):
    """``split``: one train, val and test, from DataEval's ``split_dataset`` on the input's metadata."""

    name: ClassVar[str] = "split"
    title: ClassVar[str] = "Split"
    description: ClassVar[str] = "Splits a Dataset into train, val and test, optionally stratified or grouped."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (
        Port("train", DataType.DATASET),
        Port("val", DataType.DATASET),
        Port("test", DataType.DATASET),
    )

    @classmethod
    def empty_outputs(cls, config: SplitConfig) -> frozenset[str]:
        """``val`` when ``val_frac`` is 0, ``test`` when ``test_frac`` is 0."""
        return frozenset({name for name, frac in (("val", config.val_frac), ("test", config.test_frac)) if frac == 0})

    def run(self, config: SplitConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        """Each part as a view of the input.

        DataEval's one fold always holds out a ``val``: without ``val_frac``, that holdout is the ``test``.
        """
        dataset = inputs["input"].value
        if config.val_frac == 0:
            holdout = _parts(config, inputs, context, folds=1, test_frac=0.0, val_frac=config.test_frac)
            (fold,) = holdout.folds
            return {"train": _view(dataset, fold.train), "val": _view(dataset, ()), "test": _view(dataset, fold.val)}
        splits = _parts(config, inputs, context, folds=1, test_frac=config.test_frac, val_frac=config.val_frac)
        (fold,) = splits.folds
        return {
            "train": _view(dataset, fold.train),
            "val": _view(dataset, fold.val),
            "test": _view(dataset, splits.test),
        }

    def digest(
        self,
        config: SplitConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],
    ) -> str:
        """Each part's indices."""
        return "|".join(indices_digest(outputs[name].resolve_indices()) for name in ("train", "val", "test"))

    def details(
        self,
        config: SplitConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Each part's indices into the dataset at the bottom of the input's views (data-splitting spec §5.4)."""
        return {"indices": {name: root_indices(outputs[name]) for name in ("train", "val", "test")}}

    def section(self, record: Any) -> list[Any]:
        """Each part's size."""
        from dataeval_flow._blocks import Fields

        parts = record.output
        return [Fields(items=[(name.title(), len(parts[name])) for name in ("train", "val", "test")])]


class KFoldTransform(Transform[KFoldConfig]):
    """``kfold``: `folds` train and val pairs, as lists keyed ``"0"`` to ``"k-1"``, and one test."""

    name: ClassVar[str] = "kfold"
    title: ClassVar[str] = "K-Fold"
    description: ClassVar[str] = "Splits a Dataset into k train and val folds, and one test."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (
        Port("train", DataType.DATASET, is_list=True),
        Port("val", DataType.DATASET, is_list=True),
        Port("test", DataType.DATASET),
    )

    @classmethod
    def output_keys(cls, config: KFoldConfig) -> Mapping[str, tuple[str, ...]]:
        """``"0"`` to ``"k-1"`` for both lists."""
        keys = tuple(str(index) for index in range(config.folds))
        return {"train": keys, "val": keys}

    @classmethod
    def empty_outputs(cls, config: KFoldConfig) -> frozenset[str]:
        """``test`` when ``test_frac`` is 0."""
        return frozenset({"test"}) if config.test_frac == 0 else frozenset()

    def run(self, config: KFoldConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        """Each fold's train and val, and the test, as views of the input."""
        splits = _parts(config, inputs, context, folds=config.folds, test_frac=config.test_frac, val_frac=0.0)
        dataset = inputs["input"].value
        return {
            "train": {str(index): _view(dataset, fold.train) for index, fold in enumerate(splits.folds)},
            "val": {str(index): _view(dataset, fold.val) for index, fold in enumerate(splits.folds)},
            "test": _view(dataset, splits.test),
        }

    def digest(
        self,
        config: KFoldConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],
    ) -> str:
        """Every fold's indices, and the test's."""
        parts = [indices_digest(view.resolve_indices()) for name in ("train", "val") for view in outputs[name].values()]
        return "|".join([*parts, indices_digest(outputs["test"].resolve_indices())])

    def details(
        self,
        config: KFoldConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Each fold's train and val indices, and the test's, into the dataset at the bottom of the input's views."""
        return {
            "indices": {
                "train": {key: root_indices(view) for key, view in outputs["train"].items()},
                "val": {key: root_indices(view) for key, view in outputs["val"].items()},
                "test": root_indices(outputs["test"]),
            }
        }

    def section(self, record: Any) -> list[Any]:
        """A row of part sizes per fold, then each part's spread across the folds, the test shared."""
        from dataeval_flow._blocks import Cell, Column, Fields, Table

        parts = record.output
        test = len(parts["test"])
        trains = {key: len(view) for key, view in parts["train"].items()}
        vals = {key: len(view) for key, view in parts["val"].items()}
        rows: list[dict[str, Cell]] = [
            {"fold": key, "train": trains[key], "val": vals[key], "test": test} for key in trains
        ]
        columns = [Column(key=key, header=key.title()) for key in ("fold", "train", "val", "test")]
        spread = Fields(
            items=[
                ("Train", _spread(list(trains.values()))),
                ("Val", _spread(list(vals.values()))),
                ("Test", f"{test} (shared across folds)"),
            ]
        )
        return [Table(columns=columns, rows=rows), spread]
