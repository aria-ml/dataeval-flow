"""`split` and `kfold`: DataEval's split_dataset, each part a view of the input (spec §6.1)."""

__all__ = ["KFoldConfig", "KFoldTransform", "SplitConfig", "SplitTransform"]

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import Field, model_validator

from dataeval_flow._chain._identity import indices_digest
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext


class _SplitSettings(TransformConfig, MetadataConfigMixin):
    input: str = Field(description="The Dataset to split.")
    test_frac: float = Field(default=0.0, ge=0.0, lt=1.0, description="The share held out as `test`.")
    stratify: bool = Field(default=False, description="Whether each part keeps the input's class proportions.")
    split_on: list[str] | None = Field(
        default=None, description="Metadata factors whose values never straddle parts, such as a scene or site."
    )


class SplitConfig(_SplitSettings):
    """A `split` step's settings: one train, val and test."""

    val_frac: float = Field(default=0.0, ge=0.0, lt=1.0, description="The share held out as `val`.")

    @model_validator(mode="after")
    def _fractions_leave_a_train(self) -> "SplitConfig":
        if self.test_frac + self.val_frac >= 1.0:
            raise ValueError("`test_frac` and `val_frac` together must leave something to train on.")
        return self


class KFoldConfig(_SplitSettings):
    """A `kfold` step's settings: `folds` train and val pairs, and one test."""

    folds: int = Field(ge=2, description="How many train and val pairs.")


def _parts(
    config: _SplitSettings, inputs: Mapping[str, Any], context: TransformContext, folds: int, val_frac: float
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
        test_frac=config.test_frac,
        val_frac=val_frac,
    )


def _view(dataset: Any, indices: Any) -> Any:
    from dataeval.data import Indices, View

    return View(dataset, Indices([int(index) for index in indices]))


class SplitTransform(Transform[SplitConfig]):
    """``split``: one train, val and test, from DataEval's ``split_dataset`` on the input's metadata."""

    name: ClassVar[str] = "split"
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
        """Each part as a view of the input."""
        splits = _parts(config, inputs, context, folds=1, val_frac=config.val_frac)
        dataset = inputs["input"].value
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


class KFoldTransform(Transform[KFoldConfig]):
    """``kfold``: `folds` train and val pairs, as lists keyed ``"0"``..``"k-1"``, and one test."""

    name: ClassVar[str] = "kfold"
    description: ClassVar[str] = "Splits a Dataset into k train and val folds, and one test."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (
        Port("train", DataType.DATASET, is_list=True),
        Port("val", DataType.DATASET, is_list=True),
        Port("test", DataType.DATASET),
    )

    @classmethod
    def output_keys(cls, config: KFoldConfig) -> Mapping[str, tuple[str, ...]]:
        """``"0"``..``"k-1"`` for both lists."""
        keys = tuple(str(index) for index in range(config.folds))
        return {"train": keys, "val": keys}

    @classmethod
    def empty_outputs(cls, config: KFoldConfig) -> frozenset[str]:
        """``test`` when ``test_frac`` is 0."""
        return frozenset({"test"}) if config.test_frac == 0 else frozenset()

    def run(self, config: KFoldConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        """Each fold's train and val, and the test, as views of the input."""
        splits = _parts(config, inputs, context, folds=config.folds, val_frac=0.0)
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
