"""The inline step registries: transforms, combines and checks, built-in tables and entry points (spec §5.2)."""

__all__ = [
    "CHECKS",
    "COMBINES",
    "TRANSFORMS",
    "get_check",
    "get_combine",
    "get_transform",
    "inline_registry",
    "list_checks",
    "list_combines",
    "list_transforms",
]

from typing import Any

from dataeval_flow._registry import Registry
from dataeval_flow.steps._check import Check
from dataeval_flow.steps._combine import Combine
from dataeval_flow.steps._step import Transform

_BUILTINS: dict[str, str] = {
    "conform": "dataeval_flow.steps.transforms._conform:ConformTransform",
    "export": "dataeval_flow.steps.transforms._export:ExportTransform",
    "kfold": "dataeval_flow.steps.transforms._split:KFoldTransform",
    "merge": "dataeval_flow.steps.transforms._merge:MergeTransform",
    "remove": "dataeval_flow.steps.transforms._remove:RemoveTransform",
    "select": "dataeval_flow.steps.transforms._select:SelectTransform",
    "split": "dataeval_flow.steps.transforms._split:SplitTransform",
    "view": "dataeval_flow.steps.transforms._view:ViewTransform",
    "wrap": "dataeval_flow.steps.transforms._wrap:WrapTransform",
}
_COMBINE_BUILTINS: dict[str, str] = {
    "classwise-outliers": "dataeval_flow.steps.combines._classwise:ClasswiseOutliersCombine",
    "factor-deviation": "dataeval_flow.steps.combines._factors:FactorDeviationCombine",
    "factor-predictors": "dataeval_flow.steps.combines._factors:FactorPredictorsCombine",
    "ood-union": "dataeval_flow.steps.combines._ood:OODUnionCombine",
}
_CHECK_BUILTINS: dict[str, str] = {
    "class-imbalance": "dataeval_flow.steps.checks._labels:ClassImbalanceCheck",
    "classwise-outlier-rate": "dataeval_flow.steps.checks._outliers:ClasswiseOutlierRateCheck",
    "completeness-score": "dataeval_flow.steps.checks._coverage:CompletenessScoreCheck",
    "drift": "dataeval_flow.steps.checks._drift:DriftCheck",
    "duplicate-rate": "dataeval_flow.steps.checks._duplicates:DuplicateRateCheck",
    "label-conformance": "dataeval_flow.steps.checks._label_space:LabelConformanceCheck",
    "leaf-coverage": "dataeval_flow.steps.checks._label_space:LeafCoverageCheck",
    "mergeability": "dataeval_flow.steps.checks._alignment:MergeabilityCheck",
    "metadata-issues": "dataeval_flow.steps.checks._triage:MetadataIssuesCheck",
    "ood": "dataeval_flow.steps.checks._ood:OODCheck",
    "ood-agreement": "dataeval_flow.steps.checks._ood:OODAgreementCheck",
    "ontology-structure": "dataeval_flow.steps.checks._label_space:OntologyStructureCheck",
    "outlier-rate": "dataeval_flow.steps.checks._outliers:OutlierRateCheck",
    "stratification": "dataeval_flow.steps.checks._stratification:StratificationCheck",
    "target-outlier-rate": "dataeval_flow.steps.checks._outliers:TargetOutlierRateCheck",
    "uncovered-rate": "dataeval_flow.steps.checks._coverage:UncoveredRateCheck",
}

TRANSFORMS: Registry[Transform[Any]] = Registry(
    kind="transform",
    group="dataeval_flow.transforms",
    base=lambda: Transform,
    builtins=_BUILTINS,
)
COMBINES: Registry[Combine[Any]] = Registry(
    kind="combine",
    group="dataeval_flow.combines",
    base=lambda: Combine,
    builtins=_COMBINE_BUILTINS,
)
CHECKS: Registry[Check[Any]] = Registry(
    kind="check",
    group="dataeval_flow.checks",
    base=lambda: Check,
    builtins=_CHECK_BUILTINS,
)


def get_transform(name: str) -> type[Transform[Any]]:
    """The dataset transform registered as `name`, built-in or plugin.

    Raises
    ------
    ValueError
        When nothing is registered under `name`, or the plugin registered under it failed to load.
    """
    return TRANSFORMS.get(name)


def list_transforms() -> list[type[Transform[Any]]]:
    """Every installed dataset transform, sorted by name."""
    return TRANSFORMS.list()


def get_combine(name: str) -> type[Combine[Any]]:
    """The combine registered as `name`, built-in or plugin.

    Raises
    ------
    ValueError
        When nothing is registered under `name`, or the plugin registered under it failed to load.
    """
    return COMBINES.get(name)


def list_combines() -> list[type[Combine[Any]]]:
    """Every installed combine, sorted by name."""
    return COMBINES.list()


def get_check(name: str) -> type[Check[Any]]:
    """The check registered as `name`, built-in or plugin.

    Raises
    ------
    ValueError
        When nothing is registered under `name`, or the plugin registered under it failed to load.
    """
    return CHECKS.get(name)


def list_checks() -> list[type[Check[Any]]]:
    """Every installed check, sorted by name."""
    return CHECKS.list()


def inline_registry(kind: str) -> Registry[Any]:
    """The registry of `kind`, one of the kinds whose steps write their settings inline: transform, combine, check."""
    return {"transform": TRANSFORMS, "combine": COMBINES, "check": CHECKS}[kind]
