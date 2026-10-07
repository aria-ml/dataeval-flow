"""The ``audit`` preset: train and each evaluation split judged before training, with a verdict, a record of what was
audited, and findings under five questions (audit spec §4, §5, §11)."""

__all__ = ["NO_EVALUATION_SPLIT", "PER_EVALUATION_SPLIT", "AuditWorkflow"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

from dataeval_flow.evaluators.quality import (
    ContentDigestConfig,
    DuplicatesConfig,
    FactorLeakageConfig,
    FactorTriageConfig,
    LabelHealthConfig,
    OutliersConfig,
)
from dataeval_flow.evaluators.scope import LabelReconciliationConfig
from dataeval_flow.evaluators.shift import DivergenceConfig, OODKNeighborsConfig
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import NextSteps, Preset, PresetChain, Record, ReportGroup
from dataeval_flow.workflows.audit._config import AuditConfig
from dataeval_flow.workflows.data_bias._workflow import bias_evaluators, factor_steps, gap_steps
from dataeval_flow.workflows.data_coverage._workflow import coverage_evaluators, embedding_steps

if TYPE_CHECKING:
    from dataeval_flow._chain._nodes import Node, NodeList

NO_EVALUATION_SPLIT = "no evaluation split given"
"""Why a check over the evaluation splits judged nothing when the task names train alone."""

_ROLES = ("train", "evals")

PER_EVALUATION_SPLIT = frozenset(
    {
        "class-imbalance-evals",
        "image-outliers-evals",
        "image-duplicates-evals",
        "factor-issues-evals",
        "label-conformance-evals",
        "eval-coverage",
        "embedding-divergence",
        "class-stratification",
    }
)
"""The check steps the chain runs once per evaluation split, so the only ones an `accepted` key `step[split]` can
name; every other check step runs once."""

_GROUPS = (
    ReportGroup("Is the data clean?", ("image-outliers", "image-duplicates", "factor-issues")),
    ReportGroup(
        "Are the labels sound?", ("class-imbalance", "class-sufficiency", "untrained-classes", "label-conformance")
    ),
    ReportGroup(
        "Does the data cover what the model must handle?",
        ("class-coverage", "uncovered-items", "dimensional-completeness", "factor-coverage-gaps"),
        ("factor-summary", "diversity"),
    ),
    ReportGroup("Could the model learn a shortcut?", ("shortcut-risk",)),
    ReportGroup(
        "Are the splits fit to evaluate on?",
        ("leakage", "eval-coverage", "class-stratification", "embedding-divergence"),
    ),
)

_DATA_COVERAGE = "Run data-coverage for the uncovered items; collect data for them."
_DATA_BIAS = "Run data-bias for the gaps; collect data for them."
_RESPLIT = "Re-split so evaluation draws from train's distribution, or document the shift as intended."
_MORE_LABELS = "Collect more of the named classes, or rebalance train with a `view` step and `ClassBalance`."
_NEXT_STEPS = NextSteps(
    by_check={
        "image-outliers": "Run data-cleaning to list and remove them.",
        "image-duplicates": "Run data-cleaning to list and remove them.",
        "factor-issues": "Run metadata-triage for a policy that repairs them.",
        "class-imbalance": _MORE_LABELS,
        "class-sufficiency": _MORE_LABELS,
        "untrained-classes": "Add the named classes to train, or remove them from the evaluation splits.",
        "label-conformance": "Run label-space to resolve the class names.",
        "class-coverage": _DATA_COVERAGE,
        "uncovered-items": _DATA_COVERAGE,
        "factor-coverage-gaps": _DATA_BIAS,
        "dimensional-completeness": _DATA_COVERAGE,
        "shortcut-risk": "Balance the named factors across classes, or confirm they are causal.",
        "leakage": "Re-split with `split`'s `split_on` on the leaking factor, or remove the cross-split duplicates.",
        "eval-coverage": _RESPLIT,
        "embedding-divergence": _RESPLIT,
        "class-stratification": _RESPLIT,
    },
    by_reason={
        NO_EVALUATION_SPLIT: "Give an evaluation split, or make one with data-splitting.",
        "requires an extractor": "Name an extractor to assess these checks.",
        # DataEval's words for metadata with no factors; should they change, the reason reads as "failed" below.
        "No factors found in provided metadata": (
            "Name a metadata policy, or add metadata factors, to assess these checks."
        ),
        "failed": "See why each listed step stopped in the Steps table.",
    },
)


def _each_split(kind: str, type_id: str, reads: str | None = None, **settings: Any) -> list[dict[str, Any]]:
    """`type_id` over train and over every evaluation split: `<type>-train` and `<type>-evals`, each reading its
    split, or the step of type `reads` over that split."""
    return [
        {"name": f"{type_id}-{role}", kind: type_id, "input": f"{reads}-{role}" if reads else role, **settings}
        for role in _ROLES
    ]


class AuditWorkflow(Preset, Workflow[AuditConfig, ChainResult]):
    """Judges train and each evaluation split before training, and gives a verdict over the findings.

    The task's first source is ``train``; every later source is an element of ``evals``, which may be empty. The
    settings expand to:

    - on each split, ``label-health``, ``outliers``, ``duplicates``, ``factor-triage`` and ``content-digest``, with
      ``class-imbalance``, ``image-outliers``, ``image-duplicates`` and ``factor-issues``; and, where ``ontology``
      is set, ``label-reconciliation`` with ``label-conformance``;
    - train with each evaluation split: ``ood-kneighbors`` with ``eval-coverage`` and ``divergence`` with
      ``embedding-divergence`` (optional, asking for train's whole-image embeddings first), and ``duplicates-cross``;
      each pair of evaluation splits: ``duplicates-pairs``; with ``factor-leakage``, the same two for
      ``factor-leakage``; and ``leakage`` over them all;
    - ``class-sufficiency``, ``untrained-classes`` and ``class-stratification``, train's labels against each split's;
    - on train only: ``crops``, then ``coverage`` and ``completeness`` (optional) with ``class-coverage``,
      ``uncovered-items`` under ``naive`` coverage, and ``dimensional-completeness``; ``factor-summary``, ``balance``
      and ``diversity`` (optional) with ``shortcut-risk``; and ``factor-gaps`` (optional) with
      ``factor-coverage-gaps``, unless ``factor-gaps`` is false.

    Every split's metadata is encoded like train's. The report gives the verdict, a record of what was audited, and
    the findings under five questions, with next steps. It makes no Dataset, so it declares no outputs.
    """

    name: ClassVar[str] = "audit"
    title: ClassVar[str] = "Audit"
    description: ClassVar[str] = (
        "Audits one or more splits before training: a verdict, a record of what was audited, and findings under five "
        "questions."
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = (
        "train",
        InputSlot.model_validate({"name": "evals", "list": True, "empty": NO_EVALUATION_SPLIT}),
    )

    @classmethod
    def preflight(cls, config: AuditConfig, inputs: "Mapping[str, Node | NodeList]") -> None:
        """Refuse a split with no items, naming it by its source, or, for a Dataset a chain made, by its slot or
        element key; and splits of different kinds (audit spec §4.1, §13; audit-as-a-step spec §4.2)."""
        from dataeval_flow._chain._graph import GraphError
        from dataeval_flow._chain._nodes import NodeList

        named: list[tuple[str, Node]] = []
        for slot, value in inputs.items():
            if isinstance(value, NodeList):
                named.extend(value.present.items())
            else:
                named.append((value.source or slot, value))
        for name, node in named:
            if len(node.value) == 0:
                raise GraphError(f"Split `{name}` holds no items; an audit judges only splits with data.")
        # A Dataset a chain made carries no kind; check_kinds has judged it already.
        if len({node.kind for _, node in named if node.kind is not None}) > 1:
            listing = ", ".join(f"{name}: {node.kind}" for name, node in named)
            raise GraphError(f"An audit's splits must be one kind: {listing}.")
        evals = inputs.get("evals")
        names = list(evals.present) if isinstance(evals, NodeList) else []
        for accepted in config.accepted:
            if "[" in accepted and (element := accepted[accepted.index("[") + 1 : -1]) not in names:
                raise GraphError(
                    f"`accepted` names `{accepted}`, but this task has no evaluation split `{element}`. Its "
                    f"evaluation splits: {', '.join(names)}."
                )

    @classmethod
    def chain(cls, config: AuditConfig) -> PresetChain:
        """Each split's steps, then train against the evaluation splits, then train's own."""
        c = config.checks
        o = config.outliers
        evaluators: list[Any] = [
            LabelHealthConfig(name="label-health", metadata=config.metadata),
            OutliersConfig(
                name="outliers",
                flags=list(o.flags),
                outlier_threshold=o.outlier_threshold,
                cluster_threshold=o.cluster_threshold,
                cluster_algorithm=o.cluster_algorithm,
                n_clusters=o.n_clusters,
                per_target=False,
                stats=config.stats,
            ),
            DuplicatesConfig(name="duplicates", stats=config.stats),
            FactorTriageConfig(name="factor-triage", metadata=config.metadata),
            ContentDigestConfig(name="content-digest"),
            OODKNeighborsConfig(name="ood-kneighbors", **config.ood_kneighbors.model_dump()),
            DivergenceConfig(name="divergence", method=config.divergence.method),
            *coverage_evaluators(config),
            *bias_evaluators(config),
        ]
        steps: list[dict[str, Any]] = [
            *_each_split("evaluator", "label-health"),
            *_each_split("check", "class-imbalance", "label-health", **c.class_imbalance.model_dump()),
            *_each_split("evaluator", "outliers"),
            *_each_split("check", "image-outliers", "outliers", **c.image_outliers.model_dump()),
            *_each_split("evaluator", "duplicates"),
            *_each_split("check", "image-duplicates", "duplicates", **c.image_duplicates.model_dump()),
            *_each_split("evaluator", "factor-triage"),
            *_each_split("check", "factor-issues", "factor-triage", **c.factor_issues.model_dump()),
            *_each_split("evaluator", "content-digest"),
        ]
        if config.ontology is not None:
            evaluators.append(LabelReconciliationConfig(name="label-reconciliation", ontology=config.ontology))
            steps += [
                *_each_split("evaluator", "label-reconciliation"),
                *_each_split("check", "label-conformance", "label-reconciliation", **c.label_conformance.model_dump()),
            ]
        steps += [  # train with each evaluation split; train's whole-image embeddings are asked for first (spec §4.5)
            {"name": "ood-kneighbors", "evaluator": "ood-kneighbors", "input": ["train", "evals"], "optional": True},
            {
                "name": "eval-coverage",
                "check": "eval-coverage",
                "input": "ood-kneighbors",
                **c.eval_coverage.model_dump(),
            },
            {"name": "divergence", "evaluator": "divergence", "input": ["train", "evals"], "optional": True},
            {
                "name": "embedding-divergence",
                "check": "embedding-divergence",
                "input": "divergence",
                **c.embedding_divergence.model_dump(),
            },
            {"name": "duplicates-cross", "evaluator": "duplicates", "input": ["train", "evals"]},
            {"name": "duplicates-pairs", "evaluator": "duplicates", "input": "evals", "pairs": True},
        ]
        leakage: dict[str, Any] = {
            "name": "leakage",
            "check": "leakage",
            "duplicates": ["duplicates-cross", "duplicates-pairs"],
            **c.leakage.model_dump(),
        }
        if config.factor_leakage is not None:
            evaluators.append(
                FactorLeakageConfig(
                    name="factor-leakage", factors=list(config.factor_leakage.factors), metadata=config.metadata
                )
            )
            steps += [
                {"name": "factor-leakage-cross", "evaluator": "factor-leakage", "input": ["train", "evals"]},
                {"name": "factor-leakage-pairs", "evaluator": "factor-leakage", "input": "evals", "pairs": True},
            ]
            leakage["factors"] = ["factor-leakage-cross", "factor-leakage-pairs"]
        steps += [
            leakage,
            {
                "name": "class-sufficiency",
                "check": "class-sufficiency",
                "input": "label-health-train",
                "evals": "label-health-evals",
                **c.class_sufficiency.model_dump(),
            },
            {
                "name": "untrained-classes",
                "check": "untrained-classes",
                "input": "label-health-train",
                "evals": "label-health-evals",
                **c.untrained_classes.model_dump(),
            },
            {
                "name": "class-stratification",
                "check": "class-stratification",
                "input": "label-health-train",
                "parts": "label-health-evals",
                **c.class_stratification.model_dump(),
            },
            # train only (spec §4.3); detection data is cropped before coverage (spec §4.5)
            *embedding_steps(config, "train"),
            *factor_steps("train"),
            {"name": "shortcut-risk", "check": "shortcut-risk", "input": "balance", **c.shortcut_risk.model_dump()},
            *gap_steps(config, "train"),
        ]
        return PresetChain(
            steps=steps,
            evaluators=evaluators,
            reference="train",
            groups=_GROUPS,
            record=Record("What was audited", ("label-health", "content-digest")),
            blocking=tuple(config.blocking),
            accepted=dict(config.accepted),
            next_steps=_NEXT_STEPS,
        )
