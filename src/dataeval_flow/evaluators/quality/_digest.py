"""The ``content-digest`` evaluator: SHA-256 digests of every item a Dataset holds, read from the Dataset itself."""

__all__ = ["ContentDigestEvaluator"]

import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

from dataeval_flow._digest import dataset_digest, dataset_manifest
from dataeval_flow._input_spec import InputKind
from dataeval_flow.evaluators._core import execution
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.quality._config import ContentDigestConfig
from dataeval_flow.evaluators.quality._result import ContentDigestOutput


class ContentDigestEvaluator(Evaluator[ContentDigestConfig, ContentDigestOutput]):
    """``content-digest``: a Dataset's content and metadata digests over every item (audit spec §7.1)."""

    name: ClassVar[str] = "content-digest"
    title: ClassVar[str] = "Content Digest"
    description: ClassVar[str] = "SHA-256 digests of every item's image and labels, and of its metadata."
    dataeval_class: ClassVar[Any] = dataset_digest
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.DATASET: "__call__"}

    def run(self, config: ContentDigestConfig, inputs: Sequence[EvaluatorInputs]) -> ContentDigestOutput:  # noqa: ARG002
        """Read every item of the source's Dataset, never through a cache, and digest it."""
        (source,) = inputs
        dataset = require(source.dataset, "its dataset", source.source)
        started, clock = datetime.now(UTC), time.monotonic()
        manifest = dataset_manifest(dataset)
        meta = execution("dataeval_flow.dataset_digest", started, time.monotonic() - clock, {})
        digest = manifest.digest
        return ContentDigestOutput(
            {"content": digest.content, "metadata": digest.metadata, "items": digest.items, "scheme": digest.scheme},
            meta,
            manifest,
        )
