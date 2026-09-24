"""What every task run returns, whichever kind of task it ran.

A workflow and an evaluator build their results differently, but a caller reads either the
same way: :meth:`Result.report` for text, :meth:`Result.to_dict` for the machine-readable
payload, and :meth:`Result.export` to write JSON or YAML. A new output format belongs here,
once, built from those two.
"""

__all__ = ["Result"]

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar, overload

from dataeval_flow.config.schemas import ResultMetadata, TaskKind

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset

# The text helpers below are imported where they are used: they live in the workflow package,
# which imports this module to define WorkflowResult, so importing them here would be circular.

TMetadata = TypeVar("TMetadata", bound=ResultMetadata)


def _source_lines(meta: ResultMetadata) -> list[str]:
    """Render source/dataset lines for the metadata block."""
    from dataeval_flow.workflow.base import render_label_source

    lines: list[str] = []
    source_descs = meta.source_descriptions
    if source_descs:
        label = "  Source:       "
        continuation = " " * len(label)
        for i, desc in enumerate(source_descs):
            lines.append(f"{label if i == 0 else continuation}{desc}")
    elif meta.dataset_id or meta.selection_id:
        if meta.dataset_id:
            ds_line = f"  Dataset:      {meta.dataset_id}"
            if meta.label_source:
                ds_line += f"  ({render_label_source(meta.label_source)})"
            lines.append(ds_line)
        if meta.selection_id:
            lines.append(f"  Selection:    {meta.selection_id}")
    return lines


def _metadata_block(meta: ResultMetadata) -> list[str]:
    """Human-readable envelope lines, ending with a separator when there are any."""
    from dataeval_flow.workflow._text_report import _WIDTH

    lines: list[str] = []
    if meta.timestamp:
        lines.append(f"  Timestamp:    {meta.timestamp.isoformat()}")
    if meta.execution_time_s is not None:
        lines.append(f"  Duration:     {meta.execution_time_s:.2f}s")
    lines.extend(_source_lines(meta))
    if meta.model_id:
        lines.append(f"  Model:        {meta.model_id}")
    if meta.preprocessor_id:
        lines.append(f"  Preprocessor: {meta.preprocessor_id}")
    if lines:
        lines.append("-" * _WIDTH)
    return lines


def _failure_lines(errors: Sequence[str]) -> list[str]:
    """A failed run's report body: ``FAILED``, then each error."""
    return ["  FAILED", *(f"    {error}" for error in errors)]


def _write_result(payload: dict[str, object], path: str | Path | None, *, fmt: Literal["json", "yaml"]) -> str | Path:
    """Serialize a result dict to JSON or YAML, as a string or into a file.

    A directory (or a suffix-less path) receives ``results.<ext>``.
    """
    if fmt == "json":
        from json import dumps

        content = dumps(payload, indent=2)
        ext = "json"
    else:
        from yaml import safe_dump

        content = safe_dump(payload, default_flow_style=False)
        ext = "yaml"

    if path is None:
        return content

    dest = Path(path)
    if dest.is_dir() or not dest.suffix:
        dest.mkdir(parents=True, exist_ok=True)
        dest = dest / f"results.{ext}"
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)

    dest.write_text(content, encoding="utf-8")
    return dest


@dataclass
class Result(ABC, Generic[TMetadata]):
    """One task's result, whichever kind of task ran.

    ``kind`` says which: ``"workflow"`` (a :class:`~dataeval_flow.workflow.result.WorkflowResult`,
    which judges health) or ``"evaluator"`` (an :class:`~dataeval_flow.evaluator.result.EvaluatorResult`,
    which reports DataEval's determinations and judges nothing). Health belongs to workflows
    alone, so it is not here.

    ``metadata`` carries the JATIC envelope. ``dataset`` and ``sources`` hold the resolved,
    post-selection datasets for Python callers and are never serialized.

    ``metadata`` and the fields after it are keyword-only. Dataclass inheritance puts shared
    fields first, and this keeps each kind's own payload (``data``, ``output``) positionally
    next to ``success``.
    """

    kind: ClassVar[TaskKind]

    name: str
    success: bool
    metadata: TMetadata = field(kw_only=True)
    errors: Sequence[str] = field(default_factory=list, kw_only=True)
    dataset: "AnnotatedDataset[Any] | None" = field(default=None, kw_only=True)
    sources: "dict[str, AnnotatedDataset[Any]] | None" = field(default=None, kw_only=True)

    def report(self, *, detailed: bool = True) -> str:
        """Return a plain-text report: a banner, the run's envelope, the body, then the configuration.

        Parameters
        ----------
        detailed : bool
            When ``False``, the body is the short form the console shows.

        Returns
        -------
        str
            Formatted text report suitable for ``print()``.
        """
        from dataeval_flow.workflow._text_report import _WIDTH, _render_config_section

        lines = ["", "=" * _WIDTH]
        lines.extend(f"  {part.strip().upper()}" for part in self._report_title().split("\n"))
        lines.append("=" * _WIDTH)
        lines.extend(self._report_envelope())
        lines.extend(self._report_body(detailed=detailed))
        lines.extend(_render_config_section(self.metadata.resolved_config))
        lines.extend(["", "=" * _WIDTH])
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Return the result as a plain dict: its kind, the envelope, then what this kind reports."""
        return {"kind": self.kind, "metadata": self.metadata.model_dump(mode="json"), **self._dict_body()}

    @overload
    def export(self, path: str | Path, *, fmt: Literal["json", "yaml"] = "json") -> Path: ...
    @overload
    def export(self, path: None = None, *, fmt: Literal["json", "yaml"] = "json") -> str: ...

    def export(self, path: str | Path | None = None, *, fmt: Literal["json", "yaml"] = "json") -> str | Path:
        """Serialize :meth:`to_dict` to JSON or YAML.

        Parameters
        ----------
        path : str | Path | None
            File or directory path. A directory receives ``results.<ext>``. ``None`` returns
            the serialized string.
        fmt : {"json", "yaml"}
            Serialization format. Defaults to ``"json"``.

        Returns
        -------
        str | Path
            The string when ``path`` is ``None``, otherwise the written file's path.
        """
        return _write_result(self.to_dict(), path, fmt=fmt)

    def _report_envelope(self) -> list[str]:
        """The envelope lines under the banner, ending with a separator."""
        return _metadata_block(self.metadata)

    def _report_body(self, *, detailed: bool) -> list[str]:
        """The report's body: what this kind reports, or ``FAILED`` and each error for a failed run."""
        return self._report_output(detailed=detailed) if self.success else _failure_lines(self.errors)

    @abstractmethod
    def _report_title(self) -> str:
        """The banner's title; a multi-line title renders one line per banner row."""

    @abstractmethod
    def _report_output(self, *, detailed: bool) -> list[str]:
        """The body of a successful run's report."""

    @abstractmethod
    def _dict_body(self) -> dict[str, object]:
        """What this kind adds to :meth:`to_dict` after ``kind`` and ``metadata``."""
