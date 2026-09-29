"""What every task run returns, whichever kind of task it ran.

A workflow and an evaluator build their results differently, but a caller reads either the
same way: :meth:`Result.report` for text, :meth:`Result.to_dict` for the machine-readable
payload, and :meth:`Result.export` to write JSON or YAML. A new output format belongs here,
once, built from those two.
"""

__all__ = ["LabelSpaceRecord", "LineageRecord", "Result", "ResultMetadata"]

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, Self, TypeVar, cast, overload

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python

from dataeval_flow._blocks import Asset, Block, Fields, Paragraph, Scalar, Section, Tree
from dataeval_flow._blocks._html import html_page
from dataeval_flow._blocks._text import DEFAULT_WIDTH, MIN_WIDTH, Frame, render_text

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset

    from dataeval_flow.config._schemas._task import TaskKind


class LabelSpaceRecord(BaseModel):
    """The vocabulary one source's labels were rewritten into.

    Built from the `Relabel` in a source's view, not from an alignment result, so the
    digest can be computed from the config alone. A `data-coverage` audit computes the
    same digest from its own alignment, so a result conformed by that audit's stanza
    carries the audit's value and can be matched back to it.
    """

    source: str = Field(description="Source whose view applied the Relabel.")
    ontology: str | None = Field(
        default=None,
        description=(
            "How the ontology was named: a pool entry's name, a resolved path, or `inline`. "
            "Null where the task named none, or where the one it named failed to load."
        ),
    )
    ontology_digest: str | None = Field(
        default=None,
        description=(
            "Digest of that ontology's concept ids. Null where the task named none, or "
            "where the one it named failed to load."
        ),
    )
    class_remap: Mapping[str, str] = Field(
        default_factory=dict,
        description="Source class name to target concept, as the Relabel applied it.",
    )
    target: Sequence[str] = Field(
        default=(),
        description=(
            "Target vocabulary in index order. Order is the integer indexing, so two "
            "datasets conformed against differently ordered targets carry labels that "
            "mean different things."
        ),
    )
    digest: str = Field(
        default="",
        description=(
            "Identity of this vocabulary, over the ontology digest, the class_remap and "
            "the target. Compare it against a coverage audit's label_space_digest to find "
            "the audit that justified this vocabulary."
        ),
    )


class LineageRecord(BaseModel):
    """Where one Dataset in a chain came from: the step that made it, from which Datasets, and how many items."""

    name: str = Field(
        description="The Dataset's address in the chain, such as `clean`, `split.train` or `kfold.train[0]`."
    )
    step: str | None = Field(default=None, description="The step that made it; `null` for a chain input.")
    type: str | None = Field(default=None, description="That step's type, such as `remove`; `null` for a chain input.")
    inputs: list[str] = Field(default_factory=list, description="The addresses of the Datasets it was made from.")
    source: str | None = Field(default=None, description="For a chain input, the source the task bound to it.")
    digest: str = Field(
        description="12 hex characters of its key: two results whose Datasets share it read the same data."
    )
    items: int = Field(description="How many items it holds.")


class ResultMetadata(BaseModel):
    """A result's JATIC envelope: when and how the result was made, and from what.

    Every result carries one as ``metadata``: the envelope's version and timestamp, the tool that made it, the
    datasets, views, extractor and configuration the run read, and how its metadata was encoded. Once the run
    returns, Flow fills in ``dataset_id``, ``source_descriptions``, ``tool_version``, ``execution_time_s`` and
    ``resolved_config``, and, where they apply, the view, extractor, label and diagnostics fields. The encoding
    fields are the workflow's to set. ``to_dict()`` and ``export()`` write the envelope under ``metadata``.

    Subclassing
    -----------
    A workflow may subclass it to record facts about its run beside the envelope, as pydantic fields with
    descriptions, and name the subclass as the metadata type argument of its
    :class:`~dataeval_flow.workflows.WorkflowResult`. Every added field needs a default: Flow builds a failed
    result's metadata with no arguments. Evaluator results carry an envelope of their own and take no subclass.

    Examples
    --------
    >>> from pydantic import Field
    >>> from dataeval_flow import ResultMetadata
    >>> class CountMetadata(ResultMetadata):
    ...     smallest_source: str | None = Field(default=None, description="The source holding the fewest items.")
    >>> CountMetadata().tool
    'dataeval-flow'
    """

    version: str = Field(default="1.0", description="Version of the envelope's format.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When the result was made, in UTC."
    )
    dataset_id: str | Sequence[str] = Field(
        default="", description="The `datasets:` entries the run read, comma-joined where there are several."
    )
    label_source: str | Sequence[str] | None = Field(
        default=None,
        description=(
            "Where the labels came from, such as `annotations`: one value, or one per operand where a merged "
            "source reads more than one provenance. Null where no source knows its provenance."
        ),
    )
    model_id: str | None = Field(
        default=None, description="The task's extractor, as `name (model)`. Null where the task names none."
    )
    preprocessor_id: str | None = Field(
        default=None, description="The preprocessor the task's extractor applies. Null where it names none."
    )
    selection_id: str | None = Field(
        default=None, description="The views the sources read through, comma-joined. Null where they read none."
    )
    source_descriptions: Sequence[str] = Field(
        default=(), description="Each source the task read, with the datasets and views behind it, in task order."
    )
    resolved_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The configuration the run used: its sources, with their datasets and views expanded; the workflow or "
            "evaluator entry and the extractor as configured, naming any preprocessor, policy or ontology they "
            "use; and the seed, where the pipeline sets one."
        ),
    )
    tool: str = Field(default="dataeval-flow", description="The tool that made the result.")
    tool_version: str = Field(default="", description="The version of dataeval-flow that made the result.")
    execution_time_s: float | None = Field(default=None, description="How long the run took, in seconds.")
    metadata_binning: dict[str, Any] | None = Field(
        default=None,
        description=(
            "How each metadata factor was encoded, and how well that encoding fits. Per factor: its type and "
            "level, the `encoding` applied (edges or vocabulary, who chose them, and how they were placed), the "
            "name each code reads as, and the `fit` this run's rows made against it: counts, occupied spans, and "
            "declared bins nothing reached. Null for workflows that build no metadata. Recorded because the "
            "encoding decides what every evaluator reads, and is otherwise reported only in logs."
        ),
    )
    encoding_digest: str | None = Field(
        default=None,
        description=(
            "Fingerprint of the encoding every factor was read under. Null where the workflow built no metadata "
            "or its splits did not share one. Two runs are comparable only if each can say which cuts produced "
            "it: without this, a bias score that moved cannot be told apart between the override working and "
            "the data changing. Same digest and same data means the numbers are comparable."
        ),
    )
    label_space_digest: str | None = Field(
        default=None,
        description=(
            "Identity of the vocabulary this run's labels were read under. Null when no conforming happened. "
            "The label-space counterpart of `encoding_digest`: a bias score computed over a collapsed "
            "vocabulary differs from one over an uncollapsed vocabulary, and no other field tells them apart. "
            "Also the join key to the `data-coverage` audit whose alignment produced the vocabulary."
        ),
    )
    label_space: Sequence[LabelSpaceRecord] = Field(
        default=(),
        description=(
            "One entry per source whose view conformed its labels, in the order the sources are read. A merge "
            "applies a different `class_remap` per operand against one shared target, so a single entry would "
            "have to union the mappings and would hash to a value no audit produced. Empty for a run that "
            "conformed nothing."
        ),
    )
    diagnostics: Sequence[str] = Field(
        default=(),
        description=(
            "Library diagnostics raised while the run executed: the decisions DataEval made on the caller's "
            "behalf and the ranges it could not resolve. Empty when the run raised none."
        ),
    )


TMetadata = TypeVar("TMetadata", bound=ResultMetadata)
TOutput = TypeVar("TOutput")


def _source_items(meta: ResultMetadata) -> list[tuple[str, Scalar]]:
    """What the run read: every source, or else the dataset and selection."""
    from dataeval_flow.workflows._base import render_label_source

    if meta.source_descriptions:
        return [("Source", "\n".join(meta.source_descriptions))]
    items: list[tuple[str, Scalar]] = []
    if meta.dataset_id:
        label = f"  ({render_label_source(meta.label_source)})" if meta.label_source else ""
        items.append(("Dataset", f"{meta.dataset_id}{label}"))
    if meta.selection_id:
        items.append(("Selection", meta.selection_id))
    return items


def _envelope_items(meta: ResultMetadata) -> list[tuple[str, Scalar]]:
    """The envelope a report opens with: when the run happened, how long it took, and what it read."""
    items: list[tuple[str, Scalar]] = []
    if meta.timestamp:
        items.append(("Timestamp", meta.timestamp.isoformat()))
    if meta.execution_time_s is not None:
        items.append(("Duration", f"{meta.execution_time_s:.2f}s"))
    items.extend(_source_items(meta))
    if meta.model_id:
        items.append(("Model", meta.model_id))
    if meta.preprocessor_id:
        items.append(("Preprocessor", meta.preprocessor_id))
    return items


def failure_message(error: BaseException) -> str:
    """How a failed result records the exception its run raised, whichever kind ran: ``ValueError: msg``."""
    return f"{type(error).__name__}: {error}"


def failure_section(errors: Sequence[str]) -> Section:
    """A failed run's report body: ``FAILED``, then each error."""
    return Section(title="Failed", blocks=[Paragraph(text=error) for error in errors])


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


def _page_title(document: Section) -> str:
    """A report's title on one line, as a page's title bar shows it."""
    return " — ".join(part.strip() for part in document.title.split("\n"))


def results_html(results: Sequence["Result[Any, Any]"], *, detailed: bool = True) -> str:
    """Every result's report on one page, as a run's ``result.html`` holds all of its tasks; *detailed* as `to_html`."""
    # The runner writes one page for the whole run, from the documents each result draws.
    documents = [result._document(detailed=detailed) for result in results]  # noqa: SLF001 - one page, many reports
    title = _page_title(documents[0]) if len(documents) == 1 else "dataeval-flow results"
    return html_page(title, documents, [result.assets for result in results])


class Result(ABC, Generic[TMetadata, TOutput]):
    """One task's result, whichever kind of task ran.

    ``kind`` says which: ``"workflow"`` (a :class:`~dataeval_flow.workflows.WorkflowResult`, which judges
    health) or ``"evaluator"`` (an :class:`~dataeval_flow.evaluators.EvaluatorResult`, which reports DataEval's
    determinations and judges nothing).

    Read :attr:`output` only when ``success`` is true: on a failed run it raises, so a failure can never be read as
    a clean result. :meth:`report`, :meth:`to_dict` and :meth:`export` work either way.

    Parameters
    ----------
    type : str
        The type id of what ran, e.g. ``"data-cleaning"``.
    success : bool
        Whether the run completed.
    metadata : ResultMetadata
        The JATIC envelope.
    output : object, optional
        What the run produced. Required when ``success`` is true, and refused when it is false.
    errors : Sequence[str]
        Why the run failed, or problems a completed run met along the way.
    dataset, sources : AnnotatedDataset, optional
        The resolved, post-view datasets, for Python callers. Never serialized.

    Attributes
    ----------
    kind : {"workflow", "evaluator"}
        Which kind of task made the result; a class variable of each subclass.
    type : str
        The type id of what ran.
    success : bool
        Whether the run completed.
    metadata : ResultMetadata
        The JATIC envelope, which Flow fills in once the run returns.
    errors : list[str]
        Why the run failed, or problems a completed run met along the way.
    dataset : AnnotatedDataset or None
        The dataset a one-source task read, after its view.
    sources : dict[str, AnnotatedDataset] or None
        The datasets a task of several sources read, after their views, by source name.
    assets : list
        Thumbnails of the items the report names, which Flow captures once the run is done. The HTML
        report shows them, and :meth:`to_dict` writes them under ``assets``. Empty where the report names
        no items, or thumbnails were turned off.

    Subclassing
    -----------
    Do not subclass ``Result`` directly: subclass :class:`~dataeval_flow.workflows.WorkflowResult` for a workflow
    or :class:`~dataeval_flow.evaluators.EvaluatorResult` for an evaluator. Each implements how its kind reports
    and serializes a successful run, which is all a direct subclass would add, and Flow builds only those two
    kinds of result.

    Examples
    --------
    >>> from dataeval_flow import load_config, run_tasks
    >>> results = run_tasks(load_config("pipeline.yaml"))  # doctest: +SKIP
    >>> for name, result in results.items():  # doctest: +SKIP
    ...     print(name, result.kind, result.type, "ok" if result.success else result.errors)
    >>> results["clean"].export("output/clean.json")  # doctest: +SKIP
    """

    kind: "ClassVar[TaskKind]"
    metadata_type: ClassVar[type[ResultMetadata]]

    def __init__(
        self,
        *,
        type: str,  # noqa: A002
        success: bool,
        metadata: TMetadata,
        output: TOutput | None = None,
        errors: Sequence[str] = (),
        dataset: "AnnotatedDataset[Any] | None" = None,
        sources: "dict[str, AnnotatedDataset[Any]] | None" = None,
    ) -> None:
        if success and output is None:
            raise ValueError("A successful result carries its output.")
        if not success and output is not None:
            raise ValueError("A failed result carries no output: the run did not complete.")
        self.type = type
        self.success = success
        self.metadata = metadata
        self._output = output
        self.errors: list[str] = list(errors)
        self.dataset = dataset
        self.sources = sources
        self.assets: list[Asset] = []

    @classmethod
    def failed(cls, *, type: str, errors: Sequence[str]) -> Self:  # noqa: A002
        """A failed result of this class, with default metadata for the orchestrator to fill."""
        return cls(type=type, success=False, metadata=cast("TMetadata", cls.metadata_type()), errors=errors)

    @property
    def output(self) -> TOutput:
        """What the run produced. Raises ``RuntimeError`` naming the errors when the run failed."""
        if not self.success or self._output is None:
            raise RuntimeError(f"{self.type} did not complete, so it has no output: " + "; ".join(self.errors))
        return self._output

    def __repr__(self) -> str:
        return f"{type(self).__name__}(type={self.type!r}, success={self.success})"

    def report(self, *, detailed: bool = True, width: int = DEFAULT_WIDTH) -> str:
        """Return a plain-text report: a banner, the run's envelope, the body, then the configuration.

        Parameters
        ----------
        detailed : bool
            When ``False``, the body is the short form the console shows.
        width : int
            Characters per line, at least 40. Prose wraps and charts shrink to fit it.

        Returns
        -------
        str
            Formatted text report suitable for ``print()``.

        Raises
        ------
        ValueError
            If ``width`` is below 40.
        """
        if width < MIN_WIDTH:
            raise ValueError(f"width must be at least {MIN_WIDTH}, got {width}")
        return "\n".join(render_text([self._document(detailed=detailed)], Frame(width=width)))

    def to_html(self, *, detailed: bool = True) -> str:
        """Return the report as one self-contained HTML page, drawn from the blocks the text report draws.

        The page loads nothing from a URL. Its one inline script only adds sorting, filtering and
        expand-all to a page complete without it, so it opens offline, reads the same with scripts
        blocked, and prints (or saves as PDF) the way it shows.

        Parameters
        ----------
        detailed : bool
            When ``False``, the body is the short form the console shows.

        Returns
        -------
        str
            A complete HTML document.
        """
        document = self._document(detailed=detailed)
        return html_page(_page_title(document), [document], [self.assets])

    def to_dict(self) -> dict[str, object]:
        """The result as a plain dict: its kind and envelope, then its output — or, for a failed run, its errors.

        The thumbnails of the items its report names close it, as ``assets``, where it has any.
        """
        body = self._dict_body() if self.success else {"errors": list(self.errors)}
        payload: dict[str, object] = {"kind": self.kind, "metadata": self.metadata.model_dump(mode="json"), **body}
        if self.assets:
            payload["assets"] = [asset.model_dump(mode="json") for asset in self.assets]
        return payload

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

    def _document(self, *, detailed: bool) -> Section:
        """The whole report as blocks: the banner title, the envelope, the body, then the configuration."""
        title = self._report_title() if self.success else self.type
        blocks: list[Block] = [*self._report_envelope(), *self._report_body(detailed=detailed)]
        if self.metadata.resolved_config:
            # As export would write it: a Path or other non-JSON leaf becomes its text, not an error.
            config = to_jsonable_python(self.metadata.resolved_config, fallback=str)
            blocks.append(Section(title="Configuration", reference=True, blocks=[Tree(value=config)]))
        return Section(title=title, blocks=blocks)

    def _report_envelope(self) -> list[Block]:
        """The envelope under the banner, or nothing when the metadata holds none of it."""
        items = _envelope_items(self.metadata)
        return [Fields(items=items)] if items else []

    def _report_body(self, *, detailed: bool) -> list[Block]:
        """The report's body: what this kind reports, or ``FAILED`` and each error for a failed run."""
        return self._report_output(detailed=detailed) if self.success else [failure_section(self.errors)]

    @abstractmethod
    def _report_title(self) -> str:
        """A successful run's banner title; a multi-line title renders one line per banner row."""

    @abstractmethod
    def _report_output(self, *, detailed: bool) -> list[Block]:
        """The body of a successful run's report."""

    @abstractmethod
    def _dict_body(self) -> dict[str, object]:
        """What a successful run of this kind adds to :meth:`to_dict` after ``kind`` and ``metadata``."""
