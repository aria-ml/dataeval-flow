"""Write a resolved source out as a dataset, with the provenance that produced it.

An export is not lossless. Detections are rebuilt from the realized corpus rather than
copied from the source records, because a view can filter and relabel them and nothing
then maps an output detection back to the annotation it came from. So an export carries
each box's geometry and class, and drops the source annotation id, `area`,
`segmentation`, `iscrowd` and the per-detection attributes. Read a source's own files
where you need those. Scores are not carried either: an export writes ground truth, and a
`score` records a prediction's confidence.
"""

__all__ = ["build_od_dataset", "export_provenance", "write_export", "write_exports"]

import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import numpy as np
    from datamaite import DatasetMetadata, ImageObjectDetectionSample, ObjectDetectionDataset

    from dataeval_flow.config import PipelineConfig
    from dataeval_flow.config.schemas import ExportConfig, LabelSpaceRecord
    from dataeval_flow.sources import ResolvedSource, SourceOperand
    from dataeval_flow.workflow import ResolvedOntology

_logger: logging.Logger = logging.getLogger(__name__)

#: Keys datamaite writes from the typed fields. Passing them through would duplicate them.
_DERIVED_KEYS = frozenset({"id", "file_name", "width", "height"})

#: Channels datamaite decodes a referenced image to. It reads every one with
#: `cv2.IMREAD_COLOR`, so a realized image of any other channel count is a view's doing.
_DECODED_CHANNELS = 3


class _ImageReference(NamedTuple):
    """How one output sample carries its imagery: by path, or by encoded pixels."""

    file_name: str
    path_or_uri: str | None
    image_bytes: bytes | None
    width: int
    height: int
    split: str | None


def build_od_dataset(
    resolved: "ResolvedSource",
    *,
    dataset_metadata: "DatasetMetadata",
) -> "ObjectDetectionDataset":
    """Materialize a resolved source as the dataset datamaite's writers take.

    Reference images by path wherever the source carries one and its view left the
    imagery alone, so only annotations and metadata are rewritten. Encode the realized
    pixels otherwise — for a source that is not file-backed, and for a view that resized
    or cropped the imagery, where the file on disk no longer matches the boxes.

    Parameters
    ----------
    resolved : ResolvedSource
        The source to write. Its own view is applied here.
    dataset_metadata : DatasetMetadata
        Provenance to attach, including the taxonomy. The taxonomy is filled in from the
        corpus's vocabulary when it is not already set.

    Returns
    -------
    ObjectDetectionDataset
        A concrete dataset whose samples carry integer ids and unique file names.

    Raises
    ------
    ValueError
        If a datum's imagery has to be encoded and is not 8-bit.
    """
    from datamaite import ObjectDetectionDataset

    dataset = resolved.realized()
    index2label = dict(dataset.metadata.get("index2label", {}))
    samples_by_id = _samples_by_datum_id(resolved)

    samples = tuple(
        _sample(ordinal, dataset[ordinal], samples_by_id, index2label, merged=resolved.is_merged)
        for ordinal in range(len(dataset))
    )
    return ObjectDetectionDataset(
        samples=samples,
        dataset_metadata=_with_taxonomy(dataset_metadata, index2label, resolved.name),
        dataset_id=resolved.name,
    )


def _samples_by_datum_id(resolved: "ResolvedSource") -> dict[str, tuple[int, Any]]:
    """Map each datum id to the operand position and datamaite sample behind it.

    Key on the id rather than on position, because a merged source's own view reorders
    and drops datums after the merge. `merge_datasets` prefixes each id with its
    operand's position, so build the same key here. A source whose dataset is not
    datamaite-backed contributes nothing and takes the encoded-bytes path.
    """
    index: dict[str, tuple[int, Any]] = {}
    for position, operand in enumerate(resolved.operands):
        samples = getattr(operand.raw, "samples", None)
        if samples is None:
            continue
        for sample in samples:
            key = f"{position}:{sample.image_id}" if resolved.is_merged else str(sample.image_id)
            index[key] = (position, sample)
    return index


def _file_name(ordinal: int, position: int, source: Any, *, merged: bool) -> str:
    """Name one output image, prefixing the operand position for a merged source.

    Prefix per operand: two operands numbering their images independently both hold
    img_000.png, and the writer would copy both to one path.
    """
    name = source.file_name
    if not name:
        name = PurePosixPath(source.path_or_uri).name if source.path_or_uri else f"{ordinal:08d}.png"
    return f"{position}/{name}" if merged else name


def _declared_size(source: Any, meta: "Mapping[str, Any]") -> "tuple[int | None, int | None]":
    """Return the width and height the source declares for one datum.

    A sample that stores neither falls back to the datum metadata, which datamaite fills
    in by decoding the source image. Both describe the image on disk, not the view's.
    """
    width = source.width if source.width is not None else meta.get("width")
    height = source.height if source.height is not None else meta.get("height")
    return (None if width is None else int(width), None if height is None else int(height))


def _image_reference(
    ordinal: int,
    found: "tuple[int, Any] | None",
    image: "np.ndarray",
    meta: "Mapping[str, Any]",
    *,
    merged: bool,
) -> _ImageReference:
    """Decide how one output sample carries its imagery.

    Reference the file on disk only where the realized image still matches it — same
    size, same channel count. A view operation transforms the imagery a run analyzed
    without touching the file, so referencing that file would emit imagery nobody
    evaluated: `Crop` pairs the original pixels with boxes drawn for smaller ones, and
    `SelectChannels` pairs one band with a file holding three. datamaite decodes every
    image it references as 3-channel RGB, so any other channel count is a view's doing.
    """
    channels, height, width = (int(size) for size in image.shape[:3])
    if found is None:
        _logger.debug("Datum %d has no datamaite sample behind it; encoding its pixels.", ordinal)
        return _ImageReference(f"{ordinal:08d}.png", None, _encode(image), width, height, None)

    position, source = found
    file_name = _file_name(ordinal, position, source, merged=merged)
    if source.path_or_uri is None or channels != _DECODED_CHANNELS or _declared_size(source, meta) != (width, height):
        return _ImageReference(file_name, None, _encode(image), width, height, source.split)
    return _ImageReference(file_name, source.path_or_uri, None, width, height, source.split)


def _sample(
    ordinal: int,
    datum: "tuple[Any, Any, Mapping[str, Any]]",
    samples_by_id: dict[str, tuple[int, Any]],
    index2label: "Mapping[int, str]",
    *,
    merged: bool,
) -> "ImageObjectDetectionSample":
    """Build one output sample from one datum of the realized corpus."""
    from datamaite import ImageObjectDetectionSample

    image, target, meta = datum
    datum_id = str(meta.get("id", ordinal))
    reference = _image_reference(ordinal, samples_by_id.get(datum_id), image, meta, merged=merged)

    return ImageObjectDetectionSample(
        # Allocate an integer: COCO's writer skips a sample whose image_id is not an int,
        # and a merged corpus's ids are strings.
        image_id=ordinal,
        path_or_uri=reference.path_or_uri,
        image_bytes=reference.image_bytes,
        file_name=reference.file_name,
        width=reference.width,
        height=reference.height,
        # Carry the split: a writer resolves `sample.split or default_split`, so dropping
        # it writes every split into one directory and leaks val into train.
        split=reference.split,
        metadata={
            **{k: v for k, v in meta.items() if k not in _DERIVED_KEYS},
            "source_id": datum_id,
        },
        detections=_detections(target, index2label),
    )


def _detections(target: Any, index2label: "Mapping[int, str]") -> tuple[Any, ...]:
    """Convert a MAITE target's xyxy boxes into datamaite's xywh annotations."""
    from datamaite import ObjectDetectionAnnotation

    detections = []
    for box, label in zip(target.boxes, target.labels, strict=True):
        x0, y0, x1, y1 = (float(v) for v in box)
        code = int(label)
        detections.append(
            ObjectDetectionAnnotation(
                bbox=(x0, y0, x1 - x0, y1 - y0),
                category_id=code,
                category_name=index2label.get(code),
            )
        )
    return tuple(detections)


def _encode(image: "np.ndarray") -> bytes:
    """Encode a CHW RGB array as PNG bytes.

    OpenCV arrives with `datamaite[od]`, which is a hard dependency, and is what datamaite
    decodes with. Import it here so the module loads without it.
    """
    import cv2  # type: ignore[import-untyped]
    import numpy as np

    array = np.asarray(image)
    if array.dtype != np.uint8:
        # cv2 casts silently and warns only to stderr, so a 0..1 view would export black.
        raise ValueError(
            f"Cannot export imagery of dtype {array.dtype}. An export writes 8-bit images, and "
            "encoding any other dtype rounds every pixel — a view normalized to 0..1 comes out "
            "black. Drop the view operation that changes the bit depth."
        )
    ok, buffer = cv2.imencode(".png", _encodable(array))
    if not ok:  # pragma: no cover — cv2 encodes any well-formed uint8 array
        raise ValueError("Could not encode an image for export.")
    return bytes(buffer)


def _encodable(array: "np.ndarray") -> "np.ndarray":
    """Lay a CHW array out the way cv2 encodes it, refusing a shape PNG cannot hold.

    cv2 writes one, three and four channels; two raises inside OpenCV. Refuse it here
    with a message that names the count, rather than surfacing that error.
    """
    import numpy as np

    channels = array.shape[0]
    if channels == 1:
        return np.ascontiguousarray(array[0])
    if channels == _DECODED_CHANNELS:
        return np.ascontiguousarray(np.transpose(array, (1, 2, 0))[:, :, ::-1])
    raise ValueError(
        f"Cannot export imagery with {channels} channels. An export writes greyscale or "
        "three-channel images. Drop the view operation that changes the channel count, or "
        "narrow it to one channel."
    )


def _with_taxonomy(
    dataset_metadata: "DatasetMetadata",
    index2label: "Mapping[int, str]",
    source_name: str,
) -> "DatasetMetadata":
    """Fill in the taxonomy from the corpus's vocabulary, keeping any already set."""
    import dataclasses

    from datamaite import CategoryEntry, Taxonomy

    if dataset_metadata.taxonomy is not None:
        return dataset_metadata
    codes = sorted(index2label)
    taxonomy = Taxonomy(
        entries=tuple(CategoryEntry(source_id=code, name=index2label[code]) for code in codes),
        source_dataset=source_name,
        # A conformed vocabulary is a list index, so its codes run from zero. A source read
        # straight from COCO keeps that format's sparse ids, so read the codes rather than
        # assuming either.
        id_density="dense" if codes == list(range(len(codes))) else "sparse",
        ordered_names=tuple(index2label[code] for code in codes),
    )
    return dataclasses.replace(dataset_metadata, taxonomy=taxonomy)


def export_provenance(
    resolved: "ResolvedSource",
    *,
    ontology: "ResolvedOntology | None" = None,
) -> "DatasetMetadata":
    """Build the dataset metadata an export carries.

    Record what a reader needs to answer where this corpus came from: each operand's
    dataset and the mapping that conformed it, the ontology it was conformed to, and the
    label-space digests. The digests are the join — the same values the result envelope
    carries and a coverage audit stamped, so an emitted corpus can be matched back to
    the run and the audit that justified its vocabulary.

    Parameters
    ----------
    resolved : ResolvedSource
        The source being written.
    ontology : ResolvedOntology or None
        Label space the export declared, already resolved. One that failed to load is
        recorded as no ontology, rather than claiming a vocabulary nothing was conformed
        to.

    Returns
    -------
    DatasetMetadata
        Metadata whose ``info`` block holds the provenance. A COCO write round-trips it
        verbatim into the file's own ``info``.
    """
    from datetime import datetime, timezone

    from datamaite import DatasetMetadata

    from dataeval_flow import __version__
    from dataeval_flow.sources import label_space_records

    records = label_space_records([resolved], ontology)
    # Key by the source whose view applied the Relabel. A merged source's own view gets a
    # record under the merged source's own name, which matches no operand: that Relabel
    # conformed the corpus rather than any one operand, so it belongs in `label_space` and
    # not in an operand entry.
    by_source = {record.source: record for record in records}
    name, digest = _ontology_entry(ontology)
    info: dict[str, Any] = {
        "tool": "dataeval-flow",
        "tool_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(),
        "source": resolved.name,
        "ontology": name,
        "ontology_digest": digest,
        "operands": [_operand_entry(operand, by_source.get(operand.source.name)) for operand in resolved.operands],
        "label_space": [record.model_dump(mode="json") for record in records],
    }
    return DatasetMetadata(source_dataset=resolved.name, info=info)


def _ontology_entry(ontology: "ResolvedOntology | None") -> "tuple[str | None, str | None]":
    """Name the ontology a corpus was conformed to, and digest its concepts.

    Read from the resolution rather than from the label-space records, so a corpus that
    needed no Relabel still says which vocabulary its labels are read under. Both are null
    where the export named no ontology, and where the one it named failed to load —
    recording a name then would claim a vocabulary nothing was read against.
    """
    from dataeval_flow.label_space import ontology_digest

    if ontology is None or ontology.ontology is None:
        return (None, None)
    return (ontology.source, ontology_digest(ontology.ontology.ids))


def _operand_entry(operand: "SourceOperand", record: "LabelSpaceRecord | None") -> dict[str, Any]:
    """Describe one operand: the dataset it read, the view it read through, and the remap.

    ``record`` is None where the operand's view conformed nothing, which is a corpus
    merged from sources that already shared a vocabulary.
    """
    return {
        "source": operand.source.name,
        "dataset": operand.source.dataset,
        "view": operand.view_config.name if operand.view_config else None,
        "class_remap": dict(record.class_remap) if record is not None else {},
    }


def write_export(
    export: "ExportConfig",
    config: "PipelineConfig",
    dest_root: "Path",
    *,
    data_dir: "Path | None" = None,
) -> "Path":
    """Write one export under *dest_root*, and return the directory written.

    The directory holds the corpus in the requested format and a `provenance.json`
    sidecar carrying the same mapping the COCO writer embeds in its `info` block.

    Parameters
    ----------
    export : ExportConfig
        The export to write: the source it names, the format, and what to do about a
        destination that already holds a dataset.
    config : PipelineConfig
        Pipeline holding the `sources:`, `datasets:`, `views:` and `ontologies:` pools.
    dest_root : Path
        Directory the export's own directory is created under.
    data_dir : Path or None
        Root a relative dataset or ontology path resolves against.

    Returns
    -------
    Path
        The directory written, ``dest_root / export.name``.

    Raises
    ------
    FileExistsError
        If the destination already holds a dataset and `mode` is `error`.
    ValueError
        If the export names a source the config does not define.
    """
    from datamaite import write

    from dataeval_flow.sources import resolve_source

    # An export names no workflow, so resolve its own ontology. Reuse the orchestrator's
    # resolver so a name, a path and an inline hierarchy all mean here what they mean
    # there, and so an unreadable ontology degrades to provenance without one instead of
    # losing the whole export.
    from dataeval_flow.workflow.orchestrator import _resolve_ontology

    dest = dest_root / export.name
    _refuse_occupied_destination(dest, export.mode)
    _warn_on_missing_ontology(export, config)

    resolved = resolve_source(export.source, config, data_dir=data_dir)
    ontology = _resolve_ontology(export, config, data_dir)
    provenance = export_provenance(resolved, ontology=ontology)
    dataset = build_od_dataset(resolved, dataset_metadata=provenance)

    cleared = export.mode == "replace" and _holds_a_dataset(dest)
    try:
        write(dataset, dest, output_format=export.format, mode=export.mode)
    except Exception:
        if cleared:
            _logger.error(
                "  Export '%s' cleared %s before it failed. `mode: replace` empties the "
                "destination first and does not restore it, so the corpus that was there is gone.",
                export.name,
                dest,
            )
        raise
    _write_provenance(dest, provenance)
    _logger.info("  Wrote export '%s' (%s, %d images) to %s", export.name, export.format, len(dataset.samples), dest)
    return dest


def _holds_a_dataset(dest: "Path") -> bool:
    """Whether *dest* is a directory with anything in it."""
    return dest.is_dir() and any(dest.iterdir())


def _refuse_occupied_destination(dest: "Path", mode: str) -> None:
    """Refuse an occupied destination before the corpus is built.

    datamaite applies the same policy, but only once its writer has walked the dataset —
    by which point every image has been decoded. Check it here so a re-run that cannot
    write is refused immediately instead of paying for a corpus it will throw away.
    """
    if mode != "error" or not _holds_a_dataset(dest):
        return
    raise FileExistsError(
        f"Destination {dest} already exists and is not empty. "
        "Set mode to 'replace' to clear it first, or to 'append' to write into it "
        "(append may leave stale files that a reload of the destination would pick up)."
    )


def _warn_on_missing_ontology(export: "ExportConfig", config: "PipelineConfig") -> None:
    """Warn where the config reads an ontology the export does not.

    An export names no workflow and inherits nothing from one, which is what keeps it
    independent of `tasks:`. But an export written without the ontology a workflow declares
    digests an empty label space, and the emitted corpus then carries an identity the run's
    envelope does not. Name the workflows so you can copy the ontology onto the export.
    """
    if export.ontology is not None or not config.workflows:
        return
    named = [workflow.name for workflow in config.workflows if getattr(workflow, "ontology", None) is not None]
    if not named:
        return
    _logger.warning(
        "  Export '%s' declares no ontology, but these workflows do: %s. Its label-space "
        "digests will not match the run's. Declare the same ontology on the export.",
        export.name,
        ", ".join(named),
    )


def _write_provenance(dest: "Path", provenance: "DatasetMetadata") -> None:
    """Write the provenance sidecar into a written export.

    Only the COCO writer carries `DatasetMetadata.info` into the files it writes; the
    others drop it. Write it beside the corpus for every format, so an export's provenance
    does not depend on the format you asked for. COCO then carries it in both places, and
    the embedded block stays the flat mapping datamaite round-trips.

    The sidecar holds a `runs` list, one entry per write, in the order they were written.
    `mode: append` writes into a directory that already holds a corpus, so a sidecar
    carrying one entry would describe part of that corpus and read as the whole of it.
    A fresh write leaves one entry; every other case keeps what is already recorded.

    Call this after the corpus is written, so a failed write leaves no sidecar describing
    a dataset that is not there.
    """
    import json

    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "provenance.json"
    runs = [*_recorded_runs(path), dict(provenance.info)]
    path.write_text(json.dumps({"runs": runs}, indent=2) + "\n", encoding="utf-8")


def _recorded_runs(path: "Path") -> list[Any]:
    """Return the writes a sidecar already records, or none where it records nothing.

    A sidecar that cannot be read, or that is not shaped this way, is replaced and the
    reason logged: an unreadable sidecar must not cost you the export, and history you
    could have kept must not be dropped in silence.
    """
    import json

    if not path.is_file():
        return []
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _logger.warning("  Could not read %s (%s). Recording this write alone.", path, exc)
        return []
    runs = recorded.get("runs") if isinstance(recorded, dict) else None
    if not isinstance(runs, list):
        _logger.warning("  %s records no 'runs' list. Recording this write alone.", path)
        return []
    return runs


def write_exports(
    config: "PipelineConfig",
    output_dir: "Path",
    *,
    data_dir: "Path | None" = None,
) -> int:
    """Write every declared export under ``<output_dir>/datasets/``.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline holding the `exports:` block and the pools each export reads.
    output_dir : Path
        The run's output directory. Exports are written to its `datasets` subdirectory,
        beside `results`.
    data_dir : Path or None
        Root a relative dataset or ontology path resolves against.

    Returns
    -------
    int
        How many exports failed. Report a failure rather than raising it, so one bad
        export does not cost the run its others — the caller decides what a failure
        means for the exit code.
    """
    if not config.exports:
        return 0

    dest_root = output_dir / "datasets"
    return sum(not _write_one(export, config, dest_root, data_dir) for export in config.exports)


def _write_one(
    export: "ExportConfig",
    config: "PipelineConfig",
    dest_root: "Path",
    data_dir: "Path | None",
) -> bool:
    """Write one export, reporting failure rather than raising it.

    Catch every exception. An export is one unit of work at a batch boundary, and what it
    can raise is open-ended: a `target: 5` in a `Relabel` raises `TypeError`, a
    classification source written as object detection raises `AttributeError`, and each
    writer has failures of its own. Enumerating them is unmaintainable, and one that
    escapes costs the run its other exports and its exit code. Every failure here is
    logged with the export's name and counted into that exit code.
    """
    try:
        write_export(export, config, dest_root, data_dir=data_dir)
    except Exception as exc:  # noqa: BLE001
        _logger.error("  Export '%s' failed: %s", export.name, exc)
        return False
    return True
