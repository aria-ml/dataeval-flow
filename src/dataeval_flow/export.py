"""Write a resolved source out as a dataset, with the provenance that produced it.

An export is not lossless. Detections are rebuilt from the realized corpus rather than
copied from the source records, because a view can filter and relabel them and nothing
then maps an output detection back to the annotation it came from. So an export carries
each box's geometry, class and score, and drops the source annotation id, `area`,
`segmentation`, `iscrowd` and the per-detection attributes. Read a source's own files
where you need those.
"""

__all__ = ["build_od_dataset"]

import logging
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from datamaite import DatasetMetadata, ImageObjectDetectionSample, ObjectDetectionDataset

    from dataeval_flow.sources import ResolvedSource

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
