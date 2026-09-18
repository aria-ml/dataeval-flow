"""Embeddings convenience builder wrapping DataEval."""

__all__ = [
    "build_embeddings",
    "build_extractor",
    "fitting_source",
    "is_stateful_extractor",
    "shared_extractor_scope",
]

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from dataeval import Embeddings
from dataeval.extractors import BoVWExtractor, FlattenExtractor, OnnxExtractor, TorchExtractor
from dataeval.protocols import AnnotatedDataset

_logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dataeval_flow.config.schemas import ExtractorConfig

# One entry per extractor identity within a task: the instance every source shares, and
# the selection that fitted it.
_shared_extractors: ContextVar["dict[str, _SharedExtractor] | None"] = ContextVar(
    "dataeval_flow_shared_extractors", default=None
)


class _SharedExtractor:
    """A stateful extractor and the selection whose data defined its state."""

    __slots__ = ("extractor", "fitted_by")

    def __init__(self, extractor: Callable) -> None:
        self.extractor = extractor
        self.fitted_by: str | None = None


def is_stateful_extractor(extractor_config: "ExtractorConfig") -> bool:
    """Whether this extractor's output depends on data it has already seen.

    BoVW derives a visual vocabulary from the images it is first given and then describes
    everything else in it, so two instances built from one config are not interchangeable:
    each clusters its own codebook and the histograms they produce share no basis. The
    pretrained extractors carry their representation with them and have no such state.
    """
    return getattr(extractor_config, "model", None) == "bovw"


@contextmanager
def shared_extractor_scope() -> Iterator[None]:
    """Build each stateful extractor once for everything inside this scope.

    A task compares sources with each other, so they have to be described the same way.
    The first source to ask for embeddings fits the extractor and the rest reuse it,
    which is what makes a reference/incoming comparison mean anything.
    """
    token = _shared_extractors.set({})
    try:
        yield
    finally:
        _shared_extractors.reset(token)


def fitting_source(extractor_key: str, selection_key: str) -> str | None:
    """The selection that fitted the extractor identified by `extractor_key`.

    Claims `selection_key` as the fitter when nothing has fitted it yet. Returns None
    outside a scope, where nothing is shared and each caller fits its own.

    Callers fold the result into their cache key: embeddings computed under one
    vocabulary must not be served to a run whose vocabulary came from elsewhere, and the
    extractor config alone cannot tell those apart -- it is identical either way.
    """
    registry = _shared_extractors.get()
    if registry is None:
        return None
    entry = registry.get(extractor_key)
    if entry is None:
        return None
    if entry.fitted_by is None:
        entry.fitted_by = selection_key
    return entry.fitted_by


def _make_resize_transform(height: int, width: int) -> Callable:
    """Build a CHW-image resize transform to ``(height, width)`` for IR-3.1-S-4.

    The returned callable bilinearly resizes a single CHW image so that a
    user-imposed model input size overrides the model's native input size.
    """

    def _resize(image: Any) -> Any:
        import numpy as np

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "Resizing ONNX model inputs (image_height/image_width) requires torch; "
                "install dataeval-flow[cpu] or a cuda extra."
            ) from exc

        tensor = torch.as_tensor(np.asarray(image)).float()
        if tensor.ndim != 3:
            raise ValueError(f"ONNX input resize expects CHW images; got shape {tuple(tensor.shape)}")
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
        return resized.squeeze(0).numpy()

    return _resize


def build_embeddings(
    dataset: AnnotatedDataset[Any],
    extractor_config: "ExtractorConfig",
    transforms: Callable | None = None,
    batch_size: int | None = None,
) -> Embeddings:
    """Build Embeddings from dataset and extractor config.

    Creates the appropriate extractor based on the config's model type and
    wraps it in a DataEval Embeddings instance.

    Parameters
    ----------
    dataset : MaiteDataset
        Input dataset.
    extractor_config : ExtractorConfig
        Extractor configuration with model type and params.
    transforms : Callable | None
        Preprocessing transforms to apply before encoding.
        Only used by extractor types that accept it (onnx, torch, uncertainty).

    Returns
    -------
    Embeddings
        DataEval Embeddings instance (implements FeatureExtractor).
    """

    extractor = build_extractor(extractor_config, transforms)
    return Embeddings(dataset, extractor=extractor, batch_size=batch_size)


def _shared_key(extractor_config: "ExtractorConfig", transforms: Callable | None) -> str:
    """Identity of an extractor built from this config and preprocessing."""
    return f"{extractor_config.model_dump_json()}|{transforms!r}"


def build_extractor(extractor_config: "ExtractorConfig", transforms: Callable | None = None) -> Callable:
    """Build a standalone extractor (not wrapped in Embeddings).

    Used for workflows that need to apply the extractor separately from embedding extraction
    (e.g. to extract metadata features for evaluation).

    Parameters
    ----------
    extractor_config : ExtractorConfig
        Extractor configuration with model type and params.
    transforms : Callable | None
        Preprocessing transforms to apply before encoding.
        Only used by extractor types that accept it (onnx, torch, uncertainty).

    Returns
    -------
    Callable
        A callable extractor function that takes a dataset and returns extracted features.
    """
    _logger.debug("Building %s extractor", extractor_config.model)

    registry = _shared_extractors.get()
    if registry is not None and is_stateful_extractor(extractor_config):
        key = _shared_key(extractor_config, transforms)
        entry = registry.get(key)
        if entry is None:
            entry = _SharedExtractor(_construct_extractor(extractor_config, transforms))
            registry[key] = entry
        return entry.extractor

    return _construct_extractor(extractor_config, transforms)


def _construct_extractor(extractor_config: "ExtractorConfig", transforms: Callable | None = None) -> Callable:
    """Build a new extractor instance, ignoring any sharing scope."""
    from dataeval_flow.config.schemas._extractor import (
        BoVWExtractorConfig,
        FlattenExtractorConfig,
        OnnxExtractorConfig,
        TorchExtractorConfig,
    )

    if isinstance(extractor_config, OnnxExtractorConfig):
        # IR-3.1-S-4: honor user-imposed model input size. The pinned dataeval
        # OnnxExtractor takes its input size from the model, so we resize via the
        # transform pipeline instead: append a resize (applied last, after any
        # user preprocessing) when both height and width are configured. The
        # config validator guarantees they are set together.
        onnx_transforms = transforms
        if extractor_config.image_height is not None and extractor_config.image_width is not None:
            # Append the resize last, keeping any user preprocessing ahead of it.
            resize = _make_resize_transform(extractor_config.image_height, extractor_config.image_width)
            onnx_transforms = [t for t in (transforms,) if t is not None] + [resize]
        extractor = OnnxExtractor(
            extractor_config.model_path,
            transforms=onnx_transforms,
            output_name=extractor_config.output_name,
            flatten=extractor_config.flatten,
        )
    elif isinstance(extractor_config, BoVWExtractorConfig):
        extractor = BoVWExtractor(vocab_size=extractor_config.vocab_size)
    elif isinstance(extractor_config, FlattenExtractorConfig):
        extractor = FlattenExtractor()
    elif isinstance(extractor_config, TorchExtractorConfig):
        extractor = _build_torch_extractor(extractor_config, transforms)
    else:
        raise ValueError(
            f"Extractor type '{extractor_config.model}' is not yet implemented. "
            f"Currently supported: onnx, bovw, flatten, torch."
        )
    return extractor


def _build_torch_extractor(
    config: Any,
    transforms: Callable | None = None,
) -> TorchExtractor:
    """Build a TorchExtractor from config, loading the model from disk.

    The *transforms* from ``build_preprocessing`` is a numpy→numpy wrapper.
    ``TorchExtractor`` handles tensor conversion internally and expects raw
    torchvision transforms, so we unwrap the ``v2.Compose`` when possible.
    """
    import torch
    from torchvision.transforms import v2

    # Unwrap the numpy wrapper produced by build_preprocessing to get the
    # raw v2.Compose that TorchExtractor expects (it handles tensor
    # conversion internally via torch.as_tensor).
    torch_transforms: v2.Compose | Callable | None = None
    if transforms is not None:
        inner = getattr(transforms, "__wrapped__", None)
        if isinstance(inner, v2.Compose):
            torch_transforms = inner
        elif isinstance(transforms, v2.Compose):
            torch_transforms = transforms
        else:
            torch_transforms = transforms

    model = torch.load(config.model_path, map_location="cpu", weights_only=False)
    return TorchExtractor(
        model,
        transforms=torch_transforms,
        device=config.device,
        layer_name=config.layer_name,
        use_output=config.use_output,
    )
