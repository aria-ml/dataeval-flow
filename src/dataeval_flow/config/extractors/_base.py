"""The extractor framework: the bases an extractor and its config subclass."""

__all__ = ["Extractor", "ExtractorConfig"]

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, GetJsonSchemaHandler, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from dataeval_flow._kind import bind_implementation, check_type_id, default_name, state_type_id

if TYPE_CHECKING:
    from dataeval.protocols import FeatureExtractor


class ExtractorConfig(BaseModel):
    """The settings of one ``extractors:`` entry: its name, the extractor that builds it, and how it runs.

    Each extractor has one config class. A pipeline's ``extractors:`` entry is validated with the config class its
    ``model`` names, and :func:`~dataeval_flow.run` takes an instance as its ``extractor``. A task names the entry
    by ``name``, which defaults to its ``model``. Unknown keys are refused, so a field meant for another extractor
    is caught.

    Subclassing
    -----------
    Subclass it once per extractor. Unlike a workflow or evaluator config it takes no type parameter: the
    extractor names its config instead, as ``Extractor[MeanConfig]``. Define:

    - ``model``: a ``str`` field whose default is the extractor's name (``model: str = "example.mean"``). A
      validator refuses any other value, and the JSON schema states it as a ``const``.
    - The extractor's settings, as pydantic fields, each with ``Field(description=...)``. Embeddings are cached
      under the config's JSON dump, so every setting that changes the embeddings must be a field, and none may be
      excluded from the dump.
    - A model file, if the extractor loads one, as a ``model_path`` field, relative to the data root. Before the
      extractor builds, Flow resolves it there, and in the root's ``models`` folder when it is not found there; it
      keys the cache by the file's contents as well.

    A config has no entry point of its own: Flow finds it through its extractor's ``config_type`` (see
    :class:`Extractor`).

    Examples
    --------
    >>> from dataeval_flow.config.extractors import ExtractorConfig
    >>> class MeanConfig(ExtractorConfig):
    ...     model: str = "example.mean"
    >>> MeanConfig(batch_size=64).name
    'example.mean'

    A pipeline entry for it:

    .. code-block:: yaml

        extractors:
          - name: means
            model: example.mean
            batch_size: 64
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    # Filled from ``model`` before validation when omitted. The factory marks the field optional, so type
    # checkers accept a config built without a name.
    name: str = Field(
        default_factory=str, description="Identifier for the extractor, referenced by tasks. Defaults to its `model`."
    )
    preprocessor: str | None = Field(default=None, description="Reference to a preprocessor name (optional)")
    batch_size: int | None = Field(
        default=None,
        description=(
            "Images per call to the feature extractor. Unset takes DataEval's global batch size, and a run that "
            "needs embeddings fails when that is unset too."
        ),
    )
    # Declared after the shared fields, where every extractor config has always dumped it: the embedding cache
    # keys on the dump, so moving it would miss every cached entry.
    model: str = Field(description="The extractor this entry configures.")

    @model_validator(mode="before")
    @classmethod
    def _default_name(cls, data: Any) -> Any:
        """Name an unnamed entry after its extractor: the ``model`` it gives, else the class's default."""
        return default_name(cls, data, "model")

    @model_validator(mode="after")
    def _model_is_this_class(self) -> Self:
        check_type_id(self, "model")
        return self

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """State the class's extractor name as the ``const`` its ``model`` must equal, and describe ``model``."""
        schema = handler.resolve_ref_schema(super().__get_pydantic_json_schema__(core_schema, handler))
        return state_type_id(cls, schema, "model", base=ExtractorConfig)


class _InstanceExtractorConfig(ExtractorConfig):
    """An extractor given as an object rather than configured: any DataEval ``FeatureExtractor``.

    What ``run(extractor=...)`` builds for an object. Never registered, so it cannot be written in a config file,
    appear in the JSON schema or reach the TUI. Its extractor is used as it is, never serialized, and its
    embeddings and clusters are never cached: two of these dump alike, so a key built from the dump would serve
    one extractor's embeddings to the other.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    model: str = "instance"
    extractor: Any = Field(exclude=True)


ConfigT = TypeVar("ConfigT", bound=ExtractorConfig)


class Extractor(ABC, Generic[ConfigT]):
    """A named way to turn images into embeddings: it builds a DataEval ``FeatureExtractor`` from its config.

    Each ``extractors:`` entry names an extractor by its ``model:``. When a task needs embeddings, or the clusters
    built from them, the extractor builds the feature extractor the entry describes, and Flow runs it over the
    source's images and caches what it returns. :func:`list_extractors` lists the installed extractors.

    Subclassing
    -----------
    Parameterize ``Extractor`` with the extractor's config class, ``class MeanExtractor(Extractor[MeanConfig])``,
    which binds ``config_type`` when the class is defined. The argument must be given to ``Extractor`` itself: an
    abstract base of your own may take it for its subclasses, but a generic one (``class Shared(Extractor[C])``,
    subclassed as ``Shared[MeanConfig]``) is refused. Then define:

    - ``name: ClassVar[str]``: the YAML ``model:`` value. It must equal the config's ``model`` default and the
      entry-point name.
    - ``description: ClassVar[str]``: one line, for listings.
    - :meth:`build`: returns the DataEval feature extractor.
    - ``stateful: ClassVar[bool] = True``, only when the feature extractor fits state on the first data it is
      given (BoVW fits a vocabulary). Every source in a task then shares one instance, fitted on the first source
      to ask, so all are described in one basis. A stateful fit must follow from the config, the first batch the
      extractor is given and DataEval's seed (``dataeval.config.get_seed()``) alone: its cached embeddings are
      keyed by those, so a fit that also depends on anything else, such as its own random generator, can be served
      under the wrong fit.

    A concrete extractor without ``name`` or ``description``, or not parameterized with its config, raises
    ``TypeError`` when the class is defined. Register the class under the ``dataeval_flow.extractors`` entry-point
    group, named by ``name``. Flow loads it on the first registry lookup and leaves it out, logging why, when it
    fails to import, is not an ``Extractor``, its entry-point name, ``name`` and the config's ``model`` default
    disagree, or another extractor has its name.

    Flow builds an instance with no arguments and calls :meth:`build` whenever it computes embeddings rather than
    reading them from its cache; a stateful extractor is built once per task instead. Flow guarantees that:

    - ``config`` is the entry's config, validated when it was built or loaded, with any ``model_path`` resolved
      against the data root;
    - ``transforms`` is the entry's preprocessor, or ``None`` when it names none;
    - the feature extractor is called with lists of the source's images, ``batch_size`` at a time. An entry with
      no ``batch_size`` takes DataEval's global one (``dataeval.config.set_batch_size``), and the run that asks for
      its embeddings fails when that is unset too;
    - its embeddings are cached under the config, the preprocessor's ``repr`` and the data;
    - an exception raised by :meth:`build`, or by the feature extractor it returns, reaches the workflow or
      evaluator run that asked for the embeddings, which then ends as a failed result unless a workflow catches it.

    Examples
    --------
    An extractor that embeds each image as its per-channel means, with the config it reads:

    >>> from typing import Any, ClassVar
    >>> import numpy as np
    >>> from dataeval_flow.config.extractors import Extractor, ExtractorConfig
    >>> class MeanConfig(ExtractorConfig):
    ...     model: str = "example.mean"
    >>> class Means:
    ...     def __init__(self, transforms: Any) -> None:
    ...         self.transforms = transforms
    ...
    ...     def __call__(self, images: Any, /) -> Any:
    ...         if self.transforms is not None:
    ...             images = [self.transforms(image) for image in images]
    ...         return np.stack([np.asarray(image, dtype=np.float32).mean(axis=(-2, -1)) for image in images])
    >>> class MeanExtractor(Extractor[MeanConfig]):
    ...     name: ClassVar[str] = "example.mean"
    ...     description: ClassVar[str] = "Per-channel mean of each image."
    ...
    ...     def build(self, config: MeanConfig, transforms: Any) -> Any:
    ...         return Means(transforms)
    >>> MeanExtractor().build(MeanConfig(), None)([np.ones((3, 2, 2))])
    array([[1., 1., 1.]], dtype=float32)

    Register it in the plugin's ``pyproject.toml``:

    .. code-block:: toml

        [project.entry-points."dataeval_flow.extractors"]
        "example.mean" = "my_package:MeanExtractor"

    Once the plugin is installed, a run embeds with it like a built-in:

    >>> from dataeval_flow import run
    >>> from dataeval_flow.evaluators.quality import DuplicatesConfig
    >>> extractor = MeanConfig(batch_size=64)
    >>> result = run(DuplicatesConfig(cluster_sensitivity=1.0), dataset, extractor=extractor)  # doctest: +SKIP
    """

    name: ClassVar[str]
    description: ClassVar[str]
    config_type: ClassVar[type[ExtractorConfig]]
    stateful: ClassVar[bool] = False  # True when it fits state (e.g. a vocabulary) on the data it first sees

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bind ``config_type`` from the type arguments, and require identity on a concrete extractor."""
        super().__init_subclass__(**kwargs)
        bind_implementation(cls, Extractor, with_result=False)

    @abstractmethod
    def build(self, config: ConfigT, transforms: "Callable[[Any], Any] | None") -> "FeatureExtractor":
        """Build the DataEval feature extractor `config` describes, applying `transforms` to each item first.

        Parameters
        ----------
        config : ConfigT
            The entry's settings, an instance of ``config_type``.
        transforms : Callable or None
            The preprocessing the entry's ``preprocessor`` names, one CHW array in and one out, or ``None`` when it
            names none.

        Returns
        -------
        FeatureExtractor
            A callable that takes a list of images and returns their embeddings, one row per image.

        Raises
        ------
        Exception
            Anything, to fail the run that asked for the embeddings.
        """
