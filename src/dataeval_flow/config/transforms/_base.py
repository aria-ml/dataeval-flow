"""The transform framework: the base a preprocessing step's transform subclasses."""

__all__ = ["Transform"]

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from dataeval_flow._kind import bind_implementation


class Transform(ABC):
    """One preprocessing step, named in YAML by ``step:``: a DataEval ``Transform`` that Flow can find by name.

    A preprocessor (:class:`~dataeval_flow.config.PreprocessorConfig`) is the list of steps an extractor applies to
    each image before it embeds it, and each step names a transform. A step resolves a registered transform first,
    then a ``torchvision.transforms.v2`` transform of that name. The steps run in one torchvision ``v2.Compose``,
    registered transforms alongside torchvision's own.

    A ``Transform`` satisfies DataEval's ``Transform`` protocol (``dataeval.protocols.Transform``): a callable that
    takes one item and returns it transformed. It is not a ``torchvision.transforms.v2.Transform``, which is a
    ``torch.nn.Module`` that dispatches on the types of its inputs; a Flow transform is handed the image tensor
    itself. A torchvision transform needs no registration, since a step can name it directly. To give a
    ``v2.Transform`` subclass of your own a step name, construct it in ``__init__`` and call it from
    :meth:`__call__`.

    Subclassing
    -----------
    Subclass ``Transform`` directly. It takes no type parameters and has no config class: a step's ``params`` are
    passed to the constructor as keyword arguments, as written in the config, the way a torchvision step's are.
    Define:

    - ``name: ClassVar[str]``: the YAML ``step:`` value. It must equal the entry-point name, and can never be a
      ``torchvision.transforms.v2`` name: a step naming one means torchvision's transform, so the registry refuses
      a plugin that takes one, and installing a plugin never changes what a step means. A prefix of your own, as
      in ``example.Invert``, keeps clear of torchvision's names and other plugins'.
    - ``description: ClassVar[str]``: one line, for listings.
    - ``__init__``, when the transform takes parameters, accepting them as keywords.
    - :meth:`__call__`: the transform.
    - ``__repr__``, stable across runs and naming every parameter, as torchvision's transforms have. The embedding
      cache keys on the preprocessor's ``repr``: one that changes between runs misses the cache every run, and one
      that leaves out a parameter can serve embeddings computed under another value of it.

    A concrete transform without ``name`` or ``description`` raises ``TypeError`` when the class is defined.
    Register the class under the ``dataeval_flow.transforms`` entry-point group, named by ``name``. Flow loads it on
    the first registry lookup and leaves it out, logging why, when it fails to import, is not a ``Transform``, is
    registered under another name than its own, takes a torchvision name, or takes a name another transform has.

    Flow builds each step's transform when a task whose extractor names the preprocessor starts, and hands the
    preprocessor to the extractor, which applies it to each image it embeds (the built-in ``flatten`` and ``bovw``
    extractors take no preprocessing). Flow guarantees that:

    - the constructor receives the step's ``params`` as written: the ``dtype`` and ``interpolation`` conversions
      Flow makes for torchvision's transforms do not apply;
    - :meth:`__call__` receives one image as a CHW ``torch.Tensor``, and must return a tensor;
    - an exception raised by :meth:`__call__` reaches the workflow or evaluator run that asked for the embeddings,
      which then ends as a failed result unless a workflow catches it. One raised by the constructor, such as a
      rejected parameter, is raised by the entry point that ran the task (``run``, ``run_task`` or ``run_tasks``)
      before the task runs, as a config error is.

    Examples
    --------
    >>> from typing import Any, ClassVar
    >>> from dataeval_flow.config.transforms import Transform
    >>> class Invert(Transform):
    ...     name: ClassVar[str] = "example.Invert"
    ...     description: ClassVar[str] = "Inverts each value within [0, maximum]."
    ...
    ...     def __init__(self, maximum: float = 1.0) -> None:
    ...         self.maximum = maximum
    ...
    ...     def __call__(self, data: Any, /) -> Any:
    ...         return self.maximum - data
    ...
    ...     def __repr__(self) -> str:
    ...         return f"Invert(maximum={self.maximum!r})"
    >>> Invert(maximum=255)
    Invert(maximum=255)

    Register it in the plugin's ``pyproject.toml``:

    .. code-block:: toml

        [project.entry-points."dataeval_flow.transforms"]
        "example.Invert" = "my_package:Invert"

    A preprocessor step names it, with its parameters:

    .. code-block:: yaml

        preprocessors:
          - name: inverted
            steps:
              - step: example.Invert
                params: {maximum: 255}
    """

    name: ClassVar[str]
    description: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Require identity on a concrete transform."""
        super().__init_subclass__(**kwargs)
        bind_implementation(cls, Transform, configured=False)

    @abstractmethod
    def __call__(self, data: Any, /) -> Any:
        """Transform one image and return it.

        Parameters
        ----------
        data : torch.Tensor
            One image, in CHW layout.

        Returns
        -------
        torch.Tensor
            The transformed image.
        """
