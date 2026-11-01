"""The trainer contract, and the loader that runs a trainer file from the data root's ``trainers/`` folder."""

__all__ = ["Trainer", "TrainerKit", "load_trainer"]

import hashlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataeval_flow.evaluators.performance._config import TRAINERS, TrainerConfig

if TYPE_CHECKING:
    from dataeval.protocols import EvaluationStrategy, TrainingStrategy


@dataclass(frozen=True)
class TrainerKit:
    """What a :class:`Trainer` builds: the model, and the strategies DataEval's ``Sufficiency`` trains it with.

    ``model`` is the untrained model. ``training_strategy`` has ``train(model, dataset, indices)``, which trains the
    model in place on the dataset's items at ``indices`` (drawn with replacement, so an index may repeat).
    ``evaluation_strategy`` has ``evaluate(model, dataset)``, which returns a mapping of metric name to a float, or an
    array for a per-class metric. ``reset_strategy`` takes the model and returns it untrained, before each run;
    ``None`` uses a deep copy of ``model`` as it was built.
    """

    model: Any
    training_strategy: "TrainingStrategy[Any]"
    evaluation_strategy: "EvaluationStrategy[Any]"
    reset_strategy: Callable[[Any], Any] | None = None


class Trainer(ABC):
    """The training code a ``sufficiency`` run measures, in a Python file under the data root's ``trainers/`` folder.

    A ``sufficiency`` entry's ``trainer:`` names the file and the class. When the run starts, Flow runs the file, and
    the class is constructed with the entry's ``params`` as keyword arguments and asked to :meth:`build` a
    :class:`TrainerKit`. Like ``sufficiency``, it is experimental: it may change in any release without a deprecation
    cycle.

    Subclassing
    -----------
    Define :meth:`build`, and an ``__init__`` taking the ``params`` the entry may give, as keyword arguments. Then:

    - The trainer is one file. Flow runs it under a private module name, so it cannot import a sibling file in
      ``trainers/``; it imports what the environment has installed, such as torch, scikit-learn and DataEval.
    - It reads its dataset items as they come: ``(image, target, metadata)``, the image ``CHW``.
    - It takes its device from ``dataeval.config.get_device()``, which the result's ``metadata.device`` records.
    - A ``torch.utils.data.DataLoader`` with workers needs ``num_workers=0`` or ``multiprocessing_context="fork"``: a
      worker started any other way cannot import a file Flow loaded by path.
    - Anything set process-wide when the file runs, such as a torch flag, stays set for the later tasks of the run.
    - It reads any weights it starts from under the data root: a run may have no network.

    Examples
    --------
    ``trainers/centroids.py``, a nearest-centroid classifier over the flattened images:

    >>> import numpy as np
    >>> from dataeval_flow.evaluators.performance import Trainer, TrainerKit
    >>> class Centroids:
    ...     means: np.ndarray | None = None
    >>> class Fit:
    ...     def train(self, model, dataset, indices):
    ...         items = [dataset[i] for i in indices]
    ...         images = np.stack([np.asarray(image, dtype=float).ravel() for image, _, _ in items])
    ...         labels = np.array([int(np.argmax(target)) for _, target, _ in items])
    ...         model.means = np.stack([images[labels == k].mean(axis=0) for k in range(int(labels.max()) + 1)])
    >>> class Score:
    ...     def evaluate(self, model, dataset):
    ...         hits = []
    ...         for i in range(len(dataset)):
    ...             image, target, _ = dataset[i]
    ...             distances = ((model.means - np.asarray(image, dtype=float).ravel()) ** 2).sum(axis=1)
    ...             hits.append(np.argmin(distances) == np.argmax(target))
    ...         return {"accuracy": float(np.mean(hits))}
    >>> class CentroidTrainer(Trainer):
    ...     def build(self) -> TrainerKit:
    ...         return TrainerKit(model=Centroids(), training_strategy=Fit(), evaluation_strategy=Score())

    The entry naming it:

    .. code-block:: yaml

        evaluators:
          - name: sufficiency
            type: sufficiency
            trainer: {path: trainers/centroids.py, class: CentroidTrainer}
    """

    @abstractmethod
    def build(self) -> TrainerKit:
        """Build the untrained model and the strategies that train, evaluate and reset it.

        Returns
        -------
        TrainerKit
            The model and its strategies.
        """


@contextmanager
def load_trainer(config: TrainerConfig, data_root: Path) -> Iterator[tuple[TrainerKit, dict[str, str]]]:
    """Run the trainer file `config` names, build its kit, and give it with the file's provenance for the block.

    The file must resolve, symlinks followed, inside ``<data_root>/trainers``. It runs as a module under a private
    name, ``_dataeval_flow_trainer_<first 12 hex of its SHA-256>``, entered in ``sys.modules`` before it runs (so
    dataclasses with postponed annotations, pickling and ``inspect`` work) and removed when the block ends. The bytes
    run are the bytes digested, and no bytecode is written beside the file.

    Yields
    ------
    tuple[TrainerKit, dict[str, str]]
        The kit, and the trainer's ``path`` and ``class`` as written with the file's ``sha256``.

    Raises
    ------
    ValueError
        When the file resolves outside ``trainers/``, the class is missing or is no ``Trainer``, or ``build()`` returns
        no ``TrainerKit``.
    FileNotFoundError
        When the file cannot be read.
    RuntimeError
        When the trainer's own code raises, running the file or building (constructing the class, or ``build()``);
        the original exception is its cause. Every message names the file and the class.
    """
    where = f"Trainer '{config.class_}' in {config.path}"
    folder = (data_root / TRAINERS).resolve()
    path = (data_root / config.path).resolve()
    if not path.is_relative_to(folder):
        raise ValueError(f"{where}: the file resolves to {path}, outside the data root's trainers folder {folder}.")
    try:
        source = path.read_bytes()
    except OSError as e:
        raise FileNotFoundError(f"{where}: cannot read the file: {type(e).__name__}: {e}") from e
    digest = hashlib.sha256(source).hexdigest()
    name = f"_dataeval_flow_trainer_{digest[:12]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:  # pragma: no cover - the config allows only `.py`, which always has a spec
        raise ValueError(f"{where}: not a Python file.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        try:
            exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102 - the operator's mounted trainer
        except Exception as e:
            raise RuntimeError(f"{where} failed running the file: {type(e).__name__}: {e}") from e
        cls = getattr(module, config.class_, None)
        if cls is None:
            raise ValueError(f"{where}: the file defines no '{config.class_}'.")
        if not (isinstance(cls, type) and issubclass(cls, Trainer)):
            raise ValueError(f"{where}: '{config.class_}' is not a Trainer subclass.")
        try:
            kit = cls(**config.params).build()
        except Exception as e:
            raise RuntimeError(f"{where} failed to build: {type(e).__name__}: {e}") from e
        if not isinstance(kit, TrainerKit):
            raise ValueError(f"{where}: build() returned {type(kit).__name__}, not a TrainerKit.")
        yield kit, {"path": config.path, "class": config.class_, "sha256": digest}
    finally:
        sys.modules.pop(name, None)
