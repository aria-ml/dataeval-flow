"""Datasets the docs notebooks download, in one place.

`nox -s docs_data` runs this to fetch every raw dataset the notebooks use. The docs
CI job uses it to fill the Harbor dataset cache (docs/harbor-datasets-cache.sh), and
the cache key is a hash of this file, so add a dataset here when a notebook starts
using a new one. The notebooks still call the loaders with `download=True` themselves,
which finds the files already on disk and downloads nothing.

Only raw downloads belong here. The `as_datamaite=True` exports are derived, and the
notebooks regenerate them.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from os import chdir, getcwd, makedirs, path


@contextmanager
def cd(rel_path: str) -> Iterator[None]:
    """Change folder relative to current file."""
    cur_path = getcwd()
    new_path = path.join(path.dirname(__file__), rel_path)
    print(f"push: {cur_path} -> {new_path}")
    makedirs(new_path, exist_ok=True)
    chdir(new_path)
    try:
        yield
    finally:
        print(f"pop: {cur_path} <- {new_path}")
        chdir(cur_path)


def download() -> None:
    from maite_datasets.image_classification import MilitaryVehicles, Ships
    from maite_datasets.object_detection import MILCO, SeaDrone, SkySeaLand
    from torchvision.datasets import FashionMNIST

    # Notebooks execute with docs/source/notebooks as the working directory and use root="./data".
    with cd("notebooks"):
        MilitaryVehicles(root="./data", image_set="base", download=True)
        Ships(root="./data", download=True)
        SkySeaLand(root="./data", image_set="base", download=True)
        MILCO(root="./data", image_set="base", download=True)
        SeaDrone(root="./data", image_set="val", download=True)
        FashionMNIST(root="./data", train=True, download=True)


if __name__ == "__main__":
    download()
