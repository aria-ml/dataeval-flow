"""Download the datasets the docs notebooks use, as listed in docs/datasets.toml.

`nox -s docs_data` runs this to fetch them without building the docs. The docs CI jobs
restore the same datasets from the Harbor dataset cache first (docs/datasets.toml is the
manifest they read, too), after which the loaders find the files on disk and download
nothing. The notebooks still call the loaders with `download=True` themselves, for the same
reason.

Only raw downloads belong in the manifest. The `as_datamaite=True` exports are derived, and
the notebooks regenerate them. FashionMNIST is downloaded here as well but is not in the
manifest: it comes from torchvision, which the cache's download environment does not install,
and it is small, so the notebooks fetch it themselves in CI.
"""

import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from os import chdir, getcwd, makedirs, path

MANIFEST = path.join(path.dirname(__file__), "..", "datasets.toml")


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
    from torchvision.datasets import FashionMNIST

    with open(MANIFEST, "rb") as f:
        datasets = tomllib.load(f)["dataset"]

    # Notebooks execute with docs/source/notebooks as the working directory and use root="./data".
    with cd("notebooks"):
        for entry in datasets:
            module, name = entry["loader"].rsplit(".", 1)
            kwargs = {k: v for k, v in entry.items() if k != "loader"}
            getattr(import_module(module), name)(root="./data", download=True, **kwargs)
        FashionMNIST(root="./data", train=True, download=True)


if __name__ == "__main__":
    download()
