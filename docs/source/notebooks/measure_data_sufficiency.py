# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: dataeval-flow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Measure how much training data a model needs
#
# Train a model on growing parts of the MilitaryVehicles training set, see how its score rises with the data, and find
# out whether it would reach a target score with more data, using the
# [`sufficiency`](../reference/evaluators.md#sufficiency) evaluator and the
# [`performance-sufficiency`](../reference/checks.md#performance-sufficiency) check.

# %% [markdown]
# You are a T&E engineer or data scientist deciding whether to collect or label more training data, or whether the
# model, rather than the data, is what limits its score.
#
# You should run this once you have a model and training code that work, and before you commit to collecting or labeling
# more data. It trains your model on subsets of the training set, scores each on a test set, and fits a curve to the
# scores. The check then says, for each target score, whether the model reached it and, if not, about how much data the
# curve says it would take. {doc}`check_target_feasibility` comes before it: it asks whether your labeled images show
# that a model can reach an accuracy you need. If more data would help, see {doc}`data_prioritization` to choose which
# to label first.
#
# :::{note}
# `sufficiency` and `performance-sufficiency` are experimental: they may change or be removed in any release without a
# deprecation cycle, as may the trainer file they load. The first run in a process to use either one warns with
# `ExperimentalWarning`.
# :::

# %% [markdown]
# ## What you will do
#
# - Download MilitaryVehicles and export a pretrained ResNet-18 to ONNX, into a data root that also holds your trainer.
# - Write a trainer: the model and the training and scoring code the `sufficiency` evaluator runs.
# - Chain the evaluator and the `performance-sufficiency` check in a pipeline file, and run it with `run_task()`.
# - Read the text report and the HTML report, one answer to each of four targets.
# - Read the measurements and the fitted curve in Python.

# %% [markdown]
# ## What you will learn
#
# - How the data root is laid out, and why a trainer lives under `trainers/`.
# - What a trainer is, and what to keep in mind when you write your own.
# - How `sufficiency` measures: training sizes, draws with replacement, and the `schedule`.
# - The four answers the check gives a target: reached, within the training set, beyond it, and never.
# - How to read the curve, and where the report's numbers are kept.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` with its ONNX extra (`pip install "dataeval-flow[onnx]"`), which brings ONNX Runtime.
# - Install PyTorch and torchvision, which export the model, and `dataeval-plots`, which plots the curve.
# - Install `maite-datasets` to download MilitaryVehicles.
# - Ensure network access for the first run: the dataset (~100 MB) and the ResNet-18 weights (~45 MB).

# %% [markdown]
# ## Lay out the data root
#
# A run reads everything it needs from one folder, the data root: datasets under `data/`, model files under `models/`,
# and your training code under `trainers/`. Here the data root is the folder this notebook sits in, and the notebook
# fills it as it goes:
#
# ```text
# .                                  the data root
# ├── measure_data_sufficiency.yaml  the pipeline: datasets, trainer, steps and targets
# ├── data/militaryvehicles/         MilitaryVehicles, as train_fine/ and test_fine/ (downloaded below)
# ├── models/resnet18.onnx           the pretrained model the trainer reads (exported below)
# └── trainers/backbone.py           your training code (written in "Write a trainer")
# ```
#
# A trainer path must start with `trainers/`, and the file, once symlinks are followed, must sit inside that folder. A
# path outside it fails when the config loads; a symlink out of it fails the run. The same layout works in a container,
# where you mount the project as the data root.

# %% [markdown]
# ## Download the dataset and export the model
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles) holds 24 kinds of military vehicle,
# a folder of photographs for each. The next cell downloads it into `data/`, and Flow reads those folders as they are.

# %% tags=["remove_output"]
from pathlib import Path

from maite_datasets.image_classification import MilitaryVehicles

data_root = Path(".")

# Downloads to ./data/militaryvehicles on first run (~100 MB). A re-run finds the files on disk.
MilitaryVehicles(root=data_root / "data", image_set="base", download=True)

# %% [markdown]
# The trainer scores each image with a pretrained ResNet-18, an ImageNet model. The next cell exports it to ONNX under
# `models/`, without its last layer, so that the model's output is a 512-number description of the image rather than a
# score for each of ImageNet's 1,000 classes. It also makes the `trainers/` folder the trainer goes in.

# %% tags=["remove_output"]
import warnings

import torch
import torchvision

(data_root / "trainers").mkdir(exist_ok=True)
(data_root / "models").mkdir(exist_ok=True)

onnx_path = data_root / "models" / "resnet18.onnx"
if not onnx_path.exists():
    resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    resnet.fc = torch.nn.Identity()  # keep the 512 features, not ImageNet's 1,000 class scores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the exporter used here warns that a newer one is the default
        torch.onnx.export(
            resnet,
            (torch.zeros(1, 3, 224, 224),),
            str(onnx_path),
            input_names=["input"],
            output_names=["features"],
            dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},  # any number of images at once
            dynamo=False,
        )

import onnxruntime

# Load it once here, so that a bad export fails now rather than in the middle of the run.
onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


# %% [markdown]
# Check what the data root holds now:

# %%
for name in ("train_fine", "test_fine"):
    folder = data_root / "data" / "militaryvehicles" / name
    print(f"{folder.relative_to(data_root)}: {sum(1 for _ in folder.rglob('*.jpg')):,} images")
print(f"{onnx_path.relative_to(data_root)}: {onnx_path.stat().st_size / 1e6:.0f} MB")

# %% [markdown]
# ## Write a trainer
#
# Your training code is a trainer: a subclass of {py:class}`~dataeval_flow.evaluators.performance.Trainer` in one Python
# file under `trainers/`. Flow constructs it with the entry's `params` as keyword arguments and calls its `build()`,
# which returns a {py:class}`~dataeval_flow.evaluators.performance.TrainerKit`:
#
# - the untrained model;
# - a training strategy, with `train(model, dataset, indices)`;
# - an evaluation strategy, with `evaluate(model, dataset)`, which returns each metric by name;
# - optionally a reset strategy, which makes the model untrained again for the next run. By default, it is a deep copy
#   of the model as built.
#
# This trainer describes each image once with the ONNX model, and its "model" is a nearest-centroid head: the average
# description of each class. At each training size it refits the head on the images drawn, and scores it by the share of
# test images whose nearest class average is their own. It reads the ONNX file from the data root through `params`. The
# cell below shows the file and writes it to `trainers/backbone.py`.

# %%
# %%writefile trainers/backbone.py
"""A frozen pretrained backbone with a nearest-centroid head.

The backbone's output (here ResNet-18's 512 features) is computed once per image; each training size refits only the
head, a mean feature row per class.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dataeval_flow.evaluators.performance import Trainer, TrainerKit

# The data root: this file sits in its `trainers/` folder.
DATA_ROOT = Path(__file__).parents[1]
# ImageNet's channel means and deviations, on 0-255 pixels.
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)[:, None, None]
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)[:, None, None]


class Backbone:
    """The pretrained model's outputs for each item, computed once and kept by dataset and index."""

    def __init__(self, weights: str, size: int) -> None:
        self.weights = DATA_ROOT / weights
        self.size = size
        self.session: Any = None
        self.cache: dict[tuple[int, int], NDArray[np.float32]] = {}

    def __call__(self, dataset: Any, indices: Sequence[int]) -> NDArray[np.float32]:
        """The features of the items at `indices`, one row each."""
        missing = sorted({index for index in indices if (id(dataset), index) not in self.cache})
        for start in range(0, len(missing), 64):
            batch = missing[start : start + 64]
            scores = self._run(np.stack([self._prepare(dataset[index][0]) for index in batch]))
            self.cache.update({(id(dataset), index): row for index, row in zip(batch, scores, strict=True)})
        return np.stack([self.cache[id(dataset), index] for index in indices])

    def _prepare(self, image: Any) -> NDArray[np.float32]:
        pixels = np.asarray(image, dtype=np.float32)
        rows = np.linspace(0, pixels.shape[1] - 1, self.size).astype(int)
        cols = np.linspace(0, pixels.shape[2] - 1, self.size).astype(int)
        return (pixels[:, rows][:, :, cols] - MEAN) / STD

    def _run(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.session is None:
            import onnxruntime
            from dataeval.config import get_device

            gpu = get_device().type == "cuda"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(str(self.weights), providers=providers)
        return self.session.run(None, {self.session.get_inputs()[0].name: batch})[0]


class Centroids:
    """The head: one mean feature row per class seen in training."""

    def __init__(self) -> None:
        self.labels = np.empty(0, dtype=int)
        self.means = np.empty((0, 0), dtype=np.float32)


class Fit:
    """Refit the head on the items at `indices`; an index may repeat, as DataEval draws with replacement."""

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def train(self, model: Centroids, dataset: Any, indices: Sequence[int]) -> None:
        """Average each class's features."""
        features = self.backbone(dataset, indices)
        labels = np.array([int(np.argmax(dataset[index][1])) for index in indices])
        model.labels = np.unique(labels)
        model.means = np.stack([features[labels == label].mean(axis=0) for label in model.labels])


class Score:
    """The share of the test set whose nearest centroid is its own class."""

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def evaluate(self, model: Centroids, dataset: Any) -> Mapping[str, float]:
        """Accuracy, from 0 to 1."""
        indices = list(range(len(dataset)))
        features = self.backbone(dataset, indices)
        distances = ((features[:, None, :] - model.means[None]) ** 2).sum(axis=2)
        labels = np.array([int(np.argmax(dataset[index][1])) for index in indices])
        return {"accuracy": float(np.mean(model.labels[np.argmin(distances, axis=1)] == labels))}


class BackboneTrainer(Trainer):
    """Nearest centroids over a frozen ONNX model's features; `weights` is relative to the data root."""

    def __init__(self, weights: str, size: int = 224) -> None:
        self.weights = weights
        self.size = size

    def build(self) -> TrainerKit:
        """An empty head, refit at each size; a deep copy resets it. The backbone is loaded on first use."""
        backbone = Backbone(self.weights, self.size)
        return TrainerKit(model=Centroids(), training_strategy=Fit(backbone), evaluation_strategy=Score(backbone))


# %% [markdown]
# Keep these in mind when you write your own:
#
# - Write the trainer as one file. Flow runs it under a private module name, so it cannot import a sibling file in
#   `trainers/`. It can import what the environment has: in a container, what the variant's `requirements.<variant>.txt`
#   lists, which includes PyTorch, torchvision, ONNX Runtime, OpenCV, scikit-learn, SciPy, LightGBM, and Polars.
# - Expect items as stored: `(image, target, metadata)`, the image channels first, at the dataset's own size.
# - Take the device from `dataeval.config.get_device()`, which the result's `metadata.device` records.
# - Use no network. Read pretrained weights from the data root, as `weights` does here.
# - Give a `torch.utils.data.DataLoader` either `num_workers=0` or `multiprocessing_context="fork"`: a worker started
#   any other way cannot import a file Flow loaded by path.
# - Expect process-wide settings to stay set. Something the file sets for the whole process, such as a torch flag, holds
#   for the later tasks of the same run.

# %% [markdown]
# ## Chain the evaluator and the check
#
# The pipeline file holds both steps. The `sufficiency` evaluator trains the trainer's model on growing subsets of the
# training set and scores it on the test set at each size. The `performance-sufficiency` check reads what it measured
# and judges it against your targets. The file sits beside this notebook, and the cell below prints it.

# %%
print(Path("measure_data_sufficiency.yaml").read_text())

# %% [markdown]
# Note these points about the file:
#
# - Order decides the roles. The evaluator's first input is the training set, and its second is the test set.
# - The sizes are the `schedule`. Left out, the default schedule measures five sizes, from 1% of the training set to all
#   of it, and needs a training set of at least 100 items. Here five sizes stop at 3,200 of the training set's 7,823
#   items, so that the check can tell data you already hold from data you would have to find.
# - A size counts draws, made with replacement, not distinct items: 3,200 draws cover fewer than 3,200 different images.
#   `seed` makes the draws repeatable.
# - `params` reach the trainer. They are not checked when the config loads; the trainer checks them when the run starts.
# - The four targets are chosen to show each answer the check can give, as the "Read the report" section reads them.
#   They are not goals for this model. A real target comes from what your mission needs.

# %% [markdown]
# ## Run it
#
# `run_task()` runs the task the pipeline names. The trainer's backbone describes each image once, so the first size is
# the slowest; the run takes from half a minute to a few minutes, depending on the CPU. Flow logs a line as each size is
# measured, which the cell below prints. On the command line, add `-vv` to see them.

# %%
import logging
import sys
import warnings

from dataeval_flow import ExperimentalWarning, load_config, run_task, set_device

set_device("cpu")  # the same numbers on a machine with a GPU
warnings.simplefilter("ignore", ExperimentalWarning)  # the warning the note at the top describes

progress = logging.getLogger("dataeval_flow.evaluators.performance")
progress.addHandler(logging.StreamHandler(sys.stdout))
progress.setLevel(logging.INFO)

config = load_config("measure_data_sufficiency.yaml")
result = run_task(config, "sufficiency", data_dir=data_root)

# %% tags=["remove_cell"]
assert result.success, result.errors

# %% [markdown]
# In a container, mount the project as the data root, `trainers/` and all, and the run reads the trainer from there. The
# data root is read-only there, and anything the trainer writes outside `/output` or `/cache` is lost when the container
# exits. A GPU variant trains on the GPU with `--gpus all`. See {doc}`../how_to/containerized_workflows` and the
# [Container Reference](../reference/containers.md).

# %% [markdown]
# ## Read the report
#
# `result.report()` is the text report. The check's finding is the section titled Performance Sufficiency, and the
# evaluator's own section below it names the trainer that made the numbers.

# %%
print(result.report())

# %% [markdown]
# The banner counts the targets reached at the largest size measured. Under it:
#
# - The learning curve is a row per training size: the accuracy there, and a bar for it. The bars stop at the highest
#   target, and the line under them marks the lowest and the highest, `(0.27)` and `(0.35)`. The score rises quickly
#   with the first sizes, then each size adds less than the one before. The check fits a curve to that, a power law, and
#   projects it.
# - The targets are a row for each, with whether it is reached, and where it is not, the size the fitted curve reaches
#   it at and how many times the largest size measured that is.
#
# The four targets each get a different answer:
#
# | Target | The report says | Severity | What it means |
# | --- | --- | --- | --- |
# | `accuracy ≥ 0.27` | reached | ok | The model already scores this at the largest size measured. |
# | `accuracy ≥ 0.29` | about 4,500 items (1.4x) | info | More data helps, and the training set holds it: 7,823 items. Train on more of what you have. |
# | `accuracy ≥ 0.3` | about 13,000 items (4.1x) | warning | More data would help, but more than the training set holds, so you would collect or label more. |
# | `accuracy ≥ 0.35` | never | warning | The curve levels off near 0.31: this model reaches about that however much data it has. |
#
# A size past the largest measured is an extrapolation of the fitted curve, so the further out it reaches, the less to
# trust it. The step from 0.29 to 0.3 more than doubles the data the curve asks for, because the curve flattens as it
# nears its limit. On another machine the numbers differ in the last digits, but the four answers should not change.
#
# The finding is a warning because a target is beyond the training set or never reached. A finding is `ok` where every
# target is reached, `info` where one is projected within the training set, and `warning` where one is projected past it
# or never. A target on a per-class metric, which the trainer returns as an array, is listed as not assessed. A target
# naming a metric the run did not measure fails the step, listing the metrics it did.

# %% [markdown]
# `result.findings` holds the check's findings, the ones the report's summary lists:

# %%
for finding in result.findings:
    print(f"{finding.severity:<8} {finding.title:<24} {finding.brief}")

# %% [markdown]
# ### Read the report as HTML
#
# `result.to_html()` draws the same tables as a page. Each bar's scale shows the targets as dashed lines, labeled below
# it. {doc}`view_html_reports` shows what else the page does.

# %% tags=["remove_cell"]
# The page styles its whole document, so the docs show it in an iframe of its own rather than
# letting its stylesheet restyle this site. Readers don't need this to use `to_html()`.
import html

from IPython.display import display


def show(page: str, height: int = 640) -> None:
    """*page* in an iframe, so its stylesheet and script stay inside it."""
    frame = (
        f'<iframe srcdoc="{html.escape(page, quote=True)}" title="HTML report" loading="lazy" '
        f'style="width:100%;height:{height}px;border:1px solid #d1d9e0;border-radius:6px"></iframe>'
    )
    display({"text/html": frame}, raw=True)


# %% tags=["remove_input"]
show(result.to_html())

# %% [markdown]
# ## Read the measurements in Python
#
# The report holds the measurements as a picture. The step's `output` holds them as numbers, and is DataEval's own
# `SufficiencyOutput`: its `steps` are the training sizes measured, and `averaged_measures` holds each metric at each
# size, averaged over the evaluator's `runs` (one, by default).

# %%
output = result.steps["measured"].output
print("sizes:   ", output.steps.tolist())
print("accuracy:", [round(float(value), 4) for value in output.averaged_measures["accuracy"]])
print(f"training set: {len(result.sources['train']):,} items; test set: {len(result.sources['test']):,}")

# %% [markdown]
# The fit is `1 - metric = scale * items ** -exponent + asymptote`, once per metric: `params` holds its three numbers.
# The curve can never get past `1 - asymptote`, the figure the report calls where it "levels off". `project()` reads
# the curve at any size, and `inv_project()` reads it the other way, the size at which it reaches a target.

# %%
scale, exponent, asymptote = output.params["accuracy"]
print(f"1 - accuracy = {scale:.3f} * items ** -{exponent:.3f} + {asymptote:.3f}")
print(f"The curve levels off near {1 - asymptote:.3f}\n")
print(output.project([3_200, 7_823, 20_000]))

# %% [markdown]
# `dataeval-plots` draws the measurements with the fitted curve, extended past the largest size. The dots are the sizes
# measured, the solid line is the fitted curve, and the dashed line is its limit, the figure the report calls where the
# curve "levels off":

# %%
from dataeval_plots import plot

_ = plot(output)

# %% [markdown]
# Everything the report left out is in the result's JSON, which `dataeval-flow --output` writes to
# `results/result.json`, and `result.to_dict()` gives in Python. The evaluator step's `output` holds the measurements
# and, under `extras`, the trainer's `path`, `class` and `sha256`, which pin the code that made them, and `params`, the
# fitted curve the check projected from.

# %%
extras = result.to_dict()["steps"]["measured"]["output"]["extras"]
print(extras["trainer"])
print(extras["params"])

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Lay out a data root with `data/`, `models/` and `trainers/`, and why a trainer path must start with `trainers/`.
# - Write a trainer that builds a model with a training and an evaluation strategy.
# - Chain the `sufficiency` evaluator and the `performance-sufficiency` check, and set the schedule and the targets.
# - Run it with `run_task()` and read the report: a learning curve against the targets, and a row for each.
# - Tell the four answers apart: reached, projected within the training set, projected beyond it, and never.
# - Read the measurements and the fitted curve in Python, and find the rest in the JSON.

# %% [markdown]
# ## Next steps
#
# - {doc}`data_prioritization` — Rank an unlabeled pool so the most informative samples are annotated first.
# - {doc}`../how_to/write_a_custom_workflow` — Learn the workflow's parts and chain steps into a workflow of your own.

# %% [markdown]
# ## On your own
#
# To repeat the measurement with your own data, model and settings, change these entries:
#
# - To use your own model, replace the trainer's backbone and head with your model, and keep the shape: a model, a
#   `train` that fits it on the indices it is given, and an `evaluate` that returns each metric by name. Point the
#   evaluator's `trainer` at your file with `path`, `class` and `params`.
# - To use your own data, change the `path` of the entries under `datasets`, and keep the training set first and the
#   test set second in the task's `sources`.
# - To measure other sizes, edit `schedule`: strictly increasing numbers, none larger than your training set. With no
#   `schedule`, the default measures five sizes up to the whole training set. Set `substeps` to change how many, and not
#   together with `schedule`.
# - To steady the curve, set `runs` to average several training runs at each size, or name more sizes in `schedule`.
# - To judge other targets, edit `targets` in the `performance-sufficiency` step: a `metric` that the trainer's
#   `evaluate` returns and a `value` that the metric should reach, where higher is better.
# - To repeat the same draws, keep `seed`.
