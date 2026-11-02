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
# # Check whether a performance target is reachable
#
# Ask whether your labeled images show that a model can reach an accuracy you need, using the `feasibility`
# evaluator and the `performance-feasibility` check on MilitaryVehicles.

# %% [markdown]
# **Target audience**: You are a T&E engineer or data scientist who has been given a performance requirement for an
# image classifier, such as "name the vehicle in the photo correctly 85% of the time", and who needs to know whether
# your data can support it before you spend months training.
#
# **Workflow role**: You should run this check after your data is clean and before you commit to training. See
# {doc}`Clean a dataset <data_cleaning>` and {doc}`Split a dataset <dataset_splitting>` for the steps before it, and
# {doc}`Measure how much training data a model needs <measure_data_sufficiency>` for the question that comes
# after: how much data a model you train needs. Both `feasibility` and `performance-feasibility` are experimental.

# %% [markdown]
# ## What you will do
#
# - Download MilitaryVehicles and save a pretrained image model that describes each image with 768 numbers.
# - Chain the `feasibility` evaluator and the `performance-feasibility` check in a workflow of your own.
# - Read the text report and the HTML report, and what each kind of answer means.
# - Look at the proof behind an answer: DataEval's full printout and the evidence for each bound.
# - Run the same question again with allowances, which you assert, and see what changes.

# %% [markdown]
# ## What you will learn
#
# - How to ask whether a performance target is reachable, and what the evaluator needs to answer.
# - How to read an answer: what has been shown, at what confidence, and what it assumes.
# - Why a target can come back *not demonstrated* without being *ruled out*.
# - What allowances are, why they are your assertions, and how DataEval checks them against its evidence.
# - Why running the evaluator more than once on the same data costs confidence.

# %% [markdown]
# ## Prerequisites
#
# - Install `dataeval-flow` (includes `dataeval`, `datamaite`, `pydantic`).
# - Install `maite-datasets` to download MilitaryVehicles and `torchvision` for the pretrained model.
# - Ensure network access for the first run: the images (about 100 MB) and the model (about 330 MB).

# %% [markdown]
# ## Lay out the data root
#
# Flow reads and writes files relative to a *data root*. Here that is the notebook's own folder. This is
# what it holds once the tutorial has run:
#
# ```text
# check_target_feasibility/                   # the data root: where this notebook runs
# ├── data/
# │   └── militaryvehicles/                   # the download, from maite-datasets
# │       └── train_fine/                     # the 7,823 training images, one folder per vehicle type
# ├── models/
# │   └── vit_b_16.pt                         # the pretrained image model, saved from torchvision
# └── output/
#     └── check_target_feasibility.html       # the HTML report this tutorial writes
# ```
#
# The download also holds a `test_fine/` folder of test images, which this tutorial does not read.
#
# There is no pipeline file: step 3 builds the pipeline in Python. Every setting there has the same name in a YAML
# pipeline, which `dataeval-flow --config pipeline.yaml --output output/` would read from the data root as well.

# %% [markdown]
# ### Step-by-step guide

# %% [markdown]
# ## Step 1: Get the images
#
# [MilitaryVehicles](https://huggingface.co/datasets/leibnitz-lab/military_vehicles) holds photos of 24 types of
# military vehicle, such as the T-72 tank or the BMP-2 fighting vehicle. You will use its training images: 7,823 of
# them, in one folder for each type. Flow's `image_folder` format labels each image by the folder it sits in.

# %% tags=["remove_output"]
import warnings
from pathlib import Path

from dataeval.exceptions import ExperimentalWarning
from maite_datasets.image_classification import MilitaryVehicles

# `feasibility` and `performance-feasibility` are experimental, and Flow and DataEval warn of it. The warnings are
# switched off here to keep this page readable; leave them on in your own code.
warnings.filterwarnings("ignore", category=ExperimentalWarning)

data_root = Path("./data")
MilitaryVehicles(root=data_root, image_set="base", download=True)

data_path = data_root / "militaryvehicles" / "train_fine"
print(f"Reading from {data_path}")

# %% [markdown]
# ## Step 2: Describe each image with a pretrained model
#
# The evaluator does not look at pixels. It looks at a list of numbers that describes each image, called an
# *embedding*, which a pretrained image model makes. Similar-looking images get similar numbers, and the evaluator
# trains two simple models on them, a *nearest-neighbor* model and a *linear* model, to see how well they name the
# vehicle.
#
# You will use ViT-B/16 from `torchvision`, a general-purpose model trained on the ImageNet photo collection. It has
# a final layer that turns its description into 1,000 ImageNet class scores. You replace that layer so the model hands
# back the 768-number description instead: the class scores would describe each image by ImageNet's categories, not by
# what the image shows.
#
# :::{important}
# The model must not have seen the labels of the images you judge. ImageNet's categories are not these 24 vehicle
# types, and its training set does not hold these photos as far as we know, so this model has not seen their labels.
# Flow cannot check this for you. You also have to choose the model *before* you look at how well it does. We chose
# ViT-B/16 after an informal comparison of a few models on these images, made while writing this tutorial, so its
# answers may be somewhat more favorable than those of a model chosen in advance. If you compare `k` models on your own
# data, divide `alpha` by `k` for each.
# :::

# %% tags=["remove_output"]
import torch
from torchvision.models import ViT_B_16_Weights, vit_b_16

model_dir = Path("./models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "vit_b_16.pt"
if not model_path.exists():
    backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    backbone.heads = torch.nn.Identity()  # keep the 768-number description, not the 1,000 class scores
    torch.save(backbone.eval(), model_path)

# %% [markdown]
# ## Step 3: Set the targets and chain the steps
#
# A *target* is a number you want a model to reach, written as a metric and a threshold. The evaluator takes them as
# `target:`, a mapping from metric to threshold. This tutorial sets two:
#
# - `accuracy: 0.40`: the share of images a model names right, for **any** model, including one that reads the whole
#   image. The key's default scope is `full_input`.
# - `accuracy/fixed_representation: 0.50`: the same, but only for a model that reads the 768 numbers of step 2 and
#   nothing else, such as the models the evaluator tests itself.
#
# :::{note}
# **How these targets were chosen.** A requirement for a classifier like this one would be 85% or 90%, and these images
# do not demonstrate it. The best lower bound in step 5 is about 42%: a general-purpose model's 768 numbers do not
# separate 24 look-alike vehicle types well, and the evaluator tests only the two simple models it builds from them.
# So the first target is the highest round number that this model demonstrates, 40%, to show a *demonstrated* answer.
# The second is a harder target, 50%, chosen to show an answer that is not demonstrated and, in step 8, one that is
# ruled out. The targets were chosen to show each kind of answer, and 40% is not a good requirement for a vehicle
# classifier.
# :::
#
# A few things to know about what you are asking:
#
# - **More targets widen every bound.** `alpha` is the share of the time you accept being wrong, 5% unless you set it.
#   It is one budget for the whole call, so every bound of every target holds together at 95%, and each extra target
#   takes a share of it.
# - **One call is one look.** If you run the evaluator twice on the same data, or keep the best of several runs, divide
#   `alpha` by the number of runs. This tutorial runs it twice, in step 4 and in step 8, so each run sets
#   `alpha=0.025`: each run holds at 97.5%, and the two hold together at 95%.
# - **Only labeled items count.** Every answer is about the labeled images. A class the dataset declares but that has no
#   labeled image is no part of any answer, and the report names it.
# - **Some data is refused.** A dataset made from another, such as the crops of detection boxes that `wrap` makes, is
#   refused: its items come from the same photo, so they are not independent, and the evaluator cannot tell which share
#   one. So is a dataset whose labeled images fall in fewer than two classes.
# - **Repeated photos distort an answer.** The evaluator treats every image as a separate example. In this training
#   set, 67 photos appear more than once, 139 images in all, each filed under different classes, so no model can name
#   all of them right. For a real decision you would remove them first, as {doc}`Clean a dataset <data_cleaning>` does.
#   They stay here to keep this tutorial short.
# - **A sample keeps the run short.** You will judge 3,000 images, drawn after a seeded shuffle, because the training
#   set is stored class by class and the first 3,000 would hold only some of the classes. More images narrow the
#   bounds, at the price of a longer run.
#
# The workflow has two steps. `measured` runs the evaluator on the images, and `reachable` is the check that turns its
# answers into a finding. You will run it twice, so the pipeline holds two evaluators, one with allowances, and a
# workflow and task for each. Step 8 explains the allowances.

# %% tags=["remove_output"]
from dataeval_flow import run_task
from dataeval_flow.config import (
    ImageFolderDatasetConfig,
    PipelineConfig,
    PreprocessingStep,
    PreprocessorConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.extractors import TorchExtractorConfig
from dataeval_flow.evaluators.performance import AllowancesConfig, FeasibilityConfig
from dataeval_flow.steps import CustomWorkflowConfig, StepEntry

targets = {
    "accuracy": 0.40,  # any model
    "accuracy/fixed_representation": 0.50,  # a model that reads only the embeddings
}


def reachable(name: str, evaluator: str) -> CustomWorkflowConfig:
    """A workflow of two steps: `measured` runs the evaluator, and `reachable` judges what it made."""
    return CustomWorkflowConfig(
        name=name,
        inputs=["train"],
        steps=[
            StepEntry(name="measured", evaluator=evaluator, input="train"),
            StepEntry(name="reachable", check="performance-feasibility", input="measured"),
        ],
    )


config = PipelineConfig(
    max_processes=2,
    seed=0,
    datasets=[ImageFolderDatasetConfig(name="vehicles", path=str(data_path), infer_labels=True)],
    views=[
        ViewConfig(
            name="sample",
            operations=[
                ViewOperation(type="Shuffle", params={"seed": 0}),
                ViewOperation(type="Limit", params={"size": 3000}),
            ],
        )
    ],
    sources=[SourceConfig(name="train", dataset="vehicles", view="sample")],
    # ViT-B/16 takes 224 x 224 images, normalized as the ImageNet photos were. The image folder gives pixel values
    # from 0 to 255, so ImageNet's mean and standard deviation, which are for 0 to 1, are multiplied by 255.
    preprocessors=[
        PreprocessorConfig(
            name="imagenet",
            steps=[
                PreprocessingStep(step="Resize", params={"size": [224, 224], "antialias": True}),
                PreprocessingStep(
                    step="Normalize", params={"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]}
                ),
            ],
        )
    ],
    extractors=[TorchExtractorConfig(name="vit", model_path=str(model_path), preprocessor="imagenet", batch_size=64)],
    evaluators=[
        FeasibilityConfig(name="feasibility", target=targets, alpha=0.025),
        FeasibilityConfig(name="feasibility_allowed", target=targets, alpha=0.025, allowances=AllowancesConfig(beta=0)),
    ],
    workflows=[reachable("reachable", "feasibility"), reachable("reachable_allowed", "feasibility_allowed")],
    tasks=[
        TaskConfig(name="reachable", workflow="reachable", sources=["train"], extractor="vit"),
        TaskConfig(name="reachable_allowed", workflow="reachable_allowed", sources=["train"], extractor="vit"),
    ],
)

# %% [markdown]
# ## Step 4: Run it
#
# `run_task()` describes each image with the model, which takes a while on a CPU, and the evaluator then fits its two
# models and works out the bounds. Flow keeps the embeddings in memory for the rest of the session, so the second run in
# step 8 does not compute them again.

# %% tags=["remove_output"]
result = run_task(config, config.tasks[0])

# %% tags=["remove_cell"]
if not result.success:
    print(f"Workflow failed: {result.errors}")
assert result.success

# %% [markdown]
# ## Step 5: Read the text report
#
# The finding, under the summary, is the part to read. It has a row per target, in the order you gave them. Each row
# holds a mark for the answer, `✓` for *demonstrated* and `?` for *insufficient evidence*, with the key to the marks
# under the table; the target, such as `accuracy ≥ 40%`; a bar of what the evidence decides, on a scale from 0 to 100%
# accuracy; and the bound the bar is drawn from, `≥42.4%` where only a lower bound was found. In the bar, `█` is what
# is *shown*, up to the lower bound; `▒` is what is *ruled out*, above the upper bound; and `░` is what is *open*
# between them. Under the table, a line each gives the key to the marks and says how confident the bounds are together
# and about which images, and, where you gave any, which allowances they rest on.

# %%
print(result.report())

# %% [markdown]
# The first target is **demonstrated** (`✓`). The evaluator fitted a simple model, a *linear* model, to the numbers that
# describe half of the images, and tested it on the other half. It named the vehicle right on about 46% of them. After
# allowing for having tested only 1,500 images, the lower bound is about 42%. So, with 97.5% confidence, a model of
# this kind names the vehicle right at least 40% of the time on new images from this population, if no evaluation image
# was used to train or choose a model, or to set a target. There is no upper bound, so no part of the bar is ruled out.
#
# The second target has **insufficient evidence** (`?`): it is neither shown nor ruled out. The same model is the best
# one tested, and its lower bound, about 42%, is short of 50%. Nothing bounds from above the models that read only the
# 768 numbers, so nothing rules 50% out either. To settle it, test a stronger model, to show it, or get repeated labels
# or allowances you can justify, to rule it out. Step 7 shows where DataEval says so, and step 8 uses the allowances.
#
# The finding warns, not passes, because a target is not demonstrated, so `--fail-on-warning` would fail on it. It is
# `ok` only where every target is demonstrated.

# %% [markdown]
# ## Step 6: Read the HTML report
#
# `to_html()` renders the same finding as a page. The table and the lines under it are the same blocks, drawn for a
# screen. The bar's legend, *shown*, *open* and *ruled out*, is in the table's header. The page follows your
# system's dark mode setting, so here it may not match this site's theme.
# {doc}`View a report as HTML <view_html_reports>` covers the rest of the page. You write it with `encoding="utf-8"`:

# %%
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)
page = output_dir / "check_target_feasibility.html"
page.write_text(result.to_html(), encoding="utf-8")
print(f"{page}  ({page.stat().st_size:,} bytes)")

# %% tags=["remove_cell"]
# The page styles its whole document, so the docs show it in an iframe of its own rather than
# letting its stylesheet restyle this site. Readers don't need this to use `to_html()`.
import html

from IPython.display import display


def show(page: str, height: int = 560) -> None:
    """*page* in an iframe, so its stylesheet and script stay inside it."""
    frame = (
        f'<iframe srcdoc="{html.escape(page, quote=True)}" title="HTML report" loading="lazy" '
        f'style="width:100%;height:{height}px;border:1px solid #d1d9e0;border-radius:6px"></iframe>'
    )
    display({"text/html": frame}, raw=True)


# %% tags=["remove_input"]
show(result.to_html())

# %% [markdown]
# ## Step 7: Look at the proof (optional)
#
# The finding is short on purpose. Everything behind it is in the evaluator's step, `measured`. Its `output` is
# DataEval's own `FeasibilityOutput`, and printing it gives DataEval's full printout, which states the same bounds with
# the model each one comes from, what would settle each target that is not demonstrated, and the assumption every
# answer carries:

# %%
output = result.steps["measured"].output
print(output)

# %% [markdown]
# `determinations` has a row per target, in the order you gave them, with the bounds unrounded and the evidence each
# answer rests on. `aggregate_by_evidence()` lists every bound the call made, one row per model or ceiling and target,
# which is more than the finding shows:

# %%
print(output.determinations.select("metric", "threshold", "determination", "witness", "ceiling", "limiting_factor"))

# %%
bounds = output.aggregate_by_evidence()
print(bounds.select("source", "bound_type", "bound", "information_scope", "n", "evidence_id"))

# %% [markdown]
# Each evidence item records what its bound assumes, besides the assumption every answer carries. The same text is
# in `result.json`:

# %%
for evidence_id, item in output.evidence.items():
    print(f"{evidence_id}:")
    for assumption in item.details.get("assumptions", []):
        print(f"  - {assumption}")

# %% [markdown]
# ## Step 8: Rule a target out with allowances
#
# In step 5, nothing bounded the second target from above. Without repeated labels, which Flow does not take, the only
# upper bound the evaluator can set holds for models that give one answer for each group of similar-looking images, and
# that is a narrower kind of model than either target asks about. You can widen the reach of the bound by asserting an
# *allowance*. `beta: 0` asserts that no model reading the 768 numbers beats one fixed answer for each group of similar
# images, so the upper bound then holds for any model that reads only those numbers.
#
# :::{important}
# An allowance is **your assertion**. DataEval does not test it; it records it as assumed, and the report says so.
# Assert one only where you can justify it as a bound. It is asserted here to show what it changes, not because it is
# justified for MilitaryVehicles.
# :::
#
# The pipeline's second evaluator asks about the same two targets with `allowances=AllowancesConfig(beta=0)`. It also
# sets `alpha=0.025`, since it is a second look at the same images.

# %% tags=["remove_output"]
allowed = run_task(config, config.tasks[1])

# %% tags=["remove_cell"]
if not allowed.success:
    print(f"Workflow failed: {allowed.errors}")
assert allowed.success

# %%
print(allowed.report())

# %% [markdown]
# The first target is unchanged: it is demonstrated, and no allowance was needed to show it. The second is now **ruled
# out** (`✗`). Its upper bound is about 46%, below 50%, so, if the allowance holds, no model that reads only the 768
# numbers names the vehicle right 50% of the time on new images from this population. The `*` beside the upper bound
# marks an answer that rests on assumptions, and the `Allowances` line names the one you asserted. The finding still
# warns: a ruled-out target is not a demonstrated one, and it holds only as far as the allowance does.
#
# DataEval does check your allowance against what it measured, once. Had a tested model's lower bound been above the
# upper bound the allowance sets, the answer would have been *evidence conflict* (`!`), which means that something you
# assumed is wrong. Here the lower bound, about 42%, is below the upper bound, about 46%, so the allowance survives that
# test. That does not prove it.
#
# `gamma: 0`, which needs `beta`, would go further: it asserts that the 768 numbers lose nothing a model reading the
# whole image could use, and the upper bound would then hold for any model. It is not asserted here, because a model
# trained on these vehicles could use detail that a general-purpose model's 768 numbers do not carry.

# %% [markdown]
# ## Reading the output: a short glossary
#
# - **Target**: a number you want a model to reach, such as 85% accuracy. The report writes it as `accuracy ≥ 85%`.
# - **Confidence**: how sure the evaluator is. "95% confidence" means that the method gives a wrong answer at most 5% of
#   the time. It is one figure for all the targets together, and it is for new images from the same population as the
#   ones you gave it, not for images from somewhere else.
# - **Lower bound**: a number the evaluator is confident a model's accuracy on new images is *at least*. It comes from a
#   model that was really tested, on images it was not trained on, and is rounded down, so a lower bound never claims
#   more than was found. Only a lower bound can show that a target is reachable.
# - **Upper bound**: a number that, at the stated confidence, no model of the kind it covers can exceed on new images.
#   It comes from a ceiling that the data sets: images that look alike but carry different labels cap what any model
#   can do on them. It is rounded up, so it never claims more than was found. Only an upper bound can rule a target out.
# - **Demonstrated ✓**: a tested model's lower bound reaches the target. The target is reachable, at the stated
#   confidence, if no evaluation image was used to train or choose a model, or to set the target.
# - **Insufficient evidence ?**: the target is neither shown nor ruled out. To show it, test a stronger model; to rule
#   it out, get repeated labels or allowances you can justify.
# - **Not ruled out ~**: nothing rules the target out, and no tested model has shown it. DataEval's own printout marks
#   it `?`, as it does *insufficient evidence*; the report uses `~` so that the two read apart.
# - **Ruled out ✗**: an upper bound is below the target. This holds only under the assumptions recorded with the
#   evidence, which include any allowances. A `*` beside the upper bound marks it.
# - **Evidence conflict !**: a tested model's lower bound is above an upper bound. Something you assumed is wrong, so
#   neither bound is to be trusted until you find it.
# - **Allowances**: assumptions that you assert and DataEval does not test, which let it set an upper bound for broader
#   kinds of model. `beta: 0` asserts that no model reading the embeddings beats one fixed answer for each group of
#   similar images, which sets the bound for models that read only the embeddings. Adding `gamma: 0` asserts that the
#   embeddings lose nothing a model reading the whole image could use, which sets it for any model. Assert one only
#   where you can justify it.
# - **The bar**: the whole scale from 0 to 100% accuracy, split into what is *shown* (up to the lower bound), what is
#   *ruled out* (above the upper bound) and what is *open* between them. The best accuracy that a model of the kind
#   asked about can reach is somewhere in the open part.

# %% [markdown]
# ## Conclusion
#
# In this tutorial, you learned how to:
#
# - Set performance targets and run the `feasibility` evaluator and the `performance-feasibility` check in a workflow.
# - Read the text and HTML reports: each target's answer, its bounds and the confidence.
# - Open the evaluator's output to see DataEval's full printout and the evidence behind each bound.
# - Assert an allowance, see how it changes the answer, and see how DataEval checks it against what it measured.
# - Share `alpha` between targets and between runs, since one call is one look.
# - Tell a target that is not demonstrated from one that is ruled out, and why only the second needs an assumption.

# %% [markdown]
# ## Next steps
#
# - **Settings**: [`feasibility`](../reference/evaluators.md#feasibility) and
#   [`performance-feasibility`](../reference/checks.md#performance-feasibility) list every setting. A `target:` can ask
#   about one class, `ap:<class>`, or about models that read less than the whole image, `/fixed_representation`.
# - **Training data**: {doc}`Measure how much training data a model needs <measure_data_sufficiency>` asks a
#   related question of a model you train.
# - **Your own pipeline**: {doc}`Chain steps into a workflow of your own <../how_to/write_a_custom_workflow>` explains
#   the workflow's parts, and how to write this one as YAML.

# %% [markdown]
# ## Related guides
#
# - **Concept**: DataEval's
#   [feasibility tutorial](https://dataeval.readthedocs.io/en/latest/notebooks/tt_assess_ic_feasibility.html) explains
#   the bounds and the five answers, and how to judge a model of your own.
# - **How-to**: {doc}`Reuse results with the disk cache <../how_to/reuse_results_with_cache>` keeps embeddings between
#   sessions, which this tutorial does not do.
# - **How-to**: {doc}`Sweep settings with a matrix <../how_to/run_a_matrix>` explains why a `matrix:` over `feasibility`
#   may vary only `sources`.
