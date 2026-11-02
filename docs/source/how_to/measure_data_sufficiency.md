# Measure how much training data a model needs

The `sufficiency` evaluator trains your model on growing subsets of a training set, scores it on a test set at each
size, and fits a curve to the scores. The `performance-sufficiency` check reads that curve: whether a metric reached your
target, and if not, about how much data the curve says it would take. Your training code is a trainer file in the
data root's `trainers/` folder, which Flow loads when the run starts. Both steps are experimental.

This guide mounts one trainer, runs it on MilitaryVehicles in a workflow of your own, and reads the check.

## 1. Write a trainer

A trainer is a subclass of {py:class}`~dataeval_flow.evaluators.performance.Trainer` in one Python file. Flow
constructs it with the entry's `params` as keyword arguments and calls its `build()`, which returns a
{py:class}`~dataeval_flow.evaluators.performance.TrainerKit`: the untrained model, a training strategy with
`train(model, dataset, indices)`, an evaluation strategy with `evaluate(model, dataset)` that returns each metric by
name, and optionally a reset strategy (by default, a deep copy of the model as built).

This one scores each image with a pretrained ONNX ImageNet classifier, once, and fits a nearest-centroid head on
those scores at each size. It reads its weights from the data root, through `params`:

```{literalinclude} ../../trainers/backbone.py
:language: python
```

Keep these in mind when you write your own:

- **One file.** Flow runs it under a private module name, so it cannot import a sibling file in `trainers/`. It can
  import what the environment has: in the container, what the variant's `requirements.<variant>.txt` lists, which
  includes PyTorch, torchvision, ONNX Runtime, OpenCV, scikit-learn, SciPy, LightGBM and Polars. The image has no
  `pip`.
- **Items come as stored**: `(image, target, metadata)`, the image channels first, at the dataset's own size.
- **Device.** Take it from `dataeval.config.get_device()`, which the result's `metadata.device` records.
- **No network.** Read pretrained weights from the data root, as `weights` does here.
- **DataLoader workers.** A `torch.utils.data.DataLoader` with workers needs `num_workers=0` or
  `multiprocessing_context="fork"`: a worker started any other way cannot import a file Flow loaded by path. In a
  container, forked workers share memory through `/dev/shm`, so give such a trainer `--shm-size`.
- **Process-wide settings stay set.** Something the file sets for the whole process, such as a torch flag, holds for
  the later tasks of the same run.

## 2. Lay out the data root

The trainer sits in the data root's `trainers/` folder, beside `data/` and `models/`:

```text
project/
├── pipeline.yaml
├── data/militaryvehicles/{train_fine,test_fine}/
├── models/resnet50-v1-7.onnx
└── trainers/backbone.py
```

A trainer path must start with `trainers/`, and the file, once symlinks are followed, must sit inside that folder. A
path outside it fails the config load; a symlink out of it fails the run.

## 3. Chain the evaluator and the check

```yaml
datasets:
  - {name: vehicles_train, format: image_folder, path: data/militaryvehicles/train_fine, infer_labels: true}
  - {name: vehicles_test, format: image_folder, path: data/militaryvehicles/test_fine, infer_labels: true}

sources:
  - {name: train, dataset: vehicles_train}
  - {name: test, dataset: vehicles_test}

evaluators:
  - name: sufficiency
    type: sufficiency
    trainer:
      path: trainers/backbone.py
      class: BackboneTrainer
      params: {weights: models/resnet50-v1-7.onnx}

workflows:
  - name: how_much_data
    inputs: [train, test]
    steps:
      - {name: measured, evaluator: sufficiency, input: [train, test]}
      - name: target
        check: performance-sufficiency
        input: measured
        targets:
          - {metric: accuracy, value: 0.25}
          - {metric: accuracy, value: 0.27}
          - {metric: accuracy, value: 0.4}

tasks:
  - {name: sufficiency, workflow: how_much_data, sources: [train, test]}

seed: 0
```

Notice:

- **Order decides the roles.** The evaluator's first input is the training set and its second the test set.
- **The default schedule** measures five sizes, from 1% of the training set to all of it, which needs at least 100
  training items. Set `schedule:` to name the sizes yourself, each no larger than the training set.
- **A size counts draws**, made with replacement, not distinct items: 50 draws may cover 33 different items.
- **`seed:`** makes the draws repeatable.
- **`params`** are not checked when the config loads; the trainer checks them when the run starts.

## 4. Run it

```bash
dataeval-flow -c pipeline.yaml -o out/
```

Training takes the time. Add `-vv` to see a line as each training size is measured, such as
`Sufficiency 5/5: run 1 of 1, 7823 items, accuracy=0.2517`, the last of this run's five. The service shows the same
lines in a run's logs.

In a container, mount the project as the data root, `trainers/` and all. A GPU variant trains on the GPU:

```bash
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/project",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cu130 \
    python -m dataeval_flow --config pipeline.yaml
```

The data root is read-only, and anything the trainer writes outside `/output` or `/cache` is lost when the container
exits. To mount the trainers from elsewhere, at `/dataeval/trainers`, create an empty `trainers/` folder in the data
directory first: Docker cannot make the mount point inside a read-only mount, and the container fails to start.

On a 32-thread CPU the run took under a minute.

## 5. Read the check

```text
================================================================================
  SUFFICIENCY TARGET                       1 of 3 targets reached at 7,823 items
================================================================================
  Each target is read at the largest size measured, 7,823 of the training set's
  7,823 items; higher is better. Where it is not reached, the size is where
  DataEval's fitted power law reaches it, and a size past the largest measured
  is an extrapolation of that curve, less certain the further it reaches.

  `accuracy` at least 0.25 — reached
    At 7,823 items: 0.2517
    Reached:        yes

  `accuracy` at least 0.27 — not reached; about 57,606 items
    At 7,823 items:   0.2517
    Reached:          no
    Projected size:   about 57,606 items
    Ratio to largest: 7.36

  `accuracy` at least 0.4 — not reached by the fitted curve
    At 7,823 items: 0.2517
    Reached:        no
    Projected size: not reached by the fitted curve
```

The accuracy rose from 0.14 at 78 images to 0.25 at all 7,823, and the curve flattens toward 0.28. The finding warns,
since 0.27 would take about 7.4 times this training set by the curve, an extrapolation that far, and 0.4 is past the
curve's limit. What it shows is that this model, a fixed backbone and a centroid per class, limits the accuracy more
than the amount of data does. MilitaryVehicles' test split also holds its classes in other proportions than train
(Cramér's V 0.19 between split and class), so a score on it reflects that shift as well. On another device the
numbers differ in the third decimal.

A target reached at the largest size is `ok`. One not reached there, whose projection still falls within the training
set, is `info`: rare with the default schedule, whose largest size is the whole training set, and common with a
shorter `schedule:`. A projection past the training set, or a target the curve never reaches, warns. A per-class
metric, returned as an array, cannot be a target and is not assessed; a target naming a metric the run did not measure
fails the step, listing the metrics it did.

`out/results/result.json` keeps the measurements under the step's `output`, and under its `extras` the trainer's
`path`, `class` and `sha256`, and `params`, the fitted curve the check projected from.

## Next steps

- [`sufficiency`](../reference/evaluators.md#sufficiency) and
  [`performance-sufficiency`](../reference/checks.md#performance-sufficiency) list every setting.
- [Chain steps into a workflow of your own](write_a_custom_workflow.md) explains the workflow's parts.
- [Run workflows in containers](containerized_workflows.md) covers the mounts and the variants.
