# Save a workflow from Python

Build a custom workflow in Python, try it on the data at hand, and save it as YAML that loads and runs without the code
that built it. `CustomWorkflowConfig` builds the chain, `run` tries it, and `save` writes it into a config file.
[Reuse a cleaning chain on new data](reuse_a_workflow.md) keeps the same chain in YAML instead, and
[Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) explains the chain. The chain below drops outliers and
duplicates with the [`remove`](../reference/transforms.md#remove) transform. You need a dataset in memory, such as any
`AnnotatedDataset`, and a `config/` directory.

## Build the chain and run it

Build the chain in Python and try it on the data at hand. `run` takes the entries its steps name as `definitions`:

```python
from dataeval_flow import run
from dataeval_flow.evaluators.quality import DuplicatesConfig, OutliersConfig
from dataeval_flow.steps import CustomWorkflowConfig, StepEntry

outliers = OutliersConfig(name="outliers", flags=["pixel", "visual"], outlier_threshold="zscore")
dupes = DuplicatesConfig(name="dupes")
my_clean = CustomWorkflowConfig(
    name="my_clean",
    inputs=["data"],
    steps=[
        StepEntry(name="outliers", evaluator="outliers", input="data"),
        StepEntry(name="dupes", evaluator="dupes", input="data"),
        StepEntry(
            name="clean",
            transform="remove",
            input="data",
            plans={"dupes": {"dup_types": ["exact", "near"], "keep": "first"}, "outliers": {"min_flags": 1}},
        ),
    ],
)
result = run(my_clean, september, definitions=[outliers, dupes])  # september: any dataset in memory
```

Building the workflow checks its slots, its step names, and each transform's settings. Whether `dupes` names a real
`evaluators:` entry, and whether the steps connect, is checked when the file loads into a pipeline.

## Save the chain

When it does what you want, save it with the same `definitions`. The file then holds the whole block, so it loads and
runs without the code that built it:

```python
my_clean.save("config/my_clean.yaml", definitions=[outliers, dupes])  # the config/ directory must already exist
```

It writes each entry with its name, its type and the settings that differ from their defaults, as you would write it by
hand. `my_clean.to_yaml(definitions=[outliers, dupes])` returns the same block as text:

```yaml
evaluators:
- name: outliers
  type: outliers
  flags:
  - pixel
  - visual
  outlier_threshold: zscore
- name: dupes
  type: duplicates
workflows:
- name: my_clean
  inputs:
  - data
  steps:
  - name: outliers
    evaluator: outliers
    input: data
  - name: dupes
    evaluator: dupes
    input: data
  - name: clean
    transform: remove
    input: data
    plans:
      dupes:
        dup_types:
        - exact
        - near
        keep: first
      outliers:
        min_flags: 1
```

`definitions` is optional. It takes the entries the steps name, as `run` does, such as `DuplicatesConfig(name="dupes")`
for a step that reads a `dupes` entry. Without it, `save` writes the workflow alone, and `to_yaml` returns the workflow
alone:

```python
my_clean.save("config/workflows.yaml")  # the workflow only; the config/ directory must already exist
```

`save` adds the workflow to the file, or replaces an entry of the same name in each section, and keeps the file's other
keys and entries. It creates a file that does not exist, but not a missing directory, so save to a path whose directory
exists. It refuses a file that is not a pipeline config. PyYAML keeps no comments, so a file `save` rewrites loses them.
It does not check that `definitions` holds every entry the steps name, since a chain saved into one file of a config
folder may rely on entries in another. Loading the file names any entry that is missing.

`load_config` reads a folder as one pipeline, so `dataeval-flow -c config/` runs a `workflows.yaml` beside your other
config files. Loading a config and saving it, from the TUI or the config builder, keeps every custom workflow as
written.

## Run the saved chain on new data

Later, on the next batch, load the block and run it:

```python
from dataeval_flow import load_config, run

recipe = load_config("config/my_clean.yaml")
(my_clean,) = recipe.workflows
result = run(my_clean, october, definitions=[*recipe.evaluators])  # october: the next dataset
cleaned = result.steps["clean"].output  # the October batch without what the chain removed
```

## Next steps

- [Reuse a cleaning chain on new data](reuse_a_workflow.md) — Keep the chain in your config or in a file each project
  loads.
- [Chain steps into a workflow of your own](write_a_custom_workflow.md) — Write the chain's steps and read its result.
- [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) — Learn how inputs, addresses and lists, derived data
  and failures work in a chain.
