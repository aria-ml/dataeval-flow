# Judge a custom workflow like a preset

A preset's report holds its findings under headings, such as the questions of [`audit`](../reference/presets.md#audit),
and a preset may add a verdict. A custom workflow does the same with `groups:`, `verdict:` and the keys beside them, and
judges some of its steps apart from the rest with `scopes:`.
[Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) explains how a chain is judged, and the
[Check Catalog](../reference/checks.md) lists the check types the examples name. You need a custom workflow with check
steps, as the "Judge what the workflow found" section of
[Chain steps into a workflow of your own](write_a_custom_workflow.md) builds one.

## Group the findings under headings

A custom workflow groups its findings with `groups:`. Each group has a heading, and the check types whose findings it
holds, as the steps' `check:` names them:

```yaml
evaluators:
  - {name: dupes, type: duplicates}
  - {name: labels, type: label-health}

workflows:
  - name: clean-check
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - {name: labels, evaluator: labels, input: data}
      - {name: duplicates, check: image-duplicates, input: dupes}
      - {name: imbalance, check: class-imbalance, input: labels}
    groups:
      - {heading: "Is the data clean?", checks: [image-duplicates]}
      - {heading: "Are the labels sound?", checks: [class-imbalance]}
```

Each heading's status line counts its checks' warnings, and the findings of check types no heading names follow the
groups. A group that names a check type no step runs is refused when the config loads. Groups change only the report. A
group's `evidence:` lists step types shown under it after its findings, such as `diversity`.

## Declare a verdict, a record and next steps

A custom workflow can also declare what a preset's chain declares, so it reads and reports like `audit`:

```yaml
workflows:
  - name: clean-check
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - {name: labels, evaluator: labels, input: data}
      - {name: duplicates, check: image-duplicates, input: dupes}
      - {name: imbalance, check: class-imbalance, input: labels}
    reference: data
    record: {title: What was read, steps: [label-health]}
    verdict:
      blocking: [image-duplicates]
      accepted: {class-imbalance: Rare classes by design.}
    next_steps:
      by_check: {image-duplicates: Remove them with a `remove` step.}
    preflight: {nonempty: true, one_kind: true}
```

- `reference` encodes every Dataset's metadata like the named input's.
- `record` adds a section recording what the run read.
- `verdict` gives a verdict: an unaccepted warning of a `blocking` check type makes it not ready.
- `next_steps` says what to do about each warning, by check type, and each check left unassessed, by reason. Advice by
  check type needs a `verdict`, whose warnings it advises on.
- `preflight` refuses an input with no items, or inputs of different kinds, before any step runs.

A workflow with an `audit` step takes its verdict, reference, record and next steps from that step, so it declares none
of them itself. [Check a set of splits](check_a_set_of_splits.md) runs one.

## Judge some steps together

A workflow's `verdict`, `preflight`, `reference`, `record`, `groups` and `next_steps` cover all its steps. A `scopes:`
entry holds some of them instead, judged on inputs of its own, as a preset run as a step is judged on its own (see
[Check a set of splits](check_a_set_of_splits.md)). The example reads a `splits` entry named `splitting`, which that
guide defines:

```yaml
workflows:
  - name: splitting
    type: splits
    split_on: [scene]
```

The scope below judges the train part of that split:

```yaml
workflows:
  - name: split_and_check
    inputs: [data]
    steps:
      - {name: splits, workflow: splitting, input: data}
      - {name: dupes, evaluator: dupes, input: splits.train}
      - {name: labels, evaluator: labels, input: splits.train}
      - {name: duplicates, check: image-duplicates, input: dupes}
      - {name: imbalance, check: class-imbalance, input: labels}
    scopes:
      - name: train-checks
        inputs: {train: splits.train}
        steps: [dupes, labels, duplicates, imbalance]
        reference: train
        record: {title: What was read, steps: [label-health]}
        verdict:
          blocking: [image-duplicates]
          accepted: {class-imbalance: Rare classes by design.}
        next_steps:
          by_check: {image-duplicates: Remove them with a `remove` step.}
        preflight: {nonempty: true}
```

- `inputs` names each input of the scope and binds it to an address of the workflow, which must hold a Dataset made
  before the scope's first step. The scope's steps still read the workflow's addresses: an input's name is what
  `reference` names, and what preflight messages and the record's columns call it.
- `steps` names steps of the workflow, each in one scope at most, and no preset step: inline it first. `groups`,
  `reference`, `record`, `verdict`, `next_steps` and `preflight` are declared as in the sections above, over the scope's
  steps and inputs. `reference` names one of the scope's inputs.
- The scope starts as a preset step does, before its first step runs: it binds its inputs, and its `preflight` and
  `reference` can fail it there. If an input held nothing, because a step before it failed or was skipped, every step of
  the scope is skipped and the scope gives no verdict, which the report and the JSON's `no_verdict` say.
- Only a scope that gives a `verdict` has its verdict, record, groups and next steps reported, as a nested preset's are.
  A workflow gives one verdict, so it may not declare `verdict:` itself, or its own `reference`, `record` or
  `next_steps`, beside a verdict scope.
- A matrix cannot vary a scope's `name`, `inputs` or `steps`, because they wire the chain every run shares.

The service's `inline_step` writes a scope when it inlines a preset step whose preset declares more than its steps (a
verdict, preflight, reference, record, groups or next steps), so the inlined steps keep what it declared. It writes one
holding only `inputs` and `steps` when the step reads another step's output, so the inlined steps still skip together
when that output held nothing. See [Preset decomposition](../reference/service.md#inline).

## Next steps

- [Check a set of splits](check_a_set_of_splits.md) — Run an `audit` step on the parts of a split, which takes its
  verdict from the step.
- [Gate training on an audit](gate_training_on_an_audit.md) — Gate a run on a verdict with `--require`.
- [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md) — Learn how a chain's health and failures roll up.
