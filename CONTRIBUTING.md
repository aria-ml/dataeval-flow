# Contributing

Thank you for your interest in DataEval Flow! Contributions, bug reports, and
suggestions for improvement are always welcome.

## Development Setup

DataEval Flow uses [uv](https://docs.astral.sh/uv/) for environment management and
[nox](https://nox.thea.codes/) as its task runner. Install only `uv` up front; it
fetches everything else on demand.

### Bootstrapping an environment

```bash
git clone https://github.com/aria-ml/dataeval-flow.git
cd dataeval-flow
uvx --with nox-uv nox -s dev
```

This builds `.venv` with DataEval Flow and all development dependencies. With no
arguments it prompts for the Python version and device variant; pass them as
arguments to skip the prompts:

```bash
uvx --with nox-uv nox -s dev -- --python 3.12 --device cu130
```

| Flag | Values | Default |
| ---- | ------ | ------- |
| `-p`, `--python` | `3.11` – `3.14` | `3.11` |
| `-d`, `--device` | `cpu`, `cu126`, `cu130` | `cpu` |
| `-n`, `--name` | any directory | `.venv` |

Alongside the device variant it installs the matching `onnx` extra (`onnx` for
`cpu`, `onnx-cu126` / `onnx-cu130` for CUDA) and the `app` extra — the same set the `test` and
`type` sessions build against. The chosen device is written to `.cuda-version`,
which the other sessions read so they build against the same PyTorch variant.

Activate the environment:

```bash
source .venv/bin/activate
```

:warning: **Bootstrap with `uvx`, not `uv run`.** `uv run nox -s dev` would run nox
from the environment the session deletes and rebuilds; the session detects this and
refuses to start. `uvx` fetches a throwaway nox. `--with nox-uv` is required for the
other sessions: `noxfile.py` imports `nox_uv` at module scope, so a bare `uvx nox`
cannot load it.

### Running checks

Every task is a nox session — `uvx --with nox-uv nox -l` lists them. Bare `nox` runs
the default set (`lint`, `type`, `test`, `schema`, `check`). Individually:

```bash
uv run nox -s test      # unit tests, 90% coverage gate
uv run nox -s lint      # ruff and codespell
uv run nox -s type      # pyright and type-completeness
uv run nox -s schema    # regenerate and verify config/params.schema.json
uv run nox -s check     # validate uv.lock and dependencies
uv run nox -s docs      # build the documentation
uv run nox -s verify    # FR/NFR requirements verification suite
```

Once `.venv` exists, `uv run nox ...` is the convenient form for everything except
`dev` itself; `uvx --with nox-uv nox ...` works from anywhere without a project
environment.

### Where code lives

Every public name has one home. Take the first rule that fits:

1. **The front door → `dataeval_flow`.** What you call to load and run anything, and
   what workflows and evaluators both share: `run`, `run_task`, `run_tasks`,
   `load_config`, `load_dataset`, `PipelineConfig`, `Result`, `ResultMetadata`,
   `InputSpec`, `InputKind`, `SourceCount`.
2. **A doer → `dataeval_flow.workflows` or `dataeval_flow.evaluators`.** The package
   holds its kind's framework: the base class, config and result bases, context,
   `get_*` and `list_*`. Each built-in type has a subpackage named after its type id
   (`data-cleaning` → `.data_cleaning`, `quality.*` → `.quality`) exporting what typed
   code names: its config, its result and the models inside them.
3. **Anything else a pipeline file describes → `dataeval_flow.config`.** The plain
   sections share one flat list: datasets, sources, views, preprocessors, metadata,
   stats, ontologies, tasks, exports and logging. A section whose entries are plugins
   has a subpackage: `config.extractors`, and `config.transforms` for what a
   preprocessor's `step:` names.
4. **Anything else is private:** an underscore somewhere in its import path.

A plugin registers under the entry-point group `dataeval_flow.<kind>`: `workflows`,
`evaluators`, `extractors` or `transforms`. The group names the kind, not the module.

`tests/public_api.txt` pins the result, and `tests/test_public_surface.py` checks that
each public name has one import path and every other module is private. When the
public surface changes, regenerate the snapshot deliberately and review its diff.

## How Can I Contribute?

### Reporting Bugs

The guidelines below help maintainers investigate and resolve issues quickly.

#### Crafting a Bug Report

Use the following format and include as much detail as possible.

```text
Steps to Reproduce:
 1.
 2.
 3.
 ...

Expected Behavior:

Actual Behavior:

Frequency of Behavior:

Environment:
 - dataeval-flow version:
 - Python version:
 - OS / container variant (cpu / cu126 / cu130):
 - GPU + driver (if applicable):
```

#### Submitting a Bug Report

Bugs are tracked via issues in our internal GitLab repository. Issues can also
be reported on GitHub or by emailing <dataeval-flow@ariacoustics.com>. For
issues created in GitHub, please follow the bug report template above.

#### Making it Good(tm)

The tips below help maintainers:

- Use a clear and descriptive title
- Describe the exact steps (before and during) which led to the issue
- Provide specific examples (such as data inputs, configs, or model files)
- Include the workflow YAML or relevant config snippet when possible
- Describe the behavior observed after following each step
- Explain what the expected behavior was compared to what was observed
- Include full callstacks and error messages when possible

### Suggestions for Improvement

Ideas for new workflows, extractors, or improvements are welcome. Reach out to
<dataeval-flow@ariacoustics.com>.

## Branching Strategy

See [BRANCHING.md](BRANCHING.md) for the project's branching and release
strategy. In short: feature branches off `main`, merge requests gated by CI,
semver tags drive releases.

## Code of Conduct

By participating in this project you agree to abide by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
