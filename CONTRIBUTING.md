# Contributing

Thank you for your interest in DataEval Flow! Contributions, bug reports, and
suggestions for improvement are welcome.

## Development Setup

DataEval Flow uses [uv](https://docs.astral.sh/uv/) for environment management and
[nox](https://nox.thea.codes/) as its task runner. `uv` is the only thing you need
installed up front — it fetches everything else on demand.

### Bootstrapping an environment

```bash
git clone https://github.com/aria-ml/dataeval-flow.git
cd dataeval-flow
uvx --with nox-uv nox -s dev
```

That builds `.venv` with DataEval Flow and every development dependency. Run with
no arguments it prompts for the Python version and the device variant; pass them as
arguments to skip the prompts:

```bash
uvx --with nox-uv nox -s dev -- --python 3.12 --device cu130
```

| Flag | Values | Default |
| ---- | ------ | ------- |
| `-p`, `--python` | `3.10` – `3.14` | `3.11` |
| `-d`, `--device` | `cpu`, `cu126`, `cu130` | `cpu` |
| `-n`, `--name` | any directory | `.venv` |

Alongside the device variant it installs the matching `onnx` extra (`onnx` for
`cpu`, `onnx-cu126` / `onnx-cu130` for CUDA) and the `app` extra — the same set the `test` and
`type` sessions build against. The chosen device is written to `.cuda-version`,
which the other sessions read so they build against the same PyTorch variant.

Activate the result and you are ready to work:

```bash
source .venv/bin/activate
```

:warning: **Bootstrap with `uvx`, not `uv run`.** `uv run nox -s dev` would run nox
out of the very environment the session is about to delete and rebuild; the session
detects this and refuses to start. `uvx` fetches a throwaway nox instead, so there
is nothing to pull out from under. `--with nox-uv` is not optional for the other
sessions — `noxfile.py` imports `nox_uv` at module scope, so a bare `uvx nox` cannot
even load it.

### Running checks

Every task is a nox session — `uvx --with nox-uv nox -l` lists them. Bare `nox` runs
the default set (`lint`, `type`, `test`, `schema`, `check`). Individually:

```bash
uv run nox -s test      # unit tests, 90% coverage gate
uv run nox -s lint      # ruff and codespell
uv run nox -s type      # pyright and type-completeness
uv run nox -s schema    # regenerate and verify config/params.schema.json
uv run nox -s check     # validate uv.lock and poetry.lock
uv run nox -s docs      # build the documentation
uv run nox -s verify    # FR/NFR requirements verification suite
```

Once `.venv` exists, `uv run nox ...` is the convenient form for everything except
`dev` itself; `uvx --with nox-uv nox ...` works from anywhere and needs no project
environment at all.

## How Can I Contribute?

### Reporting Bugs

Bug reports can be submitted in several ways. The guidelines below help us
investigate and resolve issues quickly.

#### Crafting a Bug Report

The bug report should be in the following format and contain as much detail as
possible.

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

Bugs can be notoriously difficult to pin down and eliminate, but following the
tips below can help the maintainers do the best they can.

- Use a clear and descriptive title
- Describe the exact steps (before and during) which led to the issue
- Provide specific examples (such as data inputs, configs, or model files)
- Include the workflow YAML or relevant config snippet when possible
- Describe the behavior observed after following each step
- Explain what the expected behavior was compared to what was observed
- Include full callstacks and error messages when possible

### Suggestions for Improvement

We are always excited to hear ideas for new workflows, extractors, or
improvements to existing features.

Feel free to reach out to <dataeval-flow@ariacoustics.com> — we would love to
hear from you.

## Branching Strategy

See [BRANCHING.md](BRANCHING.md) for the project's branching and release
strategy. In short: feature branches off `main`, merge requests gated by CI,
semver tags drive releases.

## Code of Conduct

By participating in this project you agree to abide by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
