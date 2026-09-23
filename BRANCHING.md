# DataEval Flow Branching and Release Strategy

DataEval Flow follows a lightweight **GitLab Flow** model: trunk-based
development on `main` with semver tags driving releases.

Supported minor versions are maintained on long-lived `release/vX.Y` branches, so
a line can receive patches without pulling in everything that has landed on
`main` since.

This document describes the **current state** of the project's branching and
release process. [scripts/release.py](scripts/release.py) picks the version,
promotes the changelog and tags. Label-driven bumps, where `release::*`
merge-request labels choose major/minor/patch, are still planned — see
[ROADMAP.md](ROADMAP.md#v030--release-automation--container-hardening).

## Table of Contents

- [Overview](#overview)
- [Branch Structure](#branch-structure)
- [Release Process](#release-process)
- [Release Lines and Hotfixes](#release-lines-and-hotfixes)
- [CI/CD Gates](#cicd-gates)
- [Best Practices](#best-practices)
- [Planned Improvements](#planned-improvements)

## Overview

### Key Principles

- **Single source of truth:** all features and fixes merge to `main` first
- **Semantic versioning:** [Semver 2.0.0](https://semver.org/) — `vMAJOR.MINOR.PATCH`
- **Tag-driven releases:** annotated git tags matching `v\d+.*` trigger the
  GitLab release pipeline (PyPI publish, Docker image build/sign/push,
  documentation publish), whether tagged on `main` or on a release line
- **Fixes flow forward first:** a fix lands on `main`, then is cherry-picked back
  onto any supported `release/vX.Y` line that needs it

## Branch Structure

| Branch                     | Purpose                                        | Lifetime    | Protected |
| -------------------------- | ---------------------------------------------- | ----------- | --------- |
| `main`                     | Primary development branch, releasable         | Permanent   | Yes       |
| `release/vX.Y`             | Maintenance line for a supported minor version | Long-lived  | Yes       |
| `feature/*`, `fix/*`, etc. | Feature/fix development branches               | Short-lived | No        |

### Branch Naming Conventions

While not strictly enforced, the following patterns are recommended for clarity:

- **Features:** `feature/<description>` or `<username>/<description>`
- **Bug fixes:** `fix/<issue-number>-<description>` or `<username>/<description>`
- **Documentation:** `docs/<description>`
- **Chores:** `chore/<description>`

## Release Process

### Cutting a Release

1. Ensure the branch is green and includes the work to be released.
2. Record user-visible changes under an `## Unreleased` heading in
   [CHANGELOG.md](CHANGELOG.md), using `### Added`, `### Changed`, `### Fixed`
   and `### Removed` subsections (and others, such as `### Infrastructure`, as
   needed). [scripts/release.py](scripts/release.py) renames that heading to the
   new version; it does not generate the notes, because
   [publish.yml](.github/workflows/publish.yml) lifts the section verbatim as the
   GitHub Release body.
3. **Do not edit any version field.** The package version is derived from the
   git tag by `hatch-vcs`, which writes `src/dataeval_flow/_version.py` at
   build time. `pyproject.toml` declares `dynamic = ["version"]` and carries no
   version to bump.
4. Open a release MR titled `Release vX.Y.Z`. Get review and merge to `main`.
5. Cut the release with the script, from `main` or a `release/vX.Y` branch:

   ```bash
   python scripts/release.py --dry-run  # preview; writes nothing
   python scripts/release.py            # main -> minor, release/vN.N -> patch
   ```

   It picks the version from the newest reachable tag, promotes the changelog
   section, commits `Release vX.Y.Z` and creates the annotated tag. It never
   pushes — review, then publish:

   ```bash
   git push --follow-tags origin <branch>
   ```

   A release branch carries patches only, so the script refuses `[feat]`,
   `[major]` and `[depr]` commits there and names them. Land those on `main`.

6. The tag triggers:
   - **GitLab CI** ([.gitlab-ci.yml](.gitlab-ci.yml)) — builds and signs the
     three Docker variants (cpu / cu126 / cu130), pushes them to
     Harbor, and creates per-variant release tags.
   - **GitHub Actions** ([.github/workflows/publish.yml](.github/workflows/publish.yml))
     — builds the Python package with `uv build`, extracts the matching
     CHANGELOG section as the release body, creates the GitHub Release, and
     publishes to PyPI via Trusted Publisher.

### Version Increment Guidance

Choose the version bump per Semver 2.0.0:

- **MAJOR (`X.0.0`)** — incompatible API changes, removal of public symbols,
  breaking changes to the workflow YAML schema or container interface.
- **MINOR (`0.X.0`)** — new features, new workflows, new extractors, new
  config fields that are backward-compatible.
- **PATCH (`0.0.X`)** — bug fixes only; no new public API.

While the project is **0.x (Alpha)**, minor bumps may include breaking changes;
this is consistent with Semver's pre-1.0 allowance. Breaking changes should be
called out explicitly in the CHANGELOG.

## Release Lines and Hotfixes

Fixes always land on `main` first. A fix that also belongs on a supported older
line is then cherry-picked onto that line's `release/vX.Y` branch, so `main`
never has to be held back and the older line never has to take unrelated work.

### Cutting a release line

Not every release needs a branch — cut one when a minor version must keep
receiving fixes after `main` has moved on. Branch from the line's newest release
tag, never from `main`:

```bash
git switch -c release/v0.2 v0.2.2
git push -u origin release/v0.2
```

Then mark the branch **protected** in the GitLab project settings. Beyond the
usual merge protection, the shared CI rules key security scanning off
`$CI_COMMIT_REF_PROTECTED`, so protecting the branch is what turns its scans on.

Branching from the release tag is what keeps versioning correct: `git describe`
on the new branch resolves against `v0.2.2`, so the next patch computes as
`v0.2.3` rather than inheriting whatever `main` has reached.

### Backporting a fix

After a `[fix]` MR merges to `main`, decide which lines need it:

```bash
git fetch origin
git switch -c cherry-pick/fix-to-v0-2 origin/release/v0.2
git cherry-pick <commit-sha>
# resolve conflicts if any, then:
git push -u origin cherry-pick/fix-to-v0-2
# open an MR targeting release/v0.2, titled "[fix] ..."
```

MRs into a release branch run the full lint, type, test and security suite, the
same as MRs into `main` — the shared rules gate on the merge-request event, not
on the target branch.

### Releasing a patch

Identical to the [release process above](#cutting-a-release) — record the notes
under `## Unreleased`, open an MR against the release branch, merge — but run the
script on the release branch rather than on `main`, where it bumps the patch:

```bash
git switch release/v0.2
python scripts/release.py    # v0.2.2 -> v0.2.3
git push --follow-tags origin release/v0.2
```

Release lines carry **fixes and housekeeping only**. New features, removals and
breaking changes belong on `main` — shipping one in a patch breaks the promise
the version number makes.

### What the containers do

The tag pipeline publishes `0.2.3-<variant>` and nothing else. There are no
series or "newest release" pointers to move, so cutting a v0.2 patch after v0.3
has shipped cannot drag a shared tag backwards onto the older line — the only
floating tag is `latest-<variant>`, and it follows `main` alone.

Release branches build their images but never publish them: `validate:docker`
runs the build through its test stage and stops. See the
[Container Reference](docs/source/reference/containers.md#image-tags) for the
full tag scheme.

## CI/CD Gates

All MRs to `main` are gated by the GitLab CI pipeline ([.gitlab-ci.yml](.gitlab-ci.yml)),
which runs:

- **Lint** — `ruff` + `codespell` (`nox -s lint`)
- **Type check** — `pyright` over `src/` and `tests/` (`nox -s type`)
- **Schema validation** — config schema consistency (`nox -s schema`)
- **Tests** — pytest matrix across Python 3.10, 3.11, 3.12, 3.13, and 3.14 with
  90% coverage enforcement (`nox -s test`)
- **Security scans** — Semgrep SAST, Gemnasium dependency scanning, secret
  detection, SBOM generation (Syft / CycloneDX)
- **Documentation build** — Sphinx with `--fail-on-warning`

The `main` branch is protected at the GitLab project level: MRs require a
green pipeline and approval before merging.

## Best Practices

### For Developers

1. **One MR, one purpose** — keep changes focused so the CHANGELOG entry is easy to write
2. **Update CHANGELOG.md** as part of the MR when shipping user-visible changes
3. **Write clear MR descriptions** — they are the canonical record of the change
4. **Run `nox` locally** before pushing to catch issues fast
5. **Test thoroughly** — the 90% coverage gate is a floor, not a target

### For Maintainers

1. **Verify CHANGELOG accuracy** before tagging
2. **Confirm the tag is the intended version** — it is the sole source of truth
   for the released version; nothing in the tree needs to match it
3. **Watch the release pipeline** through PyPI publish and image push completion
4. **Validate signatures** of published container images (cosign verify with
   the public key at [docker/cosign.pub](docker/cosign.pub))

## Planned Improvements

`hatch-vcs` dynamic versioning — which removed the manual `pyproject.toml`
version bump and the risk of skew between the file and the git tag — shipped in
v0.1.1. The following enhancements are still planned (see
[ROADMAP.md](ROADMAP.md) for targets):

- **Label-driven semver releases** — `release::major | feature | improvement
  | deprecation | fix | misc` MR labels drive automatic version bumps,
  changelog generation, and tag creation. Mirrors the `dataeval` library's
  approach. *(planned: v0.3.0)*
- **`release::*` label validation in CI** — ensures every MR carries a valid
  release label so the changelog is always derivable. *(planned: v0.3.0)*

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [GitLab Flow Documentation](https://about.gitlab.com/topics/version-control/what-is-gitlab-flow/)
- Project CI/CD configuration: [`.gitlab-ci.yml`](.gitlab-ci.yml)
- Release publish workflow: [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
