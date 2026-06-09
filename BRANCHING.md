# DataEval Flow Branching and Release Strategy

DataEval Flow follows a lightweight **GitLab Flow** model: trunk-based
development on `main` with semver tags driving releases.

This document describes the **current state** of the project's branching and
release process. A more sophisticated label-driven release automation model
(mirroring the `dataeval` library) is planned for v0.3.0 — see
[ROADMAP.md](ROADMAP.md#v030--release-automation--container-hardening).

## Table of Contents

- [Overview](#overview)
- [Branch Structure](#branch-structure)
- [Release Process](#release-process)
- [Hotfixes](#hotfixes)
- [CI/CD Gates](#cicd-gates)
- [Best Practices](#best-practices)
- [Planned Improvements](#planned-improvements)

## Overview

### Key Principles

- **Single source of truth:** all features and fixes merge to `main` first
- **Semantic versioning:** [Semver 2.0.0](https://semver.org/) — `vMAJOR.MINOR.PATCH`
- **Tag-driven releases:** annotated git tags on `main` matching `v\d+.*` trigger
  the GitLab release pipeline (PyPI publish, Docker image build/sign/push,
  documentation publish)

## Branch Structure

| Branch                     | Purpose                                  | Lifetime    | Protected |
| -------------------------- | ---------------------------------------- | ----------- | --------- |
| `main`                     | Primary development branch, releasable   | Permanent   | Yes       |
| `feature/*`, `fix/*`, etc. | Feature/fix development branches         | Short-lived | No        |

### Branch Naming Conventions

While not strictly enforced, the following patterns are recommended for clarity:

- **Features:** `feature/<description>` or `<username>/<description>`
- **Bug fixes:** `fix/<issue-number>-<description>` or `<username>/<description>`
- **Documentation:** `docs/<description>`
- **Chores:** `chore/<description>`

## Release Process

### Cutting a Release

1. Ensure `main` is green and includes the work to be released.
2. Update [CHANGELOG.md](CHANGELOG.md) with a new section describing the
   release. Follow the existing `## vX.Y.Z` heading pattern with `### Features`
   and `### Infrastructure` subsections (and others as needed).
3. Bump `version` in [pyproject.toml](pyproject.toml) to match the planned tag
   (without the leading `v`).
   - Note: switching to `hatch-vcs` dynamic versioning is planned for v0.2.0
     (see [ROADMAP.md](ROADMAP.md)) — once that lands, this manual step goes
     away.
4. Open a release MR titled `Release vX.Y.Z`. Get review and merge to `main`.
5. From `main`, create an annotated tag and push it:

   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

6. The tag triggers:
   - **GitLab CI** ([.gitlab-ci.yml](.gitlab-ci.yml)) — builds and signs the
     four Docker variants (cpu / cu118 / cu124 / cu128), pushes them to
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

## Hotfixes

Today, hotfixes go through the same flow as any other change: branch from
`main`, MR to `main`, then cut a new patch release.

Long-lived `release/vX.Y` maintenance branches with cherry-pick-driven patch
releases are **not** in use yet. They are planned for v0.3.0 once the project
has multiple supported minor versions in production deployments.

## CI/CD Gates

All MRs to `main` are gated by the GitLab CI pipeline ([.gitlab-ci.yml](.gitlab-ci.yml)),
which runs:

- **Lint** — `ruff` + `codespell` (`nox -s lint`)
- **Type check** — `pyright` over `src/` and `tests/` (`nox -s type`)
- **Schema validation** — config schema consistency (`nox -s schema`)
- **Tests** — pytest matrix across Python 3.10, 3.11, 3.12 with 90% coverage
  enforcement (`nox -s test`)
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
2. **Confirm `pyproject.toml` version matches the planned tag** before pushing the tag
3. **Watch the release pipeline** through PyPI publish and image push completion
4. **Validate signatures** of published container images (cosign verify with
   the public key at [docker/cosign.pub](docker/cosign.pub))

## Planned Improvements

The following enhancements are planned (see [ROADMAP.md](ROADMAP.md) for
targets):

- **`hatch-vcs` dynamic versioning** — eliminates the manual `pyproject.toml`
  version bump and the risk of skew between the file and the git tag.
  *(planned: v0.2.0)*
- **Label-driven semver releases** — `release::major | feature | improvement
  | deprecation | fix | misc` MR labels drive automatic version bumps,
  changelog generation, and tag creation. Mirrors the `dataeval` library's
  approach. *(planned: v0.3.0)*
- **Long-lived `release/vX.Y` branches with cherry-pick patches** — enables
  hotfixes to older supported minor versions without forcing an upgrade.
  *(planned: v0.3.0, contingent on multiple supported releases existing)*
- **`release::*` label validation in CI** — ensures every MR carries a valid
  release label so the changelog is always derivable. *(planned: v0.3.0)*

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [GitLab Flow Documentation](https://about.gitlab.com/topics/version-control/what-is-gitlab-flow/)
- Project CI/CD configuration: [`.gitlab-ci.yml`](.gitlab-ci.yml)
- Release publish workflow: [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
