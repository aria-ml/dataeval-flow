# Documentation cache management

This directory contains scripts for managing Jupyter notebook execution cache using Git orphan branches.

## Overview

The `.jupyter_cache` artifacts are stored in separate orphan branches named `docs-artifacts/<branch-name>`. This keeps
the main branch clean and avoids committing large binary files.

### Benefits

- Smaller repository: the main branch does not contain large cache files
- Persistent cache: the cache survives across CI runs (no 7-day expiration)
- Faster builds: cached notebook execution results are reused
- Branch isolation: each branch has its own cache

## Architecture

```text
main branch (code only)
  ↓
  CI builds docs
  ↓
docs-artifacts/main (orphan branch, cache only)
```

### Orphan branches

Orphan branches have no parent commits and are disconnected from main branch history:

- `docs-artifacts/main` holds the cache for the main branch
- `docs-artifacts/<tag>`, such as `docs-artifacts/v0.2.5`, holds the cache and the notebooks of a tagged release
- `docs-artifacts/<branch>` exists for a feature branch only if you run `push-docs-cache.sh` for it yourself

A `release/v*` branch has no artifact branch of its own. Each artifact branch holds the `.jupyter_cache/` folder, the
generated `notebooks/*.ipynb` files that the Google Colab links open, and a `README.md`.

Never merge these branches into working branches.

## Scripts

### `fetch-docs-cache.sh`

This script fetches the `.jupyter_cache` from the artifact branch before building documentation. Run it as follows:

```bash
./docs/fetch-docs-cache.sh [branch-name]
```

A branch name given as an argument comes first. Otherwise the script reads one of these environment variables:

- `READTHEDOCS_GIT_IDENTIFIER` on Read the Docs, where HEAD is detached
- `CI_COMMIT_REF_NAME` on GitLab
- `GITHUB_REF_NAME` on GitHub

Outside CI, the script falls back to the current branch name.

If the artifact branch exists, the script extracts the cache to `docs/source/.jupyter_cache/`. If it does not, the
script tries these fallbacks in order and stops at the first that exists:

1. For a tag `vX.Y.Z`, the nearest lower patch of the same line, `docs-artifacts/vX.Y.<n>`
2. For a `release/vX.Y` branch, the newest `docs-artifacts/vX.Y.Z` of that line
3. `docs-artifacts/main`

If none exists, the script continues without a cache, which is normal for a first build.

### `push-docs-cache.sh`

This script pushes the generated `.jupyter_cache` to the artifact branch after building documentation. Run it as
follows:

```bash
./docs/push-docs-cache.sh [branch-name]
```

It reads these environment variables:

- `DATAEVAL_BUILD_PAT` on GitLab, used with the `dataeval-bot` account for the authenticated push. Without it, the
  script pushes to the URL of the `origin` remote.
- `GITLAB_USER_NAME` and `GITLAB_USER_EMAIL`, used as the Git commit identity. Without them, the script uses the local
  Git configuration.

The script does the following:

- Creates or updates the orphan branch with the latest cache, the generated `.ipynb` notebooks and a `README.md`
- Includes metadata (timestamp, file count, size) in the commit message
- Force pushes (safe because the orphan branch has no shared history)
- Skips the push to the branch artifact for a `release/v*` branch, since nothing fetches `docs-artifacts/release/v*`
- Also pushes `docs-artifacts/<tag>` when HEAD carries a `v*` tag

## CI/CD integration

### GitLab CI (primary)

DataEval Flow runs CI on GitLab and mirrors to GitHub. The docs jobs live in the top-level `.gitlab-ci.yml`:

- `docs:preview` (merge requests) builds docs with `nox -s docs`. nox internally calls `fetch-docs-cache.sh`, so the MR
  build reuses the cache from `docs-artifacts/<mr-branch>` (or falls back to `docs-artifacts/main`). It does not push.
  Previews are read-only.
- `pages` (main branch only) builds docs with `nox -s docs`, then runs `push-docs-cache.sh` to update
  `docs-artifacts/main` before publishing to GitLab Pages.

Pushing requires the group-level `DATAEVAL_BUILD_PAT` CI/CD variable (a project/group access token with
`write_repository` scope, used as the `dataeval-bot` account).

### GitHub Actions (mirror)

The `.github/workflows/docs.yml` workflow runs these steps:

1. Checkout with full history (`fetch-depth: 0`)
1. Build docs with nox (nox internally calls `fetch-docs-cache.sh`)
1. Push cache is disabled. GitLab owns pushing the cache.

```yaml
permissions:
  contents: write # Required to push to artifact branches
```

## Cache policy

### When the cache is pushed

CI pushes artifact branches from one job, `pages`, which runs on the main branch only:

- `docs-artifacts/main` on every run
- `docs-artifacts/<tag>` as well, when the commit it builds carries a `v*` tag

Merge request builds and builds on `release/v*` branches push nothing. A release branch builds from the newest
`docs-artifacts/vX.Y.Z` of its line. This prevents proliferation of artifact branches for short-lived feature
branches.

### Cache invalidation

The `pages` job updates `docs-artifacts/main` on every run. To clear the cache manually, run:

```bash
# Delete the artifact branch
git push origin --delete docs-artifacts/main

# Or delete locally and remotely
git branch -D docs-artifacts/main
git push origin :docs-artifacts/main
```

The next `pages` run creates a fresh cache.

## Dataset cache

The notebooks also need their datasets. `docs/datasets.toml` lists them, and the `docs-datasets` component from
`jatic/aria/ci` restores each one from Harbor (`harbor.jatic.net/aria/cache`, one artifact per dataset) before the
notebooks run. A dataset that Harbor lacks is downloaded from its original source and stored there for the next build.
The restore is best effort: when it fails, the notebooks download their own datasets.

The docs jobs restore the datasets only when a notebook is about to execute, or on a clean build. A build that the
notebook cache above fully serves skips the restore.

## Local usage

### Fetch cache for local builds

```bash
# Fetch cache for current branch
./docs/fetch-docs-cache.sh

# Fetch cache for specific branch
./docs/fetch-docs-cache.sh main
```

### Push cache after local build

```bash
# Build docs first
nox -s docs

# Push cache for current branch
./docs/push-docs-cache.sh

# Push cache for specific branch
./docs/push-docs-cache.sh main
```

Pushing requires write access to the repository.

## Troubleshooting

### "Artifact branch does not exist yet"

This is normal for first-time builds. The cache will be created after the first successful docs build.

### "Cache directory not found"

The docs build failed or did not generate a cache. Check the build logs for errors.

### Permission denied when pushing

Ensure the CI has proper permissions:

- On GitHub, set `permissions: contents: write` in the workflow. The workflow does not push today.
- On GitLab, check that the `DATAEVAL_BUILD_PAT` variable is set and that its token has not expired and still has the
  `write_repository` scope.

### Large artifact branches

If artifact branches grow too large:

```bash
# Check size
git clone --single-branch --branch docs-artifacts/main <repo-url> cache-check
du -sh cache-check

# If too large, delete and recreate
git push origin --delete docs-artifacts/main
```

## Implementation details

### Why orphan branches?

- Isolated history: no commit ancestry, so the branches cannot be merged by accident
- Simple structure: just cache files, no code
- Garbage collection: the entire branch can be deleted without affecting main
- Atomic updates: each push is a complete snapshot

### Why force push?

Orphan branches are force-pushed because:

- They have no shared history to preserve
- Only the latest cache is needed, not historical versions
- Force pushing prevents branch bloat

### Security

- Scripts validate branch existence before operations
- Only writes to `docs-artifacts/*` branches
- Requires explicit CI permissions configuration
- Local usage requires repository write access
