#!/usr/bin/env bash
# Resolve a version from git that is both PEP 440 compliant (for hatch-vcs
# SETUPTOOLS_SCM_PRETEND_VERSION_FOR_*) and OCI-tag-safe (for `docker -t`).
#
# Emits two `export` statements on stdout; eval the output to set the vars:
#
#   eval "$(./docker/resolve-version.sh)"
#
# Resulting environment:
#   DATAEVAL_FLOW_VERSION    PEP 440 form, e.g. 0.1.0 or 0.1.1.dev5+gabc1234
#   DATAEVAL_FLOW_IMAGE_TAG  OCI-safe form, e.g. 0.1.0 or 0.1.1.dev5-gabc1234
#
# The PEP form uses `+local`; the OCI form swaps `+` for `-` (OCI tags
# disallow `+`). Both encode tag + commit distance + short sha + dirty flag.
# Requires the full tag history (CI: GIT_DEPTH=0 + `git fetch --tags`).

set -euo pipefail

raw=$(git describe --tags --long --dirty --match 'v*' --always 2>/dev/null || true)

if [[ "$raw" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)-([0-9]+)-g([0-9a-f]+)(-dirty)?$ ]]; then
  base="${BASH_REMATCH[1]}"
  dist="${BASH_REMATCH[2]}"
  sha="${BASH_REMATCH[3]}"
  dirty="${BASH_REMATCH[4]:-}"

  if [[ "$dist" == "0" && -z "$dirty" ]]; then
    pep="$base"
    tag="$base"
  else
    IFS=. read -r maj min pat <<< "$base"
    next="${maj}.${min}.$((pat + 1))"
    pep="${next}.dev${dist}+g${sha}${dirty:+.dirty}"
    tag="${next}.dev${dist}-g${sha}${dirty:+-dirty}"
  fi
else
  sha=$(git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
  pep="0.0.0.dev0+g${sha}"
  tag="0.0.0.dev0-g${sha}"
fi

printf 'export DATAEVAL_FLOW_VERSION=%q\n' "$pep"
printf 'export DATAEVAL_FLOW_IMAGE_TAG=%q\n' "$tag"
