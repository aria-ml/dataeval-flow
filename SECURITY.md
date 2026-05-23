# Security Policy

This document describes how to report vulnerabilities in `dataeval-flow`
and how the maintainers triage findings from automated scanners. It is
the project's response to JATIC SDP requirement [DSOR-1-H-3](JATIC_SDP_COMPLIANCE.md)
(false positives must be dismissed with a documented justification).

## Reporting a vulnerability

Please do **not** open a public GitLab or GitHub issue for security
vulnerabilities. Email <dataeval-flow@ariacoustics.com> with:

- A description of the issue and its impact
- Steps to reproduce (a minimal config / dataset / command line is ideal)
- Affected versions / commits / container tags
- Your name and (optionally) a credit preference for the eventual advisory

You can expect:

- An acknowledgement within **3 business days**
- A triage decision (accepted / false positive / duplicate) within **10 business days**
- A fix or mitigation plan within **30 business days** for accepted Hard
  findings, tracked against the next `v0.x.y` release

For non-security bugs, follow the regular process in [CONTRIBUTING.md](CONTRIBUTING.md).

## Supported versions

Only the latest minor release on the `main` line receives security
backports today. Once `dataeval-flow` reaches JATIC Maturity I
(target v1.0.0 — see [ROADMAP.md](ROADMAP.md)), supported-version
windows will widen per JATIC SDP `RS-5-S-1`.

| Version | Status | Security backports |
|---|---|---|
| `0.1.x` (current alpha) | active | yes |
| anything older | unsupported | no |

## Automated scanner coverage

Every merge request and every `main` pipeline runs four security scans
(see [.gitlab-ci.yml](.gitlab-ci.yml)):

| Scanner | Scope | SDP requirement | Suppression file / mechanism |
|---|---|---|---|
| Semgrep (SAST) | source code in `src/` | DSOR-1-H-1 / DSOR-1-H-2 | `SAST_EXCLUDED_PATHS` + per-rule `nosem:` comment |
| Gemnasium | Python dependency manifest | DSOR-2-H-1 / DSOR-2-H-2 | GitLab vulnerability-report dismissal (with comment) |
| Trivy | published container images | DSOR-3-H-1 / DSOR-3-H-2 / CS-2-H-2 | `.trivyignore` (one line per CVE + `Justification:` header) |
| GitLab Secret Detection | working tree | DSOR-4-H-1 / DSOR-4-H-2 | exclusions in `secret_detection` job; do not commit secrets |

Findings flow into the GitLab security dashboard. Cosign attestation
(`cosign attest --type cyclonedx`) is also published per image for
downstream verification (CS-2-H-4).

## Dismissing a false positive

The same workflow applies whether the finding came from SAST, dependency
scanning, container scanning, or secret detection.

1. **Verify it is a false positive.** Read the rule / CVE description,
   inspect the code or dependency, and confirm the finding does not apply
   to how `dataeval-flow` uses the affected component. If you are unsure,
   treat it as a true positive and fix.

2. **Get a second opinion.** Security suppressions land via a normal MR
   reviewed by a maintainer. The MR description must link to the finding
   (Semgrep rule ID, CVE, secret-rule ID) and to any upstream advisory or
   discussion that supports the dismissal.

3. **Record the suppression in the right place.** The justification text
   must be specific to this project — generic statements like "not
   exploitable in our case" without context will be rejected in review.

   - **Semgrep / SAST**
     - Per-line: add `# nosem: <rule-id>  # Justification: <reason>` on the
       triggering line.
     - Per-file or per-directory: add the path to `SAST_EXCLUDED_PATHS` in
       [.gitlab-ci.yml](.gitlab-ci.yml) with an inline `# <reason>` comment
       above the change.
     - Per-rule globally: extend `SAST_EXCLUDED_ANALYZERS` only when a
       whole analyzer is out of scope (currently we exclude the JS/Go/PHP
       analyzers because we ship only Python).

   - **Gemnasium / dependency scan**
     - Preferred fix is to bump the offending package in
       `[tool.uv].constraint-dependencies` (a constraint, not a runtime
       dep — see existing CVE pins in [pyproject.toml](pyproject.toml)).
     - When a fix is not yet available upstream, dismiss the finding in
       the GitLab vulnerability report with a comment containing:
       1. CVE ID
       2. Why the project is not exploitable (e.g. the affected API is
          never called; the vulnerable code path requires untrusted input
          that does not reach it)
       3. A re-evaluation date (typically 90 days out)

   - **Trivy / container scan**
     - Add one entry per CVE to [.trivyignore](.trivyignore) using this
       format:

       ```text
       # Justification: <one-line reason>
       # Upstream: <link or "no fix yet">
       # Re-evaluate: YYYY-MM-DD
       CVE-YYYY-NNNNN
       ```

     - Entries without all three header lines will be rejected at review.

   - **Secret Detection**
     - True positives must be rotated immediately (the secret is already
       in git history; assume it is compromised) and the commit purged
       with `git filter-repo` if it was added in the same MR.
     - False positives (test fixtures, public keys, example tokens) belong
       in `tests/fixtures/` and the path added to the secret-detection
       `paths_ignored` configuration with an inline justification.

4. **Set an expiry.** All non-CVE-fixable dismissals carry a re-evaluation
   date (90 days for HIGH/CRITICAL, 180 days for MEDIUM). When the date
   passes, the suppression must be removed or re-justified — there are no
   indefinite dismissals.

## Audit and review

The set of active suppressions is reviewed at every release cut:

- The release-prep checklist (see [BRANCHING.md](BRANCHING.md)) requires
  walking `.trivyignore`, the GitLab vulnerability-report dismissal
  history, and any in-line `nosem:` comments to confirm each justification
  is still valid.
- Suppressions whose re-evaluation date has passed without renewal block
  the release.
- The aggregate count of active suppressions is reported in the release
  notes so it is visible to downstream consumers.

## Hardcoded-secret policy

The project ships no hardcoded secrets (DSOR-4-H-2). The Cosign private
key used to sign container images is supplied via the
`COSIGN_PRIVATE_KEY_B64` CI variable; only the corresponding public key
is committed (see [docker/cosign.pub](docker/cosign.pub)). If you discover
what looks like a credential, password, token, or private key in this
repository — even in a test fixture — please report it via the channel
above; do not file a public issue.
