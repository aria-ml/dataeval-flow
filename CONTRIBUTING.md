# Contributing

Thank you for your interest in DataEval Flow! Contributions, bug reports, and
suggestions for improvement are welcome.

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
 - OS / container variant (cpu / cu118 / cu124 / cu128):
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
