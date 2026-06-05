# DataEval Flow — Metarepo Groundwork & Verification Scaffolding

**Date:** 2026-05-26
**Status:** Approved for implementation
**Scope:** Lay down the program-governance artifacts and automated-verification harness for
`dataeval-flow`, mirroring the structure already in place for `dataeval`. No self-assessment
content yet — only the groundwork that future assessments will populate.

## 1. Goal

Establish, for `dataeval-flow`, the same governance-and-verification footprint that `dataeval`
has today:

- A product folder in the metarepo (`metarepo/DataEval-Flow/`) containing the README, FR/NFR
  requirement documents, test-case files, and the Verification Cross-Reference Matrix (VCRM).
- A `verification/` directory in `dataeval-flow` containing pytest-based functional and
  non-functional tests, a `registry.yaml` describing how those tests map to requirements and
  test cases, and a `generate_metarepo.py` generator that turns automated test results into
  the metarepo artifacts.
- A synthetic-data harness so workflow verification tests can run end-to-end without external
  datasets, GPUs, or model downloads.

No assessment entries, no product reviews — those are added later by the program.

## 2. Functional Requirements

| ID    | Name                                          |
|-------|-----------------------------------------------|
| FR-1  | Installation & Cross-Environment Compatibility |
| FR-2  | Configuration Management                       |
| FR-3  | Dataset Loading                                |
| FR-4  | Feature Extraction Configuration               |
| FR-5  | Preprocessing & Selection Pipeline             |
| FR-6  | Workflow Orchestration                         |
| FR-7  | Data Cleaning Workflow                         |
| FR-8  | Drift Monitoring Workflow                      |
| FR-9  | Out-of-Distribution Detection Workflow         |
| FR-10 | Data Analysis Workflow                         |
| FR-11 | Splitting Workflow                             |
| FR-12 | Prioritization Workflow                        |
| FR-13 | Parameter Sweep Workflow                       |
| FR-14 | Reporting & Export                             |
| FR-15 | Caching                                        |
| FR-16 | CLI Interface (headless, TUI app, config builder) |
| FR-17 | Docker Containerization                        |
| FR-18 | Result Metadata Envelope                       |

All FRs use the `DR-1.5` origin (program-doc requirement) **except FR-18**, which uses
`IR-3-H-12` (model and data interoperability — hard requirement).

The seven registered workflows are: `data-analysis`, `data-cleaning`, `drift-monitoring`,
`ood-detection`, `parameter-sweep`, `data-prioritization`, `data-splitting`.

## 3. Non-Functional Requirements

| ID    | Name                              |
|-------|-----------------------------------|
| NFR-1 | Python Version Compatibility       |
| NFR-2 | Packaging & Distribution           |
| NFR-3 | Type Safety                        |
| NFR-4 | Configuration Reproducibility      |
| NFR-5 | Logging Integration                |
| NFR-6 | Test Coverage                      |

All NFRs use the `DR-1.5` origin.

## 4. Metarepo Product Folder

Create `metarepo/DataEval-Flow/` with this layout:

```
metarepo/DataEval-Flow/
├── README.md                          # tagline, needs, maturity = Level-0 Candidate (N/A assessment)
├── vcrm.md                            # generated; rows marked Pending until first verification run
├── requirements/
│   ├── FR-1-installation-compatibility.md
│   ├── FR-2-configuration-management.md
│   ├── FR-3-dataset-loading.md
│   ├── FR-4-feature-extraction.md
│   ├── FR-5-preprocessing-selection.md
│   ├── FR-6-workflow-orchestration.md
│   ├── FR-7-cleaning-workflow.md
│   ├── FR-8-drift-workflow.md
│   ├── FR-9-ood-workflow.md
│   ├── FR-10-coverage-workflows.md
│   ├── FR-11-splitting-workflow.md
│   ├── FR-12-prioritization-workflow.md
│   ├── FR-13-parameter-sweep-workflow.md
│   ├── FR-14-reporting-export.md
│   ├── FR-15-caching.md
│   ├── FR-16-cli-interface.md
│   ├── FR-17-docker-containerization.md
│   ├── NFR-1-python-compatibility.md
│   ├── NFR-2-packaging-distribution.md
│   ├── NFR-3-type-safety.md
│   ├── NFR-4-configuration-reproducibility.md
│   ├── NFR-5-logging-integration.md
│   └── NFR-6-test-coverage.md
├── test-cases/                        # generated stubs; populated after first verification run
├── assessments/                       # empty (no self-assessments yet)
└── product_reviews/                   # empty
```

The product README mirrors `metarepo/DataEval/README.md` with `dataeval-flow`-specific values:

- Tagline: workflow orchestration for DataEval evaluators.
- Maturity Level: **Level-0 Candidate**, Date of Last Program Assessment: **N/A**.
- Public Presence: `https://gitlab.jatic.net/jatic/aria/dataeval-flow` (project id 482).
- Branch / release strategy: same as DataEval (GitLab Flow with Release Branches, SemVer).

## 5. dataeval-flow `verification/` directory

```
dataeval-flow/
└── verification/
    ├── conftest.py                    # pytest hooks that emit verification_report.json
    ├── helpers.py                     # shared helpers (paths, run-pytest wrapper, etc.)
    ├── fixtures.py                    # SYNTHETIC DATA — deterministic mock dataset generators
    ├── registry.yaml                  # FR/NFR → test_case → pytest nodeid mapping
    ├── generate_metarepo.py           # registry + report → metarepo markdown
    ├── functional/
    │   ├── install/test_pip_install.py
    │   ├── contracts/test_public_api.py
    │   ├── config/test_config_load.py
    │   ├── dataset/test_dataset_configs.py
    │   ├── extractors/test_extractor_configs.py
    │   ├── pipeline/test_preprocessing_selection.py
    │   ├── orchestration/test_run_tasks.py
    │   ├── workflows/
    │   │   ├── test_analysis.py
    │   │   ├── test_cleaning.py
    │   │   ├── test_drift.py
    │   │   ├── test_ood.py
    │   │   ├── test_splitting.py
    │   │   ├── test_prioritization.py
    │   │   └── test_parameter_sweep.py
    │   ├── reporting/test_text_report.py
    │   ├── metadata/test_metadata_envelope.py
    │   ├── cache/test_cache.py
    │   ├── cli/test_cli_entrypoints.py
    │   └── docker/test_dockerfile_present.py
    └── nonfunctional/
        ├── compatibility/test_python_versions.py
        ├── packaging/test_version_metadata.py
        ├── test_type_safety.py
        ├── test_config_reproducibility.py
        └── test_logging.py
```

### 5.1 Synthetic data fixtures (`verification/fixtures.py`)

A small, deterministic, mathematically valid dataset is sufficient for workflow verification
because the goal is to confirm that workflows execute end-to-end and produce structurally
correct outputs, not to validate statistical conclusions on real data.

The fixture module exposes:

- `make_synthetic_images(n: int = 64, shape: tuple = (3, 8, 8), seed: int = 0) -> np.ndarray`
  — fixed-seed `uint8` image batch.
- `make_synthetic_labels(n: int, n_classes: int = 3, seed: int = 0) -> np.ndarray` — class
  labels in `[0, n_classes)`.
- `make_synthetic_metadata(n: int, seed: int = 0) -> dict` — factor names + factor data
  (`continuous_a`, `categorical_b`) suitable for the bias and drift workflows.
- `make_synthetic_dataset(n: int = 64, seed: int = 0)` — a MAITE-compatible image dataset
  wrapper that returns `(image, target, metadata)` triples.
- `make_synthetic_embeddings(n: int = 64, dim: int = 32, seed: int = 0) -> np.ndarray` —
  reference/test arrays for drift and OOD tests.
- `make_pipeline_config(tmp_path: Path, workflow: str, **overrides) -> Path` — writes a
  minimal `params.yaml` referencing the synthetic dataset.

Reproducibility comes from `numpy.random.default_rng(seed)`. No PIL, no torchvision
downloads, no Hugging Face hub access. Tests that require GPU-only code paths (e.g.
`Dockerfile.cu118` execution) remain `pytest.mark.skip`-marked.

### 5.2 Mapping FR/NFR → test cases

The test-case identifiers follow `dataeval`'s convention (`<requirement-number>-<sequence>`),
and some test cases cover multiple requirements (e.g. TC-1-1 covers both FR-1 and NFR-1).

| Test Case | Covers          | Module |
|-----------|-----------------|--------|
| TC-1-1    | FR-1, NFR-1     | nonfunctional/compatibility/test_python_versions.py + functional/install/test_pip_install.py |
| TC-1-2    | FR-1            | functional/install/test_pip_install.py (extras resolution) |
| TC-1-7    | FR-1, NFR-2     | nonfunctional/packaging/test_version_metadata.py |
| TC-2-1    | FR-2            | functional/config/test_config_load.py |
| TC-3-1    | FR-3            | functional/dataset/test_dataset_configs.py |
| TC-4-1    | FR-4            | functional/extractors/test_extractor_configs.py |
| TC-5-1    | FR-5            | functional/pipeline/test_preprocessing_selection.py |
| TC-6-1    | FR-6            | functional/orchestration/test_run_tasks.py |
| TC-7-1    | FR-7            | functional/workflows/test_cleaning.py |
| TC-8-1    | FR-8            | functional/workflows/test_drift.py |
| TC-9-1    | FR-9            | functional/workflows/test_ood.py |
| TC-10-1   | FR-10, NFR-3    | functional/workflows/test_analysis.py + nonfunctional/test_type_safety.py |
| TC-11-1   | FR-11, NFR-4    | functional/workflows/test_splitting.py + nonfunctional/test_config_reproducibility.py |
| TC-12-1   | FR-12           | functional/workflows/test_prioritization.py |
| TC-13-1   | FR-13           | functional/workflows/test_parameter_sweep.py |
| TC-14-1   | FR-14           | functional/reporting/test_text_report.py |
| TC-15-1   | FR-15           | functional/cache/test_cache.py |
| TC-16-1   | FR-16, NFR-5    | functional/cli/test_cli_entrypoints.py + nonfunctional/test_logging.py |
| TC-17-1   | FR-17           | functional/docker/test_dockerfile_present.py |
| TC-18-1   | FR-18           | functional/metadata/test_metadata_envelope.py |

All test-case ids use the `N-M` integer format so the generator's sort key
(`[int(p) for p in tc_id.split("-")]`) keeps working. NFR-5 (logging) overlaps with
TC-16-1 because the CLI entrypoint is what configures stdlib `logging`; running the CLI
exercises both. FR-18 (Result Metadata Envelope) owns TC-18-1.

NFR-6 (test coverage) is **not** assigned a pytest-driven test case. Instead, the generator
reads `output/coverage.xml` after `nox -s test` and records the NFR-6 row in the VCRM as
`Pass` if total coverage ≥ 90% (the threshold already enforced by `pyproject.toml`'s
`fail_under = 90`), `Fail` otherwise, or `Pending` when no coverage report is present.

### 5.3 `registry.yaml`

Same shape as `dataeval/verification/registry.yaml`:

```yaml
metarepo:
  project_id: 409                 # metarepo GitLab project

requirements:
  FR-1:
    name: Installation & Cross-Environment Compatibility
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/.../#dr-15-product-requirements-definitions
    file: FR-1-installation-compatibility.md
    test_cases: ["1-1", "1-2", "1-7"]
  ...

test_cases:
  "1-1":
    name: Python Version Compatibility
    test_type: Core Functionality
    business_case: >-
      Verify that dataeval-flow installs and functions correctly across all
      supported Python versions (3.10–3.13).
    initial_conditions:
      - Clean Python environment for the target version
      - Network access for pip/uv package installation
    expected_results:
      - Package imports without errors on the current Python version
      - Version string is set and valid (not "unknown")
      - Public symbols listed in __all__ are importable
      - py.typed marker is present
  ...
```

### 5.4 `generate_metarepo.py`

Copied from `dataeval/verification/generate_metarepo.py` with the project title changed
from "DataEval" to "DataEval Flow". Output directory: `dataeval-flow/output/metarepo/`.
Generator behaviour is unchanged: when a verification report is present, fill in P/F/S and
"Pass/Fail"; when absent, emit Pending stubs.

### 5.5 `conftest.py`

Mirrors `dataeval/verification/conftest.py` — collects pytest results into
`output/verification_report.json` keyed by test case (the test case id is derived from a
`@pytest.mark.test_case("N-M")` marker on each verification test).

## 6. Initial state after groundwork is laid

- `metarepo/DataEval-Flow/requirements/*.md` — fully written (this is hand-authored content).
- `metarepo/DataEval-Flow/test-cases/*.md` — generated as Pending stubs.
- `metarepo/DataEval-Flow/vcrm.md` — all "Verification" cells = Pending.
- `dataeval-flow/verification/` — pytest tree exists and passes locally for everything that
  doesn't require GPU or a real container build.
- After the user runs `pytest verification/ --json-report` once, the generator overwrites the
  test-case stubs and the VCRM with real Pass/Fail values.

## 7. Out of scope

- Self-assessments (`metarepo/DataEval-Flow/assessments/2026-MM-DD/`).
- Product reviews (`metarepo/DataEval-Flow/product_reviews/`).
- Guidelines exceptions (`guidelines_exceptions.yaml`).
- Updating `metarepo/README.md`'s Products list — that is a one-line addition handled as part
  of the implementation.
- Wiring `nox -s verification` — can be added later; for now verification runs are explicit
  `pytest verification/` invocations.

## 8. Verification of this groundwork

The groundwork itself is verified by:

1. `pytest verification/` runs to completion on a CPU-only development environment with no
   external data access; only GPU/container-only tests are skip-marked.
2. `python verification/generate_metarepo.py` produces files under
   `dataeval-flow/output/metarepo/` whose structure matches `metarepo/DataEval-Flow/`.
3. The generated `vcrm.md` references every FR and NFR with at least one X-marked test case.
