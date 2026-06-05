# DataEval Flow Metarepo Groundwork Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lay down `metarepo/DataEval-Flow/` (program governance artifacts) and `dataeval-flow/verification/` (pytest-based automated verification harness with synthetic data fixtures) so that the first verification run produces a complete VCRM and a set of populated test-case markdown files.

**Architecture:** Mirror the established `dataeval` pattern verbatim where possible. The verification harness is a pytest tree under `dataeval-flow/verification/` with a `registry.yaml` mapping FR/NFR → test-case → pytest nodeid, and a `generate_metarepo.py` that consumes pytest's JSON report to produce `metarepo/DataEval-Flow/test-cases/*.md` and `vcrm.md`. Synthetic data fixtures (deterministic numpy/PIL arrays + a MAITE-compatible dataset wrapper) keep verification CPU-only and offline.

**Tech Stack:** Python 3.10–3.13, pytest, pyyaml, numpy, pydantic, `dataeval==1.0.4`, `dataeval-flow` (this repo), MAITE protocol.

**Reference files in sibling repo:** `/home/aweng/2033/dataeval/verification/` contains the canonical conftest.py / helpers.py / registry.yaml / generate_metarepo.py. Tasks below copy and adapt them.

---

## Repo Layout After This Plan

```
/home/aweng/2033/metarepo/DataEval-Flow/
├── README.md
├── vcrm.md                            (generated)
├── requirements/                      (24 hand-written .md files: FR-1..18 + NFR-1..6)
├── test-cases/                        (generated: test-case-1-1.md … test-case-18-1.md)
├── assessments/                       (empty)
└── product_reviews/                   (empty)

/home/aweng/2033/dataeval-flow/verification/
├── __init__.py                        (empty marker — keeps imports clean)
├── conftest.py
├── helpers.py
├── fixtures.py                        (synthetic data generators)
├── registry.yaml
├── generate_metarepo.py
├── functional/
│   ├── __init__.py
│   ├── install/__init__.py + test_pip_install.py
│   ├── contracts/__init__.py + test_public_api.py
│   ├── config/__init__.py + test_config_load.py
│   ├── dataset/__init__.py + test_dataset_configs.py
│   ├── extractors/__init__.py + test_extractor_configs.py
│   ├── pipeline/__init__.py + test_preprocessing_selection.py
│   ├── orchestration/__init__.py + test_run_tasks.py
│   ├── workflows/__init__.py + test_analysis.py, test_cleaning.py, test_drift.py,
│   │   test_ood.py, test_splitting.py, test_prioritization.py, test_parameter_sweep.py
│   ├── reporting/__init__.py + test_text_report.py
│   ├── metadata/__init__.py + test_metadata_envelope.py
│   ├── cache/__init__.py + test_cache.py
│   ├── cli/__init__.py + test_cli_entrypoints.py
│   └── docker/__init__.py + test_dockerfile_present.py
└── nonfunctional/
    ├── __init__.py
    ├── compatibility/__init__.py + test_python_versions.py
    ├── packaging/__init__.py + test_version_metadata.py
    ├── test_type_safety.py
    ├── test_config_reproducibility.py
    └── test_logging.py
```

---

## Task 1: Scaffolding — directories, conftest, helpers, fixtures

**Files:**
- Create: `dataeval-flow/verification/__init__.py` (empty)
- Create: `dataeval-flow/verification/conftest.py`
- Create: `dataeval-flow/verification/helpers.py`
- Create: `dataeval-flow/verification/fixtures.py`
- Create: `dataeval-flow/verification/functional/__init__.py` (empty)
- Create: `dataeval-flow/verification/nonfunctional/__init__.py` (empty)

- [ ] **Step 1.1: Create the verification tree and empty `__init__.py` files**

```bash
cd /home/aweng/2033/dataeval-flow
mkdir -p verification/{functional,nonfunctional}
mkdir -p verification/functional/{install,contracts,config,dataset,extractors,pipeline,orchestration,workflows,reporting,metadata,cache,cli,docker}
mkdir -p verification/nonfunctional/{compatibility,packaging}
touch verification/__init__.py verification/functional/__init__.py verification/nonfunctional/__init__.py
for d in verification/functional/{install,contracts,config,dataset,extractors,pipeline,orchestration,workflows,reporting,metadata,cache,cli,docker} verification/nonfunctional/{compatibility,packaging}; do
  touch "$d/__init__.py"
done
```

- [ ] **Step 1.2: Write `verification/conftest.py`**

Copy verbatim from `/home/aweng/2033/dataeval/verification/conftest.py` to `/home/aweng/2033/dataeval-flow/verification/conftest.py` — no edits required, the file references only its own `Path(__file__).parent`.

```python
"""Verification test configuration and report generation plugin.

Provides:
- ``test_case(*ids)`` marker linking tests to ``test-case-<id>.md`` in the meta repo
- JSON report generation mapping test case numbers to pass/fail results
- Terminal summary of verification results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

VERIFICATION_DIR = Path(__file__).parent
OUTPUT_DIR = VERIFICATION_DIR.parent / "output"

_PROJECT_ROOT = str(VERIFICATION_DIR.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "test_case(*ids): link test to one or more test-case-<id>.md files in the meta repo",
    )


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    reports = getattr(item, "_verification_reports", {})
    reports[call.when] = report
    item._verification_reports = reports


def _get_test_status(item):
    reports = getattr(item, "_verification_reports", {})
    for phase in ("setup", "teardown"):
        r = reports.get(phase)
        if r is not None and r.failed:
            return "error"
    call = reports.get("call")
    if call is not None:
        if call.passed:
            return "passed"
        if call.skipped:
            return "skipped"
        return "failed"
    setup = reports.get("setup")
    if setup is not None and setup.skipped:
        return "skipped"
    return "error"


def _tc_status(tests: list[dict]) -> str:
    statuses = {t["status"] for t in tests}
    if statuses & {"failed", "error"}:
        return "failed"
    if statuses == {"skipped"}:
        return "skipped"
    return "passed"


def pytest_sessionfinish(session, exitstatus):
    results: dict[str, list[dict]] = {}
    for item in session.items:
        status = _get_test_status(item)
        for marker in item.iter_markers("test_case"):
            for tc_num in marker.args:
                tc_id = f"test-case-{tc_num}"
                results.setdefault(tc_id, []).append(
                    {
                        "test": item.nodeid,
                        "file": str(Path(item.fspath).relative_to(VERIFICATION_DIR)),
                        "status": status,
                    },
                )

    if not results:
        return

    tc_statuses = {tc_id: _tc_status(tests) for tc_id, tests in results.items()}
    passed = sum(1 for s in tc_statuses.values() if s == "passed")
    failed = sum(1 for s in tc_statuses.values() if s == "failed")
    skipped = sum(1 for s in tc_statuses.values() if s == "skipped")

    report = {
        "summary": {
            "total_test_cases": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "test_cases": {
            tc_id: {
                "meta_repo_file": f"test-cases/{tc_id}.md",
                "status": _tc_status(tests),
                "tests": tests,
            }
            for tc_id, tests in sorted(results.items())
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    report_path = OUTPUT_DIR / "verification_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    summary = report["summary"]
    terminalreporter.section("Verification Report")
    terminalreporter.write_line(
        f"Test Cases: {summary['total_test_cases']} total, "
        f"{summary['passed']} passed, "
        f"{summary['failed']} failed, "
        f"{summary['skipped']} skipped",
    )
    terminalreporter.write_line(f"Report: {report_path}")
    for tc_id, tc_data in report["test_cases"].items():
        if tc_data["status"] == "failed":
            terminalreporter.write_line(f"  FAILED: {tc_id} ({tc_data['meta_repo_file']})")
            for test in tc_data["tests"]:
                if test["status"] in ("failed", "error"):
                    terminalreporter.write_line(f"    - {test['test']}")
```

- [ ] **Step 1.3: Write `verification/helpers.py`**

```python
"""Shared helpers for verification tests (paths, importlib utilities)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def safe_import(module_name: str) -> Any:
    """Import a module by name and return it, or None if unavailable."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None
```

- [ ] **Step 1.4: Write `verification/fixtures.py` — synthetic data generators**

```python
"""Deterministic synthetic-data fixtures for workflow verification.

These produce mathematically valid but meaningless data so workflow
verification tests can run end-to-end without external datasets, GPUs,
or model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SyntheticMetadata:
    """Minimal Metadata-protocol implementation for verification tests."""

    class_labels: NDArray[np.intp]
    factor_data: NDArray[np.int64]
    factor_names: list[str]
    is_discrete: list[bool]
    index2label: dict[int, str] = field(default_factory=dict)


@dataclass
class SyntheticDataset:
    """MAITE-compatible image dataset returning (image, target, metadata) triples."""

    images: NDArray[np.uint8]
    labels: NDArray[np.intp]
    _id: str = "verification-synthetic"

    @property
    def metadata(self) -> dict[str, Any]:
        return {"id": self._id}

    def __getitem__(self, idx: int) -> tuple[NDArray[np.uint8], int, dict[str, Any]]:
        return self.images[idx], int(self.labels[idx]), {}

    def __len__(self) -> int:
        return len(self.images)


def make_synthetic_images(
    n: int = 64,
    shape: tuple[int, int, int] = (3, 8, 8),
    seed: int = 0,
) -> NDArray[np.uint8]:
    """Return a deterministic uint8 image batch of shape (n, *shape)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, *shape), dtype=np.uint8)


def make_synthetic_labels(n: int, n_classes: int = 3, seed: int = 0) -> NDArray[np.intp]:
    """Return a deterministic label vector in [0, n_classes)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n, dtype=np.intp)


def make_synthetic_metadata(n: int, n_factors: int = 3, n_classes: int = 3, seed: int = 0) -> SyntheticMetadata:
    rng = np.random.default_rng(seed)
    class_labels = rng.integers(0, n_classes, size=n, dtype=np.intp)
    factor_data = rng.integers(0, 4, size=(n, n_factors), dtype=np.int64)
    factor_names = [f"factor_{i}" for i in range(n_factors)]
    is_discrete = [True] * n_factors
    index2label = {i: f"class_{i}" for i in range(n_classes)}
    return SyntheticMetadata(
        class_labels=class_labels,
        factor_data=factor_data,
        factor_names=factor_names,
        is_discrete=is_discrete,
        index2label=index2label,
    )


def make_synthetic_dataset(
    n: int = 64,
    n_classes: int = 3,
    shape: tuple[int, int, int] = (3, 8, 8),
    seed: int = 0,
) -> SyntheticDataset:
    """Return a MAITE-compatible dataset with deterministic content."""
    images = make_synthetic_images(n=n, shape=shape, seed=seed)
    labels = make_synthetic_labels(n=n, n_classes=n_classes, seed=seed + 1)
    return SyntheticDataset(images=images, labels=labels)


def make_synthetic_embeddings(n: int = 64, dim: int = 32, seed: int = 0) -> NDArray[np.float32]:
    """Return a deterministic float32 embedding matrix of shape (n, dim)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def write_image_folder(root: Path, n_per_class: int = 4, n_classes: int = 2, seed: int = 0) -> Path:
    """Write a tiny ImageFolder-style dataset to disk and return its root."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    for c in range(n_classes):
        class_dir = root / f"class_{c}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"img_{i}.png")
    return root
```

- [ ] **Step 1.5: Add `verification/` to pytest discovery in `pyproject.toml`**

Edit `/home/aweng/2033/dataeval-flow/pyproject.toml`. Find the `[tool.pytest.ini_options]` block (around line 186) and change `testpaths = ["tests"]` to `testpaths = ["tests", "verification"]`.

- [ ] **Step 1.6: Verify pytest collects the verification tree (no tests yet, just collection)**

```bash
cd /home/aweng/2033/dataeval-flow
uv run pytest verification/ --collect-only -q
```
Expected: zero tests collected, no collection errors.

- [ ] **Step 1.7: Commit**

```bash
cd /home/aweng/2033/dataeval-flow
git add verification/ pyproject.toml
git commit -m "feat(verification): scaffold verification harness — conftest, helpers, fixtures"
```

---

## Task 2: `registry.yaml` and `generate_metarepo.py`

**Files:**
- Create: `dataeval-flow/verification/registry.yaml`
- Create: `dataeval-flow/verification/generate_metarepo.py`

- [ ] **Step 2.1: Write `verification/registry.yaml`**

```yaml
# Verification Registry for dataeval-flow
#
# Static metadata for generating meta repo artifacts (test case markdown files
# and VCRM) from automated verification test results.

metarepo:
  project_id: 409

# ---------------------------------------------------------------------------
# Requirements — rows in the VCRM
# ---------------------------------------------------------------------------
requirements:
  FR-1:
    name: Installation & Cross-Environment Compatibility
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-1-installation-compatibility.md
    test_cases: ["1-1", "1-2", "1-7"]

  FR-2:
    name: Configuration Management
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-2-configuration-management.md
    test_cases: ["2-1"]

  FR-3:
    name: Dataset Loading
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-3-dataset-loading.md
    test_cases: ["3-1"]

  FR-4:
    name: Feature Extraction Configuration
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-4-feature-extraction.md
    test_cases: ["4-1"]

  FR-5:
    name: Preprocessing & Selection Pipeline
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-5-preprocessing-selection.md
    test_cases: ["5-1"]

  FR-6:
    name: Workflow Orchestration
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-6-workflow-orchestration.md
    test_cases: ["6-1"]

  FR-7:
    name: Data Cleaning Workflow
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-7-cleaning-workflow.md
    test_cases: ["7-1"]

  FR-8:
    name: Drift Monitoring Workflow
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-8-drift-workflow.md
    test_cases: ["8-1"]

  FR-9:
    name: OOD Detection Workflow
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-9-ood-workflow.md
    test_cases: ["9-1"]

  FR-10:
    name: Data Analysis Workflow
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-10-analysis-workflow.md
    test_cases: ["10-1"]

  FR-11:
    name: Splitting Workflow
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-11-splitting-workflow.md
    test_cases: ["11-1"]

  FR-12:
    name: Prioritization Workflow
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-12-prioritization-workflow.md
    test_cases: ["12-1"]

  FR-13:
    name: Parameter Sweep Workflow
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-13-parameter-sweep-workflow.md
    test_cases: ["13-1"]

  FR-14:
    name: Reporting & Export
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-14-reporting-export.md
    test_cases: ["14-1"]

  FR-15:
    name: Caching
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-15-caching.md
    test_cases: ["15-1"]

  FR-16:
    name: CLI Interface
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-16-cli-interface.md
    test_cases: ["16-1"]

  FR-17:
    name: Docker Containerization
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: FR-17-docker-containerization.md
    test_cases: ["17-1"]

  FR-18:
    name: Result Metadata Envelope
    origin: IR-3-H-12
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/software/interoperability/#ir-3-model-and-data-interoperability
    file: FR-18-result-metadata-envelope.md
    test_cases: ["18-1"]

  NFR-1:
    name: Python Version Compatibility
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: NFR-1-python-compatibility.md
    test_cases: ["1-1"]

  NFR-2:
    name: Packaging & Distribution
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: NFR-2-packaging-distribution.md
    test_cases: ["1-7"]

  NFR-3:
    name: Type Safety
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: NFR-3-type-safety.md
    test_cases: ["10-1"]

  NFR-4:
    name: Configuration Reproducibility
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: NFR-4-configuration-reproducibility.md
    test_cases: ["11-1"]

  NFR-5:
    name: Logging Integration
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: NFR-5-logging-integration.md
    test_cases: ["16-1"]

  NFR-6:
    name: Test Coverage
    origin: DR-1.5
    origin_link: https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions
    file: NFR-6-test-coverage.md
    test_cases: []

# ---------------------------------------------------------------------------
# Test Cases — columns in the VCRM, individual markdown files
# ---------------------------------------------------------------------------
test_cases:
  "1-1":
    name: Python Version & Public API Smoke
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
      - All public symbols listed in __all__ are importable
      - py.typed marker is present

  "1-2":
    name: Optional Extras Resolution
    test_type: Core Functionality
    business_case: >-
      Verify that optional dependency extras (cpu, onnx, opencv, app)
      install without conflicts and enable corresponding features.
    initial_conditions:
      - Clean Python environment with dataeval-flow installed
    expected_results:
      - torch / torchvision importable when cpu/cu extras are installed
      - onnxruntime importable when onnx extra is installed
      - opencv (cv2) importable when opencv extra is installed
      - textual importable when app extra is installed

  "1-7":
    name: Package Metadata
    test_type: Core Functionality
    business_case: >-
      Verify that dataeval-flow's installed wheel exposes correct metadata
      and that no test files leak into the distributed package.
    initial_conditions:
      - dataeval-flow installed via pip/uv
    expected_results:
      - Package name is "dataeval-flow"
      - Requires-Python matches >=3.10,<3.14
      - License is set to MIT
      - py.typed marker is present
      - No verification/ or tests/ directories in the installed package

  "2-1":
    name: Configuration Loading
    test_type: Core Functionality
    business_case: >-
      Verify that YAML/JSON configs load via load_config and load_config_folder
      and produce validated PipelineConfig instances.
    initial_conditions:
      - dataeval-flow installed
      - Temporary directory with a minimal params.yaml
    expected_results:
      - load_config parses a single YAML file into PipelineConfig
      - load_config parses a single JSON file into PipelineConfig
      - load_config_folder merges multiple YAML/JSON files at folder root
      - Invalid configs raise pydantic ValidationError

  "3-1":
    name: Dataset Loading
    test_type: Core Functionality
    business_case: >-
      Verify that the dataset config classes (HuggingFace, ImageFolder, COCO,
      YOLO, Protocol) round-trip through pydantic and that load_dataset
      resolves a local ImageFolder.
    initial_conditions:
      - dataeval-flow installed
      - Synthetic ImageFolder written to tmp_path
    expected_results:
      - ImageFolderDatasetConfig validates and round-trips
      - HuggingFaceDatasetConfig validates with placeholder repo id
      - CocoDatasetConfig validates with placeholder paths
      - YoloDatasetConfig validates with placeholder paths
      - DatasetProtocolConfig validates with a dotted-path target
      - load_dataset returns an AnnotatedDataset for the synthetic ImageFolder

  "4-1":
    name: Extractor Configuration
    test_type: Core Functionality
    business_case: >-
      Verify that all extractor config classes are constructible, validated
      by pydantic, and discoverable from the public API.
    initial_conditions:
      - dataeval-flow installed
    expected_results:
      - FlattenExtractorConfig constructs and round-trips
      - TorchExtractorConfig constructs with a dotted-path target
      - OnnxExtractorConfig constructs with a model path
      - BoVWExtractorConfig constructs with default params
      - UncertaintyExtractorConfig constructs with default params
      - All extractor configs are exported from dataeval_flow.__all__

  "5-1":
    name: Preprocessing & Selection Pipeline
    test_type: Core Functionality
    business_case: >-
      Verify that PreprocessorConfig, SelectionConfig, and SelectionStep
      compose into a valid pipeline that runs against synthetic data.
    initial_conditions:
      - dataeval-flow installed
      - Synthetic dataset built via fixtures.make_synthetic_dataset
    expected_results:
      - PreprocessorConfig accepts torchvision-style transform dicts
      - SelectionStep instances stack into a SelectionConfig
      - A SelectionConfig with Limit reduces dataset length

  "6-1":
    name: Workflow Orchestration
    test_type: Core Functionality
    business_case: >-
      Verify that the workflow registry exposes seven workflows and that
      run_tasks executes them in sequence and returns WorkflowResult objects.
    initial_conditions:
      - dataeval-flow installed
    expected_results:
      - list_workflows() returns 7 entries with name + description
      - get_workflow("data-cleaning") returns a WorkflowProtocol
      - run_tasks on a synthetic pipeline returns a list of WorkflowResult
      - Each WorkflowResult has populated metadata.timestamp and tool fields

  "7-1":
    name: Data Cleaning Workflow
    test_type: Core Functionality
    business_case: >-
      Verify that DataCleaningWorkflow executes end-to-end on synthetic data
      and produces a typed WorkflowResult with cleaning findings.
    initial_conditions:
      - dataeval-flow installed
      - Synthetic dataset and FlattenExtractor configured
    expected_results:
      - DataCleaningWorkflow.execute returns success=True
      - result.data exposes outlier and duplicate findings
      - result.report() returns a non-empty string

  "8-1":
    name: Drift Monitoring Workflow
    test_type: Core Functionality
    business_case: >-
      Verify that DriftMonitoringWorkflow executes end-to-end on synthetic
      reference and test data.
    initial_conditions:
      - dataeval-flow installed
      - Two synthetic datasets (reference, test) configured as sources
    expected_results:
      - DriftMonitoringWorkflow.execute returns success=True
      - result.data exposes drift findings
      - result.report() returns a non-empty string

  "9-1":
    name: OOD Detection Workflow
    test_type: Core Functionality
    business_case: >-
      Verify that OODDetectionWorkflow executes end-to-end on synthetic
      reference and test data.
    initial_conditions:
      - dataeval-flow installed
      - Two synthetic datasets configured as sources
    expected_results:
      - OODDetectionWorkflow.execute returns success=True
      - result.data exposes OOD findings (instance_score array)
      - result.report() returns a non-empty string

  "10-1":
    name: Data Analysis Workflow & Type Safety
    test_type: Core Functionality
    business_case: >-
      Verify that DataAnalysisWorkflow executes end-to-end on synthetic
      data and that the package ships PEP 561 typing infrastructure.
    initial_conditions:
      - dataeval-flow installed
      - Synthetic dataset configured
    expected_results:
      - DataAnalysisWorkflow.execute returns success=True
      - result.data exposes analysis outputs (bias, quality, etc.)
      - py.typed marker is present in the installed package
      - dataeval_flow.__all__ is defined and non-empty

  "11-1":
    name: Splitting Workflow & Configuration Reproducibility
    test_type: Core Functionality
    business_case: >-
      Verify that DataSplittingWorkflow produces deterministic splits when
      the seed is set, demonstrating configuration reproducibility.
    initial_conditions:
      - dataeval-flow installed
      - Synthetic dataset configured with a fixed seed
    expected_results:
      - DataSplittingWorkflow.execute returns success=True
      - Two runs with the same seed produce identical splits
      - Two runs with different seeds produce different splits

  "12-1":
    name: Prioritization Workflow
    test_type: Core Functionality
    business_case: >-
      Verify that DataPrioritizationWorkflow ranks synthetic samples and
      produces a typed WorkflowResult.
    initial_conditions:
      - dataeval-flow installed
      - Synthetic dataset configured
    expected_results:
      - DataPrioritizationWorkflow.execute returns success=True
      - result.data exposes ranked indices

  "13-1":
    name: Parameter Sweep Workflow
    test_type: Core Functionality
    business_case: >-
      Verify that ParameterSweepWorkflow runs a small parameter grid and
      produces one WorkflowResult per parameter combination.
    initial_conditions:
      - dataeval-flow installed
      - Synthetic dataset configured
      - Sweep grid with 2 parameter combinations
    expected_results:
      - ParameterSweepWorkflow.execute returns success=True
      - result.data exposes per-combination outputs

  "14-1":
    name: Reporting & Export
    test_type: Core Functionality
    business_case: >-
      Verify that WorkflowResult renders human-readable text reports and
      exports JSON / YAML.
    initial_conditions:
      - dataeval-flow installed
      - A WorkflowResult from a synthetic-data workflow run
    expected_results:
      - WorkflowResult.report() returns a non-empty multi-line string
      - WorkflowResult.export(path, fmt="json") writes a valid JSON file
      - WorkflowResult.export(path, fmt="yaml") writes a valid YAML file
      - WorkflowResult.to_dict() includes metadata + data fields

  "15-1":
    name: Caching
    test_type: Core Functionality
    business_case: >-
      Verify that the DatasetCache stores and retrieves computed artifacts
      keyed by a deterministic content hash.
    initial_conditions:
      - dataeval-flow installed
      - Writable tmp_path for cache root
    expected_results:
      - DatasetCache writes an entry and round-trips it on read
      - Cache key is stable across runs for the same inputs
      - Cache miss returns None / raises a documented error

  "16-1":
    name: CLI Entrypoints & Logging
    test_type: Core Functionality
    business_case: >-
      Verify that the headless CLI, TUI app, and config builder are
      installed as entrypoints and that the CLI configures stdlib logging.
    initial_conditions:
      - dataeval-flow installed
    expected_results:
      - dataeval-flow console_script is on PATH
      - `python -m dataeval_flow --help` exits 0
      - `python -m dataeval_flow app --help` exits 0
      - `python -m dataeval_flow config --help` exits 0
      - CLI startup configures the root logger with at least one handler

  "17-1":
    name: Docker Artifacts Present
    test_type: Core Functionality
    business_case: >-
      Verify that the repo ships the expected Dockerfiles and that they
      pass a basic syntactic lint.
    initial_conditions:
      - Repo checkout with docker/ directory
    expected_results:
      - docker/Dockerfile.cpu exists and starts with FROM
      - docker/Dockerfile.cu118 exists and starts with FROM
      - Both Dockerfiles declare a non-root USER

  "18-1":
    name: Result Metadata Envelope
    test_type: Core Functionality
    business_case: >-
      Verify that every WorkflowResult populates the JATIC-required
      ResultMetadata envelope per IR-3-H-12.
    initial_conditions:
      - dataeval-flow installed
      - A WorkflowResult from a synthetic-data workflow run
    expected_results:
      - metadata.version is set
      - metadata.timestamp is a timezone-aware datetime
      - metadata.tool == "dataeval-flow"
      - metadata.tool_version is set to dataeval_flow.__version__
      - metadata.resolved_config is a dict and is non-empty
      - metadata.execution_time_s is a non-negative float
```

- [ ] **Step 2.2: Write `verification/generate_metarepo.py`**

```python
#!/usr/bin/env python3
"""Generate meta repo artifacts from verification test results."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

VERIFICATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERIFICATION_DIR.parent
REGISTRY_PATH = VERIFICATION_DIR / "registry.yaml"
REPORT_PATH = PROJECT_ROOT / "output" / "verification_report.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "metarepo"


def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def load_report() -> dict | None:
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            return json.load(f)
    return None


def _human_readable_step(nodeid: str) -> str:
    test_name = nodeid.rsplit("::", 1)[-1]
    desc = re.sub(r"^test_", "", test_name).replace("_", " ").capitalize()
    return desc


def _result_char(status: str) -> str:
    if status == "passed":
        return "P"
    if status == "skipped":
        return "S"
    return "F"


def generate_test_case_md(tc_id: str, tc_meta: dict, report: dict | None) -> str:
    tc_key = f"test-case-{tc_id}"
    today = datetime.now(tz=timezone.utc).strftime("%m/%d/%Y")
    tc_report = None
    if report and tc_key in report.get("test_cases", {}):
        tc_report = report["test_cases"][tc_key]

    lines: list[str] = []
    lines.append(f"# {tc_meta['name']}")
    lines.append("")
    lines.append("## Description")
    lines.append("")
    lines.append(f"- Test Type: {tc_meta['test_type']}")
    lines.append(f"- Business Case: {tc_meta['business_case'].strip()}")
    lines.append("")
    lines.append("**Initial Conditions:**")
    lines.append("")
    for i, cond in enumerate(tc_meta["initial_conditions"], 1):
        lines.append(f"{i}. {cond}")
    lines.append("")
    lines.append("## Test Steps")
    lines.append("")

    if tc_report:
        tests = tc_report["tests"]
        for i, test in enumerate(tests, 1):
            lines.append(f"{i}. {_human_readable_step(test['test'])}")
        lines.append(f"{len(tests) + 1}. Confirm the Expected Results by validating all steps pass")
    else:
        for i, er in enumerate(tc_meta["expected_results"], 1):
            lines.append(f"{i}. Verify: {er}")
        lines.append(
            f"{len(tc_meta['expected_results']) + 1}. Confirm the Expected Results by validating all steps pass",
        )
    lines.append("")
    lines.append("**Expected Results**")
    lines.append("")
    for i, result in enumerate(tc_meta["expected_results"], 1):
        lines.append(f"{i}. {result}")
    lines.append("")
    lines.append("## Test Results")
    lines.append("")
    lines.append("| Test Step |  Result | Notes |")
    lines.append("|:----------|:-------:|:------|")

    if tc_report:
        tests = tc_report["tests"]
        for i, test in enumerate(tests, 1):
            r = _result_char(test["status"])
            lines.append(f"|{i:<10}|    {r}    |  [^{i}] |")
        overall = tc_report["status"]
        confirm = "P" if overall == "passed" else "F"
        n = len(tests) + 1
        lines.append(f"|{n:<10}|    {confirm}    |  [^{n}] |")
        lines.append("")
        for i, test in enumerate(tests, 1):
            lines.append(f"[^{i}]: `{test['test']}` — {test['status']}")
        lines.append(f"[^{n}]: Overall verification — {overall}")
    else:
        lines.append("|1         |   P/F   |  [^1] |")
        lines.append("")
        lines.append("[^1]: Awaiting automated test results")

    lines.append("")
    lines.append(f"**Last Updated Date:** {today}")
    return "\n".join(lines) + "\n"


def _tc_sort_key(tc_id: str) -> list[int]:
    return [int(p) for p in tc_id.split("-")]


def generate_vcrm(registry: dict, report: dict | None) -> str:
    requirements = registry["requirements"]
    test_cases = registry["test_cases"]
    today = datetime.now(tz=timezone.utc).strftime("%m/%d/%Y")
    all_tc_ids = sorted(test_cases.keys(), key=_tc_sort_key)

    tc_headers = [f"[TC-{tc_id.replace('-', '.')}][{tc_id}]" for tc_id in all_tc_ids]
    header = "| Requirement ID | Requirement Origin | Coverage | " + " | ".join(tc_headers) + " |"

    sep_parts = [":--------------", ":-------------------", ":--------:"] + [":-------------:"] * len(all_tc_ids)
    separator = "| " + " | ".join(sep_parts) + " |"

    rows: list[str] = []
    for req_id, req_data in requirements.items():
        req_tcs = set(req_data.get("test_cases", []))
        coverage = "Yes" if req_tcs else "No"
        ref_key = req_id.lower().replace("-", "")
        req_cell = f"[{req_id}][{ref_key}]"
        origin = req_data.get("origin", "")
        origin_ref = origin.lower().replace("-", "").replace(".", "")
        origin_cell = f"[{origin}][{origin_ref}]" if origin else ""
        tc_cells = ["X" if tc_id in req_tcs else " " for tc_id in all_tc_ids]
        row_parts = [req_cell, origin_cell, coverage] + tc_cells
        rows.append("| " + " | ".join(row_parts) + " |")

    verification_cells: list[str] = []
    for tc_id in all_tc_ids:
        tc_key = f"test-case-{tc_id}"
        if report and tc_key in report.get("test_cases", {}):
            status = report["test_cases"][tc_key]["status"]
            verification_cells.append("Pass" if status == "passed" else "Fail")
        else:
            verification_cells.append("Pending")
    verification_row = "| **Verification** | | | " + " | ".join(verification_cells) + " |"

    tc_links = [f"[{tc_id}]:test-cases/test-case-{tc_id}.md" for tc_id in all_tc_ids]
    req_links = []
    for req_id, req_data in requirements.items():
        ref_key = req_id.lower().replace("-", "")
        filename = req_data.get("file", "#")
        req_links.append(f"[{ref_key}]:requirements/{filename}")
    origin_links: dict[str, str] = {}
    for req_data in requirements.values():
        origin = req_data.get("origin", "")
        origin_link = req_data.get("origin_link", "#")
        if origin:
            origin_ref = origin.lower().replace("-", "").replace(".", "")
            origin_links[origin_ref] = f"[{origin_ref}]:{origin_link}"

    parts = [
        "# DataEval Flow Verification Cross-Reference Matrix (VCRM)",
        "",
        header,
        separator,
        *rows,
        verification_row,
        "",
        f"**Last Updated:** {today}",
        "",
        "<!-- Links for Test Cases -->",
        "",
        *tc_links,
        "",
        "<!-- Links for Requirement IDs -->",
        "",
        *req_links,
        "",
        "<!-- Links for Requirement Origins -->",
        "",
        *sorted(origin_links.values()),
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    registry = load_registry()
    report = load_report()

    if report:
        s = report["summary"]
        print(
            f"Loaded verification report: {s['total_test_cases']} test cases "
            f"({s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped)",
        )
    else:
        print("No verification report found — generating templates only")

    tc_dir = OUTPUT_DIR / "test-cases"
    tc_dir.mkdir(parents=True, exist_ok=True)

    for tc_id, tc_meta in registry["test_cases"].items():
        content = generate_test_case_md(tc_id, tc_meta, report)
        out_path = tc_dir / f"test-case-{tc_id}.md"
        out_path.write_text(content)
        print(f"  Generated {out_path.name}")

    vcrm_content = generate_vcrm(registry, report)
    vcrm_path = OUTPUT_DIR / "vcrm.md"
    vcrm_path.write_text(vcrm_content)
    print("  Generated vcrm.md")
    print(f"\nAll artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.3: Smoke-test the generator (no report yet — should emit Pending stubs)**

```bash
cd /home/aweng/2033/dataeval-flow
uv run python verification/generate_metarepo.py
ls output/metarepo/test-cases/ output/metarepo/vcrm.md
```
Expected: 19 test-case-*.md files + vcrm.md printed, all "Verification" cells = `Pending`.

- [ ] **Step 2.4: Commit**

```bash
cd /home/aweng/2033/dataeval-flow
git add verification/registry.yaml verification/generate_metarepo.py
git commit -m "feat(verification): add registry and metarepo artifact generator"
```

---

## Task 3: TC-1-1, TC-1-2, TC-1-7 — installation, packaging, Python compatibility

**Files:**
- Create: `dataeval-flow/verification/functional/install/test_pip_install.py`
- Create: `dataeval-flow/verification/functional/contracts/test_public_api.py`
- Create: `dataeval-flow/verification/nonfunctional/compatibility/test_python_versions.py`
- Create: `dataeval-flow/verification/nonfunctional/packaging/test_version_metadata.py`

- [ ] **Step 3.1: Write `functional/install/test_pip_install.py`**

```python
"""TC-1-1 / TC-1-2 — installation, import, and optional extras."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.test_case("1-1")
class TestPipInstall:
    def test_import_dataeval_flow(self) -> None:
        import dataeval_flow

        assert dataeval_flow is not None

    def test_version_is_set(self) -> None:
        import dataeval_flow

        assert dataeval_flow.__version__
        assert dataeval_flow.__version__ != "unknown"

    def test_core_modules_importable(self) -> None:
        for name in (
            "dataeval_flow.config",
            "dataeval_flow.dataset",
            "dataeval_flow.workflow",
            "dataeval_flow.workflows",
            "dataeval_flow.runner",
            "dataeval_flow.cache",
            "dataeval_flow.preprocessing",
            "dataeval_flow.embeddings",
            "dataeval_flow.metadata",
            "dataeval_flow.selection",
        ):
            assert importlib.import_module(name) is not None


@pytest.mark.test_case("1-2")
class TestOptionalExtras:
    @pytest.mark.parametrize(
        ("module", "label"),
        [
            ("torch", "cpu/cu*"),
            ("torchvision", "cpu/cu*"),
            ("onnxruntime", "onnx"),
            ("cv2", "opencv"),
            ("textual", "app"),
        ],
    )
    def test_optional_module_importable_if_installed(self, module: str, label: str) -> None:
        mod = pytest.importorskip(module, reason=f"{label} extra not installed")
        assert mod is not None
```

- [ ] **Step 3.2: Write `functional/contracts/test_public_api.py`**

```python
"""TC-1-1 — public API surface of dataeval_flow."""

from __future__ import annotations

import pytest


@pytest.mark.test_case("1-1")
class TestPublicAPI:
    def test_top_level_all_exports(self) -> None:
        import dataeval_flow

        assert hasattr(dataeval_flow, "__all__")
        assert len(dataeval_flow.__all__) > 0

    def test_all_exports_are_importable(self) -> None:
        import dataeval_flow

        for name in dataeval_flow.__all__:
            assert hasattr(dataeval_flow, name), f"missing public symbol: {name}"

    def test_run_tasks_and_load_config_present(self) -> None:
        from dataeval_flow import load_config, load_config_folder, run_tasks

        assert callable(load_config)
        assert callable(load_config_folder)
        assert callable(run_tasks)
```

- [ ] **Step 3.3: Write `nonfunctional/compatibility/test_python_versions.py`**

```python
"""TC-1-1 — Python version compatibility."""

from __future__ import annotations

import sys

import pytest

SUPPORTED = ((3, 10), (3, 11), (3, 12), (3, 13))


@pytest.mark.test_case("1-1")
class TestPythonVersions:
    def test_running_on_supported_version(self) -> None:
        assert sys.version_info[:2] in SUPPORTED, f"Python {sys.version_info[:2]} is not supported"

    def test_import_succeeds_on_current_version(self) -> None:
        import dataeval_flow

        assert dataeval_flow is not None
```

- [ ] **Step 3.4: Write `nonfunctional/packaging/test_version_metadata.py`**

```python
"""TC-1-7 — packaging metadata."""

from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

import pytest

DIST = "dataeval-flow"


@pytest.mark.test_case("1-7")
class TestVersionMetadata:
    def test_version_not_unknown(self) -> None:
        assert md.version(DIST) != "unknown"

    def test_package_name(self) -> None:
        meta = md.metadata(DIST)
        assert meta["Name"].lower() == DIST

    def test_requires_python(self) -> None:
        meta = md.metadata(DIST)
        assert meta["Requires-Python"].strip().startswith(">=3.10")

    def test_license_set(self) -> None:
        meta = md.metadata(DIST)
        license_val = meta.get("License") or meta.get("License-Expression") or ""
        assert "MIT" in license_val.upper()

    def test_py_typed_marker_present(self) -> None:
        import dataeval_flow

        pkg_root = Path(dataeval_flow.__file__).parent
        assert (pkg_root / "py.typed").exists()

    def test_no_test_files_in_installed_package(self) -> None:
        import dataeval_flow

        pkg_root = Path(dataeval_flow.__file__).parent
        assert not any(p.name == "tests" for p in pkg_root.iterdir())
        assert not any(p.name == "verification" for p in pkg_root.iterdir())
```

- [ ] **Step 3.5: Run the new tests**

```bash
cd /home/aweng/2033/dataeval-flow
uv run pytest verification/functional/install verification/functional/contracts verification/nonfunctional/compatibility verification/nonfunctional/packaging -v
```
Expected: all PASS (some `1-2` parametrizations may be skipped depending on installed extras).

- [ ] **Step 3.6: Commit**

```bash
cd /home/aweng/2033/dataeval-flow
git add verification/functional/install verification/functional/contracts verification/nonfunctional/compatibility verification/nonfunctional/packaging
git commit -m "test(verification): TC-1-1, TC-1-2, TC-1-7 install / packaging / API"
```

---

## Task 4: TC-2-1, TC-3-1, TC-4-1, TC-5-1 — config, dataset, extractors, pipeline

**Files:**
- Create: `dataeval-flow/verification/functional/config/test_config_load.py`
- Create: `dataeval-flow/verification/functional/dataset/test_dataset_configs.py`
- Create: `dataeval-flow/verification/functional/extractors/test_extractor_configs.py`
- Create: `dataeval-flow/verification/functional/pipeline/test_preprocessing_selection.py`

- [ ] **Step 4.1: Write `functional/config/test_config_load.py`**

```python
"""TC-2-1 — configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import PipelineConfig, load_config, load_config_folder


MINIMAL_CONFIG = {
    "sources": {
        "main": {
            "dataset": {"type": "image_folder", "path": "./images"},
        },
    },
    "tasks": [],
}


@pytest.mark.test_case("2-1")
class TestConfigLoading:
    def test_load_yaml_single_file(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "params.yaml"
        cfg_path.write_text(yaml.safe_dump(MINIMAL_CONFIG))
        cfg = load_config(cfg_path)
        assert isinstance(cfg, PipelineConfig)

    def test_load_json_single_file(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "params.json"
        cfg_path.write_text(json.dumps(MINIMAL_CONFIG))
        cfg = load_config(cfg_path)
        assert isinstance(cfg, PipelineConfig)

    def test_load_config_folder_merges_files(self, tmp_path: Path) -> None:
        (tmp_path / "sources.yaml").write_text(yaml.safe_dump({"sources": MINIMAL_CONFIG["sources"]}))
        (tmp_path / "tasks.yaml").write_text(yaml.safe_dump({"tasks": []}))
        cfg = load_config_folder(tmp_path)
        assert isinstance(cfg, PipelineConfig)

    def test_invalid_config_raises_validation_error(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml.safe_dump({"sources": "this is not a dict"}))
        with pytest.raises((ValidationError, ValueError, TypeError)):
            load_config(cfg_path)
```

- [ ] **Step 4.2: Write `functional/dataset/test_dataset_configs.py`**

```python
"""TC-3-1 — dataset configs and load_dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataeval_flow import (
    CocoDatasetConfig,
    DatasetProtocolConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    YoloDatasetConfig,
    load_dataset,
)
from verification.fixtures import write_image_folder


@pytest.mark.test_case("3-1")
class TestDatasetConfigs:
    def test_image_folder_config_roundtrip(self) -> None:
        cfg = ImageFolderDatasetConfig(type="image_folder", path="./images")
        assert cfg.model_dump()["path"] == "./images"

    def test_huggingface_config_roundtrip(self) -> None:
        cfg = HuggingFaceDatasetConfig(type="huggingface", repo_id="placeholder/repo")
        assert cfg.repo_id == "placeholder/repo"

    def test_coco_config_roundtrip(self) -> None:
        cfg = CocoDatasetConfig(type="coco", images="./imgs", annotations="./ann.json")
        assert cfg.annotations.endswith("ann.json")

    def test_yolo_config_roundtrip(self) -> None:
        cfg = YoloDatasetConfig(type="yolo", path="./yolo")
        assert cfg.path == "./yolo"

    def test_protocol_config_roundtrip(self) -> None:
        cfg = DatasetProtocolConfig(type="protocol", target="package.module:factory")
        assert cfg.target == "package.module:factory"


@pytest.mark.test_case("3-1")
class TestLoadDataset:
    def test_load_dataset_image_folder(self, tmp_path: Path) -> None:
        root = write_image_folder(tmp_path / "data")
        cfg = ImageFolderDatasetConfig(type="image_folder", path=str(root))
        ds = load_dataset(cfg, data_dir=tmp_path)
        assert len(ds) > 0
```

If a config field name above differs from the actual pydantic model (e.g. `repo_id` vs `repo`), inspect `src/dataeval_flow/config/schemas/_dataset.py` once during step 4.5 and adjust before committing. The pattern (one `_roundtrip` test per dataset config + one end-to-end via `load_dataset`) does not change.

- [ ] **Step 4.3: Write `functional/extractors/test_extractor_configs.py`**

```python
"""TC-4-1 — extractor configs."""

from __future__ import annotations

import pytest

from dataeval_flow import (
    BoVWExtractorConfig,
    FlattenExtractorConfig,
    OnnxExtractorConfig,
    TorchExtractorConfig,
    UncertaintyExtractorConfig,
)


@pytest.mark.test_case("4-1")
class TestExtractorConfigs:
    def test_flatten_extractor_config_constructs(self) -> None:
        cfg = FlattenExtractorConfig(type="flatten")
        assert cfg.type == "flatten"

    def test_torch_extractor_config_constructs(self) -> None:
        cfg = TorchExtractorConfig(type="torch", target="torchvision.models:resnet18")
        assert cfg.target.startswith("torchvision")

    def test_onnx_extractor_config_constructs(self) -> None:
        cfg = OnnxExtractorConfig(type="onnx", path="./model.onnx")
        assert cfg.path.endswith(".onnx")

    def test_bovw_extractor_config_constructs(self) -> None:
        cfg = BoVWExtractorConfig(type="bovw")
        assert cfg.type == "bovw"

    def test_uncertainty_extractor_config_constructs(self) -> None:
        cfg = UncertaintyExtractorConfig(type="uncertainty")
        assert cfg.type == "uncertainty"

    def test_all_extractor_configs_exported(self) -> None:
        import dataeval_flow

        for name in (
            "FlattenExtractorConfig",
            "TorchExtractorConfig",
            "OnnxExtractorConfig",
            "BoVWExtractorConfig",
            "UncertaintyExtractorConfig",
        ):
            assert name in dataeval_flow.__all__
```

Field-name caveat from Step 4.2 applies here too — inspect `src/dataeval_flow/config/schemas/_extractor.py` if a default-required field is missing and adjust the kwargs.

- [ ] **Step 4.4: Write `functional/pipeline/test_preprocessing_selection.py`**

```python
"""TC-5-1 — preprocessing & selection pipeline."""

from __future__ import annotations

import pytest

from dataeval_flow import PreprocessorConfig, SelectionConfig, SelectionStep
from verification.fixtures import make_synthetic_dataset


@pytest.mark.test_case("5-1")
class TestPreprocessingSelection:
    def test_preprocessor_config_accepts_transforms(self) -> None:
        cfg = PreprocessorConfig(transforms=[{"target": "torchvision.transforms.ToTensor"}])
        assert len(cfg.transforms) == 1

    def test_selection_step_constructs(self) -> None:
        step = SelectionStep(target="dataeval.selection:Limit", params={"size": 4})
        assert step.target.endswith("Limit")

    def test_selection_config_stacks_steps(self) -> None:
        cfg = SelectionConfig(
            steps=[
                SelectionStep(target="dataeval.selection:Limit", params={"size": 4}),
                SelectionStep(target="dataeval.selection:Shuffle", params={}),
            ],
        )
        assert len(cfg.steps) == 2

    def test_synthetic_dataset_baseline_length(self) -> None:
        ds = make_synthetic_dataset(n=8)
        assert len(ds) == 8
```

- [ ] **Step 4.5: Run the new tests; adjust kwargs if any pydantic field name differs**

```bash
cd /home/aweng/2033/dataeval-flow
uv run pytest verification/functional/config verification/functional/dataset verification/functional/extractors verification/functional/pipeline -v
```
If a `ValidationError` reports a missing/unknown field, open the relevant `src/dataeval_flow/config/schemas/_*.py` and fix the kwargs in the test — do not change the schema. Re-run until green.

- [ ] **Step 4.6: Commit**

```bash
cd /home/aweng/2033/dataeval-flow
git add verification/functional/{config,dataset,extractors,pipeline}
git commit -m "test(verification): TC-2-1..5-1 config / dataset / extractor / pipeline"
```

---

## Task 5: TC-6-1, TC-14-1, TC-15-1, TC-18-1 — orchestration, reporting, cache, metadata envelope

**Files:**
- Create: `dataeval-flow/verification/functional/orchestration/test_run_tasks.py`
- Create: `dataeval-flow/verification/functional/reporting/test_text_report.py`
- Create: `dataeval-flow/verification/functional/cache/test_cache.py`
- Create: `dataeval-flow/verification/functional/metadata/test_metadata_envelope.py`
- Modify: `dataeval-flow/verification/conftest.py` — append shared workflow fixture

- [ ] **Step 5.1: Write a shared workflow fixture in `conftest.py`**

Append the following to the existing `verification/conftest.py` (do not overwrite the report-collection hooks from Task 1.2 — add at the bottom of the file):

```python
# ---------------------------------------------------------------------------
# Shared workflow fixtures (TC-6-1 and downstream tasks)
# ---------------------------------------------------------------------------

import pytest as _pytest


@_pytest.fixture
def synthetic_pipeline_config(tmp_path):
    """Return a PipelineConfig + data_dir that runs a single trivial workflow."""
    from dataeval_flow import (
        DataCleaningTaskConfig,
        DataCleaningWorkflowConfig,
        FlattenExtractorConfig,
        ImageFolderDatasetConfig,
        PipelineConfig,
        SourceConfig,
        TaskConfig,
    )
    from verification.fixtures import write_image_folder

    write_image_folder(tmp_path / "imgs", n_per_class=4, n_classes=2)
    cfg = PipelineConfig(
        sources={
            "main": SourceConfig(
                dataset=ImageFolderDatasetConfig(type="image_folder", path="imgs"),
                extractor=FlattenExtractorConfig(type="flatten"),
            ),
        },
        tasks=[
            TaskConfig(
                workflow=DataCleaningWorkflowConfig(
                    type="data-cleaning",
                    params=DataCleaningTaskConfig(source="main"),
                ),
            ),
        ],
    )
    return cfg, tmp_path
```

The exact pydantic field names above (e.g. `SourceConfig(dataset=..., extractor=...)`, `TaskConfig(workflow=...)`) may differ slightly from current schemas. When step 5.5 reports a `ValidationError`, open `src/dataeval_flow/config/_models.py` and `src/dataeval_flow/config/schemas/_task.py` and adjust the constructor kwargs — keep the structural shape (one source, one cleaning task).

- [ ] **Step 5.2: Write `functional/orchestration/test_run_tasks.py`**

```python
"""TC-6-1 — workflow orchestration."""

from __future__ import annotations

import pytest

from dataeval_flow import WorkflowResult, get_workflow, list_workflows, run_tasks
from dataeval_flow.workflow import WorkflowProtocol


@pytest.mark.test_case("6-1")
class TestOrchestration:
    def test_list_workflows_returns_seven(self) -> None:
        wfs = list_workflows()
        names = {w["name"] for w in wfs}
        assert names == {
            "data-analysis",
            "data-cleaning",
            "drift-monitoring",
            "ood-detection",
            "parameter-sweep",
            "data-prioritization",
            "data-splitting",
        }

    def test_get_workflow_returns_protocol(self) -> None:
        wf = get_workflow("data-cleaning")
        assert isinstance(wf, WorkflowProtocol)

    def test_get_workflow_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            get_workflow("does-not-exist")

    def test_run_tasks_returns_results(self, synthetic_pipeline_config) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        assert len(results) == 1
        assert isinstance(results[0], WorkflowResult)
        assert results[0].metadata.tool == "dataeval-flow"
```

- [ ] **Step 5.3: Write `functional/reporting/test_text_report.py`**

```python
"""TC-14-1 — reporting & export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dataeval_flow import run_tasks


@pytest.mark.test_case("14-1")
class TestReporting:
    def test_report_returns_nonempty_string(self, synthetic_pipeline_config) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        text = results[0].report()
        assert isinstance(text, str)
        assert len(text.splitlines()) > 5

    def test_export_json_writes_file(self, synthetic_pipeline_config, tmp_path: Path) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        out = results[0].export(tmp_path / "result.json", fmt="json")
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert "metadata" in parsed

    def test_export_yaml_writes_file(self, synthetic_pipeline_config, tmp_path: Path) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        out = results[0].export(tmp_path / "result.yaml", fmt="yaml")
        assert out.exists()
        parsed = yaml.safe_load(out.read_text())
        assert "metadata" in parsed

    def test_to_dict_includes_metadata_and_data(self, synthetic_pipeline_config) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        d = results[0].to_dict()
        assert "metadata" in d
```

- [ ] **Step 5.4: Write `functional/cache/test_cache.py`**

```python
"""TC-15-1 — caching."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.test_case("15-1")
class TestCache:
    def test_cache_module_importable(self) -> None:
        from dataeval_flow import cache

        assert cache is not None

    def test_dataset_cache_roundtrip(self, tmp_path: Path) -> None:
        from dataeval_flow.cache import DatasetCache

        c = DatasetCache(root=tmp_path)
        key = c.key_for(payload={"x": 1, "y": [1, 2, 3]})
        c.put(key, b"hello")
        assert c.get(key) == b"hello"

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        from dataeval_flow.cache import DatasetCache

        c = DatasetCache(root=tmp_path)
        assert c.get("nonexistent-key") is None
```

The exact `DatasetCache` API (constructor signature, `key_for`/`put`/`get`) is the most likely thing to drift from the design. When step 5.5 reports a missing attribute, open `src/dataeval_flow/cache.py` once and adapt — the test intent (put + get + miss-returns-None) does not change.

- [ ] **Step 5.5: Write `functional/metadata/test_metadata_envelope.py`**

```python
"""TC-18-1 — JATIC ResultMetadata envelope (IR-3-H-12)."""

from __future__ import annotations

from datetime import datetime

import pytest

import dataeval_flow
from dataeval_flow import run_tasks


@pytest.mark.test_case("18-1")
class TestResultMetadataEnvelope:
    def test_version_field_set(self, synthetic_pipeline_config) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.version

    def test_timestamp_is_timezone_aware(self, synthetic_pipeline_config) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        ts = result.metadata.timestamp
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_tool_identifier(self, synthetic_pipeline_config) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.tool == "dataeval-flow"

    def test_tool_version_matches_package(self, synthetic_pipeline_config) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.tool_version == dataeval_flow.__version__

    def test_resolved_config_is_dict_and_nonempty(self, synthetic_pipeline_config) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert isinstance(result.metadata.resolved_config, dict)
        assert result.metadata.resolved_config

    def test_execution_time_nonnegative(self, synthetic_pipeline_config) -> None:
        cfg, data_dir = synthetic_pipeline_config
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.metadata.execution_time_s is not None
        assert result.metadata.execution_time_s >= 0
```

- [ ] **Step 5.6: Run the new tests**

```bash
cd /home/aweng/2033/dataeval-flow
uv run pytest verification/functional/{orchestration,reporting,cache,metadata} -v
```
Expected: all PASS. If a pydantic field name issue surfaces, adjust the `synthetic_pipeline_config` fixture per the caveat in Step 5.1.

- [ ] **Step 5.7: Commit**

```bash
cd /home/aweng/2033/dataeval-flow
git add verification/conftest.py verification/functional/{orchestration,reporting,cache,metadata}
git commit -m "test(verification): TC-6-1, 14-1, 15-1, 18-1 orchestration / reporting / cache / metadata envelope"
```

---

## Task 6: TC-7-1..13-1 + NFR-3/NFR-4 — workflow verification tests

**Files:**
- Create: `dataeval-flow/verification/functional/workflows/test_cleaning.py`
- Create: `dataeval-flow/verification/functional/workflows/test_drift.py`
- Create: `dataeval-flow/verification/functional/workflows/test_ood.py`
- Create: `dataeval-flow/verification/functional/workflows/test_analysis.py`
- Create: `dataeval-flow/verification/functional/workflows/test_splitting.py`
- Create: `dataeval-flow/verification/functional/workflows/test_prioritization.py`
- Create: `dataeval-flow/verification/functional/workflows/test_parameter_sweep.py`
- Create: `dataeval-flow/verification/nonfunctional/test_type_safety.py`
- Create: `dataeval-flow/verification/nonfunctional/test_config_reproducibility.py`

All workflow tests share the same shape:

1. Build a single-workflow `PipelineConfig` using `write_image_folder` and the workflow's task config class
2. Call `run_tasks`
3. Assert `success is True`, `result.report()` is a non-empty string, and the workflow-specific output fields exist

Subtle differences per workflow: drift and ood need **two** sources (reference + test); parameter-sweep needs a sweep dimension in the task config; splitting needs a seed.

- [ ] **Step 6.1: Write `functional/workflows/test_cleaning.py`**

```python
"""TC-7-1 — data cleaning workflow."""

from __future__ import annotations

import pytest

from dataeval_flow import (
    DataCleaningTaskConfig,
    DataCleaningWorkflowConfig,
    FlattenExtractorConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    run_tasks,
)
from verification.fixtures import write_image_folder


@pytest.mark.test_case("7-1")
class TestDataCleaningWorkflow:
    def test_cleaning_workflow_runs(self, tmp_path) -> None:
        write_image_folder(tmp_path / "imgs", n_per_class=4, n_classes=2)
        cfg = PipelineConfig(
            sources={
                "main": SourceConfig(
                    dataset=ImageFolderDatasetConfig(type="image_folder", path="imgs"),
                    extractor=FlattenExtractorConfig(type="flatten"),
                ),
            },
            tasks=[TaskConfig(workflow=DataCleaningWorkflowConfig(
                type="data-cleaning",
                params=DataCleaningTaskConfig(source="main"),
            ))],
        )
        result = run_tasks(cfg, data_dir=tmp_path)[0]
        assert result.success
        assert isinstance(result.report(), str) and len(result.report()) > 0
```

- [ ] **Step 6.2: Write `functional/workflows/test_drift.py`**

```python
"""TC-8-1 — drift monitoring workflow."""

from __future__ import annotations

import pytest

from dataeval_flow import (
    DriftMonitoringTaskConfig,
    DriftMonitoringWorkflowConfig,
    FlattenExtractorConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    run_tasks,
)
from verification.fixtures import write_image_folder


@pytest.mark.test_case("8-1")
class TestDriftMonitoringWorkflow:
    def test_drift_workflow_runs(self, tmp_path) -> None:
        write_image_folder(tmp_path / "ref", n_per_class=4, n_classes=2, seed=0)
        write_image_folder(tmp_path / "test", n_per_class=4, n_classes=2, seed=99)
        cfg = PipelineConfig(
            sources={
                "ref": SourceConfig(
                    dataset=ImageFolderDatasetConfig(type="image_folder", path="ref"),
                    extractor=FlattenExtractorConfig(type="flatten"),
                ),
                "test": SourceConfig(
                    dataset=ImageFolderDatasetConfig(type="image_folder", path="test"),
                    extractor=FlattenExtractorConfig(type="flatten"),
                ),
            },
            tasks=[TaskConfig(workflow=DriftMonitoringWorkflowConfig(
                type="drift-monitoring",
                params=DriftMonitoringTaskConfig(reference="ref", target="test"),
            ))],
        )
        result = run_tasks(cfg, data_dir=tmp_path)[0]
        assert result.success
        assert isinstance(result.report(), str) and result.report().strip()
```

The `DriftMonitoringTaskConfig` field names (`reference` / `target`) may differ — inspect `src/dataeval_flow/workflows/drift/params.py` once during Step 6.10.

- [ ] **Step 6.3: Write `functional/workflows/test_ood.py`**

```python
"""TC-9-1 — OOD detection workflow."""

from __future__ import annotations

import pytest

from dataeval_flow import (
    FlattenExtractorConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    run_tasks,
)
from dataeval_flow.workflow import get_workflow
from verification.fixtures import write_image_folder


@pytest.mark.test_case("9-1")
class TestOODWorkflow:
    def test_ood_workflow_runs(self, tmp_path) -> None:
        write_image_folder(tmp_path / "ref", n_per_class=4, n_classes=2, seed=0)
        write_image_folder(tmp_path / "test", n_per_class=4, n_classes=2, seed=99)
        # Workflow registered as "ood-detection". Build TaskConfig dynamically.
        wf = get_workflow("ood-detection")
        params_schema = wf.params_schema
        assert params_schema is not None
        params = params_schema(reference="ref", target="test")
        cfg = PipelineConfig(
            sources={
                "ref": SourceConfig(
                    dataset=ImageFolderDatasetConfig(type="image_folder", path="ref"),
                    extractor=FlattenExtractorConfig(type="flatten"),
                ),
                "test": SourceConfig(
                    dataset=ImageFolderDatasetConfig(type="image_folder", path="test"),
                    extractor=FlattenExtractorConfig(type="flatten"),
                ),
            },
            tasks=[TaskConfig(workflow={"type": "ood-detection", "params": params.model_dump()})],
        )
        result = run_tasks(cfg, data_dir=tmp_path)[0]
        assert result.success
        assert result.report().strip()
```

- [ ] **Step 6.4: Write `functional/workflows/test_analysis.py`**

```python
"""TC-10-1 — data analysis workflow + NFR-3 type-safety hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataeval_flow import (
    FlattenExtractorConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    run_tasks,
)
from dataeval_flow.workflow import get_workflow
from verification.fixtures import write_image_folder


@pytest.mark.test_case("10-1")
class TestDataAnalysisWorkflow:
    def test_analysis_workflow_runs(self, tmp_path) -> None:
        write_image_folder(tmp_path / "imgs", n_per_class=4, n_classes=2)
        wf = get_workflow("data-analysis")
        params_schema = wf.params_schema
        params = params_schema(source="main") if params_schema is not None else None
        cfg = PipelineConfig(
            sources={
                "main": SourceConfig(
                    dataset=ImageFolderDatasetConfig(type="image_folder", path="imgs"),
                    extractor=FlattenExtractorConfig(type="flatten"),
                ),
            },
            tasks=[TaskConfig(workflow={"type": "data-analysis", "params": params.model_dump() if params else {}})],
        )
        result = run_tasks(cfg, data_dir=tmp_path)[0]
        assert result.success
```

- [ ] **Step 6.5: Write `functional/workflows/test_splitting.py`**

```python
"""TC-11-1 — splitting workflow."""

from __future__ import annotations

import pytest

from dataeval_flow import (
    FlattenExtractorConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    run_tasks,
)
from dataeval_flow.workflow import get_workflow
from verification.fixtures import write_image_folder


def _build_split_cfg(tmp_path, seed: int):
    write_image_folder(tmp_path / "imgs", n_per_class=8, n_classes=2)
    wf = get_workflow("data-splitting")
    params_schema = wf.params_schema
    assert params_schema is not None
    params = params_schema(source="main", seed=seed)
    return PipelineConfig(
        sources={
            "main": SourceConfig(
                dataset=ImageFolderDatasetConfig(type="image_folder", path="imgs"),
                extractor=FlattenExtractorConfig(type="flatten"),
            ),
        },
        tasks=[TaskConfig(workflow={"type": "data-splitting", "params": params.model_dump()})],
    )


@pytest.mark.test_case("11-1")
class TestDataSplittingWorkflow:
    def test_splitting_workflow_runs(self, tmp_path) -> None:
        cfg = _build_split_cfg(tmp_path, seed=42)
        result = run_tasks(cfg, data_dir=tmp_path)[0]
        assert result.success
```

- [ ] **Step 6.6: Write `functional/workflows/test_prioritization.py`**

```python
"""TC-12-1 — prioritization workflow."""

from __future__ import annotations

import pytest

from dataeval_flow import (
    FlattenExtractorConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    run_tasks,
)
from dataeval_flow.workflow import get_workflow
from verification.fixtures import write_image_folder


@pytest.mark.test_case("12-1")
class TestDataPrioritizationWorkflow:
    def test_prioritization_workflow_runs(self, tmp_path) -> None:
        write_image_folder(tmp_path / "imgs", n_per_class=8, n_classes=2)
        wf = get_workflow("data-prioritization")
        params_schema = wf.params_schema
        params = params_schema(source="main") if params_schema is not None else None
        cfg = PipelineConfig(
            sources={
                "main": SourceConfig(
                    dataset=ImageFolderDatasetConfig(type="image_folder", path="imgs"),
                    extractor=FlattenExtractorConfig(type="flatten"),
                ),
            },
            tasks=[TaskConfig(workflow={"type": "data-prioritization", "params": params.model_dump() if params else {}})],
        )
        result = run_tasks(cfg, data_dir=tmp_path)[0]
        assert result.success
```

- [ ] **Step 6.7: Write `functional/workflows/test_parameter_sweep.py`**

```python
"""TC-13-1 — parameter sweep workflow."""

from __future__ import annotations

import pytest

from dataeval_flow import (
    FlattenExtractorConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    run_tasks,
)
from dataeval_flow.workflow import get_workflow
from verification.fixtures import write_image_folder


@pytest.mark.test_case("13-1")
class TestParameterSweepWorkflow:
    def test_parameter_sweep_runs(self, tmp_path) -> None:
        write_image_folder(tmp_path / "imgs", n_per_class=4, n_classes=2)
        wf = get_workflow("parameter-sweep")
        params_schema = wf.params_schema
        params = params_schema(source="main") if params_schema is not None else None
        cfg = PipelineConfig(
            sources={
                "main": SourceConfig(
                    dataset=ImageFolderDatasetConfig(type="image_folder", path="imgs"),
                    extractor=FlattenExtractorConfig(type="flatten"),
                ),
            },
            tasks=[TaskConfig(workflow={"type": "parameter-sweep", "params": params.model_dump() if params else {}})],
        )
        result = run_tasks(cfg, data_dir=tmp_path)[0]
        assert result.success
```

- [ ] **Step 6.8: Write `nonfunctional/test_type_safety.py`**

```python
"""TC-10-1 (NFR-3) — type-safety infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.test_case("10-1")
class TestTypeSafety:
    def test_py_typed_marker_present(self) -> None:
        import dataeval_flow

        assert (Path(dataeval_flow.__file__).parent / "py.typed").exists()

    def test_top_level_all_defined(self) -> None:
        import dataeval_flow

        assert hasattr(dataeval_flow, "__all__")
        assert isinstance(dataeval_flow.__all__, list)
        assert len(dataeval_flow.__all__) > 0

    def test_workflow_protocol_runtime_checkable(self) -> None:
        from dataeval_flow.workflow import WorkflowProtocol, get_workflow

        wf = get_workflow("data-cleaning")
        assert isinstance(wf, WorkflowProtocol)
```

- [ ] **Step 6.9: Write `nonfunctional/test_config_reproducibility.py`**

```python
"""TC-11-1 (NFR-4) — configuration reproducibility."""

from __future__ import annotations

import pytest

from dataeval_flow import (
    FlattenExtractorConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    run_tasks,
)
from dataeval_flow.workflow import get_workflow
from verification.fixtures import write_image_folder


def _split_cfg(tmp_path, seed: int):
    write_image_folder(tmp_path / "imgs", n_per_class=8, n_classes=2)
    wf = get_workflow("data-splitting")
    params = wf.params_schema(source="main", seed=seed)
    return PipelineConfig(
        sources={
            "main": SourceConfig(
                dataset=ImageFolderDatasetConfig(type="image_folder", path="imgs"),
                extractor=FlattenExtractorConfig(type="flatten"),
            ),
        },
        tasks=[TaskConfig(workflow={"type": "data-splitting", "params": params.model_dump()})],
    )


@pytest.mark.test_case("11-1")
class TestConfigReproducibility:
    def test_same_seed_produces_identical_output(self, tmp_path) -> None:
        a = run_tasks(_split_cfg(tmp_path / "a", seed=42), data_dir=tmp_path / "a")[0]
        b = run_tasks(_split_cfg(tmp_path / "b", seed=42), data_dir=tmp_path / "b")[0]
        assert a.to_dict() == b.to_dict()

    def test_different_seeds_produce_different_output(self, tmp_path) -> None:
        a = run_tasks(_split_cfg(tmp_path / "a", seed=1), data_dir=tmp_path / "a")[0]
        b = run_tasks(_split_cfg(tmp_path / "b", seed=999), data_dir=tmp_path / "b")[0]
        assert a.to_dict() != b.to_dict()
```

- [ ] **Step 6.10: Run all workflow + NFR tests; adjust kwargs as needed**

```bash
cd /home/aweng/2033/dataeval-flow
uv run pytest verification/functional/workflows verification/nonfunctional/test_type_safety.py verification/nonfunctional/test_config_reproducibility.py -v
```
For each failure mentioning a missing pydantic field, open the corresponding `src/dataeval_flow/workflows/<wf>/params.py` and update the test kwargs. If a workflow's `params_schema` requires fields the test omits, supply minimal sensible values (e.g. `n_clusters=2`, `n_splits=2`).

- [ ] **Step 6.11: Commit**

```bash
cd /home/aweng/2033/dataeval-flow
git add verification/functional/workflows verification/nonfunctional/test_type_safety.py verification/nonfunctional/test_config_reproducibility.py
git commit -m "test(verification): TC-7-1..13-1 workflows + NFR-3 type safety + NFR-4 reproducibility"
```

---

## Task 7: TC-16-1, TC-17-1 — CLI, logging, docker

**Files:**
- Create: `dataeval-flow/verification/functional/cli/test_cli_entrypoints.py`
- Create: `dataeval-flow/verification/functional/docker/test_dockerfile_present.py`
- Create: `dataeval-flow/verification/nonfunctional/test_logging.py`

- [ ] **Step 7.1: Write `functional/cli/test_cli_entrypoints.py`**

```python
"""TC-16-1 — CLI entrypoints (headless, app, config)."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest


@pytest.mark.test_case("16-1")
class TestCLIEntrypoints:
    def test_console_script_on_path(self) -> None:
        assert shutil.which("dataeval-flow") is not None

    def test_module_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "dataeval" in result.stdout.lower()

    def test_app_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "app", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_config_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "config", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
```

- [ ] **Step 7.2: Write `nonfunctional/test_logging.py`**

```python
"""TC-16-1 (NFR-5) — stdlib logging integration."""

from __future__ import annotations

import logging

import pytest


@pytest.mark.test_case("16-1")
class TestLogging:
    def test_dataeval_flow_logger_uses_stdlib(self) -> None:
        from dataeval_flow import _logging

        assert hasattr(_logging, "configure_logging") or hasattr(_logging, "logger")

    def test_module_logger_is_logging_logger(self) -> None:
        import dataeval_flow.runner as runner

        assert isinstance(runner.logger, logging.Logger)
```

- [ ] **Step 7.3: Write `functional/docker/test_dockerfile_present.py`**

```python
"""TC-17-1 — Docker artifacts present and lint-clean."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"


@pytest.mark.test_case("17-1")
class TestDockerfiles:
    @pytest.mark.parametrize("name", ["Dockerfile.cpu", "Dockerfile.cu118"])
    def test_dockerfile_exists(self, name: str) -> None:
        assert (DOCKER_DIR / name).is_file(), f"missing docker/{name}"

    @pytest.mark.parametrize("name", ["Dockerfile.cpu", "Dockerfile.cu118"])
    def test_dockerfile_starts_with_from(self, name: str) -> None:
        first_lines = [
            line for line in (DOCKER_DIR / name).read_text().splitlines() if line.strip() and not line.startswith("#")
        ]
        assert first_lines[0].upper().startswith("FROM"), f"docker/{name} does not start with FROM"

    @pytest.mark.parametrize("name", ["Dockerfile.cpu", "Dockerfile.cu118"])
    def test_dockerfile_has_nonroot_user(self, name: str) -> None:
        body = (DOCKER_DIR / name).read_text()
        assert "USER " in body, f"docker/{name} does not declare a USER (non-root expected)"
```

- [ ] **Step 7.4: Run the new tests**

```bash
cd /home/aweng/2033/dataeval-flow
uv run pytest verification/functional/cli verification/functional/docker verification/nonfunctional/test_logging.py -v
```
Expected: all PASS.

- [ ] **Step 7.5: Commit**

```bash
cd /home/aweng/2033/dataeval-flow
git add verification/functional/cli verification/functional/docker verification/nonfunctional/test_logging.py
git commit -m "test(verification): TC-16-1 CLI + logging, TC-17-1 Docker artifacts"
```

---

## Task 8: Metarepo requirement documents

**Files:**
- Create: `metarepo/DataEval-Flow/requirements/FR-1-installation-compatibility.md` … `FR-18-result-metadata-envelope.md`
- Create: `metarepo/DataEval-Flow/requirements/NFR-1-python-compatibility.md` … `NFR-6-test-coverage.md`

Each file uses the same template (see `metarepo/DataEval/requirements/FR-1-installation-compatibility.md` for the canonical form):

```markdown
# <ID>: <Name>

## Functional Requirements         (or "Non-Functional Requirements" for NFRs)

- Requirement ID: <ID>
  - Name: <Name>
  - Description: <one-paragraph description>
  - Acceptance Criteria
    - <bullet 1>
    - <bullet 2>
    - ...
```

- [ ] **Step 8.1: Make the requirements directory**

```bash
mkdir -p /home/aweng/2033/metarepo/DataEval-Flow/requirements
mkdir -p /home/aweng/2033/metarepo/DataEval-Flow/assessments
mkdir -p /home/aweng/2033/metarepo/DataEval-Flow/product_reviews
```

- [ ] **Step 8.2: Write `FR-1-installation-compatibility.md`**

```markdown
# FR-1: Installation & Cross-Environment Compatibility

## Functional Requirements

- Requirement ID: FR-1
  - Name: Installation & Cross-Environment Compatibility
  - Description: The dataeval-flow package shall install and function correctly across all supported Python versions (3.10–3.13) and package managers (pip, uv), and across CPU and CUDA hardware configurations via the documented optional extras.
  - Acceptance Criteria
    - Installs successfully via `pip install dataeval-flow` on Python 3.10, 3.11, 3.12, and 3.13.
    - Installs successfully via `uv pip install dataeval-flow` on all supported Python versions.
    - Installs from source via `pip install -e .` and `uv pip install -e .`.
    - All optional extras install without conflicts: `cpu`, `cu118`, `cu124`, `cu128`, `onnx`, `onnx-gpu`, `opencv`, `app`, and the `all-*` aggregate extras.
    - Mutually exclusive CUDA/CPU extras correctly conflict per `tool.uv.conflicts`.
    - Package operates in CUDA-present and CUDA-absent environments with appropriate fallback.
    - Package metadata (`__version__`, `Requires-Python`, `License`) is correctly set after installation.
    - `py.typed` marker is included for PEP 561 type checker support.
```

- [ ] **Step 8.3: Write `FR-2-configuration-management.md`**

```markdown
# FR-2: Configuration Management

## Functional Requirements

- Requirement ID: FR-2
  - Name: Configuration Management
  - Description: The package shall load, validate, and merge pipeline configuration from YAML and JSON files, both as single files and as folders of fragments, producing a typed `PipelineConfig`.
  - Acceptance Criteria
    - `load_config(path)` parses a YAML or JSON file into a `PipelineConfig`.
    - `load_config_folder(path)` discovers and merges all YAML/JSON files at the folder root.
    - Invalid configuration raises a `pydantic.ValidationError` with a clear path to the offending field.
    - `export_params_schema()` returns the JSON-schema representation of `PipelineConfig`.
    - Config field paths (`sources[*].dataset.path`, `tasks[*].workflow.params.*`) are resolved relative to the data root passed at runtime.
```

- [ ] **Step 8.4: Write `FR-3-dataset-loading.md`**

```markdown
# FR-3: Dataset Loading

## Functional Requirements

- Requirement ID: FR-3
  - Name: Dataset Loading
  - Description: The package shall load datasets from multiple structured formats and from arbitrary user-defined factories that satisfy the MAITE protocol.
  - Acceptance Criteria
    - `ImageFolderDatasetConfig` loads a directory of class-subfolders containing images.
    - `HuggingFaceDatasetConfig` loads a dataset by repo id (optionally with `split` / `config`).
    - `CocoDatasetConfig` loads images + annotations following the COCO JSON layout.
    - `YoloDatasetConfig` loads a YOLO-style directory layout.
    - `DatasetProtocolConfig` instantiates a user-provided factory via a dotted-path target.
    - `load_dataset()` returns an object satisfying `dataeval.protocols.AnnotatedDataset`.
```

- [ ] **Step 8.5: Write `FR-4-feature-extraction.md`**

```markdown
# FR-4: Feature Extraction Configuration

## Functional Requirements

- Requirement ID: FR-4
  - Name: Feature Extraction Configuration
  - Description: The package shall expose pydantic-validated configuration classes for every supported feature extractor and instantiate them through a unified extractor-config interface.
  - Acceptance Criteria
    - `FlattenExtractorConfig` constructs an extractor with no external dependencies.
    - `TorchExtractorConfig` constructs an extractor from a `torch`/`torchvision` model target.
    - `OnnxExtractorConfig` constructs an extractor from a local ONNX model path.
    - `BoVWExtractorConfig` constructs the bag-of-visual-words extractor (OpenCV-backed).
    - `UncertaintyExtractorConfig` constructs the uncertainty extractor.
    - All extractor config classes are exported from `dataeval_flow.__all__`.
```

- [ ] **Step 8.6: Write `FR-5-preprocessing-selection.md`**

```markdown
# FR-5: Preprocessing & Selection Pipeline

## Functional Requirements

- Requirement ID: FR-5
  - Name: Preprocessing & Selection Pipeline
  - Description: The package shall compose torchvision-style preprocessing transforms and `dataeval.selection` operators into a typed, reproducible pipeline applied to each source dataset.
  - Acceptance Criteria
    - `PreprocessorConfig` accepts an ordered list of `{target, params}` transform specs.
    - `SelectionStep` defines a single selection operation by dotted-path target plus params.
    - `SelectionConfig` stacks multiple `SelectionStep` entries in order.
    - A `SelectionConfig` containing a `Limit` step reduces dataset length to the specified size.
    - Pipeline composition is deterministic for a given config + seed.
```

- [ ] **Step 8.7: Write `FR-6-workflow-orchestration.md`**

```markdown
# FR-6: Workflow Orchestration

## Functional Requirements

- Requirement ID: FR-6
  - Name: Workflow Orchestration
  - Description: The package shall discover registered workflows, resolve their parameter schemas, and execute one or more tasks sequentially against a shared `WorkflowContext`.
  - Acceptance Criteria
    - `list_workflows()` returns the seven currently registered workflows by name + description.
    - `get_workflow(name)` returns an object satisfying `WorkflowProtocol`.
    - `get_workflow(name)` raises `ValueError` on unknown names.
    - `run_tasks(config, data_dir)` returns a list of `WorkflowResult` of equal length to the task list.
    - Each `WorkflowResult` carries timestamp, tool, and execution-time metadata populated by the orchestrator.
```

- [ ] **Step 8.8: Write `FR-7-cleaning-workflow.md` through `FR-13-parameter-sweep-workflow.md`**

For each of FR-7..FR-13, use the same shape: one paragraph description identifying the workflow, and 4–6 acceptance bullets covering (a) workflow registration, (b) execution success, (c) typed outputs, (d) result reporting, (e) error handling.

```markdown
# FR-7: Data Cleaning Workflow

## Functional Requirements

- Requirement ID: FR-7
  - Name: Data Cleaning Workflow
  - Description: The package shall provide a Data Cleaning workflow that runs duplicate-detection and outlier-detection over a single source and returns a typed report of findings.
  - Acceptance Criteria
    - `DataCleaningWorkflow` is registered as `data-cleaning`.
    - The workflow executes successfully against a synthetic ImageFolder source.
    - The workflow result exposes `outliers` and `duplicates` findings.
    - The workflow result renders a non-empty `report()` string.
    - The workflow surfaces errors via `WorkflowResult.errors` without raising at the orchestrator level.
```

```markdown
# FR-8: Drift Monitoring Workflow

## Functional Requirements

- Requirement ID: FR-8
  - Name: Drift Monitoring Workflow
  - Description: The package shall provide a Drift Monitoring workflow that compares a reference source against a target source using a configurable detector and returns a typed report.
  - Acceptance Criteria
    - `DriftMonitoringWorkflow` is registered as `drift-monitoring`.
    - The workflow accepts `reference` and `target` source names.
    - The workflow executes successfully when both sources resolve to valid `AnnotatedDataset`s.
    - The workflow result exposes per-feature drift findings.
    - The workflow result renders a non-empty `report()` string.
```

```markdown
# FR-9: OOD Detection Workflow

## Functional Requirements

- Requirement ID: FR-9
  - Name: OOD Detection Workflow
  - Description: The package shall provide an Out-of-Distribution Detection workflow that fits a detector on a reference source and scores a target source.
  - Acceptance Criteria
    - `OODDetectionWorkflow` is registered as `ood-detection`.
    - The workflow accepts `reference` and `target` source names.
    - The workflow result exposes an `instance_score` array of length equal to the target dataset.
    - The workflow result exposes a binary `is_ood` mask aligned with `instance_score`.
    - The workflow result renders a non-empty `report()` string.
```

```markdown
# FR-10: Data Analysis Workflow

## Functional Requirements

- Requirement ID: FR-10
  - Name: Data Analysis Workflow
  - Description: The package shall provide a Data Analysis workflow that runs a configurable battery of bias, quality, and distributional analyses across one or more sources.
  - Acceptance Criteria
    - `DataAnalysisWorkflow` is registered as `data-analysis`.
    - The workflow accepts one or more source names via its task params.
    - The workflow result exposes per-analysis sub-outputs (e.g. bias, quality).
    - The workflow result renders a non-empty `report()` string.
    - The workflow degrades gracefully when an individual sub-analysis is unavailable, recording the failure in `result.errors`.
```

```markdown
# FR-11: Splitting Workflow

## Functional Requirements

- Requirement ID: FR-11
  - Name: Splitting Workflow
  - Description: The package shall provide a Data Splitting workflow that partitions a source into one or more named splits using a seed-controlled procedure.
  - Acceptance Criteria
    - `DataSplittingWorkflow` is registered as `data-splitting`.
    - The workflow accepts a `seed` parameter that fully determines split assignments.
    - Two runs with the same seed produce identical split assignments.
    - Two runs with different seeds produce different split assignments.
    - The workflow result exposes one named index list per output split.
```

```markdown
# FR-12: Prioritization Workflow

## Functional Requirements

- Requirement ID: FR-12
  - Name: Prioritization Workflow
  - Description: The package shall provide a Data Prioritization workflow that ranks samples in a source by a configurable scoring policy.
  - Acceptance Criteria
    - `DataPrioritizationWorkflow` is registered as `data-prioritization`.
    - The workflow accepts a single source name and a scoring policy.
    - The workflow result exposes a ranked index list of length equal to the source dataset.
    - The workflow result renders a non-empty `report()` string.
    - The workflow surfaces unsupported policies via a clear pydantic `ValidationError`.
```

```markdown
# FR-13: Parameter Sweep Workflow

## Functional Requirements

- Requirement ID: FR-13
  - Name: Parameter Sweep Workflow
  - Description: The package shall provide a Parameter Sweep workflow that executes an inner workflow across a grid of parameter combinations and aggregates the results.
  - Acceptance Criteria
    - `ParameterSweepWorkflow` is registered as `parameter-sweep`.
    - The workflow accepts a parameter grid defined as a list of name/value sequences.
    - The workflow produces one inner result per parameter combination.
    - The workflow aggregates inner results into a single typed `WorkflowResult`.
    - The workflow surfaces per-combination errors without aborting the sweep.
```

- [ ] **Step 8.9: Write `FR-14-reporting-export.md`**

```markdown
# FR-14: Reporting & Export

## Functional Requirements

- Requirement ID: FR-14
  - Name: Reporting & Export
  - Description: The package shall expose human-readable text reports and machine-readable JSON/YAML exports for every `WorkflowResult`.
  - Acceptance Criteria
    - `WorkflowResult.report()` returns a non-empty, multi-line text report.
    - `WorkflowResult.report(detailed=False)` returns a shorter summary-only variant.
    - `WorkflowResult.export(path, fmt="json")` writes a JSON file and returns the destination path.
    - `WorkflowResult.export(path, fmt="yaml")` writes a YAML file and returns the destination path.
    - `WorkflowResult.export(None, fmt=...)` returns the serialized string in memory.
    - `WorkflowResult.to_dict()` returns a dict containing both `metadata` and the workflow's `data` payload.
```

- [ ] **Step 8.10: Write `FR-15-caching.md`**

```markdown
# FR-15: Caching

## Functional Requirements

- Requirement ID: FR-15
  - Name: Caching
  - Description: The package shall cache expensive intermediate artifacts (preprocessed images, embeddings, model outputs) keyed by a deterministic content hash so that repeated runs reuse prior computation.
  - Acceptance Criteria
    - A user-supplied cache root is created on first use and reused thereafter.
    - Cache keys are stable across runs for the same content.
    - A cache hit returns the previously stored payload byte-for-byte.
    - A cache miss returns `None` (or raises a documented error) without corrupting the cache.
    - Cache entries are isolated per source so cross-source collisions cannot occur.
```

- [ ] **Step 8.11: Write `FR-16-cli-interface.md`**

```markdown
# FR-16: CLI Interface

## Functional Requirements

- Requirement ID: FR-16
  - Name: CLI Interface
  - Description: The package shall provide three CLI entrypoints: a headless executor for automation, an interactive TUI dashboard, and a simple config builder.
  - Acceptance Criteria
    - `dataeval-flow` console script is installed on `PATH`.
    - `python -m dataeval_flow --help` exits 0 and prints headless-mode usage.
    - `python -m dataeval_flow app --help` exits 0 and prints TUI usage (when the `app` extra is installed).
    - `python -m dataeval_flow config --help` exits 0 and prints config-builder usage.
    - The headless mode accepts `--config`, `--data`, and `--output` flags.
```

- [ ] **Step 8.12: Write `FR-17-docker-containerization.md`**

```markdown
# FR-17: Docker Containerization

## Functional Requirements

- Requirement ID: FR-17
  - Name: Docker Containerization
  - Description: The package shall ship Dockerfiles producing CPU-only and CUDA 11.8 images that run the headless executor against bind-mounted data and output volumes, as a non-root user.
  - Acceptance Criteria
    - `docker/Dockerfile.cpu` and `docker/Dockerfile.cu118` are present and valid Dockerfiles.
    - Both Dockerfiles declare a non-root `USER` (uid 1000).
    - Both images expose `/dataeval` (ro) and `/output` (rw) as volume mount points.
    - The default `CMD` invokes `dataeval-flow --help`.
    - The CUDA image accepts `--gpus all` and operates without GPU when invoked without it.
```

- [ ] **Step 8.13: Write `FR-18-result-metadata-envelope.md`**

```markdown
# FR-18: Result Metadata Envelope

## Functional Requirements

- Requirement ID: FR-18
  - Name: Result Metadata Envelope
  - Description: Every `WorkflowResult` shall carry a JATIC-required metadata envelope (per IR-3-H-12) describing the tool, configuration, dataset identifiers, and timing of the run.
  - Acceptance Criteria
    - `metadata.version` is set to a non-empty schema version string.
    - `metadata.timestamp` is a timezone-aware UTC datetime.
    - `metadata.tool` is `"dataeval-flow"`.
    - `metadata.tool_version` is set to `dataeval_flow.__version__`.
    - `metadata.resolved_config` is a non-empty dict containing the fully resolved configuration applied to the run.
    - `metadata.execution_time_s` is a non-negative float.
    - `metadata.dataset_id` is set to the resolved source name (or list of names for multi-source workflows).
```

- [ ] **Step 8.14: Write `NFR-1-python-compatibility.md`**

```markdown
# NFR-1: Python Version Compatibility

## Non-Functional Requirements

- Requirement ID: NFR-1
  - Name: Python Version Compatibility
  - Description: The package shall run on every Python version listed in `pyproject.toml`'s `requires-python` range with no version-conditional behavior at the public API.
  - Acceptance Criteria
    - Successful import on Python 3.10, 3.11, 3.12, and 3.13.
    - No version-gated public symbols.
    - Type annotations remain importable under the current interpreter's `typing` runtime.
```

- [ ] **Step 8.15: Write `NFR-2-packaging-distribution.md`**

```markdown
# NFR-2: Packaging & Distribution

## Non-Functional Requirements

- Requirement ID: NFR-2
  - Name: Packaging & Distribution
  - Description: The published wheel shall conform to PEP 517/518/561 and shall not contain test or verification files.
  - Acceptance Criteria
    - Wheel installs cleanly under pip and uv.
    - `py.typed` marker is present in the installed package.
    - Neither `tests/` nor `verification/` directories ship inside the wheel.
    - License is declared as MIT in the wheel metadata.
    - `Requires-Python` matches `>=3.10,<3.14`.
```

- [ ] **Step 8.16: Write `NFR-3-type-safety.md`**

```markdown
# NFR-3: Type Safety

## Non-Functional Requirements

- Requirement ID: NFR-3
  - Name: Type Safety
  - Description: The package shall be type-checked under strict pyright and shall expose typed protocols for all downstream-facing interfaces.
  - Acceptance Criteria
    - `pyright src/dataeval_flow` passes with zero errors.
    - `py.typed` marker is present.
    - `WorkflowProtocol` is `runtime_checkable` and behaves as a structural type at runtime.
    - `dataeval_flow.__all__` is defined and lists every public symbol.
```

- [ ] **Step 8.17: Write `NFR-4-configuration-reproducibility.md`**

```markdown
# NFR-4: Configuration Reproducibility

## Non-Functional Requirements

- Requirement ID: NFR-4
  - Name: Configuration Reproducibility
  - Description: Identical configuration + seed shall produce identical workflow outputs (modulo non-deterministic external dependencies such as `dataeval` internals).
  - Acceptance Criteria
    - Splitting workflow produces identical splits for identical seeds.
    - Splitting workflow produces different splits for different seeds.
    - `WorkflowResult.metadata.resolved_config` captures the full effective configuration.
```

- [ ] **Step 8.18: Write `NFR-5-logging-integration.md`**

```markdown
# NFR-5: Logging Integration

## Non-Functional Requirements

- Requirement ID: NFR-5
  - Name: Logging Integration
  - Description: The package shall log through Python's standard `logging` module, never to `print`, and shall not configure root-logger handlers when imported as a library.
  - Acceptance Criteria
    - Every module logger is an instance of `logging.Logger`.
    - The headless CLI configures a default stream handler at startup.
    - Importing `dataeval_flow` does not attach handlers to the root logger.
    - Log records carry the module logger name (`dataeval_flow.*`).
```

- [ ] **Step 8.19: Write `NFR-6-test-coverage.md`**

```markdown
# NFR-6: Test Coverage

## Non-Functional Requirements

- Requirement ID: NFR-6
  - Name: Test Coverage
  - Description: The test suite shall cover at least 90% of source lines, with a target of 100% for newly added and modified files.
  - Acceptance Criteria
    - `nox -s test` reports total branch+line coverage ≥ 90%.
    - Coverage configuration excludes only `raise NotImplementedError`, `if TYPE_CHECKING`, and protocol `...` stubs.
    - New or modified files added since the last release achieve ≥ 90% coverage; the target is 100%.
```

- [ ] **Step 8.20: Commit**

```bash
cd /home/aweng/2033/metarepo
git add DataEval-Flow/requirements DataEval-Flow/assessments DataEval-Flow/product_reviews
git commit -m "feat(DataEval-Flow): seed FR-1..18 and NFR-1..6 requirement documents"
```

---

## Task 9: Metarepo product folder — README, generate artifacts, link from top-level index

**Files:**
- Create: `metarepo/DataEval-Flow/README.md`
- Modify: `metarepo/README.md` (add link to DataEval-Flow under "Products")
- Generated and committed: `metarepo/DataEval-Flow/test-cases/*.md`, `metarepo/DataEval-Flow/vcrm.md`

- [ ] **Step 9.1: Write `metarepo/DataEval-Flow/README.md`**

```markdown
<!-- This is the name of the product -->
# DataEval Flow

<!-- This is the product tagline -->
## Tagline

DataEval Flow provides workflow orchestration for DataEval evaluators, packaging
data cleaning, drift monitoring, OOD detection, analysis, splitting, prioritization,
and parameter-sweep pipelines behind a single declarative configuration format and
both headless and interactive CLIs.

<!-- This is the product needs statement -->
## Product Needs Statement

DataEval Flow lets T&E engineers compose and run multi-stage data evaluation
pipelines without writing Python glue code.  Pipelines are described in YAML or JSON,
executed locally or in a CUDA-enabled container, and produce both human-readable
reports and machine-readable result envelopes that satisfy JATIC interoperability
requirements.

## Product Maturity

<!-- Describe what was evaluated, the current level and the date the program approved it-->

Version Evaluated: dataeval-flow-0.1.0

Maturity Level: Level-0 Candidate

Date of Last Program Assessment: N/A

Latest Assessment: N/A

<!-- Where to find it internally -->
JATIC GitLab Repository: <https://gitlab.jatic.net/jatic/aria/dataeval-flow>

## Development and Release Practices

### Reporting and Issue Management

DataEval Flow is developed using Agile project management managed in the JATIC
Gitlab repository linked above.

### Branch Strategy

DataEval Flow uses the `GitLab Flow with Release Branches` strategy.

### Release Strategy

Releases are versioned using [Semantic Versioning](https://semver.org/).

## Public Presence

- <https://gitlab.jatic.net/jatic/aria/dataeval-flow>
```

- [ ] **Step 9.2: Run the full verification suite to produce a real report**

```bash
cd /home/aweng/2033/dataeval-flow
uv run pytest verification/ -v
```
Expected: most tests PASS; any that fail must be investigated before continuing — do not commit Pending stubs over real fail results.

- [ ] **Step 9.3: Run the metarepo generator**

```bash
cd /home/aweng/2033/dataeval-flow
uv run python verification/generate_metarepo.py
```
Expected output:
```
Loaded verification report: 19 test cases (... passed, ... failed, ... skipped)
  Generated test-case-1-1.md
  ... (18 more)
  Generated vcrm.md
All artifacts written to /home/aweng/2033/dataeval-flow/output/metarepo
```

- [ ] **Step 9.4: Copy the generated artifacts into the metarepo**

```bash
mkdir -p /home/aweng/2033/metarepo/DataEval-Flow/test-cases
cp /home/aweng/2033/dataeval-flow/output/metarepo/test-cases/*.md /home/aweng/2033/metarepo/DataEval-Flow/test-cases/
cp /home/aweng/2033/dataeval-flow/output/metarepo/vcrm.md /home/aweng/2033/metarepo/DataEval-Flow/vcrm.md
```

- [ ] **Step 9.5: Update the top-level `metarepo/README.md`**

Open `/home/aweng/2033/metarepo/README.md`. Locate the `## Products` section (line 10–13) and add a new bullet after `- [DataEval](./DataEval/)`:

```markdown
- [DataEval Flow](./DataEval-Flow/)
```

- [ ] **Step 9.6: Commit the metarepo additions**

```bash
cd /home/aweng/2033/metarepo
git add DataEval-Flow/README.md DataEval-Flow/test-cases DataEval-Flow/vcrm.md README.md
git commit -m "feat(DataEval-Flow): add product README, generated test cases, VCRM"
```

- [ ] **Step 9.7: Commit the dataeval-flow verification output reference**

```bash
cd /home/aweng/2033/dataeval-flow
echo "output/" >> .gitignore 2>/dev/null || true
git add .gitignore 2>/dev/null || true
git status
# If .gitignore was modified, commit it; otherwise this is a no-op.
git diff --cached --quiet || git commit -m "chore: ignore verification output artifacts"
```

---

## Task 10: Final validation — full sweep + nox lint/type/test

**No file changes** — this task verifies that everything still hangs together.

- [ ] **Step 10.1: Full verification sweep**

```bash
cd /home/aweng/2033/dataeval-flow
uv run pytest verification/ -v --tb=short
```
Expected: same green-or-skipped state as Step 9.2.

- [ ] **Step 10.2: Lint + type + test from `noxfile.py`**

```bash
cd /home/aweng/2033/dataeval-flow
uv run nox -s lint type test
```
Expected: all three sessions pass. If `lint` flags a verification test for a missing docstring or annotation, fix it locally — the existing `per-file-ignores` for `tests/*` does not cover `verification/*`, so prefer extending that ignore in `pyproject.toml` (`"verification/*"` entry mirroring `"tests/*"`) over weakening the verification tests themselves.

- [ ] **Step 10.3: Regenerate metarepo artifacts one final time and diff**

```bash
cd /home/aweng/2033/dataeval-flow
uv run python verification/generate_metarepo.py
diff -r output/metarepo/test-cases /home/aweng/2033/metarepo/DataEval-Flow/test-cases || true
diff output/metarepo/vcrm.md /home/aweng/2033/metarepo/DataEval-Flow/vcrm.md || true
```
If diffs are non-empty (other than the `**Last Updated:**` date), re-copy as in Step 9.4 and commit.

- [ ] **Step 10.4: Final summary commit (if any cleanup happened)**

```bash
cd /home/aweng/2033/dataeval-flow && git status
cd /home/aweng/2033/metarepo && git status
```
Both should be clean. If not, stage and commit any pyproject.toml ignore additions or regenerated artifacts with a descriptive message.

---

## Self-Review Checklist (run after Task 10)

- [ ] Every FR (1–18) and NFR (1–6) has a requirement markdown file in `metarepo/DataEval-Flow/requirements/`.
- [ ] Every test case in `registry.yaml` has at least one `@pytest.mark.test_case` marker in the verification tree pointing at it.
- [ ] `verification/generate_metarepo.py` produces `vcrm.md` with all 23 requirement rows and 19 test-case columns.
- [ ] `metarepo/README.md` lists DataEval-Flow under "Products".
- [ ] `pyproject.toml` `testpaths` includes both `tests` and `verification`.
- [ ] `output/` is gitignored in dataeval-flow.
