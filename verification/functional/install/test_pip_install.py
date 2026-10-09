"""TC-1-1 — installation, the command line entry points, and optional extras."""

from __future__ import annotations

import importlib
import pkgutil
import shutil
import subprocess
import sys
import tomllib
from importlib.metadata import distribution
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from verification.helpers import REPO_ROOT, run_cli

# Every extra the package declares, and what each one adds (requirement names, lower case).
EXTRAS: dict[str, set[str]] = {
    "cpu": {"torch", "torchvision"},
    "cu126": {"torch", "torchvision"},
    "cu130": {"torch", "torchvision"},
    "onnx": {"onnx", "onnxruntime"},
    "onnx-cu126": {"onnx", "onnxruntime-gpu"},
    "onnx-cu130": {"onnx", "onnxruntime-gpu"},
    "opencv": {"opencv-python-headless"},
    "app": {"textual"},
    "service": {"fastapi", "uvicorn"},
    "ontology": {"dataeval"},
}


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())


def _requirement_names(extra: str) -> set[str]:
    return {Requirement(spec).name.lower() for spec in _pyproject()["project"]["optional-dependencies"][extra]}


@pytest.mark.required
class TestPipInstall:
    def test_import_dataeval_flow(self) -> None:
        import dataeval_flow

        assert dataeval_flow is not None

    def test_version_is_set(self) -> None:
        import dataeval_flow

        assert dataeval_flow.__version__
        assert dataeval_flow.__version__ != "unknown"

    def test_every_public_module_imports_without_optional_extras_being_needed(self) -> None:
        import dataeval_flow

        names = [
            info.name
            for info in pkgutil.walk_packages(dataeval_flow.__path__, "dataeval_flow.")
            if not any(part.startswith("_") for part in info.name.split(".")[1:])
        ]

        assert {
            "dataeval_flow.config",
            "dataeval_flow.workflows",
            "dataeval_flow.evaluators",
            "dataeval_flow.steps",
        } <= (set(names))
        for name in names:
            assert importlib.import_module(name) is not None

    def test_console_script_is_declared_and_runs(self) -> None:
        scripts = [ep for ep in distribution("dataeval-flow").entry_points if ep.group == "console_scripts"]
        assert {ep.name: ep.value for ep in scripts} == {"dataeval-flow": "dataeval_flow.__main__:main"}

        script = shutil.which("dataeval-flow", path=str(Path(sys.executable).parent))
        assert script is not None, "the dataeval-flow console script is not installed next to this interpreter"
        run = subprocess.run([script, "--version"], capture_output=True, text=True, check=False)  # noqa: S603

        import dataeval_flow

        assert run.returncode == 0, run.stderr
        assert run.stdout.strip() == f"dataeval-flow {dataeval_flow.__version__}"

    def test_python_dash_m_runs_and_reports_the_same_version(self) -> None:
        import dataeval_flow

        run = run_cli("--version")

        assert run.returncode == 0, run.stderr
        assert run.stdout.strip() == f"dataeval-flow {dataeval_flow.__version__}"


@pytest.mark.required
class TestDeclaredExtras:
    def test_the_declared_extras_are_exactly_the_documented_ones(self) -> None:
        declared = set(_pyproject()["project"]["optional-dependencies"])

        assert declared == set(EXTRAS)
        assert set(distribution("dataeval-flow").metadata.get_all("Provides-Extra")) == set(EXTRAS)

    @pytest.mark.parametrize("extra", sorted(EXTRAS))
    def test_each_extra_adds_its_packages(self, extra: str) -> None:
        assert _requirement_names(extra) == EXTRAS[extra]

    def test_the_cuda_and_cpu_torch_extras_exclude_each_other_and_so_do_the_onnx_extras(self) -> None:
        conflicts = [{entry["extra"] for entry in group} for group in _pyproject()["tool"]["uv"]["conflicts"]]

        assert {"cpu", "cu126", "cu130"} in conflicts
        assert {"onnx", "onnx-cu126", "onnx-cu130"} in conflicts

    def test_the_cuda_onnx_extras_drop_their_runtime_when_both_are_requested(self) -> None:
        """Under pip, which does not enforce the exclusion, asking for both leaves no conflicting pin behind."""
        for extra, other in (("onnx-cu126", "onnx-cu130"), ("onnx-cu130", "onnx-cu126")):
            specs = [Requirement(s) for s in _pyproject()["project"]["optional-dependencies"][extra]]
            runtime = [s for s in specs if s.name == "onnxruntime-gpu"]

            assert runtime
            for spec in runtime:
                assert spec.marker is not None
                # python_version is pinned: some pins apply only on newer interpreters than the one running.
                assert not spec.marker.evaluate({"extra": other, "python_version": "3.14"})
                assert spec.marker.evaluate({"extra": extra, "python_version": "3.14"})

    def test_the_container_variants_install_only_declared_extras(self) -> None:
        import yaml

        variants = yaml.safe_load((REPO_ROOT / "docker" / "variants.yaml").read_text())["variants"]

        assert set(variants) == {"cpu", "cu126", "cu130"}
        for name, variant in variants.items():
            assert set(variant["extras"]) <= set(EXTRAS), name
            assert "service" in variant["extras"], name


class TestOptionalExtras:
    @pytest.mark.optional
    @pytest.mark.parametrize(
        ("module", "label"),
        [
            ("torch", "cpu/cu*"),
            ("torchvision", "cpu/cu*"),
            ("onnx", "onnx"),
            ("onnxruntime", "onnx"),
            ("cv2", "opencv"),
            ("textual", "app"),
            ("fastapi", "service"),
            ("uvicorn", "service"),
            ("rdflib", "ontology"),
        ],
    )
    def test_optional_module_importable_if_installed(self, module: str, label: str) -> None:
        mod = pytest.importorskip(module, reason=f"{label} extra not installed")
        assert mod is not None

    @pytest.mark.required
    def test_serve_without_the_service_extra_exits_one_and_names_the_extra(self, tmp_path: Path) -> None:
        code = (
            "import sys\n"
            "sys.modules['fastapi'] = None\n"
            "sys.modules['uvicorn'] = None\n"
            f"sys.argv = ['dataeval_flow', 'serve', '--data', {str(tmp_path)!r}, '--output', {str(tmp_path)!r}]\n"
            "from dataeval_flow.__main__ import main\n"
            "main()\n"
        )

        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603

        assert run.returncode == 1, run.stdout + run.stderr
        assert "requires the 'service' extra" in run.stderr
        assert "pip install dataeval-flow[service]" in run.stderr

    @pytest.mark.required
    def test_app_without_the_app_extra_exits_one_and_names_the_extra(self) -> None:
        code = (
            "import sys\n"
            "sys.modules['textual'] = None\n"
            "sys.argv = ['dataeval_flow', 'app']\n"
            "from dataeval_flow.__main__ import main\n"
            "main()\n"
        )

        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603

        assert run.returncode == 1, run.stdout + run.stderr
        assert "requires the 'app' extra" in run.stdout
        assert "pip install dataeval-flow[app]" in run.stdout

    @pytest.mark.required
    def test_reading_an_ontology_file_without_the_ontology_extra_names_the_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataeval_flow.workflows._ontology import OntologyLoadError, load_ontology

        path = tmp_path / "taxonomy.ttl"
        path.write_text(
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n"
            "@prefix ex: <http://example.org/> .\n"
            'ex:vehicle a skos:Concept ; skos:prefLabel "vehicle" .\n'
            'ex:car a skos:Concept ; skos:prefLabel "car" ; skos:broader ex:vehicle .\n'
        )
        monkeypatch.setitem(sys.modules, "rdflib", None)

        with pytest.raises(OntologyLoadError, match=r"dataeval\[ontology\]"):
            load_ontology(str(path))

    @pytest.mark.optional
    def test_reading_an_ontology_file_works_with_the_ontology_extra(self, tmp_path: Path) -> None:
        pytest.importorskip("rdflib", reason="ontology extra not installed")
        from dataeval_flow.workflows._ontology import load_ontology

        path = tmp_path / "taxonomy.ttl"
        path.write_text(
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n"
            "@prefix ex: <http://example.org/> .\n"
            'ex:vehicle a skos:Concept ; skos:prefLabel "vehicle" .\n'
            'ex:car a skos:Concept ; skos:prefLabel "car" ; skos:broader ex:vehicle .\n'
        )

        ontology, source = load_ontology(str(path))

        assert source == str(path)
        assert set(ontology.leaves) == {"http://example.org/car"}
