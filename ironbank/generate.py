"""Generate Iron Bank variant directories, Dockerfiles, and manifests from templates."""

import re
import shutil
import subprocess
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

root = Path(__file__).resolve().parent.parent
ironbank_dir = root / "ironbank"
config = yaml.safe_load((ironbank_dir / "variants.yaml").read_text())


def _default_version() -> str:
    """Resolve the last published tag for version default."""
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"],  # noqa: S607
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0.2.2"
    return re.sub(r"^v", "", tag)


version = _default_version()

env = Environment(
    loader=FileSystemLoader(ironbank_dir),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=select_autoescape(default=False, default_for_string=False),
)

dockerfile_tmpl = env.get_template("Dockerfile.j2")
manifest_tmpl = env.get_template("hardening_manifest.yaml.j2")
readme_tmpl = env.get_template("README.md.j2")

repository_name = config.get("repository_name", "dataeval/dataeval-flow")
release_url_prefix = config.get("release_url_prefix", "https://github.com/aria-ml/dataeval-flow/releases/download")
maintainers = config.get("maintainers", [])

for name, variant in config["variants"].items():
    variant_dir = ironbank_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Check if a wheelhouse sha256 file already exists
    sha_file = root / "dist" / "ironbank" / name / f"dataeval-flow-{version}-wheels-{name}.tar.gz.sha256"
    if sha_file.exists():
        wheelhouse_sha256 = sha_file.read_text().strip().split()[0]
    else:
        wheelhouse_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"

    context = {
        "variant_name": name,
        "base_image": variant["base_image"],
        "base_tag": variant["base_tag"],
        "label_title": variant["label_title"],
        "label_description": variant["label_description"],
        "mode": variant.get("mode", "cpu"),
        "version": version,
        "repository_name": repository_name,
        "release_url_prefix": release_url_prefix,
        "maintainers": maintainers,
        "wheelhouse_sha256": wheelhouse_sha256,
    }

    # Render Dockerfile
    (variant_dir / "Dockerfile").write_text(dockerfile_tmpl.render(**context))

    # Render hardening_manifest.yaml
    (variant_dir / "hardening_manifest.yaml").write_text(manifest_tmpl.render(**context))

    # Render README.md
    (variant_dir / "README.md").write_text(readme_tmpl.render(**context))

    # Copy LICENSE, entrypoint.sh, and the locked+hashed dependency closure the
    # wheelhouse was built against (pins the same `dataeval @ git+...` commit
    # the wheelhouse's dataeval wheel was built from, so the Dockerfile can
    # install it via --find-links instead of pip re-resolving dataeval-flow's
    # own unpinned `@main` reference over the network).
    if (root / "LICENSE").exists():
        shutil.copy2(root / "LICENSE", variant_dir / "LICENSE")
    if (root / "docker" / "entrypoint.sh").exists():
        shutil.copy2(root / "docker" / "entrypoint.sh", variant_dir / "entrypoint.sh")
    requirements_file = root / f"requirements.{name}.txt"
    if requirements_file.exists():
        shutil.copy2(requirements_file, variant_dir / "requirements.txt")

    print(f"Generated Iron Bank bundle in {variant_dir}")
