"""Generate Dockerfile.<variant> files from docker/Dockerfile.j2 template."""

from pathlib import Path

import tomllib
import yaml
from jinja2 import Environment, FileSystemLoader

root = Path(__file__).resolve().parent.parent
config = yaml.safe_load((root / "docker" / "variants.yaml").read_text())

# Read project version from pyproject.toml
pyproject = tomllib.loads((root / "pyproject.toml").read_text())
version = pyproject["project"]["version"]

env = Environment(
    loader=FileSystemLoader(root / "docker"),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=True,
)
template = env.get_template("Dockerfile.j2")

uv_version = config["uv_version"]
python_version = config["python_version"]

# Pre-render the base_image values (they may reference uv_version/python_version)
base_env = Environment(autoescape=True)

for name, variant in config["variants"].items():
    build_base_image = base_env.from_string(variant["build_base_image"]).render(
        uv_version=uv_version,
        python_version=python_version,
    )
    prod_base_image = base_env.from_string(variant["prod_base_image"]).render(
        uv_version=uv_version,
        python_version=python_version,
    )
    extras_flags = " ".join(f"--extra {e}" for e in variant["extras"])

    rendered = template.render(
        variant_name=name,
        build_base_image=build_base_image,
        prod_base_image=prod_base_image,
        uv_version=uv_version,
        python_version=python_version,
        extras_flags=extras_flags,
        label_title=variant["label_title"],
        label_description=variant["label_description"],
        version=version,
        security_patches=variant.get("security_patches", []),
    )

    out = root / f"Dockerfile.{name}"
    out.write_text(rendered)
    print(f"Generated {out}")
