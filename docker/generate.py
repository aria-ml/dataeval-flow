"""Generate Dockerfile.<variant> files from docker/Dockerfile.j2 template."""

import re
import subprocess
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

root = Path(__file__).resolve().parent.parent
config = yaml.safe_load((root / "docker" / "variants.yaml").read_text())


def _default_version() -> str:
    """Resolve the last published tag for the `DATAEVAL_FLOW_VERSION` build-arg default.

    `--abbrev=0` returns the most recent tag *without* the `-N-g<sha>[-dirty]`
    suffix that `git describe` normally appends. Using the bare tag keeps the
    committed Dockerfile.<variant> defaults stable across regenerations — the
    rendered ARG only churns when an actual release tag lands, not when a
    contributor regenerates from a dirty working tree.

    This default is only consumed by local `docker build` invocations that omit
    `--build-arg`; CI release builds always pass an explicit version derived
    from `git describe`, which becomes the source of truth in the published
    image. See README "Versioning" for details.
    """
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
        return "unknown"
    return re.sub(r"^v", "", tag)


version = _default_version()

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

    out = root / "docker" / f"Dockerfile.{name}"
    out.write_text(rendered)
    print(f"Generated {out}")
