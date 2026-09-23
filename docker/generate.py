"""Generate Dockerfile.<variant> files from docker/Dockerfile.j2 template."""

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

root = Path(__file__).resolve().parent.parent
config = yaml.safe_load((root / "docker" / "variants.yaml").read_text())


# Fallback for the `DATAEVAL_FLOW_VERSION` build arg, used only by a local
# `docker build` that omits `--build-arg`. CI always passes an explicit version
# from docker/resolve-version.sh, and that is what a published image carries.
#
# A fixed string rather than the last git tag, so rendering is a pure function of
# Dockerfile.j2 and variants.yaml. Deriving it from `git describe` made the
# output depend on surrounding repository state: CI clones shallow and without
# tags, so it rendered "unknown", which is not valid PEP 440; and the value
# differed per branch, so cherry-picking container changes between main and a
# release branch always carried a spurious diff.
PLACEHOLDER_VERSION = "0.0.0.dev0"


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
    base_image = base_env.from_string(variant["base_image"]).render(
        uv_version=uv_version,
        python_version=python_version,
    )
    extras_flags = " ".join(f"--extra {e}" for e in variant["extras"])

    rendered = template.render(
        variant_name=name,
        base_image=base_image,
        uv_version=uv_version,
        python_version=python_version,
        extras_flags=extras_flags,
        label_title=variant["label_title"],
        label_description=variant["label_description"],
        version=PLACEHOLDER_VERSION,
        security_patches=variant.get("security_patches", []),
    )

    out = root / "docker" / f"Dockerfile.{name}"
    out.write_text(rendered)
    print(f"Generated {out}")
