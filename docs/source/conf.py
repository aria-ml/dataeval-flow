"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import datetime
from typing import Any

# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

project = "DataEval Workflows"
copyright = f"{datetime.date.today().year}, ARiA"  # noqa: A001
author = "ARiA"

root_doc = "index"
language = "en"

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_nb",
    "sphinx_design",
    "sphinx_immaterial",
    "sphinx_new_tab_link",
]

source_suffix = [".rst", ".md"]

nb_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "build",
    "notebooks/*.ipynb",
    "cache",
    "data",
    "models",
    "conf.py",
]

suppress_warnings = ["ref.python", "autoapi.python_import_resolution"]

# -----------------------------------------------------------------------------
# AutoAPI settings
# -----------------------------------------------------------------------------

autoapi_dirs = ["../../src/dataeval_flow/"]
autoapi_type = "python"
autoapi_root = "reference/autoapi"
autoapi_file_pattern = "*.py"
autoapi_python_class_content = "class"
autoapi_options = [
    "members",
    "show-module-summary",
    "imported-members",
    "inherited-members",
]
autoapi_generate_api_docs = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autoapi_own_page_level = "function"
autoapi_member_order = "groupwise"
autoapi_add_toctree_entry = False

# -----------------------------------------------------------------------------
# MyST settings
# -----------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "html_admonition",
]
myst_heading_anchors = 4

nb_execution_mode = "auto"  # "off" to disable execution, "auto" to execute only if outputs are missing
nb_execution_timeout = -1  # No timeout for notebook execution

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

html_theme = "sphinx_immaterial"

html_show_sourcelink = True
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "white",
            "accent": "indigo",
            "toggle": {
                "icon": "material/toggle-switch-off-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "black",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/toggle-switch",
                "name": "Switch to light mode",
            },
        },
    ],
    "features": [
        "navigation.expand",
        "navigation.instant",
        "navigation.sections",
        "navigation.tabs",
        "navigation.tabs.sticky",
    ],
}

# -----------------------------------------------------------------------------
# AutoAPI skip logic
# -----------------------------------------------------------------------------


def _inherits_from(obj: Any, full_name: str) -> bool:
    parent = obj.obj.get("inherited_from")
    return bool(parent and parent.get("full_name") == full_name)


def autoapi_skip_member(app: Any, what: str, name: str, obj: Any, skip: bool, options: Any) -> bool:  # noqa: ARG001
    """Skip undocumented attributes, pydantic internals, and empty modules."""
    if what == "attribute" and obj.docstring == "":
        skip = True
    if what in ("module", "package") and (obj.all is None or len(obj.all) == 0):
        skip = True
    if _inherits_from(obj, "pydantic.main.BaseModel"):
        skip = True
    if what == "attribute" and name.endswith(".model_config"):
        skip = True
    return skip


def setup(app: Any) -> None:
    """Connect the autoapi-skip-member event to our custom skip logic."""
    app.connect("autoapi-skip-member", autoapi_skip_member)
