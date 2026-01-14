# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add project root to PYTHONPATH so modules can be imported
sys.path.insert(0, os.path.abspath("../.."))


project = 'DeepPathway'
copyright = '2026, Muhammad Ahtazaz Ahsan'
author = 'Muhammad Ahtazaz Ahsan'
release = '0.1'

# ---- General configuration ----

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# Support Markdown as well as reStructuredText
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# ---- HTML output ----

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# ---- Autodoc options ----

autodoc_member_order = "bysource"
autodoc_typehints = "description"