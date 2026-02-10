project = "SigAlg"
copyright = "2026, John Myers"
author = "John Myers"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_proof",
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
    "jupyter_sphinx",
]

autosummary_generate = True

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "dollarmath",  # Enable $...$ and $$...$$ for math
    "amsmath",  # Enable advanced math environments
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.abspath("../src"))

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",
]
