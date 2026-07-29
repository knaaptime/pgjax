# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../src"))

# Mock the native C++ extension so pgjax can be imported without a build.
sys.modules.setdefault("pgjax_cpp", MagicMock())

import pgjax  # noqa: E402

project = "pgjax"
copyright = "2024-, pgjax developers"  # noqa: A001
author = "pgjax developers"

version = pgjax.__version__
release = version

language = "en"
html_title = project

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html\#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# Mock the native C++ extension so autodoc can import pgjax without a build.
autodoc_mock_imports = ["pgjax_cpp"]

bibtex_bibfiles = ["_static/references.bib"]
bibtex_reference_style = "author_year"

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html\#options-for-html-output

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}
suppress_warnings = ["ref.ref"]

html_theme = "furo"
html_static_path = ["_static"]

autodoc_typehints = "none"


def linkcode_resolve(domain, info):
    def find_source():
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(pgjax.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "pgjax/%s#L%d-L%d" % find_source()  # noqa: UP031
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    tag = "dev" if "dev" in release else ("v" + release)
    return f"https://github.com/knaaptime/pgjax/blob/{tag}/{filename}"
