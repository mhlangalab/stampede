# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

from datetime import date

# -- Project information -----------------------------------------------------

project = "stampede"
author = "Niels Velthuijs & Siebren Frölich"
copyright = f"{date.today().year}, {author}"
with open("../stampede/__init__.py") as f:
    for line in f:
        if line.startswith("__version__ = "):
            version = line.replace("__version__ =", "").strip()
            break
    else:
        raise ValueError("__version__ not found in stampede/__init__.py!")


# -- General configuration ---------------------------------------------------

needs_sphinx = "4.0"

exclude_patterns = [
    "**.ipynb_checkpoints",
    ".github",
    "README.md",
    "_build",
    "adatas",
    "data",
    "dist",
    "jupyter_execute",
    "testenv",
    "tests",
    "venv",
]
extensions = [
    "m2r2",  # mdinclude, needs to be before myst_nb
    "myst_nb",  # md and ipynb files
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",  # for Google-style docstrings
    # "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx_autodoc_typehints",  # needs to be after napoleon
]


# Configuration of sphinx.ext.autodoc
autodoc_mock_imports = [
    "adjustText",
    "anndata",
    "matplotlib",
    "natsort",
    "numpy",
    "pandas",
    "igraph",
    "scanpy",
    "sklearn",
    "scipy",
    "seaborn",

    "geosketch",
    "pydeseq2",
    "statsmodels",
]
# autodoc_member_order = "bysource"
autodoc_typehints = "both"  # "description"  # "none"
autodoc_typehints_format = "short"


# Configuration of sphinx.ext.napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None  # [("Params", "Parameters")]

# # Configuration of sphinx.ext.autosummary
# autosummary_generate = True

# Configuration of m2r2 & myst-nb
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",  # handled by m2r2
    ".ipynb": "myst-nb",  # handled by myst-nb
}

# configure myst_bb
nb_execution_mode = "off"

# Configuration of sphinx.ext.intersphinx
intersphinx_mapping = dict(
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    matplotlib=("https://matplotlib.org/stable/", None),
    # fix: py:class reference target not found: pandas.core.frame.DataFrame [ref.class]
    # pandas=("https://pandas.pydata.org/pandas-docs/stable/", None),
    pandas=("https://pandas.pydata.org/pandas-docs/version/2.3/", None),
    pydeseq2=("https://pydeseq2.readthedocs.io/en/stable/", None),
    python=("https://docs.python.org/3", None),
)
nitpicky = True


# -- Options for HTML output -------------------------------------------------

# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html
html_theme = "pydata_sphinx_theme"
html_sidebars = {
    "**": ["sidebar-collapse", "page-toc"],
    # "**": [],  # no primary sidebar (left)
}
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mhlangalab/stampede",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "text": "STAMPede",
        "alt_text": "STAMPede - Home",
        # "image_light": "img/logo_no_bg.png",  # logo using light mode
        # "image_dark": "img/logo_no_bg.png",  # logo using dark mode
    },
    "secondary_sidebar_items": [],  # no secondary sidebar (right)
    "show_toc_level": 2,  # page-toc depth
    "show_nav_level": 0,  # sidebar-nav-bs depth
    "show_prev_next": False,
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": ["last-updated"],
}
# html_static_path = ["_static"]
# html_css_files = ["custom.css"]
# html_show_sphinx = False
html_logo = "img/logo_no_bg.png"  # universal logo
html_favicon = "img/logo_no_bg.png"  # logo in the browser tab
html_title = "STAMPede"  # title in the browser tab
html_last_updated_fmt = "%Y-%m-%d, %H:%M (UTC)"
