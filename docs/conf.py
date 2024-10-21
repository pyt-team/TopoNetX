"""Sphinx configuration file."""

# -- Project information -----------------------------------------------------

project = "TopoNetX"
copyright = "2022-2023, PyT-Team, Inc."  # noqa: A001
author = "PyT-Team Authors"
language = "en"

# -- General configuration ---------------------------------------------------

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
]

templates_path = ["_templates"]
source_suffix = [".rst"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Configure nbsphinx for notebook execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

# nbsphinx_thumbnails = {
#     "notebooks/01_simplicial_complexes": "notebooks/sc.png",
#     "notebooks/02_cell_complexes": "notebooks/cc.png",
#     "notebooks/03_combinatorial_complexes": "notebooks/ccc.png",
# }

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_baseurl = "https://pyt-team.github.io/toponetx/"

html_context = {
    "github_user": "pyt-team",
    "github_repo": "TopoNetX",
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pyt-team/TopoNetX",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
}

html_favicon = "_static/favicon-48.png"

html_show_sourcelink = False

# Exclude copy button from appearing over notebook cell numbers by using :not()
# The default copybutton selector is `div.highlight pre`
# https://github.com/executablebooks/sphinx-copybutton/blob/master/sphinx_copybutton/__init__.py#L82
copybutton_selector = ":not(.prompt) > div.highlight pre"

# -- Options for EPUB output -------------------------------------------------

epub_exclude_files = ["search.html"]
