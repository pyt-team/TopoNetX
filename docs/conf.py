"""Sphinx configuration file."""

project = "TopoNetX"
copyright = "2022-2023, PyT-Team, Inc."
author = "PyT-Team Authors"

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.load_style",
]

# Configure nbsphinx for notebook execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

nbsphinx_execute = "never"

# To get a prompt similar to the Classic Notebook, use
nbsphinx_input_prompt = " In [%s]:"
nbsphinx_output_prompt = " Out [%s]:"

nbsphinx_allow_errors = True

templates_path = ["_templates"]

source_suffix = [".rst"]

master_doc = "index"

language = "en"

# nbsphinx_thumbnails = {
#     "notebooks/01_simplicial_complexes": "notebooks/sc.png",
#     "notebooks/02_cell_complexes": "notebooks/cc.png",
#     "notebooks/03_combinatorial_complexes": "notebooks/ccc.png",
# }

nbsphinx_prolog = (
    r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div class="admonition note">
      <p>Notebook source code:
        <a class="reference external" href="https://github.com/pyt-team/"""
    r"""TopoNetX/blob/main/{{ docname|e }}">{{ docname|e }}</a>
      </p>
    </div>

.. raw:: latex
    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
    """
)
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

pygments_style = None

html_theme = "pydata_sphinx_theme"
html_baseurl = "pyt-team.github.io"
htmlhelp_basename = "pyt-teamdoc"
html_last_updated_fmt = "%c"

latex_elements = {}


latex_documents = [
    (
        master_doc,
        "toponetx.tex",
        "TopoNetX Documentation",
        "PyT-Team",
        "manual",
    ),
]

man_pages = [(master_doc, "toponetx", "TopoNetX Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "toponetx",
        "TopoNetX Documentation",
        author,
        "toponetx",
        "One line description of project.",
        "Miscellaneous",
    ),
]

epub_title = project
epub_exclude_files = ["search.html"]

# configure intersphinx
intersphinx_mapping = {
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# configure numpydoc
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
