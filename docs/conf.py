# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import sparcity

project = 'SparCity'
copyright = '2022, Yuji Okano'
author = 'Yuji Okano'
version = sparcity.__version__
release = sparcity.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'nbsphinx', 'sphinx_gallery.load_style']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = 'sparcity_doc'
master_doc = 'index'
latex_documents = [
    (master_doc, 'sparcity.tex',
     'SparCity Documentation',
     'Yuji Okano', 'manual'),
]

man_pages = [
    (master_doc, 'sparcity',
     'SparCity Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'sparcity',
     'SparCity Documentation',
     author,
     'sparcity',
     'Sparse estimator for geographical information',
     'Miscellaneous'),
]
