import os
import sys

# Add the src/ directory to the path so Sphinx can find our Python modules
sys.path.insert(0, os.path.abspath('../src'))

project = 'R3-FL'
copyright = '2026, Daniel'
author = 'Daniel'
version = '1.0'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',   # Auto-generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx_rtd_theme',     # ReadTheDocs Theme
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
