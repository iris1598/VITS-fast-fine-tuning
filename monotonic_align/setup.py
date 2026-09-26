"""Build script for the monotonic_align Cython extension.

Two things changed since this project was written:

* ``distutils`` was removed from the standard library in Python 3.12, so
  ``from distutils.core import setup`` no longer works — we use ``setuptools``.
* The extension is registered as a plain top-level module ``core`` so that
  ``build_ext --inplace`` drops ``core.<abi>.pyd/.so`` right next to this file
  (the historical ``Extension("monotonic_align.core", ...)`` layout nested the
  output into a second ``monotonic_align/`` directory).

Usage (from inside this directory)::

    python setup.py build_ext --inplace
"""

from setuptools import Extension, setup

import numpy
from Cython.Build import cythonize

extensions = [
    Extension(
        "core",
        sources=["core.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="monotonic_align",
    version="1.0.0",
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={"boundscheck": False, "wraparound": False, "cdivision": True},
    ),
)
