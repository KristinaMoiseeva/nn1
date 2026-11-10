from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='lab3_layers',
    ext_modules=cythonize(
        [Extension('im2col_cython', ['im2col_cython.pyx'], include_dirs=[np.get_include()])],
        compiler_directives={'language_level': 3},
    ),
)
