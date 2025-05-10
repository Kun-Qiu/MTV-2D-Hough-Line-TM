from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        "parametric_X",
        ["parametric_X_c.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-ffast-math"]
    )
]

setup(
     ext_modules=cythonize(extensions),
)