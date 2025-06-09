from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
from pathlib import Path

# Since setup.py is in cython_build, we look for files in parametric_x subdirectory
pyx_file = Path("ParametricX") / "ParametricX.pyx"

if not pyx_file.exists():
    raise FileNotFoundError(f"Cython file not found at: {pyx_file}")

# Platform-specific compiler flags
if os.name == 'nt':  # Windows
    extra_compile_args = ['/O2']
else:  # Linux/Mac
    extra_compile_args = ['-O3', '-ffast-math']

extensions = [
    Extension(
        "ParametricX",
        sources=[str(pyx_file)],
        include_dirs=[
            np.get_include(),
            str(Path("ParametricX"))  # For .pxd files
        ],
        extra_compile_args=extra_compile_args
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
    package_dir={'': '.'},  # Look for packages in current directory
    packages=['ParametricX']  # Package containing the extension
)