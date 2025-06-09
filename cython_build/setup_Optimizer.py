from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
from pathlib import Path

# Define base directory
BASE_DIR = Path(__file__).parent

module_paths = {
    "ParametricOpt": BASE_DIR / "ParametricOpt" / "ParametricOpt.pyx"
}

# Verify files exist
for name, path in module_paths.items():
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

# Compiler settings
extra_compile_args = ['/O2'] if os.name == 'nt' else ['-O3', '-ffast-math']

extensions = [
    Extension(
        name,
        sources=[str(path)],
        include_dirs=[
            np.get_include(),
            str(BASE_DIR / "ParametricX")  # For C headers if needed
        ],
        extra_compile_args=extra_compile_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
    for name, path in module_paths.items()
]

setup(
    name="parametric_modules",
    version="0.1",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True
        },
        include_path=[
            str(BASE_DIR / "ParametricX"),  # Must include this
            str(BASE_DIR / "ParametricOpt"),
            str(BASE_DIR)
        ]
    ),
    packages=[],  # No Python packages, just extensions
    install_requires=[
        'numpy>=1.17.0',
        'cython>=0.29.0'
    ],
    python_requires='>=3.6',
    zip_safe=False
)