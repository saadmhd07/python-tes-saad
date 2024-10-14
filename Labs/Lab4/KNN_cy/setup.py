from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
 

setup(
    ext_modules=cythonize(
        Extension(
            "knn",  # Name of the extension
            sources=["knn.pyx"],  # Path to the Cython file
            include_dirs=[numpy.get_include()],  # Include Numpy headers
        ),
        annotate=True,  # Optional: Generates an HTML file with source code annotations
        compiler_directives={'language_level': 3}  # Python 3 syntax
    ),
)