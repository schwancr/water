from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

extensions = [
              Extension("pyxutils", ["pyxutils.pyx"]),
              Extension("cywater", ["cywater.pyx"]),
              Extension("cywatertot", ["cywatertot.pyx"])
             ]

setup(
    cmdclass = {'build_ext' : build_ext},
    ext_modules = extensions,
    include_dirs = [np.get_include()]
)

