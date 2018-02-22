# TODO
# ** should outputs be arrays instead of Peak class
# ** psalg should be built with cmake to produce a .so
# ** the cython .pyx  should be in psana and built in psana
# ** what array class are we going to use?
# *  remove main class from pebble, so it only gets created once
# remove the "if" statement around placement-new and non-placement-new
# every method gets a drp-pointer
# calling push_back could malloc?
# pebble overwrite protection (bounds check) if arrays get too big

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
ext = Extension("peakFinder",
                sources=["peakFinder.pyx", "psalgos/src/PeakFinderAlgos.cpp", "psalgos/src/LocalExtrema.cpp"],
                language="c++",
                extra_compile_args=['-g', '-std=c++11'],
                include_dirs=[numpy.get_include(), "psalgos/include/PeakFinderAlgos.h", "psalgos/include/LocalExtrema.h"]
)

setup(name="peakFinder",
      ext_modules=cythonize(ext))


