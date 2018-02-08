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


