
import os
import sys
import numpy as np
from setuptools import setup, Extension, find_packages

# HIDE WARNING:
# cc1plus: warning: command line option "-Wstrict-prototypes" is valid for C/ObjC but not for C++
from distutils.sysconfig import get_config_vars
cfg_vars = get_config_vars()
for k, v in cfg_vars.items():
    if type(v) == str:
        cfg_vars[k] = v.replace("-Wstrict-prototypes", "")

print('Begin: %s' % ' '.join(sys.argv))

arg = [arg for arg in sys.argv if arg.startswith('--instdir')]
if not arg:
    raise Exception('Parameter --instdir is missing')
instdir = arg[0].split('=')[1]
sys.argv.remove(arg[0])


# Shorter BUILD_LIST can be used to speedup development loop.
#Command example: ./build_all.sh -b PEAKFINDER:HEXANODE:CFD -md
BUILD_LIST = ('PSANA','SHMEM','PEAKFINDER','HEXANODE','DGRAM','HSD','CFD','NDARRAY')# ,'XTCAV')
arg = [arg for arg in sys.argv if arg.startswith('--ext_list')]
if arg:
    s_exts = arg[0].split('=')[1]
    sys.argv.remove(arg[0])
    if s_exts : BUILD_LIST = s_exts.split(':')
    #print('Build c++ python-extensions: %s' % s_exts)


# allows a version number to be passed to the setup
VERSION = '0.0.0'
arg = [arg for arg in sys.argv if arg.startswith('--version')]
if arg:
    VERSION = arg[0].split('=')[1]
    sys.argv.remove(arg[0])


print('-- psana.setup.py build extensions  : %s' % ' '.join(BUILD_LIST))
print('-- psana.setup.py install directory : %s' % instdir)
print('-- psana.setup.py include sys.prefix: %s' % sys.prefix)
print('-- psana.setup.py np.get_include()  : %s' % np.get_include())


if sys.platform == 'darwin':
    # Flag -Wno-cpp hides warning:
    #warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
    macos_sdk_version_arg = '-mmacosx-version-min=10.9'
    extra_c_compile_args = ['-Wno-#warnings', macos_sdk_version_arg]
    extra_cxx_compile_args = ['-std=c++11', '-Wno-#warnings', macos_sdk_version_arg]
    extra_link_args = [macos_sdk_version_arg]
    # Use libgomp instead of the version provided by the compiler. Passing plain -fopenmp uses the llvm version of OpenMP
    # which appears to have a conflict with the numpy we use from conda. numpy uses Intel MKL which itself uses OpenMP,
    # but this seems to cause crashes if you use the llvm OpenMP in the same process.
    openmp_compile_args = ['-fopenmp=libgomp']
    openmp_link_args = ['-fopenmp=libgomp']
else:
    # Flag -Wno-cpp hides warning:
    #warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
    extra_c_compile_args=['-Wno-cpp']
    extra_cxx_compile_args=['-std=c++11', '-Wno-cpp']
    extra_link_args = []
    # Use the version of openmp provided by the compiler
    openmp_compile_args = ['-fopenmp']
    openmp_link_args = ['-fopenmp']

extra_link_args_rpath = extra_link_args + ['-Wl,-rpath,'+ os.path.abspath(os.path.join(instdir, 'lib'))]

CYT_BLD_DIR = 'build'

from Cython.Build import cythonize

# defaults if the build list is empty
PACKAGES = []
EXTS = []
CYTHON_EXTS = []
INSTALL_REQS = []
PACKAGE_DATA = {}
ENTRY_POINTS = {}


if 'PSANA' in BUILD_LIST :
    dgram_module = Extension('psana.dgram',
                            sources = ['src/dgram.cc'],
                            libraries = ['xtc','shmemcli'],
                            include_dirs = ['src', np.get_include(), os.path.join(instdir, 'include')],
                            library_dirs = [os.path.join(instdir, 'lib')],
                            extra_link_args = extra_link_args_rpath,
                            extra_compile_args = extra_cxx_compile_args)

    container_module = Extension('psana.container',
                            sources = ['src/container.cc'],
                            libraries = ['xtc'],
                            include_dirs = [np.get_include(), os.path.join(instdir, 'include')],
                            library_dirs = [os.path.join(instdir, 'lib')],
                            extra_link_args = extra_link_args_rpath,
                            extra_compile_args = extra_cxx_compile_args)

    PACKAGES = find_packages()
    PACKAGE_DATA = {'psana.graphqt': ['data/icons/*.png','data/icons/*.gif']}
    EXTS = [dgram_module, container_module]
    INSTALL_REQS = [
        'numpy',
    ]
    ENTRY_POINTS = {
        'console_scripts': [
            'convert_npy_to_txt  = psana.pyalgos.app.convert_npy_to_txt:do_main',
            'convert_txt_to_npy  = psana.pyalgos.app.convert_txt_to_npy:do_main',
            'merge_mask_ndarrays = psana.pyalgos.app.merge_mask_ndarrays:do_main',
            'merge_max_ndarrays  = psana.pyalgos.app.merge_max_ndarrays:do_main',
            'cdb                 = psana.pscalib.app.cdb:cdb_cli',
            'proc_info           = psana.pscalib.app.proc_info:do_main',
            'proc_control        = psana.pscalib.app.proc_control:do_main',
            'proc_new_datasets   = psana.pscalib.app.proc_new_datasets:do_main',
            'det_raw_dark_proc   = psana.pscalib.app.det_raw_dark_proc:do_main',
            'timeconverter       = psana.graphqt.app.timeconverter:timeconverter',
            'calibman            = psana.graphqt.app.calibman:calibman_gui',
            'hdf5explorer        = psana.graphqt.app.hdf5explorer:hdf5explorer_gui',
            'screengrabber       = psana.graphqt.ScreenGrabberQt5:run_GUIScreenGrabber',
            'detnames            = psana.app.detnames:detnames',
            'xtcavDark           = psana.xtcav.app.xtcavDark',
            'xtcavLasingOff      = psana.xtcav.app.xtcavLasingOff',
            'xtcavLasingOn       = psana.xtcav.app.xtcavLasingOn',
            'xtcavDisplay        = psana.xtcav.app.xtcavDisplay',
            'shmemClientSimple   = psana.app.shmemClientSimple:main',
        ]
    }


if 'SHMEM' in BUILD_LIST :
    ext = Extension('shmem',
                    sources=["psana/shmem/shmem.pyx"],
                    libraries = ['xtc','shmemcli'],
                    include_dirs = [np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)


if 'PEAKFINDER' in BUILD_LIST :
    ext = Extension("peakFinder",
                    sources=["psana/peakFinder/peakFinder.pyx",
                             "../psalg/psalg/peaks/src/PeakFinderAlgos.cc",
                             "../psalg/psalg/peaks/src/LocalExtrema.cc"],
                    libraries = ['utils'], # for SysLog
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
                    include_dirs=[np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
    )
    CYTHON_EXTS.append(ext)

    # direct LCLS1 version of peak-finders
    ext = Extension("psalg_ext",
                    sources=["psana/peakFinder/psalg_ext.pyx",
                             "../psalg/psalg/peaks/src/PeakFinderAlgosLCLS1.cc",
                             "../psalg/psalg/peaks/src/LocalExtrema.cc"],
                    libraries = ['utils'], # for SysLog
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
                    include_dirs=[np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("peakfinder8",
                    sources=["psana/peakFinder/peakfinder8.pyx",
                             "psana/peakFinder/peakfinder8.cc"],
                    libraries = ['utils'], # for SysLog
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
                    include_dirs=[np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
    )
    CYTHON_EXTS.append(ext)

if 'HEXANODE' in BUILD_LIST :
    # ugly: only build hexanode apps if the roentdek software exists.
    # this is a rough python equivalent of the way cmake finds out whether
    # packages exist. - cpo
    if(os.path.isfile(os.path.join(sys.prefix, 'lib', 'libResort64c_x64.a'))):
        ext = Extension("hexanode",
                        sources=["psana/hexanode/hexanode_ext.pyx",
                                 "../psalg/psalg/hexanode/src/cfib.cc",
                                 "../psalg/psalg/hexanode/src/wrap_resort64c.cc",
                                 "../psalg/psalg/hexanode/src/SortUtils.cc",
                                 "../psalg/psalg/hexanode/src/LMF_IO.cc"],
                        language="c++",
                        extra_compile_args = extra_cxx_compile_args,
                        include_dirs=[os.path.join(sys.prefix,'include'), np.get_include(), os.path.join(instdir, 'include')],
                        library_dirs = [os.path.join(instdir, 'lib'), os.path.join(sys.prefix, 'lib')],
                        libraries=['Resort64c_x64'],
                        extra_link_args = extra_link_args,
        )
        CYTHON_EXTS.append(ext)


if 'HEXANODE_TEST' in BUILD_LIST :
  if(os.path.isfile(os.path.join(sys.prefix, 'lib', 'libResort64c_x64.a'))):
    ext = Extension("hexanode",
                    sources=["psana/hexanode/test_ext.pyx",
                             "../psalg/psalg/hexanode/src/LMF_IO.cc",
                             "../psalg/psalg/hexanode/src/cfib.cc"],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    include_dirs=[np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
                    extra_link_args = extra_link_args,
    )
    CYTHON_EXTS.append(ext)


if 'CFD' in BUILD_LIST :
    ext = Extension("constFracDiscrim",
                    sources=["psana/constFracDiscrim/constFracDiscrim.pyx",
                             "../psalg/psalg/constFracDiscrim/src/ConstFracDiscrim.cc"],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args,
                    include_dirs=[os.path.join(sys.prefix,'include'), np.get_include(), os.path.join(instdir, 'include')],
    )
    CYTHON_EXTS.append(ext)


if 'DGRAM' in BUILD_LIST :
    ext = Extension('dgramCreate',
                    #packages=['psana.peakfinder',],
                    sources=["psana/peakFinder/dgramCreate.pyx"],
                    libraries = ['xtc'],
                    include_dirs = [np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
                    # include_dirs=[np.get_include(), "../install/include"]
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana.dgramchunk",
                    sources=["src/dgramchunk.pyx"],
                    extra_compile_args=extra_c_compile_args,
                    extra_link_args=extra_link_args,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana.smdreader",
                    sources=["psana/smdreader.pyx"],
                    include_dirs=["psana"],
                    extra_compile_args=extra_c_compile_args,
                    extra_link_args=extra_link_args,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana.eventbuilder",
                    sources=["psana/eventbuilder.pyx"],
                    include_dirs=["psana"],
                    extra_compile_args=extra_c_compile_args,
                    extra_link_args=extra_link_args,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana.parallelreader",
                    sources=["psana/parallelreader.pyx"],
                    include_dirs=["psana"],
                    extra_compile_args=extra_c_compile_args + openmp_compile_args,
                    extra_link_args=extra_link_args + openmp_link_args,
    )
    CYTHON_EXTS.append(ext)


if 'HSD' in BUILD_LIST :
    ext = Extension("hsd",
                    sources=["psana/hsd/hsd.pyx",
                             "../psalg/psalg/peaks/src/PeakFinderAlgos.cc",
                             "../psalg/psalg/peaks/src/LocalExtrema.cc"],
                    libraries=['xtc','psalg','digitizer','utils'],
                    language="c++",
                    extra_compile_args=extra_cxx_compile_args,
                    include_dirs=[np.get_include(),
                                  "../install/include",
                                  os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)


if 'NDARRAY' in BUILD_LIST :
    ext = Extension("ndarray",
                    sources=["psana/pycalgos/NDArray_ext.pyx",
                             "../psalg/psalg/peaks/src/WFAlgos.cc"],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    include_dirs=[os.path.join(sys.prefix,'include'), np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
                    libraries=[],
                    extra_link_args = extra_link_args,
    )
    CYTHON_EXTS.append(ext)


setup(
    name = 'psana',
    version = VERSION,
    license = 'LCLS II',
    description = 'LCLS II analysis package',
    install_requires = INSTALL_REQS,
    packages = PACKAGES,
    package_data = PACKAGE_DATA,
    #cmdclass={'build_ext': my_build_ext},
    ext_modules = EXTS + cythonize(CYTHON_EXTS, build_dir=CYT_BLD_DIR, language_level=2),
    entry_points = ENTRY_POINTS,
)


# ===== EOF ======
