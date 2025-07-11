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

instdir_env = os.environ.get('INSTDIR')
if not instdir_env:
    raise Exception('Parameter --instdir is missing')
instdir = instdir_env

xtcdatadir_env = os.environ.get('XTCDATADIR')
if not xtcdatadir_env:
    xtcdatadir = instdir_env
else:
    xtcdatadir = xtcdatadir_env

psalgdir_env = os.environ.get('PSALGDIR')
if not psalgdir_env:
    psalgdir = instdir_env
else:
    psalgdir = psalgdir_env

# Shorter BUILD_LIST can be used to speedup development loop.
#Command example: ./build_all.sh -b PEAKFINDER:HEXANODE:CFD -md
BUILD_LIST = ('PSANA','SHMEM','PEAKFINDER','HEXANODE','DGRAM','HSD','CFD','NDARRAY', 'PYCALGOS')# ,'XTCAV')
build_list_env = os.environ.get('BUILD_LIST')
if build_list_env:
    BUILD_LIST = build_list_env.split(':')
    #print('Build c++ python-extensions: %s' % str(BUILD_LIST))


# allows a version number to be passed to the setup
VERSION = '0.0.0'
version_env = os.environ.get('VERSION')
if version_env:
    VERSION = version_env


print('-- psana2.setup.py build extensions  : %s' % ' '.join(BUILD_LIST))
print('-- psana2.setup.py install directory : %s' % instdir)
print('-- psana2.setup.py include sys.prefix: %s' % sys.prefix)
print('-- psana2.setup.py np.get_include()  : %s' % np.get_include())


if sys.platform == 'darwin':
    # Flag -Wno-cpp hides warning:
    #warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-W#warnings]
    macos_sdk_version_arg = '-mmacosx-version-min=10.9'
    extra_c_compile_args = ['-Wno-#warnings', macos_sdk_version_arg]
    extra_cxx_compile_args = ['-std=c++20', '-Wno-#warnings', macos_sdk_version_arg]
    extra_link_args = [macos_sdk_version_arg]
    # Use libgomp instead of the version provided by the compiler. Passing plain -fopenmp uses the llvm version of OpenMP
    # which appears to have a conflict with the numpy we use from conda. numpy uses Intel MKL which itself uses OpenMP,
    # but this seems to cause crashes if you use the llvm OpenMP in the same process.
    openmp_compile_args = ['-fopenmp=libgomp']
    openmp_link_args = ['-fopenmp=libgomp']
else:
    # Flag -Wno-cpp hides warning:
    #warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
    extra_c_compile_args=['-Wno-cpp', '-Wno-address']
    extra_cxx_compile_args=['-std=c++20', '-Wno-cpp', '-Wno-volatile']
    extra_link_args = []
    # Use the version of openmp provided by the compiler
    openmp_compile_args = ['-fopenmp']
    openmp_link_args = ['-fopenmp']

extra_link_args_rpath = extra_link_args + ['-Wl,-rpath,'+ os.path.abspath(os.path.join(instdir, 'lib'))]
if xtcdatadir_env:
    extra_link_args_rpath = extra_link_args_rpath + ['-Wl,-rpath,'+ os.path.abspath(os.path.join(xtcdatadir, 'lib'))]
if psalgdir_env:
    extra_link_args_rpath = extra_link_args_rpath + ['-Wl,-rpath,'+ os.path.abspath(os.path.join(psalgdir, 'lib'))]

CYT_BLD_DIR = 'build'

from Cython.Build import cythonize

# defaults if the build list is empty
PACKAGES = []
EXTS = []
CYTHON_EXTS = []
INSTALL_REQS = []
PACKAGE_DATA = {}
ENTRY_POINTS = {}

if xtcdatadir_env:
    xtc_headers =  os.path.join(xtcdatadir, 'include')
    xtc_lib_path = os.path.join(xtcdatadir, 'lib')
else:
    xtc_headers =  os.path.join(instdir, 'include')
    xtc_lib_path = os.path.join(instdir, 'lib')

if psalgdir_env:
    psalg_headers =  os.path.join(psalgdir, 'include')
    psalg_lib_path =  os.path.join(psalgdir, 'lib')
else:
    psalg_headers =  os.path.join(instdir, 'include')
    psalg_lib_path =  os.path.join(instdir, 'lib')

if 'PSANA' in BUILD_LIST :
    dgram_module = Extension('psana2.dgram',
                            sources = ['src/dgram.cc'],
                            libraries = ['xtc'],
                            #include_dirs = ['src', np.get_include(), os.path.join(instdir, 'include'), ],
                            #library_dirs = [os.path.join(instdir, 'lib')],
                            include_dirs = ['src', np.get_include(), os.path.join(instdir, 'include'), xtc_headers ],
                            library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path],
                            extra_link_args = extra_link_args_rpath,
                            extra_compile_args = extra_cxx_compile_args)

    container_module = Extension('psana2.container',
                            sources = ['src/container.cc'],
                            libraries = ['xtc'],
                            #include_dirs = [np.get_include(), os.path.join(instdir, 'include')],
                            #library_dirs = [os.path.join(instdir, 'lib')],
                            include_dirs = [np.get_include(), os.path.join(instdir, 'include'), xtc_headers ],
                            library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path],
                            extra_link_args = extra_link_args_rpath,
                            extra_compile_args = extra_cxx_compile_args)

    PACKAGES = find_packages()
    PACKAGE_DATA = {'psana2.graphqt': ['data/icons/*.png','data/icons/*.gif'], "psana2.pscalib": ["geometry/data/*data"]}
    EXTS = [dgram_module, container_module]
    INSTALL_REQS = [
        'numpy',
    ]
    ENTRY_POINTS = {
        'console_scripts': [
            'convert_npy_to_txt  = psana2.pyalgos.app.convert_npy_to_txt:do_main',
            'convert_txt_to_npy  = psana2.pyalgos.app.convert_txt_to_npy:do_main',
            'merge_mask_ndarrays = psana2.pyalgos.app.merge_mask_ndarrays:do_main',
            'merge_max_ndarrays  = psana2.pyalgos.app.merge_max_ndarrays:do_main',
            'cdb                 = psana2.pscalib.app.cdb:cdb_cli',
            'proc_info           = psana2.pscalib.app.proc_info:do_main',
            'proc_control        = psana2.pscalib.app.proc_control:do_main',
            'proc_new_datasets   = psana2.pscalib.app.proc_new_datasets:do_main',
            'timeconverter       = psana2.graphqt.app.timeconverter:timeconverter',
            'calibman            = psana2.graphqt.app.calibman:calibman_gui',
            'hdf5explorer        = psana2.graphqt.app.hdf5explorer:hdf5explorer_gui',
            'screengrabber       = psana2.graphqt.ScreenGrabberQt5:run_GUIScreenGrabber',
            'detnames            = psana2.app.detnames:detnames',
            'config_dump         = psana2.app.config_dump:config_dump',
            'xtcavDark           = psana2.xtcav.app.xtcavDark:__main__',
            'xtcavLasingOff      = psana2.xtcav.app.xtcavLasingOff:__main__',
            'xtcavLasingOn       = psana2.xtcav.app.xtcavLasingOn:__main__',
            'xtcavDisplay        = psana2.xtcav.app.xtcavDisplay:__main__',
            'shmemClientSimple   = psana2.app.shmemClientSimple:main',
            'epix10ka_pedestals_calibration = psana2.app.epix10ka_pedestals_calibration:do_main',
            'epix10ka_charge_injection = psana2.app.epix10ka_charge_injection:do_main',
            'epix10ka_deploy_constants = psana2.app.epix10ka_deploy_constants:do_main',
            'epix10ka_raw_calib_image  = psana2.app.epix10ka_raw_calib_image:do_main',
            'epix10ka_calib_components = psana2.app.epix10ka_calib_components:__main__',
            'epixm320_dark_proc        = psana2.app.epixm320_dark_proc:do_main',
            'epixm320_charge_injection = psana2.app.epixm320_charge_injection:do_main',
            'datinfo             = psana2.app.datinfo:do_main',
            'det_dark_proc       = psana2.app.det_dark_proc:do_main',
            'det_pixel_status    = psana2.app.det_pixel_status:do_main',
            'parallel_proc       = psana2.app.parallel_proc:do_main',
            'iv                  = psana2.graphqt.app.iv:image_viewer',
            'masked              = psana2.graphqt.app.masked:mask_editor',
            'roicon              = psana2.app.roicon:__main__',
            'psplot_live         = psana2.app.psplot_live.main:start',
            'timestamp_sort_h5   = psana2.app.timestamp_sort_h5.main:start',
            'psanaplot = psana2.app.psana_plot:_do_main',
            'jungfrau_dark_proc  = psana2.app.jungfrau_dark_proc:do_main',
            'jungfrau_deploy_constants = psana2.app.jungfrau_deploy_constants:do_main',
            'caliblogs           = psana2.app.caliblogs:do_main',
            'calibrepo           = psana2.app.calibrepo:do_main',
            'calib_prefetch      = psana2.pscalib.app.calib_prefetch.__main__:main',
        ]
    }


if 'SHMEM' in BUILD_LIST and sys.platform != 'darwin':
    ext = Extension('shmem',
                    sources=["psana2/shmem/shmem.pyx"],
                    libraries = ['xtc','shmemcli'],
                    #include_dirs = [np.get_include(), os.path.join(instdir, 'include')],
                    #library_dirs = [os.path.join(instdir, 'lib')],
                    include_dirs = [np.get_include(), os.path.join(instdir, 'include'), xtc_headers, psalg_headers ],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path, psalg_lib_path],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)


if 'PEAKFINDER' in BUILD_LIST :
    ext = Extension("peakFinder",
                    sources=["psana2/peakFinder/peakFinder.pyx",
                             "psana2/peakFinder/src/PeakFinderAlgos.cc",
                             "psana2/peakFinder/src/LocalExtrema.cc"],
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
                    sources=["psana2/peakFinder/psalg_ext.pyx",
                             "psana2/peakFinder/src/PeakFinderAlgosLCLS1.cc",
                             "psana2/peakFinder/src/LocalExtrema.cc"],
                    libraries = ['utils'], # for SysLog
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
                    include_dirs=[np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("peakfinder8",
                    sources=["psana2/peakFinder/peakfinder8.pyx",
                             "psana2/peakFinder/peakfinder8.cc"],
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
                        sources=["psana2/hexanode/hexanode_ext.pyx",
                                 "psana2/hexanode/src/cfib.cc",
                                 "psana2/hexanode/src/wrap_resort64c.cc",
                                 "psana2/hexanode/src/SortUtils.cc",
                                 "psana2/hexanode/src/LMF_IO.cc"],
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
                    sources=["psana2/hexanode/test_ext.pyx",
                             "psana2/hexanode/src/LMF_IO.cc",
                             "psana2/hexanode/src/cfib.cc"],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    include_dirs=[np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
                    extra_link_args = extra_link_args,
    )
    CYTHON_EXTS.append(ext)


if 'CFD' in BUILD_LIST :
    ext = Extension("constFracDiscrim",
                    sources=["psana2/constFracDiscrim/constFracDiscrim.pyx",
                             "psana2/constFracDiscrim/src/ConstFracDiscrim.cc"],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args,
                    include_dirs=[os.path.join(sys.prefix,'include'), np.get_include(), os.path.join(instdir, 'include')],
    )
    CYTHON_EXTS.append(ext)


if 'DGRAM' in BUILD_LIST :
    ext = Extension('dgramCreate',
                    #packages=['psana.peakfinder',],
                    sources=["psana2/peakFinder/dgramCreate.pyx"],
                    libraries = ['xtc'],
                    #include_dirs = [np.get_include(), os.path.join(instdir, 'include')],
                    #library_dirs = [os.path.join(instdir, 'lib')],
                    include_dirs = [np.get_include(), os.path.join(instdir, 'include'), xtc_headers ],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
                    # include_dirs=[np.get_include(), "../install/include"]
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana2.dgramchunk",
                    sources=["src/dgramchunk.pyx"],
                    extra_compile_args=extra_c_compile_args,
                    extra_link_args=extra_link_args,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana2.smdreader",
                    sources=["psana2/smdreader.pyx"],
                    libraries = ['xtc'],
                    #include_dirs=["psana"],
                    #include_dirs = [np.get_include(), os.path.join(instdir, 'include')],
                    include_dirs = ["psana2", np.get_include(), os.path.join(instdir, 'include'), xtc_headers ],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path],
                    #extra_compile_args=extra_c_compile_args,
                    extra_compile_args=extra_c_compile_args + openmp_compile_args,
                    #extra_link_args=extra_link_args,
                    extra_link_args=extra_link_args + openmp_link_args + extra_link_args_rpath
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana2.eventbuilder",
                    sources=["psana2/eventbuilder.pyx"],
                    include_dirs=["psana2"],
                    extra_compile_args=extra_c_compile_args,
                    extra_link_args=extra_link_args,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana2.parallelreader",
                    sources=["psana2/parallelreader.pyx"],
                    include_dirs=["psana2"],
                    extra_compile_args=extra_c_compile_args + openmp_compile_args,
                    extra_link_args=extra_link_args + openmp_link_args,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana2.dgramedit",
                    sources=["psana2/dgramedit.pyx"],
                    libraries = ['xtc'],
                    #include_dirs=["psana",np.get_include(), os.path.join(instdir, 'include')],
                    #library_dirs = [os.path.join(instdir, 'lib')],
                    include_dirs = ["psana2", np.get_include(), os.path.join(instdir, 'include'), xtc_headers ],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("quadanode",
                    sources=["psana2/quadanode.pyx"],
                    include_dirs=["psana2",np.get_include(), os.path.join(instdir, 'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension("psana2.dgramlite",
                    sources=["psana2/dgramlite.pyx"],
                    libraries = ['xtc'],
                    #include_dirs=["psana",np.get_include(), os.path.join(instdir, 'include')],
                    #iibrary_dirs = [os.path.join(instdir, 'lib')],
                    include_dirs = ["psana2", np.get_include(), os.path.join(instdir, 'include'), xtc_headers ],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)


if 'HSD' in BUILD_LIST :
    ext = Extension("hsd",
                    sources=["psana2/hsd/hsd.pyx"],
                    libraries=[],
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
                    sources=["psana2/pycalgos/NDArray_ext.pyx",
                             "psana2/peakFinder/src/WFAlgos.cc"],\
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    include_dirs=["psana2",os.path.join(sys.prefix,'include'),np.get_include(),os.path.join(instdir,'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
                    libraries=[],
                    extra_link_args = extra_link_args,
    )
    CYTHON_EXTS.append(ext)


if 'PYCALGOS' in BUILD_LIST :
    ext = Extension("utilsdetector_ext",
                    sources=["psana2/pycalgos/utilsdetector_ext.pyx",
                             "psana2/pycalgos/UtilsDetector.cc"],
                    #libraries = ['utils'], # for SysLog
                    libraries = [],
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args + ['-O3',],
                    extra_link_args = extra_link_args_rpath,
                    #include_dirs=[np.get_include(), os.path.join(instdir, 'include')],
                    include_dirs=["psana2",os.path.join(sys.prefix,'include'),np.get_include(),os.path.join(instdir,'include')],
                    library_dirs = [os.path.join(instdir, 'lib')],
    )
    CYTHON_EXTS.append(ext)

setup(
    name = 'psana2',
    version = VERSION,
    license = 'LCLS II',
    description = 'LCLS II analysis package',
    install_requires = INSTALL_REQS,
    packages = PACKAGES,
    package_data = PACKAGE_DATA,
    #cmdclass={'build_ext': my_build_ext},
    ext_modules = EXTS + cythonize(CYTHON_EXTS, build_dir=CYT_BLD_DIR, language_level=2, annotate=True),
    entry_points = ENTRY_POINTS,
)
