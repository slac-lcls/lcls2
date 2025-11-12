import os
import sys
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
psdaqdir_env = os.environ.get('PSDAQDIR')
if not psdaqdir_env:
    psdaqdir = instdir_env
else:
    psdaqdir = psdaqdir_env

# Shorter BUILD_LIST can be used to speedup development loop.
#Command example: ./build_all.sh -b PEAKFINDER:HEXANODE:CFD -md
BUILD_LIST = ('PSDAQ','TRIGGER')
build_list_env = os.environ.get('BUILD_LIST')
if build_list_env:
    BUILD_LIST = build_list_env.split(':')
    #print('Build c++ python-extensions: %s' % str(BUILD_LIST))


VERSION = '0.0.0'
version_env = os.environ.get('VERSION')
if version_env:
    VERSION = version_env


print('-- psana.setup.py build extensions  : %s' % ' '.join(BUILD_LIST))
print('-- psana.setup.py install directory : %s' % instdir)
print('-- psana.setup.py include sys.prefix: %s' % sys.prefix)


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
    extra_c_compile_args=['-Wno-cpp']
    extra_cxx_compile_args=['-std=c++20', '-Wno-cpp']
    extra_link_args = []
    # Use the version of openmp provided by the compiler
    openmp_compile_args = ['-fopenmp']
    openmp_link_args = ['-fopenmp']

extra_link_args_rpath = extra_link_args + ['-Wl,-rpath,'+ os.path.abspath(os.path.join(instdir, 'lib'))]
if xtcdatadir_env:
    extra_link_args_rpath = extra_link_args_rpath + ['-Wl,-rpath,'+ os.path.abspath(os.path.join(xtcdatadir, 'lib'))]
if psdaqdir_env:
    extra_link_args_rpath = extra_link_args_rpath + ['-Wl,-rpath,'+ os.path.abspath(os.path.join(psdaqdir, 'lib'))]

CYT_BLD_DIR = 'build'

from Cython.Build import cythonize

# defaults if the build list is empty
PACKAGES = []
CYTHON_EXTS = []
PACKAGE_DATA = {}
SCRIPTS = []
ENTRY_POINTS = {}

this_file = os.path.abspath(__file__)
repo_root = os.path.abspath(os.path.join(this_file, "../.."))

if xtcdatadir_env:
    xtc_headers =  os.path.join(xtcdatadir, 'include')
    xtc_lib_path = os.path.join(xtcdatadir, 'lib')
else:
    xtc_headers = os.path.join(repo_root, "xtcdata")
    xtc_lib_path = os.path.join(repo_root, "build", "xtcdata", "xtcdata", "xtc")

if psdaqdir_env:
    psdaq_headers = os.path.join(psdaqdir, 'include')
    psdaq_lib_dirs = [os.path.join(psdaqdir, 'lib')]
else:
    psdaq_headers = os.path.join(repo_root, 'psdaq')
    psdaq_lib_dirs = [
        os.path.join(repo_root, "build", "psdaq", "psdaq"),           
        os.path.join(repo_root, "build", "psdaq", "psdaq", "aes-stream-drivers"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "app"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "bld"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "eb"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "epicstools"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "hsd"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "mmhw"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "monreq"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "pgp"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "service"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "tpr"),
        os.path.join(repo_root, "build", "psdaq", "psdaq", "trigger"),
    ]


if 'PSDAQ' in BUILD_LIST :
    PACKAGES = find_packages()
    PACKAGE_DATA = {'psdaq.control_gui': ['data/icons/*.png','data/icons/*.gif']}
    SCRIPTS = ['psdaq/procmgr/procmgr','psdaq/procmgr/procstat','psdaq/procmgr/condaProcServ']
    ENTRY_POINTS = {
        'console_scripts': [
            'control = psdaq.control.control:main',
            'selectPlatform = psdaq.control.selectPlatform:main',
            'showPlatform = psdaq.control.showPlatform:main',
            'daqstate = psdaq.control.daqstate:main',
            'currentexp = psdaq.control.currentexp:main',
            'testClient2 = psdaq.control.testClient2:main',
            'testAsyncErr = psdaq.control.testAsyncErr:main',
            'testFileReport = psdaq.control.testFileReport:main',
            'configdb = psdaq.configdb.configdb:main',
            'configdb_GUI = psdaq.configdb.configdb_GUI:main',
            'configdb_multimod = psdaq.configdb.configdb_multimod:configdb_multimod',
            'epixquad_store_gainmap = psdaq.configdb.epixquad_store_gainmap:main',
            'epixquad_create_pixelmask = psdaq.configdb.epixquad_create_pixelmask:main',
            'epixhr_config_from_yaml_set = psdaq.configdb.epixhr_config_from_yaml_set:main',
            'epixhr_pedestal_scan = psdaq.cas.epixhr_pedestal_scan:main',
            'epixhr_chargeinj_scan = psdaq.cas.epixhr_chargeinj_scan:main',
            'epixhr_timing_scan = psdaq.cas.epixhr_timing_scan:main',
            'epixhr_config_scan = psdaq.cas.epixhr_config_scan:main',
            'epixhr_r0acq_scan = psdaq.cas.epixhr_r0acq_scan:main',
            'epixquad_chargeinj_scan = psdaq.cas.epixquad_chargeinj_scan:main',
            'epixquad_pedestal_scan = psdaq.cas.epixquad_pedestal_scan:main',
            'epixuhr_chargeinj_scan = psdaq.cas.epixuhr_chargeinj_scan:main',
            'epixuhr_pedestal_scan = psdaq.cas.epixuhr_pedestal_scan:main',
            'seq_epixhr = psdaq.seq.seq_epixhr:main',
            'generic_timing_scan = psdaq.cas.generic_timing_scan:main',
            'getrun = psdaq.control.getrun:main',
            'groupca = psdaq.cas.groupca:main',
            'xpmpva = psdaq.cas.xpmpva:main',
            'xpmpatt = psdaq.cas.xpmpatt:main',
            'hsdca = psdaq.cas.hsdca:main',
            'hsdcas = psdaq.cas.hsdcas:main',
            'hsdpva = psdaq.cas.hsdpva:main',
            'hsdpvs = psdaq.cas.hsdpvs:main',
            'pvatable = psdaq.cas.pvatable:main',
            'pvant = psdaq.cas.pvant:main',
            'campvs = psdaq.cas.campvs:main',
            'tprca = psdaq.cas.tprca:main',
            'tprcas = psdaq.cas.tprcas:main',
            'xpmioc = psdaq.cas.xpmioc:main',
            'bldcas = psdaq.cas.bldcas:main',
            'bldgmd = psdaq.cas.bldgmd:main',
            'hpsdbuscas = psdaq.cas.hpsdbuscas:main',
            'wave8pvs = psdaq.cas.wave8pvs:main',
            'setup_tdrp = psdaq.cas.setup_tdrp:main',
            'pytdrp = psdaq.cas.pytdrp:main',
            'pytdet_collector = psdaq.cas.pytdet_collector:main',
            'pyxpm = psdaq.pyxpm.pyxpm:main',
            'pykcuxpm = psdaq.pyxpm.pykcuxpm:main',
            'amccpromload = psdaq.pyxpm.amccpromload:main',
            'pykcu = psdaq.pykcu.pykcu:main',
            'control_gui = psdaq.control_gui.app.control_gui:control_gui',
            'bluesky_simple = psdaq.control.bluesky_simple:main',
            'opal_config_scan = psdaq.control.opal_config_scan:main',
            'ts_config_scan = psdaq.control.ts_config_scan:main',
            'timed_run = psdaq.control.timed_run:main',
            'epics_exporter = psdaq.cas.epics_exporter:main',
            'hsd_epics_exporter = psdaq.cas.hsd_epics_exporter:main',
            'hsd_input_select = psdaq.cas.hsd_input_select:main',
            'fixed_rate_table = psdaq.seq.fixed_rate_table:main',
            'periodicgenerator = psdaq.seq.periodicgenerator:main',
            'rixgenerator = psdaq.seq.rixgeneratory:main',
            'seqplot = psdaq.seq.seqplot:main',
            'seqprogram = psdaq.seq.seqprogram:main',
            'traingenerator = psdaq.seq.traingenerator:main',
            'ued_seq_setup = psdaq.seq.ued_seq_setup:main',
            'bos = psdaq.bos.bos:main',
            'prometheus2pvs = psdaq.cas.prometheus2pvs:main',
            'prometheusIOC = psdaq.cas.prometheusIOC:main',
            'daqmgr = psdaq.slurm.main:_do_main',
            'cnf2py = psdaq.slurm.cnftopy:_do_main',
            'daqstat = psdaq.slurm.daqstat:_do_main',
            'daqlog_header = psdaq.slurm.daqlog_header:_do_main',
            'jungfrau_pedestal_scan = psdaq.cas.jungfrau_pedestal_scan:main',
            'jungfrau_gain_mode = psdaq.configdb.jungfrau_gain_mode:change_gain_mode',
            'jungfrau_protection = psdaq.configdb.jungfrau_protection:change_protection_settings',
            'detector_protection_monitor = psdaq.trigger.detector_protection_monitor:main',
        ]
    }

if 'TRIGGER' in BUILD_LIST and sys.platform != 'darwin':
    ext = Extension('EbDgram',
                    sources=["psdaq/trigger/EbDgram.pyx"],
                    libraries = ['xtc','service','trigger'],
                    include_dirs = [os.path.join(instdir, 'include'), xtc_headers, psdaq_headers],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path]+psdaq_lib_dirs,
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension('ResultDgram',
                    sources=["psdaq/trigger/ResultDgram.pyx"],
                    libraries = ['xtc','service','trigger'],
                    include_dirs = [os.path.join(instdir, 'include'), xtc_headers, psdaq_headers],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path]+psdaq_lib_dirs,
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension('TmoTebData',
                    sources=["psdaq/trigger/TmoTebData.pyx"],
                    libraries = ['xtc','service','trigger'],
                    include_dirs = [os.path.join(instdir, 'include'), xtc_headers, psdaq_headers],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path]+psdaq_lib_dirs,
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)

    ext = Extension('TimingTebData',
                    sources=["psdaq/trigger/TimingTebData.pyx"],
                    libraries = ['xtc','service','trigger'],
                    include_dirs = [os.path.join(instdir, 'include'), xtc_headers, psdaq_headers],
                    library_dirs = [os.path.join(instdir, 'lib'), xtc_lib_path]+psdaq_lib_dirs,
                    language="c++",
                    extra_compile_args = extra_cxx_compile_args,
                    extra_link_args = extra_link_args_rpath,
    )
    CYTHON_EXTS.append(ext)


setup(
    name = 'psdaq',
    license = 'LCLS II',
    description = 'LCLS II DAQ package',
    version = VERSION,
    packages = PACKAGES,
    package_data = PACKAGE_DATA,
    scripts = SCRIPTS,
    ext_modules = cythonize(CYTHON_EXTS, build_dir=CYT_BLD_DIR, language_level=2, annotate=True),
    entry_points = ENTRY_POINTS,
)
