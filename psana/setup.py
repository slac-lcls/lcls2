
import os
import sys
import numpy as np
from setuptools import setup, Extension #, find_packages

print('Begin: %s' % ' '.join(sys.argv))

arg = [arg for arg in sys.argv if arg.startswith('--xtcdata')]
if not arg:
    raise Exception('Parameter --xtcdata is missing')
xtcdata = arg[0].split('=')[1]
sys.argv.remove(arg[0])

arg = [arg for arg in sys.argv if arg.startswith('--legion')]
if arg:
    legion = arg[0].split('=')[1]
    sys.argv.remove(arg[0])

    legion_src = ['src/legion_helper.cc']
    legion_lib = ['legion']
    legion_inc_dir = [os.path.join(legion, 'include')]
    legion_lib_dir = [os.path.join(legion, 'lib'), os.path.join(legion, 'lib64')]
    legion_link_args = ['-Wl,-rpath='+ os.path.abspath(os.path.join(legion, 'lib'))]
    legion_compile_args = ['-DPSANA_USE_LEGION']
else:
    legion_src = []
    legion_lib = []
    legion_inc_dir = []
    legion_lib_dir = []
    legion_link_args = []
    legion_compile_args = []

dgram_module = Extension('psana.dgram',
                         sources = ['src/dgram.cc'] + legion_src,
                         libraries = ['xtc'] + legion_lib,
                         include_dirs = ['src', np.get_include(), os.path.join(xtcdata, 'include')] + legion_inc_dir,
                         library_dirs = [os.path.join(xtcdata, 'lib')] + legion_lib_dir,
                         extra_link_args = ['-Wl,-rpath='+ os.path.abspath(os.path.join(xtcdata, 'lib'))] + legion_link_args,
                         extra_compile_args=['-std=c++11'] + legion_compile_args)

seq_module = Extension('psana.seq',
                         sources = ['src/seq.cc'],
                         libraries = ['xtc'],
                         include_dirs = [np.get_include(), os.path.join(xtcdata, 'include')],
                         library_dirs = [os.path.join(xtcdata, 'lib')],
                         extra_link_args = ['-Wl,-rpath='+ os.path.abspath(os.path.join(xtcdata, 'lib'))],
                         extra_compile_args=['-std=c++11'])

container_module = Extension('psana.container',
                         sources = ['src/container.cc'],
                         libraries = ['xtc'],
                         include_dirs = [np.get_include(), os.path.join(xtcdata, 'include')],
                         library_dirs = [os.path.join(xtcdata, 'lib')],
                         extra_link_args = ['-Wl,-rpath='+ os.path.abspath(os.path.join(xtcdata, 'lib'))],
                         extra_compile_args=['-std=c++11'])

setup(name = 'psana',
       version = '0.1',
       license = 'LCLS II',
       description = 'LCLS II analysis package',
       install_requires=[
         'numpy',
       ],
       #packages = find_packages(),
       packages = ['psana',
                   'psana.detector',
                   'psana.pscalib.calib',
                   'psana.pscalib.geometry',
                   'psana.pyalgos.generic',
                   'psana.graphqt',
       ],
       include_package_data = True,
       package_data={'graphqt': ['data/icons/*.png','data/icons/*.gif'],
       },

       #cmdclass = {'build': dgram_build, 'build_ext': dgram_build_ext},
       ext_modules = [dgram_module, seq_module, container_module],
       entry_points={
            'console_scripts': [
                'convert_npy_to_txt  = psana.pyalgos.app.convert_npy_to_txt:do_main',
                'convert_txt_to_npy  = psana.pyalgos.app.convert_txt_to_npy:do_main',
                'merge_mask_ndarrays = psana.pyalgos.app.merge_mask_ndarrays:do_main',
                'merge_max_ndarrays  = psana.pyalgos.app.merge_max_ndarrays:do_main',
                'cdb                 = psana.pscalib.app.cdb:cdb_cli',
                'proc_info           = psana.pscalib.app.proc_info:do_main',
                'proc_control        = psana.pscalib.app.proc_control:do_main',
                'proc_new_datasets   = psana.pscalib.app.proc_new_datasets:do_main',
                'timeconverter       = psana.graphqt.app.timeconverter:timeconverter',
                'calibman            = psana.graphqt.app.calibman:calibman_gui',
             ]
       },
)

CYT_BLD_DIR = 'build'

from Cython.Build import cythonize
ext = Extension("peakFinder",
                sources=["psana/peakFinder/peakFinder.pyx",
                         "../psalg/psalg/peakFinder/src/PeakFinderAlgos.cc",
                         "../psalg/psalg/peakFinder/src/LocalExtrema.cc"],
                language="c++",
                extra_compile_args=['-std=c++11'],
                include_dirs=[np.get_include(),
                              "../install/include"],
)

setup(name="peakFinder",
      ext_modules=cythonize(ext, build_dir=CYT_BLD_DIR))

ext = Extension('dgramCreate',
                #packages=['psana.peakfinder',],
                sources=["psana/peakFinder/dgramCreate.pyx"],
                libraries = ['xtc'],
                include_dirs = [np.get_include(), os.path.join(xtcdata, 'include')],
                library_dirs = [os.path.join(xtcdata, 'lib')],
                language="c++",
                extra_compile_args=['-std=c++11'],
                extra_link_args = ['-Wl,-rpath='+ os.path.abspath(os.path.join(xtcdata, 'lib'))],
                # include_dirs=[np.get_include(),
                              # "../install/include"]
)

setup(name='dgramCreate',
      ext_modules=cythonize(ext, build_dir=CYT_BLD_DIR))

setup(name='dgramchunk',
      ext_modules = cythonize(Extension(
                    "psana.dgramchunk",
                    sources=["src/dgramchunk.pyx"],
      ), build_dir=CYT_BLD_DIR))

setup(name='smdreader',
      ext_modules = cythonize(Extension(
                    "psana.smdreader",
                    sources=["psana/smdreader.pyx"],
                    include_dirs=["psana"],
      ), build_dir=CYT_BLD_DIR))

setup(name='eventbuilder', 
      ext_modules = cythonize(Extension(
                    "psana.eventbuilder",                                 
                    sources=["psana/eventbuilder.pyx"],  
                    include_dirs=["psana"],
      ), build_dir=CYT_BLD_DIR))

ext = Extension("hsd",
                sources=["psana/hsd/hsd.pyx",
                         "../psalg/psalg/peakFinder/src/PeakFinderAlgos.cc",
                         "../psalg/psalg/peakFinder/src/LocalExtrema.cc"],
                libraries=['xtc','psalg'],
                language="c++",
                extra_compile_args=['-std=c++11'],
                include_dirs=[np.get_include(),
                              "../install/include",
                              "../psdaq/psdaq/hsd",
                              os.path.join(xtcdata, 'include')],
                library_dirs = [os.path.join(xtcdata, 'lib')],
                extra_link_args = ['-Wl,-rpath='+ os.path.abspath(os.path.join(xtcdata, 'lib'))],
)

setup(name="hsd",
      ext_modules=cythonize(ext, build_dir=CYT_BLD_DIR),
)

'''
from setuptools.command.build_ext import build_ext
class dgram_build_ext(build_ext):
    user_options = build_ext.user_options
    user_options.extend([('xtcdata=', None, 'base folder of xtcdata installation')])

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.xtcdata = None
    
    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.xtcdata is not None:
            for ext in self.extensions:
                ext.library_dirs.append(os.path.join(self.xtcdata, 'lib'))
                ext.include_dirs.append(os.path.join(self.xtcdata, 'include'))
                ext.extra_link_args.append('-Wl,-rpath='+ os.path.join(self.xtcdata, 'lib'))
        else:
            print('missing')
            #raise Exception("Parameter --xtcdata is missing")
        print(self.extensions)
'''   
