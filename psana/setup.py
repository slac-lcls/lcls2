import os
import sys
import numpy as np
from setuptools import setup, Extension #, find_packages

arg = [arg for arg in sys.argv if arg.startswith('--xtcdata')]
if not arg:
    raise Exception('Parameter --xtcdata is missing')
xtcdata = arg[0].split('=')[1]
sys.argv.remove(arg[0])

dgram_module = Extension('psana.dgram',
                         sources = ['src/dgram.cc'],
                         libraries = ['xtcdata'],
                         include_dirs = [np.get_include(), os.path.join(xtcdata, 'include')],
                         library_dirs = [os.path.join(xtcdata, 'lib')],
                         extra_link_args = ['-Wl,-rpath='+ os.path.join(xtcdata, 'lib')],
                         extra_compile_args=['-std=c++11'])

setup(name = 'psana',
       version = '0.1',
       description = 'LCLS II analysis package',
       #include_package_data = True,
       #packages = find_packages(),
       packages = ['psana', 'psana.detector', 'psana.pscalib', 'psana.pyalgos'],
       #cmdclass = {'build': dgram_build, 'build_ext': dgram_build_ext},
       ext_modules = [dgram_module])




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
