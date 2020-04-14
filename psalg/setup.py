import sys
from setuptools import setup, find_packages


# allows a version number to be passed to the setup
VERSION = '0.0.0'
arg = [arg for arg in sys.argv if arg.startswith('--version')]
if arg:
    VERSION = arg[0].split('=')[1]
    sys.argv.remove(arg[0])


setup(
       name = 'psalg',
       version = VERSION,
       license = 'LCLS II',
       description = 'LCLS II DAQ/ANA base package',

       packages = find_packages(),

       entry_points={
            'console_scripts': [
                'configdb = psalg.configdb.configdb:main',
                'syslog = psalg.utils.syslog:main',
              ]
       },
)
