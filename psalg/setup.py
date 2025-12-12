import os
import sys
from setuptools import setup, find_packages


# allows a version number to be passed to the setup
VERSION = '0.0.0'
version_env = os.environ.get('VERSION')
if version_env:
    VERSION = version_env


setup(
       name = 'psalg',
       version = VERSION,
       license = 'LCLS II',
       description = 'LCLS II DAQ/ANA base package',

       packages = find_packages(),

       entry_points={
            'console_scripts': [
                'syslog   = psalg.utils.syslog:main',
                'daqPipes = psalg.daqPipes.daqPipes:main',
                'daqStats = psalg.daqPipes.daqStats:main',
              ]
       },
)
