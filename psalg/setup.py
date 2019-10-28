from setuptools import setup, find_packages

setup(
       name = 'psalg',
       license = 'LCLS II',
       description = 'LCLS II DAQ/ANA base package',

       packages = find_packages(),

       entry_points={
            'console_scripts': [
                'syslog = psalg.utils.syslog:main',
              ]
       },
)
