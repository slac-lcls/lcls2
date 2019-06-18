from setuptools import setup, find_packages
import versioneer

setup(
       name = 'psalg',
       license = 'LCLS II',
       description = 'LCLS II DAQ/ANA base package',

       packages = find_packages(),
)
