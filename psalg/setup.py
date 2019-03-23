from setuptools import setup, find_packages
import versioneer

setup(
       name = 'psalg',
       license = 'LCLS II',
       description = 'LCLS II DAQ/ANA base package',

       version=versioneer.get_version(),
       cmdclass=versioneer.get_cmdclass(),
       packages = find_packages(),
)
