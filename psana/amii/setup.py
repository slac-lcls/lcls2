import os
import re
from setuptools import setup


def get_version(pkg):
    """Scrap __version__  from __init__.py"""
    vfilename = os.path.join(os.getcwd(), pkg, '__init__.py')
    vfile = open(vfilename).read()
    m = re.search(r'__version__ = (\S+)\n', vfile)
    if m is None or len(m.groups()) != 1:
        raise Exception("Cannot determine __version__ from init file: '%s'!" % vfilename)
    version = m.group(1).strip('\'\"')
    return version

setup(
    name='ami',
    version=get_version('ami'),
    description='LCLS analysis monitoring',
    long_description='The package used at LCLS-II for online analysis monitoring',
    author='Daniel Damiani',
    author_email='ddamiani@slac.stanford.edu',
    url='https://confluence.slac.stanford.edu/display/PSDMInternal/AMI+Replacement',
    package_dir={'ami.examples': 'examples'},
    packages=['ami', 'ami.operation', 'ami.examples'],
    install_requires=[
        'pyzmq',
        'numpy',
        'ipython',
    ],
    entry_points={
        'console_scripts': [
            'ami-worker = ami.worker:main',
            'ami-manager = ami.manager:main',
            'ami-cleint = ami.client:main',
        ]
    },
    classifiers=[
        'Development Status :: 1 - Planning'
        'Environment :: Other Environment',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Utilities',
    ],
)
