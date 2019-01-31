from setuptools import setup, find_packages
import versioneer

setup(
       name = 'psdaq',
       license = 'LCLS II',
       description = 'LCLS II DAQ package',

       version=versioneer.get_version(),
       cmdclass=versioneer.get_cmdclass(),
       packages = find_packages(),

       scripts = ['psdaq/procmgr/procmgr','psdaq/procmgr/procstat','psdaq/procmgr/condaProcServ'],

       entry_points={
            'console_scripts': [
                'collection = psdaq.control.collection:main',
                'dti_proxy = psdaq.control.dti_proxy:main',
                'showPlatform = psdaq.control.showPlatform:main',
                'partca = psdaq.cas.partca:main',
                'partcas = psdaq.cas.partcas:main',
                'modcas = psdaq.cas.modcas:main',
                'xpmca = psdaq.cas.xpmca:main',
                'deadca = psdaq.cas.deadca:main',
                'dtica = psdaq.cas.dtica:main',
                'dticas = psdaq.cas.dticas:main',
                'hsdca = psdaq.cas.hsdca:main',
                'hsdcas = psdaq.cas.hsdcas:main',
                'control_gui = psdaq.control_gui.app.control_gui:control_gui',
              ]
       },
)
