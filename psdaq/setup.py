from setuptools import setup, find_packages

setup(
       name = 'psdaq',
       license = 'LCLS II',
       description = 'LCLS II DAQ package',

       packages = find_packages(),
       package_data={'control_gui': ['data/icons/*.png','data/icons/*.gif'],},

       scripts = ['psdaq/procmgr/procmgr','psdaq/procmgr/procstat','psdaq/procmgr/condaProcServ'],

       entry_points={
            'console_scripts': [
                'control = psdaq.control.control:main',
                'selectPlatform = psdaq.control.selectPlatform:main',
                'showPlatform = psdaq.control.showPlatform:main',
                'daqstate = psdaq.control.daqstate:main',
                'currentexp = psdaq.control.currentexp:main',
                'testClient2 = psdaq.control.testClient2:main',
                'testAsyncErr = psdaq.control.testAsyncErr:main',
                'getrun = psdaq.control.getrun:main',
                'groupca = psdaq.cas.groupca:main',
                'partca = psdaq.cas.partca:main',
                'partcas = psdaq.cas.partcas:main',
                'modcas = psdaq.cas.modcas:main',
                'modca = psdaq.cas.modca:main',
                'xpmca = psdaq.cas.xpmca:main',
                'deadca = psdaq.cas.deadca:main',
                'dtica = psdaq.cas.dtica:main',
                'dticas = psdaq.cas.dticas:main',
                'hsdca = psdaq.cas.hsdca:main',
                'hsdcas = psdaq.cas.hsdcas:main',
                'hsdpva = psdaq.cas.hsdpva:main',
                'hsdpvs = psdaq.cas.hsdpvs:main',
                'campvs = psdaq.cas.campvs:main',
                'tprca = psdaq.cas.tprca:main',
                'tprcas = psdaq.cas.tprcas:main',
                'xpmioc = psdaq.cas.xpmioc:main',
                'bldcas = psdaq.cas.bldcas:main',
                'hpsdbuscas = psdaq.cas.hpsdbuscas:main',
                'pyxpm = psdaq.pyxpm.pyxpm:main',
                'pykcu = psdaq.pykcu.pykcu:main',
                'control_gui = psdaq.control_gui.app.control_gui:control_gui',
                'bluesky_simple = psdaq.control.bluesky_simple:main',
              ]
       },
)
