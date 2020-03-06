# this sys.path.append is a hack to allow each package to import its
# own version of device-specific submodules (surf/axi-pcie-core etc.) that
# have been placed as subdirectories here by setup.py.  ryan herbst
# thinks of these packages as a device-specific "board support package".
# this allows one to put multiple devices in the same conda env.
# a cleaner approach would be to use relative imports everywhere, but
# that would be a lot of work for the tid-air people - cpo.
import sys
import os
# extra wrinkle: pick up the pyxpm's version of the packages to avoid
# duplication (currently pyxpm/pykcu can use the same version of surf)
sys.path.insert(0,os.path.join(os.path.dirname(os.path.realpath(__file__)),'../pyxpm'))
