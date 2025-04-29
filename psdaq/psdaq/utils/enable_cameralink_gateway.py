import pyrogue as pr
import os
import sys

submoduledir = os.environ["SUBMODULEDIR"]

top_level = submoduledir + "/cameralink-gateway/"

sys.path.append("%s/firmware/python".format(top_level))

pr.addLibraryPath(top_level+'firmware/submodules/axi-pcie-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/lcls2-pgp-fw-lib/python')
pr.addLibraryPath(top_level+'firmware/submodules/lcls-timing-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/l2si-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/clink-gateway-fw-lib/python')
pr.addLibraryPath(top_level+'firmware/submodules/surf/python')
pr.addLibraryPath(top_level+'firmware/python')
