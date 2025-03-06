import pyrogue as pr
import os
import sys

submoduledir = os.environ["SUBMODULEDIR"]

top_level = submoduledir + "/epix-uhr-gtreadout-dev/"

sys.path.append("%s/firmware/python".format(top_level))

pr.addLibraryPath(top_level+'firmware/submodules/AsicRegMapping/python')
pr.addLibraryPath(top_level+'firmware/submodules/axi-pcie-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/pixel-camera-readout-common/python')
pr.addLibraryPath(top_level+'firmware/submodules/lcls-timing-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/l2si-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/surf/python')
pr.addLibraryPath(top_level+'firmware/submodules/ePixViewer/python')
pr.addLibraryPath(top_level+'firmware/python'

