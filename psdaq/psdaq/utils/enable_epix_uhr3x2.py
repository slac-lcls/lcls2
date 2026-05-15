import pyrogue as pr
import os
import sys

# submoduledir = os.environ["SUBMODULEDIR"]
# submoduledir = "/cds/sw/package/conda_envs/dorlhiac"
submoduledir = "/cds/home/d/dorlhiac/Repos"

top_level = submoduledir + "/epixuhr-3x2-readout-testing/"

sys.path.append("%s/firmware/python".format(top_level))

pr.addLibraryPath(top_level+'firmware/submodules/AsicRegMapping/python')
pr.addLibraryPath(top_level+'firmware/submodules/axi-pcie-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/pixel-camera-readout-common/python')
pr.addLibraryPath(top_level+'firmware/submodules/lcls-timing-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/l2si-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/surf/python')
pr.addLibraryPath(top_level+'firmware/submodules/ePixViewer/python')
pr.addLibraryPath(top_level+'firmware/submodules/hdl-cores-lib-dev/firmware/python')
pr.addLibraryPath(top_level+'firmware/submodules/lcls2-pgp-fw-lib/python')
pr.addLibraryPath(top_level+'firmware/python')

