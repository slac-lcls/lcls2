import pyrogue as pr
import os
import sys

submoduledir = os.environ["SUBMODULEDIR"]

top_level = submoduledir + "/epix-hr-m-320k/"

sys.path.append("%s/firmware/python".format(top_level))

pr.addLibraryPath(top_level+'firmware/submodules/epix-hr-leap-common/python')
pr.addLibraryPath(top_level+'firmware/submodules/epix-hr-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/lcls-timing-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/l2si-core/python')
pr.addLibraryPath(top_level+'firmware/submodules/surf/python')
pr.addLibraryPath(top_level+'firmware/python')
pr.addLibraryPath(top_level+'firmware/python/ePixViewer/software')