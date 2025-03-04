import pyrogue as pr
import os
import sys

submoduledir = os.environ["SUBMODULEDIR"]

top_level = submoduledir + "/epix_hr_single_10k/"

sys.path.append("%s/firmware/python".format(top_level))

pr.addLibraryPath(top_level + 'firmware/submodules/axi-pcie-core/python')
pr.addLibraryPath(top_level + 'firmware/submodules/epix-hr-core/python')
pr.addLibraryPath(top_level + 'firmware/submodules/surf/python')
pr.addLibraryPath(top_level + 'firmware/python')
pr.addLibraryPath(top_level + 'software/python')
pr.addLibraryPath(top_level + 'firmware/submodules/l2si-core/python')
pr.addLibraryPath(top_level + 'firmware/submodules/lcls-timing-core/python')
try :
    pr.addLibraryPath(top_level+'firmware/submodules/ePixViewer/python')
except :
    print("pr.addLibraryPath Import ePixViewer failed")
    pass
