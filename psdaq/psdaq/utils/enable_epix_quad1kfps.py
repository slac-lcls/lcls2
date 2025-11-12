import pyrogue as pr
import os
import sys

submoduledir = os.environ["SUBMODULEDIR"]

top_level = submoduledir + "/epix-quad-1kfps/"

sys.path.append("%s/firmware/python" % top_level)

pr.addLibraryPath(top_level + 'firmware/submodules/axi-pcie-core/python')
pr.addLibraryPath(top_level + 'firmware/submodules/surf/python')
pr.addLibraryPath(top_level + 'firmware/python')
pr.addLibraryPath(top_level + 'software/python')