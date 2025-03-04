import pyrogue as pr
import os
import sys

submoduledir = os.environ["SUBMODULEDIR"]

top_level = submoduledir + "/epix_quad/"
pr.addLibraryPath(top_level + 'firmware/submodules/axi-pcie-core/python')
pr.addLibraryPath(top_level + 'firmware/submodules/surf/python')
pr.addLibraryPath(top_level + 'firmware/python')
pr.addLibraryPath(top_level + 'software/python')