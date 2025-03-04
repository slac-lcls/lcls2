import pyrogue as pr
import os
import sys

submoduledir = os.environ["SUBMODULEDIR"]

top_level = submoduledir + "/epix_100a_gen2/"

sys.path.append("%s/firmware/python".format(top_level))

pr.addLibraryPath(top_level+'firmware/submodules/surf/python')
pr.addLibraryPath(top_level+'software/python')
