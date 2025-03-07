import pyrogue as pr
import os
import sys

submoduledir = os.environ["SUBMODULEDIR"]

top_level = submoduledir + "/lcls2_timetool/"

sys.path.append("%s/firmware/python".format(top_level))


