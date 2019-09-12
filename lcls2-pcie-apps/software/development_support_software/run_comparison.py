#####################################################
#####################################################
######copy pasta usage example below#################
#####################################################
#####################################################

import compare_pyrogue_vdhl_memory_offsets
import importlib
import TimeToolDev
import sys
import time
import gc


cl = TimeToolDev.TimeToolDev(True)
cl.stop()

importlib.reload(compare_pyrogue_vdhl_memory_offsets)
my_dict = {}
compare_pyrogue_vdhl_memory_offsets.recursive_memory_validator(cl,0,my_dict,"","","")
my_object_names = [i for i in my_dict if "110" in my_dict[i]]

