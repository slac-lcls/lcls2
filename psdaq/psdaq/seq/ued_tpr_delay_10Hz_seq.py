"""
software imperfections:
running out out ian and chris's home directories
andor epics prefix is hardwired to 3
the xpm epix prefix is hardwired 
we should use argparse instead of sys.argv
we dont check the status of the seqprogram command

experimenter complexities:
relies on the user setting up the daq environment
need to stop daq to run script
BlueSky number of events must be set by the user to L+M+1+N
"""


import os
import sys
import epics
from psdaq.cas.pvedit import Pv

pre_delay = int(sys.argv[1])
L_dark = int(sys.argv[2])
M_pre = int(sys.argv[3])
N_post = int(sys.argv[4])

"""
Trigger execution delay time and duration for shutters in single-shot
experiment.

Args:
- pre_delay: Number of 'event' bin lengths to delay before starting the
  sequence. Defaults to 1=100ms
- L: Number of Dark shots
- M: Number of pre ebeam shots
- N: Number of post ebeam shots
"""

ebeam_delay = L_dark + pre_delay + 0.4 # 90ms to open Lambda shutter
ebeam_width = M_pre + 1.0 + N_post
laser_delay = L_dark + M_pre + pre_delay + 0.85 # 15ms to open Uniblitz
laser_width = 0.2 # 0.2 10Hz shots = 20ms

# Assumes a 10Hz event rate
event_to_ns_factor = 1E+8 # 100ms

ebeam_delay = int(round(ebeam_delay * event_to_ns_factor))
ebeam_width = int(round(ebeam_width * event_to_ns_factor))
laser_delay = int(round(laser_delay * event_to_ns_factor))
laser_width = int(round(laser_width * event_to_ns_factor))

# TPR channel 5,6 (shutter) Enabled and Normal just in case the scientists
# have disabled to keep shutters open or closed
epics.caput("UED:CAM:TPR:01:CH05_SYS2_TCTL", "Enabled")
epics.caput("UED:CAM:TPR:01:TRG05_TPOL", "Rising Edge")
epics.caput("UED:CAM:TPR:01:CH06_SYS2_TCTL", "Enabled")
epics.caput("UED:CAM:TPR:01:TRG06_TPOL", "Rising Edge")

#Calculated shutter width and delay values
epics.caput("UED:CAM:TPR:01:TRG06_SYS2_TWID", ebeam_width)
epics.caput("UED:CAM:TPR:01:TRG06_SYS2_TDES", ebeam_delay)
epics.caput("UED:CAM:TPR:01:TRG05_SYS2_TWID", laser_width)
epics.caput("UED:CAM:TPR:01:TRG05_SYS2_TDES", laser_delay)

pvSeqMask=Pv("DAQ:UED:XPM:0:PART:0:SeqMask")
# causes two engines to start at the same time (3 is 0b11 in binary)
pvSeqMask.put(3)

f=open("/path/to/ued/daq/release/psdaq/psdaq/seq/ued_engine_10hz.py")
enginelines=f.read()
f.close()

for engine in range(2):
    firstline = f"pre_delay={pre_delay}; L_dark={L_dark}; M_pre={M_pre}; N_post={N_post}; engine={engine}\n"
    f=open(f"engine{engine}.py", "w")
    f.write(firstline)
    f.write(enginelines)
    f.close()

os.system("seqprogram --seq 0:engine0.py 1:engine1.py --pv DAQ:UED:XPM:0")

epics.caput("UED:ANDOR:CAM:03:Acquire", 0)
epics.caput("UED:ANDOR:CAM:03:AcquireTime", .002) # 2ms exposure
epics.caput("UED:ANDOR:CAM:03:Acquire", 1)
